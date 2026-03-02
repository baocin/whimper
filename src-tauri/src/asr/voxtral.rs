//! Voxtral-Mini-4B Realtime ASR module
//!
//! Ported from timeline_app. Uses Burn/wgpu GPU backend for streaming ASR.
//! Q4_0 GGUF quantization (~2.51 GB).

use anyhow::{anyhow, Result};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;
use tracing::info;

use burn::backend::Wgpu;
use burn::tensor::{Tensor, TensorData};
use voxtral_mini_realtime::{
    audio::{pad_audio, AudioBuffer, MelConfig, MelSpectrogram, PadConfig},
    gguf::{Q4ModelLoader, Q4VoxtralModel},
    models::time_embedding::TimeEmbedding,
    tokenizer::VoxtralTokenizer,
};

type WgpuBackend = Wgpu;

/// Transcription result from batch mode
#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub text: String,
    pub processing_time_ms: u64,
    pub audio_duration_ms: u64,
}

/// Voxtral-Mini-4B ASR engine
pub struct VoxtralAsr {
    model: Mutex<Option<Q4VoxtralModel>>,
    tokenizer: Mutex<Option<VoxtralTokenizer>>,
    mel_extractor: MelSpectrogram,
    pad_config: PadConfig,
    time_embed: Tensor<WgpuBackend, 3>,
    transcript_parts: Mutex<Vec<String>>,
}

impl VoxtralAsr {
    pub fn new() -> Self {
        let device = burn::backend::wgpu::WgpuDevice::default();
        let time_embedding = TimeEmbedding::new(3072);
        let t_embed = time_embedding.embed::<WgpuBackend>(6.0, &device);
        Self {
            model: Mutex::new(None),
            tokenizer: Mutex::new(None),
            mel_extractor: MelSpectrogram::new(MelConfig::voxtral()),
            pad_config: PadConfig::voxtral(),
            time_embed: t_embed,
            transcript_parts: Mutex::new(Vec::new()),
        }
    }

    /// Load model from directory containing voxtral-q4.gguf and tekken.json
    pub fn load_model(&self, model_dir: &str) -> Result<()> {
        let dir = PathBuf::from(model_dir);

        let gguf_path = dir.join("voxtral-q4.gguf");
        let tokenizer_path = dir.join("tekken.json");

        if !gguf_path.exists() {
            return Err(anyhow!("GGUF model not found at {:?}", gguf_path));
        }
        if !tokenizer_path.exists() {
            return Err(anyhow!("Tokenizer not found at {:?}", tokenizer_path));
        }

        info!("Loading Voxtral-Mini-4B from {:?}", dir);
        let start = Instant::now();

        let device = burn::backend::wgpu::WgpuDevice::default();
        let mut loader = Q4ModelLoader::from_file(&gguf_path)
            .map_err(|e| anyhow!("Failed to open GGUF: {}", e))?;

        let model = loader
            .load(&device)
            .map_err(|e| anyhow!("Failed to load Q4 model: {}", e))?;

        let tokenizer = VoxtralTokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        let elapsed = start.elapsed();
        info!("Voxtral-Mini-4B loaded in {:.1}s", elapsed.as_secs_f32());

        *self.model.lock().unwrap() = Some(model);
        *self.tokenizer.lock().unwrap() = Some(tokenizer);

        Ok(())
    }

    pub fn is_loaded(&self) -> bool {
        self.model.lock().unwrap().is_some()
    }

    /// Build mel spectrogram tensor from f32 samples (16kHz mono)
    fn mel_tensor(&self, samples: &[f32]) -> Result<Tensor<WgpuBackend, 3>> {
        let device = burn::backend::wgpu::WgpuDevice::default();
        let audio = AudioBuffer::new(samples.to_vec(), 16000);
        let padded = pad_audio(&audio, &self.pad_config);
        let mel = self.mel_extractor.compute_log(&padded.samples);
        let n_frames = mel.len();
        let n_mels = if n_frames > 0 { mel[0].len() } else { 0 };

        if n_frames == 0 {
            return Err(anyhow!("Audio too short to produce mel frames"));
        }

        let mut mel_transposed = vec![vec![0.0f32; n_frames]; n_mels];
        for (frame_idx, frame) in mel.iter().enumerate() {
            for (mel_idx, &val) in frame.iter().enumerate() {
                mel_transposed[mel_idx][frame_idx] = val;
            }
        }
        let mel_flat: Vec<f32> = mel_transposed.into_iter().flatten().collect();
        Ok(Tensor::from_data(
            TensorData::new(mel_flat, [1, n_mels, n_frames]),
            &device,
        ))
    }

    /// Filter control tokens and decode to text
    fn decode_tokens(&self, tokenizer: &VoxtralTokenizer, generated: &[i32]) -> Result<String> {
        let text_tokens: Vec<u32> = generated
            .iter()
            .filter(|&&t| t >= 1000)
            .map(|&t| t as u32)
            .collect();
        tokenizer
            .decode(&text_tokens)
            .map_err(|e| anyhow!("Token decoding failed: {}", e))
    }

    /// Transcribe complete audio (16kHz mono f32) — batch mode
    pub fn transcribe(&self, samples: &[f32]) -> Result<TranscriptionResult> {
        let model = self.model.lock().unwrap();
        let model = model
            .as_ref()
            .ok_or_else(|| anyhow!("Voxtral model not loaded"))?;
        let tokenizer = self.tokenizer.lock().unwrap();
        let tokenizer = tokenizer
            .as_ref()
            .ok_or_else(|| anyhow!("Voxtral tokenizer not loaded"))?;

        let audio_duration_ms = (samples.len() as f64 / 16.0) as u64;
        let start = Instant::now();

        let mel_tensor = self.mel_tensor(samples)?;
        let tokens = model.transcribe_streaming(mel_tensor, self.time_embed.clone());
        let text = self.decode_tokens(tokenizer, &tokens)?;

        let processing_time_ms = start.elapsed().as_millis() as u64;

        Ok(TranscriptionResult {
            text: text.trim().to_string(),
            processing_time_ms,
            audio_duration_ms,
        })
    }

    /// Transcribe a single audio chunk (peak-normalized, 16kHz mono f32).
    /// Returns the decoded text for this chunk only.
    pub fn transcribe_chunk(&self, samples: &[f32]) -> Result<String> {
        let model = self.model.lock().unwrap();
        let model = model
            .as_ref()
            .ok_or_else(|| anyhow!("Voxtral model not loaded"))?;
        let tokenizer = self.tokenizer.lock().unwrap();
        let tokenizer = tokenizer
            .as_ref()
            .ok_or_else(|| anyhow!("Voxtral tokenizer not loaded"))?;

        let inference_start = Instant::now();
        let mel_tensor = self.mel_tensor(samples)?;
        let new_tokens = model.transcribe_streaming(mel_tensor, self.time_embed.clone());
        let inference_ms = inference_start.elapsed().as_millis();
        let audio_duration_ms = (samples.len() as f64 / 16.0) as u64;

        let text_count = new_tokens.iter().filter(|&&t| t >= 1000).count();
        let control_count = new_tokens.len() - text_count;
        tracing::info!(
            total = new_tokens.len(),
            text_tokens = text_count,
            control_tokens = control_count,
            "ASR raw tokens (first 20): {:?}",
            &new_tokens[..new_tokens.len().min(20)]
        );
        tracing::info!(
            "ASR inference: {}ms for {}ms audio, {} tokens returned",
            inference_ms,
            audio_duration_ms,
            new_tokens.len()
        );

        let text = self.decode_tokens(tokenizer, &new_tokens)?;
        Ok(text.trim().to_string())
    }

    /// Process a single audio chunk: transcribe it and append to accumulated transcript.
    /// Returns the full transcript so far (all chunks joined).
    pub fn process_audio_chunk(&self, samples: &[f32]) -> Result<String> {
        let chunk_text = self.transcribe_chunk(samples)?;

        let mut parts = self.transcript_parts.lock().unwrap();
        if !chunk_text.is_empty() {
            parts.push(chunk_text);
        }

        Ok(parts.join(" "))
    }

    /// Get current accumulated transcript
    pub fn current_transcript(&self) -> String {
        let parts = self.transcript_parts.lock().unwrap();
        parts.join(" ")
    }

    /// Reset streaming state for a new recording session
    pub fn reset(&self) {
        self.transcript_parts.lock().unwrap().clear();
    }
}

unsafe impl Send for VoxtralAsr {}
unsafe impl Sync for VoxtralAsr {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voxtral_asr_creation() {
        let asr = VoxtralAsr::new();
        assert!(!asr.is_loaded());
    }

    #[test]
    fn test_voxtral_asr_not_loaded_error() {
        let asr = VoxtralAsr::new();
        let result = asr.transcribe(&[0.0f32; 16000]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not loaded"));
    }

    #[test]
    fn test_voxtral_streaming_reset() {
        let asr = VoxtralAsr::new();
        asr.reset();
        assert!(asr.current_transcript().is_empty());
    }
}
