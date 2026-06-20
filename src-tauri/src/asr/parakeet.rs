//! Parakeet TDT 0.6B v2 INT8 - Automatic Speech Recognition
//!
//! Direct ORT inference with TDT greedy decoding.
//! The TDT architecture has a joiner with 1030-dimensional output:
//! - 1025 vocabulary logits (1024 tokens + blank)
//! - 5 duration logits (skip 1-5 frames)

use super::mel_features::{compute_log_mel, MelSpectrogramConfig};
use super::preprocess::{highpass_80hz, peak_normalize, trim_silence};
use super::vad::SileroVad;
use anyhow::{anyhow, Result};
use ndarray::Array3;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

const ENCODER_DIM: usize = 1024;
const DECODER_DIM: usize = 640;
const VOCAB_SIZE: usize = 1025; // 1024 tokens + blank
const NUM_DURATIONS: usize = 5;
const BLANK_ID: usize = 1024;

struct TdtModel {
    encoder: Session,
    decoder: Session,
    joiner: Session,
    vocab: Vec<String>,
}

/// Physical CPU core count, used to size ONNX Runtime's intra-op thread pool.
/// INT8 CPU inference scales with physical (not logical) cores; oversubscribing
/// with hyperthreads adds sync overhead without throughput. Falls back
/// conservatively if detection fails.
fn physical_cores() -> usize {
    sysinfo::System::new()
        .physical_core_count()
        .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
        .unwrap_or(4)
        .max(1)
}

/// Read a usize from an env var, falling back to `default` if unset/invalid.
fn env_threads(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(default)
}

/// Build an ONNX Runtime session with full graph optimization and a fixed
/// intra-op thread count. Inter-op threads are left at 1: execution is
/// sequential, so inter-op parallelism would only add idle pool threads.
fn build_session(path: &Path, intra_threads: usize) -> Result<Session> {
    Session::builder()
        .map_err(|e| anyhow!("Session builder error: {}", e))?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|e| anyhow!("Optimization level error: {}", e))?
        .with_intra_threads(intra_threads)
        .map_err(|e| anyhow!("Thread config error: {}", e))?
        .commit_from_file(path)
        .map_err(|e| anyhow!("Failed to load {:?}: {}", path, e))
}

fn load_vocabulary(tokens_path: &PathBuf) -> Result<Vec<String>> {
    let content = fs::read_to_string(tokens_path)
        .map_err(|e| anyhow!("Failed to read tokens.txt: {}", e))?;

    let mut vocab = Vec::with_capacity(VOCAB_SIZE);
    for line in content.lines() {
        let token = line.split_whitespace().next().unwrap_or("").to_string();
        vocab.push(token);
    }

    if vocab.len() < VOCAB_SIZE {
        return Err(anyhow!(
            "Vocabulary too small: {} < {}",
            vocab.len(),
            VOCAB_SIZE
        ));
    }

    Ok(vocab)
}

/// Transcription result with timing info
#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub text: String,
    pub processing_time_ms: u64,
    pub audio_duration_ms: u64,
}

/// Parakeet TDT ASR engine using direct ORT inference
pub struct ParakeetAsr {
    model: Mutex<Option<TdtModel>>,
    vad: Mutex<SileroVad>,
}

impl ParakeetAsr {
    pub fn new() -> Self {
        Self {
            model: Mutex::new(None),
            vad: Mutex::new(SileroVad::new()),
        }
    }

    /// Load model from directory containing encoder, decoder, joiner, and tokens
    pub fn load_model(&self, model_dir: &str) -> Result<()> {
        let mut model_guard = self
            .model
            .lock()
            .map_err(|e| anyhow!("Model mutex poisoned: {}", e))?;
        if model_guard.is_some() {
            return Ok(());
        }

        let models_dir = PathBuf::from(model_dir);

        let encoder_path = if models_dir.join("encoder.int8.onnx").exists() {
            models_dir.join("encoder.int8.onnx")
        } else {
            models_dir.join("encoder.onnx")
        };

        let decoder_path = if models_dir.join("decoder.int8.onnx").exists() {
            models_dir.join("decoder.int8.onnx")
        } else {
            models_dir.join("decoder.onnx")
        };

        let joiner_path = if models_dir.join("joiner.int8.onnx").exists() {
            models_dir.join("joiner.int8.onnx")
        } else {
            models_dir.join("joiner.onnx")
        };

        let tokens_path = models_dir.join("tokens.txt");

        tracing::info!("Loading Parakeet TDT model from {:?}...", models_dir);

        for (name, path) in [
            ("encoder", &encoder_path),
            ("decoder", &decoder_path),
            ("joiner", &joiner_path),
            ("tokens", &tokens_path),
        ] {
            if !path.exists() {
                return Err(anyhow!("Parakeet TDT {} not found at {:?}", name, path));
            }
        }

        let vocab = load_vocabulary(&tokens_path)?;
        tracing::info!("Loaded vocabulary: {} tokens", vocab.len());

        // Thread sizing: the encoder is one big batched op that scales with
        // physical cores; the decoder and joiner are tiny tensors invoked
        // hundreds of times in the greedy loop, where a large thread pool only
        // adds per-call sync overhead — so they get a small fixed count.
        // Both are overridable via env for benchmarking.
        let cores = physical_cores();
        let enc_threads = env_threads("WHIMPER_ENCODER_THREADS", cores);
        let small_threads = env_threads("WHIMPER_DECODER_THREADS", 2);
        tracing::info!(
            "ONNX sessions: encoder intra-threads={}, decoder/joiner intra-threads={} (physical_cores={}, opt=All)",
            enc_threads,
            small_threads,
            cores
        );

        let encoder = build_session(&encoder_path, enc_threads)
            .map_err(|e| anyhow!("encoder: {}", e))?;
        let decoder = build_session(&decoder_path, small_threads)
            .map_err(|e| anyhow!("decoder: {}", e))?;
        let joiner = build_session(&joiner_path, small_threads)
            .map_err(|e| anyhow!("joiner: {}", e))?;

        // Load Silero VAD for silence trimming
        if let Ok(mut vad_guard) = self.vad.lock() {
            if !vad_guard.is_loaded() {
                if let Err(e) = vad_guard.load_embedded() {
                    tracing::warn!("Failed to load Silero VAD, trimming disabled: {}", e);
                }
            }
        }

        tracing::info!("Parakeet TDT model loaded successfully");

        *model_guard = Some(TdtModel {
            encoder,
            decoder,
            joiner,
            vocab,
        });

        Ok(())
    }

    pub fn is_loaded(&self) -> bool {
        self.model.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    /// Transcribe audio samples (16kHz mono f32)
    pub fn transcribe(&self, samples: &[f32]) -> Result<TranscriptionResult> {
        if samples.is_empty() {
            return Err(anyhow!("Audio samples are empty"));
        }

        let audio_duration_ms = (samples.len() as f32 / 16.0) as u64;
        let start = Instant::now();

        // Preprocessing: highpass → normalize → trim silence
        let skip_preprocess = std::env::var("WHIMPER_SKIP_PREPROCESS").is_ok();
        let mut buf = samples.to_vec();

        if skip_preprocess {
            tracing::info!("WHIMPER_SKIP_PREPROCESS set — bypassing highpass/normalize/trim");
        } else {
            highpass_80hz(&mut buf);
            peak_normalize(&mut buf);
        }

        let raw_peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let processed_peak = buf.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        tracing::info!(
            "Audio stats: raw_len={} processed_len={} raw_peak={:.4} processed_peak={:.4}",
            samples.len(),
            buf.len(),
            raw_peak,
            processed_peak
        );

        let processed = if skip_preprocess {
            &buf[..]
        } else if let Ok(mut vad_guard) = self.vad.lock() {
            if vad_guard.is_loaded() {
                let (trim_start, trim_end) = trim_silence(&buf, &mut vad_guard);
                if trim_start >= trim_end {
                    tracing::debug!("No speech detected after VAD trimming");
                    return Ok(TranscriptionResult {
                        text: String::new(),
                        processing_time_ms: start.elapsed().as_millis() as u64,
                        audio_duration_ms,
                    });
                }
                &buf[trim_start..trim_end]
            } else {
                &buf[..]
            }
        } else {
            &buf[..]
        };

        let mel_config = MelSpectrogramConfig::parakeet_tdt();
        let mel_features = compute_log_mel(processed, &mel_config)?;
        tracing::debug!(
            "Extracted mel features: {} frames x {} bins",
            mel_features.nrows(),
            mel_features.ncols()
        );

        let mut model_guard = self
            .model
            .lock()
            .map_err(|e| anyhow!("Model mutex poisoned: {}", e))?;
        let model = model_guard
            .as_mut()
            .ok_or_else(|| anyhow!("Model not loaded"))?;

        let encoder_out = Self::run_encoder(model, &mel_features)?;
        let token_ids = Self::tdt_greedy_decode(model, &encoder_out)?;
        let text = Self::tokens_to_text(model, &token_ids);

        let processing_time_ms = start.elapsed().as_millis() as u64;
        let rtf = if audio_duration_ms > 0 {
            processing_time_ms as f32 / audio_duration_ms as f32
        } else {
            0.0
        };

        tracing::info!(
            "Transcription: \"{}\" ({} tokens, RTF: {:.2})",
            &text[..text.len().min(80)],
            token_ids.len(),
            rtf
        );

        Ok(TranscriptionResult {
            text,
            processing_time_ms,
            audio_duration_ms,
        })
    }

    fn run_encoder(
        model: &mut TdtModel,
        mel_features: &ndarray::Array2<f32>,
    ) -> Result<Array3<f32>> {
        let (num_frames, mel_bins) = mel_features.dim();

        // Encoder expects [batch, mel_bins, time]
        let mut encoder_input_data = vec![0.0f32; mel_bins * num_frames];
        for t in 0..num_frames {
            for m in 0..mel_bins {
                encoder_input_data[m * num_frames + t] = mel_features[[t, m]];
            }
        }

        let length_data = vec![num_frames as i64];

        let encoder_value =
            Tensor::from_array(([1usize, mel_bins, num_frames], encoder_input_data))
                .map_err(|e| anyhow!("Encoder input error: {}", e))?;
        let length_value = Tensor::from_array(([1usize], length_data))
            .map_err(|e| anyhow!("Length input error: {}", e))?;

        let outputs = model
            .encoder
            .run(ort::inputs![encoder_value, length_value])
            .map_err(|e| anyhow!("Encoder failed: {}", e))?;

        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("Extract encoder output: {}", e))?;

        let enc_frames = shape[2] as usize;

        let mut encoder_out = Array3::<f32>::zeros((1, ENCODER_DIM, enc_frames));
        for i in 0..ENCODER_DIM {
            for t in 0..enc_frames {
                encoder_out[[0, i, t]] = data[i * enc_frames + t];
            }
        }

        tracing::debug!(
            "Encoder: {} mel frames -> {} encoded frames",
            num_frames,
            enc_frames
        );

        Ok(encoder_out)
    }

    fn init_decoder_states() -> (Vec<f32>, Vec<f32>) {
        let states = vec![0.0f32; 2 * DECODER_DIM];
        let slice_state = vec![0.0f32; 2 * DECODER_DIM];
        (states, slice_state)
    }

    fn run_decoder(
        model: &mut TdtModel,
        targets: &[i32],
        states: Vec<f32>,
        slice_state: Vec<f32>,
    ) -> Result<(Array3<f32>, Vec<f32>, Vec<f32>)> {
        let seq_len = targets.len().max(1);

        let targets_data: Vec<i32> = if targets.is_empty() {
            vec![0i32]
        } else {
            targets.to_vec()
        };

        let target_length = vec![seq_len as i32];

        let targets_value = Tensor::from_array(([1usize, seq_len], targets_data))
            .map_err(|e| anyhow!("Decoder targets error: {}", e))?;
        let length_value = Tensor::from_array(([1usize], target_length))
            .map_err(|e| anyhow!("Decoder length error: {}", e))?;
        let states_value = Tensor::from_array(([2usize, 1usize, DECODER_DIM], states))
            .map_err(|e| anyhow!("Decoder states error: {}", e))?;
        let slice_value = Tensor::from_array(([2usize, 1usize, DECODER_DIM], slice_state))
            .map_err(|e| anyhow!("Decoder slice error: {}", e))?;

        let outputs = model
            .decoder
            .run(ort::inputs![
                targets_value,
                length_value,
                states_value,
                slice_value
            ])
            .map_err(|e| anyhow!("Decoder failed: {}", e))?;

        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("Extract decoder output: {}", e))?;

        let out_seq = shape[2] as usize;

        let mut decoder_out = Array3::<f32>::zeros((1, DECODER_DIM, out_seq));
        for i in 0..DECODER_DIM {
            for t in 0..out_seq {
                decoder_out[[0, i, t]] = data[i * out_seq + t];
            }
        }

        let (_, states_data) = outputs[2]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("Extract decoder states: {}", e))?;
        let new_states = states_data.to_vec();

        let (_, slice_data) = outputs[3]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("Extract decoder slice: {}", e))?;
        let new_slice = slice_data.to_vec();

        Ok((decoder_out, new_states, new_slice))
    }

    fn run_joiner(
        model: &mut TdtModel,
        encoder_frame: &Array3<f32>,
        decoder_out: &Array3<f32>,
    ) -> Result<(usize, usize)> {
        let mut enc_data = vec![0.0f32; ENCODER_DIM];
        for i in 0..ENCODER_DIM {
            enc_data[i] = encoder_frame[[0, i, 0]];
        }

        let mut dec_data = vec![0.0f32; DECODER_DIM];
        for i in 0..DECODER_DIM {
            dec_data[i] = decoder_out[[0, i, 0]];
        }

        let enc_value = Tensor::from_array(([1usize, ENCODER_DIM, 1usize], enc_data))
            .map_err(|e| anyhow!("Joiner enc error: {}", e))?;
        let dec_value = Tensor::from_array(([1usize, DECODER_DIM, 1usize], dec_data))
            .map_err(|e| anyhow!("Joiner dec error: {}", e))?;

        let outputs = model
            .joiner
            .run(ort::inputs![enc_value, dec_value])
            .map_err(|e| anyhow!("Joiner failed: {}", e))?;

        let (_, logits) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("Extract joiner output: {}", e))?;

        let mut max_token_idx = 0;
        let mut max_token_val = logits[0];
        for i in 1..VOCAB_SIZE {
            if logits[i] > max_token_val {
                max_token_val = logits[i];
                max_token_idx = i;
            }
        }

        let mut max_dur_idx = 0;
        let mut max_dur_val = logits[VOCAB_SIZE];
        for i in 1..NUM_DURATIONS {
            if logits[VOCAB_SIZE + i] > max_dur_val {
                max_dur_val = logits[VOCAB_SIZE + i];
                max_dur_idx = i;
            }
        }

        let duration = if max_dur_idx == 0 { 1 } else { max_dur_idx };

        Ok((max_token_idx, duration))
    }

    fn tdt_greedy_decode(model: &mut TdtModel, encoder_out: &Array3<f32>) -> Result<Vec<usize>> {
        let enc_frames = encoder_out.shape()[2];
        let mut token_ids: Vec<usize> = Vec::new();

        let (mut states, mut slice_state) = Self::init_decoder_states();

        // Initial decoder call with blank token
        let (mut decoder_out, mut states_next, mut slice_next) =
            Self::run_decoder(model, &[BLANK_ID as i32], states.clone(), slice_state.clone())?;

        let mut t = 0;
        let max_iterations = enc_frames * 10;
        let mut iterations = 0;

        while t < enc_frames && iterations < max_iterations {
            iterations += 1;

            let mut enc_frame = Array3::<f32>::zeros((1, ENCODER_DIM, 1));
            for i in 0..ENCODER_DIM {
                enc_frame[[0, i, 0]] = encoder_out[[0, i, t]];
            }

            // Use cached decoder output for joiner
            let dec_seq_len = decoder_out.shape()[2];
            let mut dec_frame = Array3::<f32>::zeros((1, DECODER_DIM, 1));
            for i in 0..DECODER_DIM {
                dec_frame[[0, i, 0]] = decoder_out[[0, i, dec_seq_len - 1]];
            }

            let (token_idx, duration) = Self::run_joiner(model, &enc_frame, &dec_frame)?;

            if token_idx != BLANK_ID {
                token_ids.push(token_idx);
                // Accept pending states, then run decoder with new token only
                states = states_next;
                slice_state = slice_next;
                let result = Self::run_decoder(
                    model,
                    &[token_idx as i32],
                    states.clone(),
                    slice_state.clone(),
                )?;
                decoder_out = result.0;
                states_next = result.1;
                slice_next = result.2;
            }

            t += duration;
        }

        tracing::debug!(
            "TDT decode: {} frames -> {} tokens in {} iterations",
            enc_frames,
            token_ids.len(),
            iterations
        );

        Ok(token_ids)
    }

    fn tokens_to_text(model: &TdtModel, token_ids: &[usize]) -> String {
        let mut text = String::new();

        for &id in token_ids {
            if id < model.vocab.len() {
                let token = &model.vocab[id];
                if token.starts_with('▁') {
                    if !text.is_empty() {
                        text.push(' ');
                    }
                    text.push_str(&token[3..]); // Skip ▁ (3 bytes UTF-8)
                } else {
                    text.push_str(token);
                }
            }
        }

        text.trim().to_string()
    }

    /// Check if transcription is a known hallucination pattern
    pub fn is_hallucination(text: &str) -> bool {
        let text_lower = text.to_lowercase().trim().to_string();

        let hallucinations = [
            "",
            ".",
            "..",
            "...",
            "thank you",
            "thanks for watching",
            "please subscribe",
            "bye",
            "bye bye",
            "[music]",
            "[silence]",
            "(silence)",
            "(music)",
            "you",
            "the",
        ];

        hallucinations.contains(&text_lower.as_str()) || text_lower.len() < 3
    }
}

unsafe impl Send for ParakeetAsr {}
unsafe impl Sync for ParakeetAsr {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parakeet_asr_creation() {
        let asr = ParakeetAsr::new();
        assert!(!asr.is_loaded());
    }

    #[test]
    fn test_parakeet_asr_not_loaded_error() {
        let asr = ParakeetAsr::new();
        let result = asr.transcribe(&[0.0f32; 16000]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not loaded"));
    }

    #[test]
    fn test_hallucination_detection() {
        assert!(ParakeetAsr::is_hallucination(""));
        assert!(ParakeetAsr::is_hallucination("Thank you"));
        assert!(ParakeetAsr::is_hallucination("[music]"));
        assert!(ParakeetAsr::is_hallucination(".."));
        assert!(!ParakeetAsr::is_hallucination("I ordered the salmon"));
    }

    #[test]
    fn test_env_threads_parsing() {
        std::env::set_var("WHIMPER_TEST_THREADS_X", "8");
        assert_eq!(env_threads("WHIMPER_TEST_THREADS_X", 4), 8);
        std::env::remove_var("WHIMPER_TEST_THREADS_X");
        assert_eq!(env_threads("WHIMPER_TEST_THREADS_X", 4), 4);
        std::env::set_var("WHIMPER_TEST_THREADS_X", "0"); // invalid -> default
        assert_eq!(env_threads("WHIMPER_TEST_THREADS_X", 4), 4);
        std::env::remove_var("WHIMPER_TEST_THREADS_X");
    }

    #[test]
    fn test_physical_cores_at_least_one() {
        assert!(physical_cores() >= 1);
    }

    /// Real-model RTF benchmark. Ignored by default (needs the downloaded model
    /// and a raw f32le 16kHz mono audio fixture). Run e.g.:
    ///
    ///   WHIMPER_BENCH_AUDIO=/tmp/whimper_bench/jfk.f32 \
    ///     cargo test --release -p whimper bench_rtf -- --ignored --nocapture
    ///
    /// Set WHIMPER_ENCODER_THREADS to compare thread configs on one build.
    #[test]
    #[ignore]
    fn bench_rtf() {
        let audio_path = std::env::var("WHIMPER_BENCH_AUDIO")
            .expect("set WHIMPER_BENCH_AUDIO to a raw f32le 16k mono file");
        let bytes = std::fs::read(&audio_path).expect("read audio fixture");
        let samples: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        eprintln!(
            "BENCH: {} samples ({:.2}s) from {}",
            samples.len(),
            samples.len() as f32 / 16000.0,
            audio_path
        );

        let model_dir = crate::state::model_dir();
        let asr = ParakeetAsr::new();

        let load_start = Instant::now();
        asr.load_model(&model_dir.to_string_lossy())
            .expect("load model");
        eprintln!(
            "BENCH: cold-start model load = {} ms",
            load_start.elapsed().as_millis()
        );

        for i in 0..3 {
            let r = asr.transcribe(&samples).expect("transcribe");
            let rtf = r.processing_time_ms as f32 / r.audio_duration_ms.max(1) as f32;
            eprintln!(
                "BENCH run {}: processing={} ms, audio={} ms, RTF={:.3}, text=\"{}\"",
                i,
                r.processing_time_ms,
                r.audio_duration_ms,
                rtf,
                r.text
            );
            // Exercise the durable log end-to-end with the real model output.
            let rec = crate::transcript::TranscriptRecord::new(
                r.text,
                r.audio_duration_ms,
                r.processing_time_ms,
                false,
            );
            crate::transcript::append(&rec);
        }

        // Also exercise the empty/VAD-trimmed branch if a silence fixture is set.
        if let Ok(silence_path) = std::env::var("WHIMPER_BENCH_SILENCE") {
            let bytes = std::fs::read(&silence_path).expect("read silence fixture");
            let sil: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let r = asr.transcribe(&sil).expect("transcribe silence");
            eprintln!(
                "BENCH silence: processing={} ms, text=\"{}\" (empty={})",
                r.processing_time_ms,
                r.text,
                r.text.trim().is_empty()
            );
            let rec = crate::transcript::TranscriptRecord::new(
                r.text,
                r.audio_duration_ms,
                r.processing_time_ms,
                false,
            );
            crate::transcript::append(&rec);
        }
    }
}
