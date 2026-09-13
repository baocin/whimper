//! NVIDIA Nemotron Speech Streaming English 0.6B - Automatic Speech Recognition
//!
//! Cache-aware FastConformer encoder + RNNT decoder, run through the `parakeet-rs`
//! crate, which owns the ONNX sessions, mel extraction, SentencePiece vocabulary and
//! the streaming cache. Audio preprocessing (highpass, VAD-gated GTCRN, spectral gate,
//! RMS normalization, silence trim) stays here because it is tuned for this app's mic path.

use super::preprocess::{
    dilate_mask, highpass_80hz, rms_normalize, spectral_gate, trim_silence, vad_speech_mask,
};
use super::vad::SileroVad;
use anyhow::{anyhow, Result};
use parakeet_rs::Nemotron;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Files that must be present in the model directory.
pub const MODEL_FILES: &[&str] = &["encoder.onnx", "decoder_joint.onnx", "tokenizer.model"];

/// Transcription result with timing info
#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub text: String,
    pub processing_time_ms: u64,
    pub audio_duration_ms: u64,
}

/// Nemotron streaming ASR engine.
pub struct NemotronAsr {
    model: Mutex<Option<Nemotron>>,
    vad: Arc<Mutex<SileroVad>>,
    gtcrn: Arc<Mutex<super::gtcrn::GtcrnEnhancer>>,
}

impl NemotronAsr {
    pub fn new() -> Self {
        Self {
            model: Mutex::new(None),
            vad: Arc::new(Mutex::new(SileroVad::new())),
            gtcrn: Arc::new(Mutex::new(super::gtcrn::GtcrnEnhancer::new())),
        }
    }

    /// Load the model from a directory holding encoder, decoder_joint and tokenizer.
    pub fn load_model(&self, model_dir: &str) -> Result<()> {
        let mut model_guard = self
            .model
            .lock()
            .map_err(|e| anyhow!("Model mutex poisoned: {}", e))?;
        if model_guard.is_some() {
            return Ok(());
        }

        let models_dir = PathBuf::from(model_dir);

        for name in MODEL_FILES {
            let path = models_dir.join(name);
            if !path.exists() {
                return Err(anyhow!("Nemotron {} not found at {:?}", name, path));
            }
        }

        tracing::info!("Loading Nemotron streaming model from {:?}...", models_dir);

        // parakeet-rs auto-detects the variant (English-only vs multilingual 3.5)
        // from the encoder graph, so no mode selection is needed here.
        let model = Nemotron::from_pretrained(&models_dir, None)
            .map_err(|e| anyhow!("Failed to load Nemotron model: {}", e))?;

        // Load Silero VAD for silence trimming
        if let Ok(mut vad_guard) = self.vad.lock() {
            if !vad_guard.is_loaded() {
                if let Err(e) = vad_guard.load_embedded() {
                    tracing::warn!("Failed to load Silero VAD, trimming disabled: {}", e);
                }
            }
        }

        // Load GTCRN for noise reduction
        if let Ok(mut gtcrn_guard) = self.gtcrn.lock() {
            if !gtcrn_guard.is_loaded() {
                if let Err(e) = gtcrn_guard.load_embedded() {
                    tracing::warn!("Failed to load GTCRN, noise reduction disabled: {}", e);
                }
            }
        }

        tracing::info!("Nemotron model loaded successfully");

        *model_guard = Some(model);

        Ok(())
    }

    pub fn is_loaded(&self) -> bool {
        self.model.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    /// Transcribe audio samples to text (16kHz mono f32).
    ///
    /// If `pre_enhanced` is true, the audio has already been through highpass + VAD-gated GTCRN
    /// (via streaming preprocessing), so we skip those steps and only run spectral gate,
    /// RMS normalization, and silence trimming.
    pub fn transcribe(&self, samples: &[f32], pre_enhanced: bool) -> Result<TranscriptionResult> {
        if samples.is_empty() {
            return Err(anyhow!("Audio samples are empty"));
        }

        let audio_duration_ms = (samples.len() as f32 / 16.0) as u64;
        let start = Instant::now();

        let skip_preprocess = std::env::var("WHIMPER_SKIP_PREPROCESS").is_ok();
        let mut buf = samples.to_vec();

        // Speech mask for VAD-gated processing (computed pre-GTCRN, reused for spectral gate)
        let mut speech_mask: Option<Vec<bool>> = None;

        if skip_preprocess {
            tracing::info!("WHIMPER_SKIP_PREPROCESS set — bypassing all preprocessing");
        } else if pre_enhanced {
            // Streaming mode: highpass + VAD-gated GTCRN already done.
            // Still run spectral gate (needs full buffer), RMS normalize, and silence trim.
            tracing::info!("Pre-enhanced audio: skipping highpass and GTCRN");

            if let Ok(mut vad_guard) = self.vad.lock() {
                if vad_guard.is_loaded() {
                    let mask = vad_speech_mask(&buf, &mut vad_guard);
                    speech_mask = Some(dilate_mask(&mask));
                }
            }

            if let Some(ref mask) = speech_mask {
                spectral_gate(&mut buf, mask);
            }

            rms_normalize(&mut buf);
        } else {
            // Batch mode: full preprocessing pipeline
            highpass_80hz(&mut buf);

            // Pre-GTCRN VAD scan: identify speech regions for gating
            if let Ok(mut vad_guard) = self.vad.lock() {
                if vad_guard.is_loaded() {
                    let mask = vad_speech_mask(&buf, &mut vad_guard);
                    speech_mask = Some(dilate_mask(&mask));
                }
            }
            // vad_guard released here — important for lock ordering

            // GTCRN noise reduction: only process speech frames
            if let Ok(mut gtcrn_guard) = self.gtcrn.lock() {
                if gtcrn_guard.is_loaded() {
                    gtcrn_guard.reset_state();
                    let result = match &speech_mask {
                        Some(mask) => gtcrn_guard.enhance_vad_gated(&buf, mask),
                        None => gtcrn_guard.enhance(&buf),
                    };
                    match result {
                        Ok(enhanced) => {
                            tracing::info!(
                                "GTCRN noise reduction applied ({} samples)",
                                enhanced.len()
                            );
                            buf = enhanced;
                        }
                        Err(e) => {
                            tracing::warn!("GTCRN enhancement failed, using original audio: {}", e);
                        }
                    }
                }
            }

            // Spectral gate: clean up residual GTCRN artifacts using noise floor from non-speech frames
            if let Some(ref mask) = speech_mask {
                spectral_gate(&mut buf, mask);
            }

            rms_normalize(&mut buf);
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

        // Post-GTCRN VAD trim: more accurate on cleaned audio (second VAD pass)
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

        let mut model_guard = self
            .model
            .lock()
            .map_err(|e| anyhow!("Model mutex poisoned: {}", e))?;
        let model = model_guard
            .as_mut()
            .ok_or_else(|| anyhow!("Model not loaded"))?;

        // Clear streaming caches so this utterance does not inherit the previous one's state.
        model.reset();

        let text = model
            .transcribe_audio(processed)
            .map_err(|e| anyhow!("Nemotron transcription failed: {}", e))?
            .trim()
            .to_string();

        let processing_time_ms = start.elapsed().as_millis() as u64;
        let rtf = if audio_duration_ms > 0 {
            processing_time_ms as f32 / audio_duration_ms as f32
        } else {
            0.0
        };

        tracing::info!(
            "Transcription: \"{}\" (RTF: {:.2})",
            &text[..text.len().min(80)],
            rtf
        );

        Ok(TranscriptionResult {
            text,
            processing_time_ms,
            audio_duration_ms,
        })
    }

    /// Clear the streaming cache so the next chunk starts a fresh utterance.
    pub fn reset_stream(&self) -> Result<()> {
        let mut guard = self
            .model
            .lock()
            .map_err(|e| anyhow!("Model mutex poisoned: {}", e))?;
        if let Some(model) = guard.as_mut() {
            model.reset();
        }
        Ok(())
    }

    /// Feed one chunk of 16kHz mono audio to the streaming decoder and return any
    /// newly emitted text. Output is append-only — RNNT never retracts a token, so
    /// the caller can type the return value straight out.
    pub fn transcribe_chunk(&self, samples: &[f32]) -> Result<String> {
        let mut guard = self
            .model
            .lock()
            .map_err(|e| anyhow!("Model mutex poisoned: {}", e))?;
        let model = guard
            .as_mut()
            .ok_or_else(|| anyhow!("Model not loaded"))?;

        model
            .transcribe_chunk(samples)
            .map_err(|e| anyhow!("Nemotron streaming chunk failed: {}", e))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nemotron_asr_creation() {
        let asr = NemotronAsr::new();
        assert!(!asr.is_loaded());
    }

    #[test]
    fn test_nemotron_asr_not_loaded_error() {
        let asr = NemotronAsr::new();
        let result = asr.transcribe(&[0.0f32; 16000], false);
        assert!(result.is_err());
    }

    /// End-to-end check against the real model: decode a known phrase and transcribe it.
    /// Ignored by default because it needs the ~662 MB model in ~/.whimper/models/nemotron-int4.
    /// Run with: cargo test transcribes_known_phrase -- --ignored --nocapture
    #[test]
    #[ignore]
    fn test_transcribes_known_phrase() {
        let dir = crate::state::model_dir();
        assert!(dir.join("encoder.onnx").exists(), "model missing at {:?}", dir);

        let asr = NemotronAsr::new();
        asr.load_model(&dir.to_string_lossy()).expect("load model");

        let wav = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/quick_brown_fox.wav");
        let samples = crate::audio::file_decoder::decode_audio_file(wav).expect("decode wav");

        let result = asr.transcribe(&samples, false).expect("transcribe");
        println!("transcript: {:?}", result.text);

        let lower = result.text.to_lowercase();
        assert!(
            lower.contains("quick brown fox"),
            "expected 'quick brown fox', got {:?}",
            result.text
        );
    }

    /// Measures whether streaming can keep up in real time.
    ///
    /// The crate recomputes the mel spectrogram over its ENTIRE buffered audio on
    /// every call, so per-chunk cost grows with utterance length. Each 560ms chunk
    /// must transcribe in well under 560ms or live typing falls behind and the
    /// pipeline's bounded channel starts dropping mic samples.
    ///
    /// Run with: cargo test streaming_chunk_latency -- --ignored --nocapture
    #[test]
    #[ignore]
    fn test_streaming_chunk_latency() {
        let dir = crate::state::model_dir();
        assert!(dir.join("encoder.onnx").exists(), "model missing at {:?}", dir);

        let asr = NemotronAsr::new();
        asr.load_model(&dir.to_string_lossy()).expect("load model");

        let wav = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/quick_brown_fox.wav");
        let base = crate::audio::file_decoder::decode_audio_file(wav).expect("decode wav");

        // Repeat the fixture out to ~60s of continuous speech.
        let mut audio: Vec<f32> = Vec::new();
        while audio.len() < 60 * 16_000 {
            audio.extend_from_slice(&base);
        }

        asr.reset_stream().expect("reset");

        const CHUNK: usize = 8960; // 560ms
        const BUDGET_MS: u128 = 560;

        // The first chunks pay ORT arena/allocator warm-up and are not
        // representative of the steady state, so they are measured but excluded.
        const WARMUP_CHUNKS: usize = 10;

        let mut timings: Vec<u128> = Vec::new();
        for (i, chunk) in audio.chunks(CHUNK).enumerate() {
            let started = Instant::now();
            let text = asr.transcribe_chunk(chunk).expect("chunk");
            let ms = started.elapsed().as_millis();
            timings.push(ms);
            if i % 10 == 0 {
                println!(
                    "chunk {:3} ({:5.1}s in): {:4}ms {:?}",
                    i,
                    i as f32 * 0.56,
                    ms,
                    text
                );
            }
        }

        let steady = &timings[WARMUP_CHUNKS.min(timings.len())..];
        let worst = steady.iter().copied().max().unwrap_or(0);
        let median = {
            let mut s = steady.to_vec();
            s.sort_unstable();
            s.get(s.len() / 2).copied().unwrap_or(0)
        };

        println!(
            "steady state over {} chunks: median {}ms, worst {}ms (budget {}ms)",
            steady.len(),
            median,
            worst,
            BUDGET_MS
        );
        println!("warm-up peak: {}ms", timings[..WARMUP_CHUNKS.min(timings.len())]
            .iter()
            .copied()
            .max()
            .unwrap_or(0));

        // Median is what determines whether the stream keeps up over an
        // utterance; isolated spikes are absorbed by the pipeline's buffering.
        assert!(
            median < BUDGET_MS,
            "streaming cannot keep up: median chunk {}ms exceeds the {}ms real-time budget",
            median,
            BUDGET_MS
        );
    }

    /// Endurance: can a single unbroken stream run for a long session?
    ///
    /// `transcribe_chunk` appends to an internal audio buffer and recomputes the mel
    /// spectrogram over ALL of it on every call, so both memory and per-chunk CPU
    /// grow with stream length. This feeds minutes of continuous audio and reports
    /// latency drift plus RSS so the trend can be extrapolated to hours.
    ///
    /// Run with: cargo test --release streaming_endurance -- --ignored --nocapture
    #[test]
    #[ignore]
    fn test_streaming_endurance() {
        let dir = crate::state::model_dir();
        assert!(dir.join("encoder.onnx").exists(), "model missing at {:?}", dir);

        let asr = NemotronAsr::new();
        asr.load_model(&dir.to_string_lossy()).expect("load model");

        let wav = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/quick_brown_fox.wav");
        let base = crate::audio::file_decoder::decode_audio_file(wav).expect("decode wav");

        const CHUNK: usize = 8960; // 560ms
        const BUDGET_MS: u128 = 560;
        const MINUTES: usize = 5;
        let total_chunks = MINUTES * 60 * 16_000 / CHUNK;

        let pid = std::process::id().to_string();
        let rss_mb = || -> u64 {
            std::process::Command::new("ps")
                .args(["-o", "rss=", "-p", &pid])
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .and_then(|s| s.trim().parse::<u64>().ok())
                .map(|kb| kb / 1024)
                .unwrap_or(0)
        };

        asr.reset_stream().expect("reset");

        let rss_start = rss_mb();
        println!("start rss={}MB, feeding {} minutes continuous", rss_start, MINUTES);

        let mut timings: Vec<u128> = Vec::with_capacity(total_chunks);
        let mut cursor = 0usize;

        for i in 0..total_chunks {
            let chunk: Vec<f32> = (0..CHUNK).map(|j| base[(cursor + j) % base.len()]).collect();
            cursor += CHUNK;

            let started = Instant::now();
            asr.transcribe_chunk(&chunk).expect("chunk");
            timings.push(started.elapsed().as_millis());

            if i % 50 == 0 {
                println!(
                    "chunk {:4} ({:6.1}s audio): {:6}ms  rss={}MB",
                    i,
                    i as f32 * 0.56,
                    timings[i],
                    rss_mb()
                );
            }
        }

        let rss_end = rss_mb();
        let median = |s: &[u128]| {
            let mut v = s.to_vec();
            v.sort_unstable();
            v.get(v.len() / 2).copied().unwrap_or(0)
        };

        let head = median(&timings[..60.min(timings.len())]);
        let tail = median(&timings[timings.len().saturating_sub(60)..]);

        println!(
            "first-minute median {}ms -> last-minute median {}ms ({:.1}x drift)",
            head,
            tail,
            tail as f32 / head.max(1) as f32
        );
        println!(
            "rss {}MB -> {}MB (+{}MB over {} minutes)",
            rss_start,
            rss_end,
            rss_end.saturating_sub(rss_start),
            MINUTES
        );

        assert!(
            tail < BUDGET_MS,
            "stream degrades: last-minute median {}ms exceeds {}ms budget after {} minutes",
            tail,
            BUDGET_MS,
            MINUTES
        );
    }

    #[test]
    fn test_hallucination_detection() {
        assert!(NemotronAsr::is_hallucination(""));
        assert!(NemotronAsr::is_hallucination("Thank you"));
        assert!(NemotronAsr::is_hallucination("[music]"));
        assert!(NemotronAsr::is_hallucination(".."));
        assert!(!NemotronAsr::is_hallucination("I ordered the salmon"));
    }
}
