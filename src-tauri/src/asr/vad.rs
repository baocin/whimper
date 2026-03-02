//! Silero VAD (Voice Activity Detection) using ONNX Runtime
//!
//! Simplified port from timeline_app — macOS only, no Android/memory safety.
//! Detects speech in audio to gate ASR processing.

use anyhow::{Context, Result};
use std::sync::atomic::{AtomicBool, Ordering};

/// Embedded Silero VAD ONNX model (~2.3MB)
const MODEL_BYTES: &[u8] = include_bytes!("../../models/silero_vad.onnx");

/// Speech detection threshold (0.0-1.0)
const THRESHOLD: f32 = 0.35;

/// Silero VAD processor
pub struct SileroVad {
    session: Option<ort::session::Session>,
    model_loaded: AtomicBool,
    /// LSTM hidden state persisted across process_chunk() calls [2, 1, 128] = 256 floats
    state: Vec<f32>,
}

impl SileroVad {
    pub fn new() -> Self {
        Self {
            session: None,
            model_loaded: AtomicBool::new(false),
            state: vec![0.0f32; 256],
        }
    }

    /// Reset LSTM state (call on stream boundaries)
    pub fn reset_state(&mut self) {
        self.state.fill(0.0);
    }

    /// Load VAD model from embedded bytes
    pub fn load_embedded(&mut self) -> Result<()> {
        use ort::session::{builder::GraphOptimizationLevel, Session};

        let session = Session::builder()
            .context("Failed to create session builder")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .context("Failed to set optimization level")?
            .with_intra_threads(1)
            .context("Failed to set intra threads")?
            .commit_from_memory(MODEL_BYTES)
            .context("Failed to load embedded Silero VAD model")?;

        self.session = Some(session);
        self.model_loaded.store(true, Ordering::SeqCst);
        tracing::info!("Silero VAD model loaded");
        Ok(())
    }

    pub fn is_loaded(&self) -> bool {
        self.model_loaded.load(Ordering::SeqCst)
    }

    /// Simple energy-based VAD fallback
    fn energy_vad(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let energy: f32 = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
        let db = 10.0 * energy.max(1e-10).log10();
        // Map -60dB to 0.0 and -20dB to 1.0
        ((db + 60.0) / 40.0).clamp(0.0, 1.0)
    }

    /// Process a single 512-sample chunk and return speech probability
    fn process_chunk(&mut self, samples: &[f32]) -> Result<f32> {
        use ort::value::Tensor;

        if self.session.is_none() {
            return Ok(self.energy_vad(samples));
        }

        let input = Tensor::from_array(([1usize, samples.len()], samples.to_vec()))
            .context("Failed to create input tensor")?;

        let state = Tensor::from_array(([2usize, 1usize, 128usize], self.state.clone()))
            .context("Failed to create state tensor")?;

        let sr = Tensor::from_array(([1usize], vec![16000i64]))
            .context("Failed to create sr tensor")?;

        let session = self.session.as_mut().unwrap();
        let outputs = session
            .run(ort::inputs![
                "input" => input,
                "state" => state,
                "sr" => sr
            ])
            .context("Silero VAD inference failed")?;

        // Persist output state for next call
        if let Some(state_out) = outputs.get("stateN") {
            if let Ok((_shape, state_data)) = state_out.try_extract_tensor::<f32>() {
                let new_state: Vec<f32> = state_data.iter().copied().collect();
                if new_state.len() == 256 {
                    self.state = new_state;
                }
            }
        }

        let output = outputs
            .get("output")
            .context("VAD model output 'output' not found")?;

        let (_shape, data) = output
            .try_extract_tensor::<f32>()
            .context("Failed to extract VAD output tensor")?;

        Ok(data.first().copied().unwrap_or(0.0))
    }

    /// Real-time speech detection for audio buffers.
    ///
    /// Runs Silero VAD on 512-sample chunks, returns true if any chunk
    /// exceeds the speech threshold. Undersized tail chunks are zero-padded.
    pub fn is_speech_realtime(&mut self, samples: &[f32]) -> Result<bool> {
        if samples.is_empty() {
            return Ok(false);
        }

        const CHUNK_SIZE: usize = 512; // 32ms at 16kHz

        for chunk in samples.chunks(CHUNK_SIZE) {
            let padded;
            let input = if chunk.len() < CHUNK_SIZE {
                padded = {
                    let mut buf = vec![0.0f32; CHUNK_SIZE];
                    buf[..chunk.len()].copy_from_slice(chunk);
                    buf
                };
                &padded[..]
            } else {
                chunk
            };
            let prob = self.process_chunk(input).unwrap_or(0.0);
            if prob >= THRESHOLD {
                return Ok(true);
            }
        }

        Ok(false)
    }
}

impl Default for SileroVad {
    fn default() -> Self {
        Self::new()
    }
}
