//! Streaming audio preprocessor: applies highpass + VAD-gated GTCRN incrementally.
//!
//! Designed to run on a dedicated processing thread during recording,
//! so the mic callback never blocks on ONNX inference.

use super::gtcrn::GtcrnEnhancer;
use super::stft::{hann_window, istft, stft, FFT_SIZE};
use super::vad::SileroVad;
use rustfft::num_complex::Complex32;
use std::sync::Mutex;

/// Biquad high-pass filter state for sample-by-sample processing.
struct BiquadState {
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl BiquadState {
    fn new() -> Self {
        Self { x1: 0.0, x2: 0.0, y1: 0.0, y2: 0.0 }
    }

    /// Process one sample through the 80Hz Butterworth HPF.
    fn process(&mut self, x0: f32) -> f32 {
        const B0: f32 = 0.9853512;
        const B1: f32 = -1.9707024;
        const B2: f32 = 0.9853512;
        const A1: f32 = -1.9706320;
        const A2: f32 = 0.9707729;

        let y0 = B0 * x0 + B1 * self.x1 + B2 * self.x2 - A1 * self.y1 - A2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x0;
        self.y2 = self.y1;
        self.y1 = y0;
        y0
    }
}

const VAD_CHUNK_SIZE: usize = 512;

/// Streaming preprocessor that applies highpass + VAD-gated GTCRN incrementally.
///
/// Audio flows: highpass (sample-by-sample) → buffer to VAD chunks →
/// STFT → GTCRN on speech frames → overlap-add output.
pub struct StreamingPreprocessor {
    hp: BiquadState,
    /// Buffer accumulating highpassed samples for VAD + STFT processing
    input_buf: Vec<f32>,
    /// Previous VAD result for 1-chunk dilation lookahead
    prev_vad: bool,
    /// Buffered samples from previous chunk that need the next VAD decision (dilation delay)
    deferred_samples: Vec<f32>,
    /// Whether deferred_samples were classified as speech
    deferred_is_speech: bool,
    window: Vec<f32>,
    total_input: usize,
    total_output: usize,
}

impl StreamingPreprocessor {
    pub fn new() -> Self {
        Self {
            hp: BiquadState::new(),
            input_buf: Vec::with_capacity(4096),
            prev_vad: false,
            deferred_samples: Vec::new(),
            deferred_is_speech: false,
            window: hann_window(FFT_SIZE),
            total_input: 0,
            total_output: 0,
        }
    }

    /// Feed incoming 16kHz mono samples. Returns enhanced samples ready for accumulation.
    /// May return an empty Vec while buffering.
    pub fn process_chunk(
        &mut self,
        samples: &[f32],
        vad: &Mutex<SileroVad>,
        gtcrn: &Mutex<GtcrnEnhancer>,
    ) -> Vec<f32> {
        self.total_input += samples.len();

        // Step 1: Highpass filter sample-by-sample
        let filtered: Vec<f32> = samples.iter().map(|&s| self.hp.process(s)).collect();
        self.input_buf.extend_from_slice(&filtered);

        let mut output = Vec::new();

        // Step 2: Process complete VAD chunks (512 samples each)
        while self.input_buf.len() >= VAD_CHUNK_SIZE {
            let chunk: Vec<f32> = self.input_buf.drain(..VAD_CHUNK_SIZE).collect();

            // Run VAD on this chunk
            let current_vad = if let Ok(mut vad_guard) = vad.lock() {
                if vad_guard.is_loaded() {
                    vad_guard.process_chunk_threshold(&chunk)
                } else {
                    true // assume speech if VAD unavailable
                }
            } else {
                true
            };

            // Dilation: a chunk is "speech" if it OR its neighbor is speech.
            // We emit the *previous* chunk using: prev_vad || current_vad
            // This gives us 1-chunk lookahead dilation.
            if !self.deferred_samples.is_empty() {
                let is_speech = self.deferred_is_speech || current_vad;
                let enhanced = self.enhance_chunk(&self.deferred_samples.clone(), is_speech, gtcrn);
                output.extend(enhanced);
            }

            // Defer current chunk — we need the next VAD result for dilation
            self.deferred_samples = chunk;
            self.deferred_is_speech = self.prev_vad || current_vad;
            self.prev_vad = current_vad;
        }

        self.total_output += output.len();
        output
    }

    /// Flush remaining buffered samples at end of recording.
    pub fn flush(
        &mut self,
        _vad: &Mutex<SileroVad>,
        gtcrn: &Mutex<GtcrnEnhancer>,
    ) -> Vec<f32> {
        let mut output = Vec::new();

        // Emit deferred chunk with last known VAD state
        if !self.deferred_samples.is_empty() {
            let samples = std::mem::take(&mut self.deferred_samples);
            let enhanced = self.enhance_chunk(&samples, self.deferred_is_speech, gtcrn);
            output.extend(enhanced);
        }

        // Process any remaining input buffer (pad to VAD chunk size)
        if !self.input_buf.is_empty() {
            let mut remaining = std::mem::take(&mut self.input_buf);
            remaining.resize(VAD_CHUNK_SIZE, 0.0);
            let enhanced = self.enhance_chunk(&remaining, self.prev_vad, gtcrn);
            output.extend(enhanced);
        }

        tracing::info!(
            "StreamingPreprocessor flush: {} total input, {} total output samples",
            self.total_input,
            self.total_output + output.len()
        );

        output
    }

    /// Enhance a VAD-chunk-sized block of samples through GTCRN (if speech) or passthrough.
    fn enhance_chunk(
        &self,
        samples: &[f32],
        is_speech: bool,
        gtcrn: &Mutex<GtcrnEnhancer>,
    ) -> Vec<f32> {
        if !is_speech {
            return samples.to_vec();
        }

        // For speech chunks, run STFT → per-frame GTCRN → iSTFT
        if samples.len() < FFT_SIZE {
            return samples.to_vec();
        }

        let frames = stft(samples, &self.window);
        if frames.is_empty() {
            return samples.to_vec();
        }

        let enhanced_frames: Vec<Vec<Complex32>> = if let Ok(mut gtcrn_guard) = gtcrn.lock() {
            if gtcrn_guard.is_loaded() {
                frames
                    .iter()
                    .map(|frame| {
                        gtcrn_guard.enhance_frame(frame).unwrap_or_else(|_| frame.clone())
                    })
                    .collect()
            } else {
                frames
            }
        } else {
            frames
        };

        let reconstructed = istft(&enhanced_frames, &self.window, samples.len());
        reconstructed
    }
}
