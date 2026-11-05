//! GTCRN (Gated Temporal Convolutional Recurrent Network) speech enhancement.
//!
//! Ultra-lightweight noise reduction (~48K params, ~523KB ONNX).
//! Operates in STFT domain: each frame [1, 257, 1, 2] (real+imag) is enhanced
//! while maintaining recurrent state across frames.

use super::stft::{hann_window, istft, stft, FFT_SIZE, HOP_SIZE, NUM_BINS};
use anyhow::{Context, Result};
use rustfft::num_complex::Complex32;
use std::sync::atomic::{AtomicBool, Ordering};

/// Embedded GTCRN ONNX model (~523KB)
const MODEL_BYTES: &[u8] = include_bytes!("../../models/gtcrn_simple.onnx");

// Cache dimensions
const CONV_CACHE_LEN: usize = 2 * 1 * 16 * 16 * 33; // [2, 1, 16, 16, 33]
const TRA_CACHE_LEN: usize = 2 * 3 * 1 * 1 * 16; // [2, 3, 1, 1, 16]
const INTER_CACHE_LEN: usize = 2 * 1 * 33 * 16; // [2, 1, 33, 16]

/// GTCRN speech enhancement processor
pub struct GtcrnEnhancer {
    session: Option<ort::session::Session>,
    model_loaded: AtomicBool,
    conv_cache: Vec<f32>,
    tra_cache: Vec<f32>,
    inter_cache: Vec<f32>,
}

impl GtcrnEnhancer {
    pub fn new() -> Self {
        Self {
            session: None,
            model_loaded: AtomicBool::new(false),
            conv_cache: vec![0.0; CONV_CACHE_LEN],
            tra_cache: vec![0.0; TRA_CACHE_LEN],
            inter_cache: vec![0.0; INTER_CACHE_LEN],
        }
    }

    pub fn reset_state(&mut self) {
        self.conv_cache.fill(0.0);
        self.tra_cache.fill(0.0);
        self.inter_cache.fill(0.0);
    }

    pub fn load_embedded(&mut self) -> Result<()> {
        use ort::session::{builder::GraphOptimizationLevel, Session};

        let session = Session::builder()
            .context("Failed to create session builder")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("Failed to set optimization level: {e}"))?
            .with_intra_threads(1)
            .map_err(|e| anyhow::anyhow!("Failed to set intra threads: {e}"))?
            .commit_from_memory(MODEL_BYTES)
            .map_err(|e| anyhow::anyhow!("Failed to load embedded GTCRN model: {e}"))?;

        self.session = Some(session);
        self.model_loaded.store(true, Ordering::SeqCst);
        tracing::info!("GTCRN speech enhancement model loaded");
        Ok(())
    }

    pub fn is_loaded(&self) -> bool {
        self.model_loaded.load(Ordering::SeqCst)
    }

    /// Enhance a single STFT frame, returning the enhanced complex bins.
    /// Used by the streaming preprocessor for per-frame processing.
    pub fn enhance_frame(&mut self, frame: &[Complex32]) -> Result<Vec<Complex32>> {
        let session = self.session.as_mut().context("GTCRN model not loaded")?;

        let mut mix_data = vec![0.0f32; NUM_BINS * 2];
        for (i, &c) in frame.iter().enumerate() {
            mix_data[i * 2] = c.re;
            mix_data[i * 2 + 1] = c.im;
        }

        let mix = ort::value::Tensor::from_array(
            ([1usize, NUM_BINS, 1usize, 2usize], mix_data),
        ).context("Failed to create mix tensor")?;

        let conv = ort::value::Tensor::from_array(
            ([2usize, 1usize, 16usize, 16usize, 33usize], self.conv_cache.clone()),
        ).context("Failed to create conv_cache tensor")?;

        let tra = ort::value::Tensor::from_array(
            ([2usize, 3usize, 1usize, 1usize, 16usize], self.tra_cache.clone()),
        ).context("Failed to create tra_cache tensor")?;

        let inter = ort::value::Tensor::from_array(
            ([2usize, 1usize, 33usize, 16usize], self.inter_cache.clone()),
        ).context("Failed to create inter_cache tensor")?;

        let outputs = session.run(ort::inputs![
            "mix" => mix,
            "conv_cache" => conv,
            "tra_cache" => tra,
            "inter_cache" => inter
        ]).context("GTCRN inference failed")?;

        // Update caches
        if let Some(out) = outputs.get("conv_cache_out") {
            if let Ok((_shape, data)) = out.try_extract_tensor::<f32>() {
                let new: Vec<f32> = data.iter().copied().collect();
                if new.len() == CONV_CACHE_LEN {
                    self.conv_cache = new;
                }
            }
        }
        if let Some(out) = outputs.get("tra_cache_out") {
            if let Ok((_shape, data)) = out.try_extract_tensor::<f32>() {
                let new: Vec<f32> = data.iter().copied().collect();
                if new.len() == TRA_CACHE_LEN {
                    self.tra_cache = new;
                }
            }
        }
        if let Some(out) = outputs.get("inter_cache_out") {
            if let Ok((_shape, data)) = out.try_extract_tensor::<f32>() {
                let new: Vec<f32> = data.iter().copied().collect();
                if new.len() == INTER_CACHE_LEN {
                    self.inter_cache = new;
                }
            }
        }

        let enh_output = outputs.get("enh").context("GTCRN output 'enh' not found")?;
        let (_shape, enh_data) = enh_output.try_extract_tensor::<f32>()
            .context("Failed to extract enhanced tensor")?;
        let enh_vec: Vec<f32> = enh_data.iter().copied().collect();

        Ok((0..NUM_BINS)
            .map(|i| {
                let re = enh_vec.get(i * 2).copied().unwrap_or(0.0);
                let im = enh_vec.get(i * 2 + 1).copied().unwrap_or(0.0);
                Complex32::new(re, im)
            })
            .collect())
    }

    /// Enhance a full audio buffer (16kHz mono f32) by denoising in the STFT domain.
    pub fn enhance(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        if samples.len() < FFT_SIZE {
            return Ok(samples.to_vec());
        }

        let session = self
            .session
            .as_mut()
            .context("GTCRN model not loaded")?;

        // STFT
        let window = hann_window(FFT_SIZE);
        let frames = stft(samples, &window);
        let num_frames = frames.len();

        // Process each frame through the model
        let mut enhanced_frames: Vec<[f32; NUM_BINS * 2]> = Vec::with_capacity(num_frames);

        for frame in &frames {
            // Build input tensor: [1, 257, 1, 2] from complex frame
            let mut mix_data = vec![0.0f32; NUM_BINS * 2];
            for (i, &c) in frame.iter().enumerate() {
                mix_data[i * 2] = c.re;
                mix_data[i * 2 + 1] = c.im;
            }

            let mix = ort::value::Tensor::from_array(
                ([1usize, NUM_BINS, 1usize, 2usize], mix_data),
            )
            .context("Failed to create mix tensor")?;

            let conv = ort::value::Tensor::from_array(
                ([2usize, 1usize, 16usize, 16usize, 33usize], self.conv_cache.clone()),
            )
            .context("Failed to create conv_cache tensor")?;

            let tra = ort::value::Tensor::from_array(
                ([2usize, 3usize, 1usize, 1usize, 16usize], self.tra_cache.clone()),
            )
            .context("Failed to create tra_cache tensor")?;

            let inter = ort::value::Tensor::from_array(
                ([2usize, 1usize, 33usize, 16usize], self.inter_cache.clone()),
            )
            .context("Failed to create inter_cache tensor")?;

            let outputs = session
                .run(ort::inputs![
                    "mix" => mix,
                    "conv_cache" => conv,
                    "tra_cache" => tra,
                    "inter_cache" => inter
                ])
                .context("GTCRN inference failed")?;

            // Update caches
            if let Some(out) = outputs.get("conv_cache_out") {
                if let Ok((_shape, data)) = out.try_extract_tensor::<f32>() {
                    let new: Vec<f32> = data.iter().copied().collect();
                    if new.len() == CONV_CACHE_LEN {
                        self.conv_cache = new;
                    }
                }
            }
            if let Some(out) = outputs.get("tra_cache_out") {
                if let Ok((_shape, data)) = out.try_extract_tensor::<f32>() {
                    let new: Vec<f32> = data.iter().copied().collect();
                    if new.len() == TRA_CACHE_LEN {
                        self.tra_cache = new;
                    }
                }
            }
            if let Some(out) = outputs.get("inter_cache_out") {
                if let Ok((_shape, data)) = out.try_extract_tensor::<f32>() {
                    let new: Vec<f32> = data.iter().copied().collect();
                    if new.len() == INTER_CACHE_LEN {
                        self.inter_cache = new;
                    }
                }
            }

            // Extract enhanced frame [1, 257, 1, 2] → [257 * 2]
            let enh_output = outputs
                .get("enh")
                .context("GTCRN output 'enh' not found")?;
            let (_shape, enh_data) = enh_output
                .try_extract_tensor::<f32>()
                .context("Failed to extract enhanced tensor")?;

            let mut frame_data = [0.0f32; NUM_BINS * 2];
            let enh_vec: Vec<f32> = enh_data.iter().copied().collect();
            let copy_len = frame_data.len().min(enh_vec.len());
            frame_data[..copy_len].copy_from_slice(&enh_vec[..copy_len]);
            enhanced_frames.push(frame_data);
        }

        // Convert enhanced frames back to complex and iSTFT
        let enhanced_complex: Vec<Vec<Complex32>> = enhanced_frames
            .iter()
            .map(|f| {
                (0..NUM_BINS)
                    .map(|i| Complex32::new(f[i * 2], f[i * 2 + 1]))
                    .collect()
            })
            .collect();

        let result = istft(&enhanced_complex, &window, samples.len());
        Ok(result)
    }

    /// Enhance audio, skipping ONNX inference on silence frames identified by the VAD mask.
    /// Each mask entry covers 512 samples (2 STFT frames at HOP_SIZE=256).
    pub fn enhance_vad_gated(
        &mut self,
        samples: &[f32],
        speech_mask: &[bool],
    ) -> Result<Vec<f32>> {
        if samples.len() < FFT_SIZE {
            return Ok(samples.to_vec());
        }

        let session = self
            .session
            .as_mut()
            .context("GTCRN model not loaded")?;

        let window = hann_window(FFT_SIZE);
        let frames = stft(samples, &window);
        let num_frames = frames.len();

        let mut enhanced_frames: Vec<Vec<Complex32>> = Vec::with_capacity(num_frames);
        let mut frames_enhanced = 0usize;
        let mut frames_skipped = 0usize;

        for (frame_idx, frame) in frames.iter().enumerate() {
            // Map STFT frame to VAD chunk: each VAD chunk is 512 samples,
            // each STFT hop is 256 samples, so ~2 frames per chunk
            let sample_offset = frame_idx * HOP_SIZE;
            let vad_idx = sample_offset / 512;
            let is_speech = speech_mask.get(vad_idx).copied().unwrap_or(false);

            if !is_speech {
                // Pass through original frame without inference
                enhanced_frames.push(frame.clone());
                frames_skipped += 1;
                continue;
            }

            frames_enhanced += 1;

            // Build input tensor [1, 257, 1, 2]
            let mut mix_data = vec![0.0f32; NUM_BINS * 2];
            for (i, &c) in frame.iter().enumerate() {
                mix_data[i * 2] = c.re;
                mix_data[i * 2 + 1] = c.im;
            }

            let mix = ort::value::Tensor::from_array(
                ([1usize, NUM_BINS, 1usize, 2usize], mix_data),
            )
            .context("Failed to create mix tensor")?;

            let conv = ort::value::Tensor::from_array(
                ([2usize, 1usize, 16usize, 16usize, 33usize], self.conv_cache.clone()),
            )
            .context("Failed to create conv_cache tensor")?;

            let tra = ort::value::Tensor::from_array(
                ([2usize, 3usize, 1usize, 1usize, 16usize], self.tra_cache.clone()),
            )
            .context("Failed to create tra_cache tensor")?;

            let inter = ort::value::Tensor::from_array(
                ([2usize, 1usize, 33usize, 16usize], self.inter_cache.clone()),
            )
            .context("Failed to create inter_cache tensor")?;

            let outputs = session
                .run(ort::inputs![
                    "mix" => mix,
                    "conv_cache" => conv,
                    "tra_cache" => tra,
                    "inter_cache" => inter
                ])
                .context("GTCRN inference failed")?;

            // Update caches
            if let Some(out) = outputs.get("conv_cache_out") {
                if let Ok((_shape, data)) = out.try_extract_tensor::<f32>() {
                    let new: Vec<f32> = data.iter().copied().collect();
                    if new.len() == CONV_CACHE_LEN {
                        self.conv_cache = new;
                    }
                }
            }
            if let Some(out) = outputs.get("tra_cache_out") {
                if let Ok((_shape, data)) = out.try_extract_tensor::<f32>() {
                    let new: Vec<f32> = data.iter().copied().collect();
                    if new.len() == TRA_CACHE_LEN {
                        self.tra_cache = new;
                    }
                }
            }
            if let Some(out) = outputs.get("inter_cache_out") {
                if let Ok((_shape, data)) = out.try_extract_tensor::<f32>() {
                    let new: Vec<f32> = data.iter().copied().collect();
                    if new.len() == INTER_CACHE_LEN {
                        self.inter_cache = new;
                    }
                }
            }

            // Extract enhanced frame
            let enh_output = outputs
                .get("enh")
                .context("GTCRN output 'enh' not found")?;
            let (_shape, enh_data) = enh_output
                .try_extract_tensor::<f32>()
                .context("Failed to extract enhanced tensor")?;

            let enh_vec: Vec<f32> = enh_data.iter().copied().collect();
            let enh_complex: Vec<Complex32> = (0..NUM_BINS)
                .map(|i| {
                    let re = enh_vec.get(i * 2).copied().unwrap_or(0.0);
                    let im = enh_vec.get(i * 2 + 1).copied().unwrap_or(0.0);
                    Complex32::new(re, im)
                })
                .collect();
            enhanced_frames.push(enh_complex);
        }

        tracing::info!(
            "GTCRN VAD-gated: {} frames enhanced, {} skipped",
            frames_enhanced,
            frames_skipped
        );

        let result = istft(&enhanced_frames, &window, samples.len());
        Ok(result)
    }
}

impl Default for GtcrnEnhancer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhance_short_audio_passthrough() {
        let mut enhancer = GtcrnEnhancer::new();
        let short = vec![0.1f32; 100];
        let result = enhancer.enhance(&short).unwrap();
        assert_eq!(result, short);
    }
}
