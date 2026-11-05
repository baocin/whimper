//! Audio preprocessing: high-pass filter, peak normalization, silence trimming.

use super::stft::{hann_window, istft, stft, FFT_SIZE, HOP_SIZE, NUM_BINS};
use super::vad::SileroVad;
use rustfft::num_complex::Complex32;

/// Second-order Butterworth high-pass biquad at 80Hz for 16kHz sample rate.
/// Removes desk vibrations, AC rumble, and handling noise.
pub fn highpass_80hz(samples: &mut [f32]) {
    // Pre-computed coefficients: 2nd-order Butterworth HPF, fc=80Hz, fs=16000Hz
    const B0: f32 = 0.9853512;
    const B1: f32 = -1.9707024;
    const B2: f32 = 0.9853512;
    const A1: f32 = -1.9706320;
    const A2: f32 = 0.9707729;

    let (mut x1, mut x2, mut y1, mut y2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for s in samples.iter_mut() {
        let x0 = *s;
        let y0 = B0 * x0 + B1 * x1 + B2 * x2 - A1 * y1 - A2 * y2;
        x2 = x1;
        x1 = x0;
        y2 = y1;
        y1 = y0;
        *s = y0;
    }
}

/// Scale buffer so max |sample| = 0.95.
/// Skips silence (peak < 0.001). Does not attenuate already-loud signals.
pub fn peak_normalize(samples: &mut [f32]) {
    let peak = samples.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    if peak < 0.001 {
        return; // silence, don't amplify noise
    }
    let gain = 0.95 / peak;
    if gain > 1.0 {
        for s in samples.iter_mut() {
            *s *= gain;
        }
    }
}

/// Trim leading/trailing silence using Silero VAD.
/// Returns the trimmed slice indices. If no speech found, returns empty range.
pub fn trim_silence(samples: &[f32], vad: &mut SileroVad) -> (usize, usize) {
    const CHUNK_SIZE: usize = 512; // 32ms at 16kHz
    const PAD_SAMPLES: usize = 320; // 20ms at 16kHz

    vad.reset_state();

    let num_chunks = (samples.len() + CHUNK_SIZE - 1) / CHUNK_SIZE;
    let mut speech_chunks: Vec<bool> = Vec::with_capacity(num_chunks);

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
        let is_speech = vad.process_chunk_threshold(input);
        speech_chunks.push(is_speech);
    }

    let first_speech = match speech_chunks.iter().position(|&s| s) {
        Some(i) => i,
        None => return (0, 0), // no speech → empty
    };
    let last_speech = speech_chunks.iter().rposition(|&s| s).unwrap();

    let start = (first_speech * CHUNK_SIZE).saturating_sub(PAD_SAMPLES);
    let end = ((last_speech + 1) * CHUNK_SIZE + PAD_SAMPLES).min(samples.len());

    tracing::debug!(
        "Silence trim: {}→{} samples (was {})",
        start,
        end,
        samples.len()
    );

    (start, end)
}

/// Scan audio with VAD and return a per-chunk speech mask.
/// Each entry covers 512 samples (32ms at 16kHz).
pub fn vad_speech_mask(samples: &[f32], vad: &mut SileroVad) -> Vec<bool> {
    const CHUNK_SIZE: usize = 512;

    vad.reset_state();

    let num_chunks = (samples.len() + CHUNK_SIZE - 1) / CHUNK_SIZE;
    let mut mask = Vec::with_capacity(num_chunks);

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
        mask.push(vad.process_chunk_threshold(input));
    }

    mask
}

/// Expand speech regions by 1 chunk in each direction to protect attack/release transients.
pub fn dilate_mask(mask: &[bool]) -> Vec<bool> {
    let mut out = vec![false; mask.len()];
    for (i, &is_speech) in mask.iter().enumerate() {
        if is_speech {
            if i > 0 {
                out[i - 1] = true;
            }
            out[i] = true;
            if i + 1 < mask.len() {
                out[i + 1] = true;
            }
        }
    }
    out
}

/// Spectral gate post-filter: attenuate frequency bins below the noise floor.
/// Uses non-speech frames (from the VAD mask) to estimate per-bin noise floor,
/// then applies a soft gain to suppress residual GTCRN artifacts.
pub fn spectral_gate(samples: &mut Vec<f32>, speech_mask: &[bool]) {
    if samples.len() < FFT_SIZE {
        return;
    }

    const GATE_FACTOR: f32 = 1.5;
    const MIN_NOISE_FRAMES: usize = 3;

    let window = hann_window(FFT_SIZE);
    let mut frames = stft(samples, &window);
    let num_frames = frames.len();

    if num_frames == 0 {
        return;
    }

    // Identify non-speech frames for noise estimation
    let mut noise_floor = vec![0.0f32; NUM_BINS];
    let mut noise_count = 0usize;

    for (frame_idx, frame) in frames.iter().enumerate() {
        let sample_offset = frame_idx * HOP_SIZE;
        let vad_idx = sample_offset / 512;
        let is_speech = speech_mask.get(vad_idx).copied().unwrap_or(true);

        if !is_speech {
            for (bin, &c) in frame.iter().enumerate() {
                noise_floor[bin] += (c.re * c.re + c.im * c.im).sqrt();
            }
            noise_count += 1;
        }
    }

    if noise_count < MIN_NOISE_FRAMES {
        return; // Not enough noise reference, skip gating
    }

    // Average noise floor
    for nf in noise_floor.iter_mut() {
        *nf /= noise_count as f32;
    }

    // Apply soft spectral gate in-place (avoids a second ~462 MB allocation)
    for frame in frames.iter_mut() {
        for (bin, c) in frame.iter_mut().enumerate() {
            let mag = (c.re * c.re + c.im * c.im).sqrt();
            let threshold = noise_floor[bin] * GATE_FACTOR;
            if threshold > 0.0 {
                let gain = (mag / threshold).clamp(0.0, 1.0);
                *c = Complex32::new(c.re * gain, c.im * gain);
            }
        }
    }

    let result = istft(&frames, &window, samples.len());
    *samples = result;
}

/// RMS-based normalization targeting a consistent speech loudness level.
/// Unlike peak_normalize, this both amplifies and attenuates, and is robust
/// to transient spikes.
pub fn rms_normalize(samples: &mut [f32]) {
    const TARGET_RMS_DBFS: f32 = -20.0;
    const MAX_GAIN: f32 = 30.0;

    if samples.is_empty() {
        return;
    }

    // Compute RMS with f64 precision
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    let rms = (sum_sq / samples.len() as f64).sqrt() as f32;

    if rms < 0.001 {
        return; // Below -60 dBFS, effectively silence
    }

    let rms_db = 20.0 * rms.log10();
    let gain_db = TARGET_RMS_DBFS - rms_db;
    let gain = (10.0f32).powf(gain_db / 20.0).min(MAX_GAIN);

    for s in samples.iter_mut() {
        *s = (*s * gain).clamp(-1.0, 1.0);
    }

    tracing::debug!("RMS normalize: rms={:.4} ({:.1} dB), gain={:.2}x", rms, rms_db, gain);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_highpass_preserves_length() {
        let mut buf = vec![0.1f32; 1600];
        highpass_80hz(&mut buf);
        assert_eq!(buf.len(), 1600);
    }

    #[test]
    fn test_normalize_quiet_signal() {
        let mut buf = vec![0.01f32; 100];
        peak_normalize(&mut buf);
        let peak = buf.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        assert!((peak - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_normalize_amplifies_moderate_signal() {
        let mut buf = vec![0.5f32; 100];
        peak_normalize(&mut buf);
        let peak = buf.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        assert!((peak - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_normalize_skips_loud_signal() {
        let mut buf = vec![1.0f32; 100];
        let original = buf.clone();
        peak_normalize(&mut buf);
        assert_eq!(buf, original);
    }

    #[test]
    fn test_normalize_skips_silent() {
        let mut buf = vec![0.0001f32; 100];
        let original = buf.clone();
        peak_normalize(&mut buf);
        assert_eq!(buf, original);
    }

    #[test]
    fn test_rms_normalize_quiet_signal() {
        // 0.01 amplitude → RMS ≈ 0.01 → -40 dBFS → need +20 dB gain → 10x
        let mut buf = vec![0.01f32; 1600];
        rms_normalize(&mut buf);
        let rms: f32 = (buf.iter().map(|s| s * s).sum::<f32>() / buf.len() as f32).sqrt();
        let rms_db = 20.0 * rms.log10();
        assert!((rms_db - (-20.0)).abs() < 1.0, "RMS should be ~-20 dBFS, got {:.1}", rms_db);
    }

    #[test]
    fn test_rms_normalize_loud_signal() {
        // 0.9 amplitude → RMS ≈ 0.9 → ~-0.9 dBFS → should attenuate
        let mut buf = vec![0.9f32; 1600];
        rms_normalize(&mut buf);
        let rms: f32 = (buf.iter().map(|s| s * s).sum::<f32>() / buf.len() as f32).sqrt();
        let rms_db = 20.0 * rms.log10();
        assert!((rms_db - (-20.0)).abs() < 1.0, "RMS should be ~-20 dBFS, got {:.1}", rms_db);
    }

    #[test]
    fn test_rms_normalize_silence_unchanged() {
        let mut buf = vec![0.0001f32; 100];
        let original = buf.clone();
        rms_normalize(&mut buf);
        assert_eq!(buf, original);
    }

    #[test]
    fn test_rms_normalize_clamps() {
        // Very quiet signal that would need huge gain — check output stays in [-1, 1]
        let mut buf = vec![0.002f32; 1600];
        rms_normalize(&mut buf);
        assert!(buf.iter().all(|&s| s >= -1.0 && s <= 1.0));
    }

    #[test]
    fn test_dilate_mask_expands() {
        let mask = vec![false, false, true, false, false];
        let dilated = dilate_mask(&mask);
        assert_eq!(dilated, vec![false, true, true, true, false]);
    }

    #[test]
    fn test_dilate_mask_edges() {
        let mask = vec![true, false, false, false, true];
        let dilated = dilate_mask(&mask);
        assert_eq!(dilated, vec![true, true, false, true, true]);
    }

    #[test]
    fn test_dilate_mask_empty() {
        let dilated = dilate_mask(&[]);
        assert!(dilated.is_empty());
    }
}
