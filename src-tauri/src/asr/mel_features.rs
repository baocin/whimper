//! Log-mel spectrogram feature extraction for Parakeet TDT
//!
//! 128 mel bins, 16kHz, 25ms window, 10ms hop, per-band normalization.

use anyhow::{anyhow, Result};
use ndarray::Array2;
use rustfft::{num_complex::Complex32, FftPlanner};

/// Configuration for log-mel feature extraction.
#[derive(Debug, Clone)]
pub struct MelSpectrogramConfig {
    pub sample_rate: u32,
    pub fft_size: usize,
    pub frame_length: usize,
    pub hop_length: usize,
    pub mel_bins: usize,
    pub mel_min_hz: f32,
    pub mel_max_hz: f32,
    pub preemphasis: f32,
    pub log_epsilon: f32,
}

impl MelSpectrogramConfig {
    /// Configuration matching NVIDIA Parakeet TDT (128 mel bins, 25ms window, 10ms hop).
    pub fn parakeet_tdt() -> Self {
        Self {
            sample_rate: 16_000,
            fft_size: 512,
            frame_length: 400, // 25ms at 16kHz
            hop_length: 160,   // 10ms at 16kHz
            mel_bins: 128,
            mel_min_hz: 0.0,
            mel_max_hz: 8_000.0,
            preemphasis: 0.97,
            log_epsilon: 5.960_464_477_539_063e-8, // 2^-24, matches NeMo
        }
    }
}

/// Compute log-mel spectrogram features with per-bin mean/variance normalization.
///
/// Returns Array2<f32> with shape [num_frames, mel_bins]
pub fn compute_log_mel(samples: &[f32], config: &MelSpectrogramConfig) -> Result<Array2<f32>> {
    if samples.is_empty() {
        return Err(anyhow!("Audio buffer is empty"));
    }

    let filtered = apply_preemphasis(samples, config.preemphasis);

    let frame_length = config.frame_length;
    let hop_length = config.hop_length.max(1);
    let fft_size = config.fft_size.max(frame_length.next_power_of_two());
    let freq_bins = fft_size / 2 + 1;

    let num_frames = if filtered.len() <= frame_length {
        1
    } else {
        (filtered.len() - frame_length) / hop_length + 1
    };

    let window = hann_window(frame_length);
    let mel_filters = build_mel_filterbank(
        freq_bins,
        config.mel_bins,
        config.sample_rate,
        config.mel_min_hz,
        config.mel_max_hz.min(config.sample_rate as f32 / 2.0),
    )?;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    let mut features = Array2::<f32>::zeros((num_frames, config.mel_bins));
    let mut fft_buffer = vec![Complex32::new(0.0, 0.0); fft_size];

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_length;

        for i in 0..fft_size {
            let sample_idx = start + i;
            let sample = if i < frame_length && sample_idx < filtered.len() {
                filtered[sample_idx] * window[i]
            } else {
                0.0
            };
            fft_buffer[i].re = sample;
            fft_buffer[i].im = 0.0;
        }

        fft.process(&mut fft_buffer);

        let mut power = vec![0.0f32; freq_bins];
        for bin in 0..freq_bins {
            power[bin] = fft_buffer[bin].norm_sqr();
        }

        for (mel_idx, filter) in mel_filters.iter().enumerate() {
            let mut energy = 0.0f32;
            for (bin, weight) in filter.iter().enumerate() {
                energy += power[bin] * weight;
            }
            features[(frame_idx, mel_idx)] = (energy + config.log_epsilon).ln();
        }
    }

    if num_frames > 0 {
        normalize_per_band(&mut features);
    }

    Ok(features)
}

fn apply_preemphasis(samples: &[f32], coefficient: f32) -> Vec<f32> {
    if coefficient <= 0.0 {
        return samples.to_vec();
    }
    let mut out = Vec::with_capacity(samples.len());
    out.push(samples[0]);
    for i in 1..samples.len() {
        out.push(samples[i] - coefficient * samples[i - 1]);
    }
    out
}

fn hann_window(length: usize) -> Vec<f32> {
    if length == 0 {
        return vec![];
    }
    (0..length)
        .map(|i| {
            0.5 - 0.5
                * (2.0 * std::f32::consts::PI * i as f32 / length as f32).cos()
        })
        .collect()
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

fn build_mel_filterbank(
    freq_bins: usize,
    mel_bins: usize,
    sample_rate: u32,
    mel_min_hz: f32,
    mel_max_hz: f32,
) -> Result<Vec<Vec<f32>>> {
    if mel_bins == 0 {
        return Err(anyhow!("mel_bins must be greater than zero"));
    }

    let mel_min = hz_to_mel(mel_min_hz.max(0.0));
    let mel_max = hz_to_mel(mel_max_hz.max(mel_min_hz));
    let mel_points: Vec<f32> = (0..=mel_bins + 1)
        .map(|i| mel_min + (mel_max - mel_min) * (i as f32 / (mel_bins + 1) as f32))
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    let nyquist = sample_rate as f32 / 2.0;
    let mut bin_points: Vec<usize> = hz_points
        .iter()
        .map(|&hz| {
            let ratio = (hz / nyquist).clamp(0.0, 1.0);
            (ratio * (freq_bins as f32 - 1.0)).round() as usize
        })
        .collect();

    for i in 1..bin_points.len() {
        if bin_points[i] <= bin_points[i - 1] {
            bin_points[i] = (bin_points[i - 1] + 1).min(freq_bins - 1);
        }
    }

    let mut filters = Vec::with_capacity(mel_bins);

    for i in 0..mel_bins {
        let left = bin_points[i];
        let center = bin_points[i + 1];
        let right = bin_points[i + 2];

        let mut filter = vec![0.0f32; freq_bins];

        if center > left {
            for (idx, bin) in (left..=center).enumerate() {
                filter[bin] = idx as f32 / (center - left) as f32;
            }
        }

        if right > center {
            for (idx, _bin) in (center..right).enumerate() {
                let denom = (right - center) as f32;
                filter[center + idx] = (denom - idx as f32) / denom;
            }
        }

        // Slaney normalization: normalize each filter by its bandwidth in Hz
        // so narrow low-frequency and wide high-frequency filters contribute equally.
        // This matches librosa.filters.mel(norm='slaney') used by NeMo during training.
        let bandwidth = hz_points[i + 2] - hz_points[i];
        if bandwidth > 0.0 {
            let norm = 2.0 / bandwidth;
            for val in filter.iter_mut() {
                *val *= norm;
            }
        }

        filters.push(filter);
    }

    Ok(filters)
}

fn normalize_per_band(features: &mut Array2<f32>) {
    let (num_frames, num_bins) = features.dim();
    if num_frames == 0 {
        return;
    }

    for mel in 0..num_bins {
        let mut mean = 0.0f32;
        for frame in 0..num_frames {
            mean += features[(frame, mel)];
        }
        mean /= num_frames as f32;

        let mut variance = 0.0f32;
        for frame in 0..num_frames {
            let diff = features[(frame, mel)] - mean;
            variance += diff * diff;
        }
        variance /= num_frames as f32;
        let std = variance.sqrt() + 1e-5;

        for frame in 0..num_frames {
            features[(frame, mel)] = (features[(frame, mel)] - mean) / std;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_audio() {
        let cfg = MelSpectrogramConfig::parakeet_tdt();
        let err = compute_log_mel(&[], &cfg).unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn test_sine_wave_frames() {
        let cfg = MelSpectrogramConfig::parakeet_tdt();
        let sr = cfg.sample_rate as f32;
        let duration_s = 0.5;
        let samples: Vec<f32> = (0..(sr * duration_s) as usize)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr).sin())
            .collect();

        let feats = compute_log_mel(&samples, &cfg).expect("mel");
        assert!(feats.nrows() > 0);
        assert_eq!(feats.ncols(), cfg.mel_bins);
    }

    #[test]
    fn test_config_parakeet_tdt() {
        let cfg = MelSpectrogramConfig::parakeet_tdt();
        assert_eq!(cfg.mel_bins, 128);
        assert_eq!(cfg.sample_rate, 16_000);
        assert_eq!(cfg.frame_length, 400);
        assert_eq!(cfg.hop_length, 160);
    }
}
