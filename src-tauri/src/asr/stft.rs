//! Shared STFT / iSTFT primitives for audio processing.

use rustfft::{num_complex::Complex32, FftPlanner};

pub(super) const FFT_SIZE: usize = 512;
pub(super) const HOP_SIZE: usize = 256;
pub(super) const NUM_BINS: usize = FFT_SIZE / 2 + 1; // 257

pub(super) fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|n| {
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / size as f32).cos())
        })
        .collect()
}

/// Forward STFT: returns Vec of complex frames, each with NUM_BINS bins.
pub(super) fn stft(samples: &[f32], window: &[f32]) -> Vec<Vec<Complex32>> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);

    let num_frames = if samples.len() >= FFT_SIZE {
        (samples.len() - FFT_SIZE) / HOP_SIZE + 1
    } else {
        0
    };

    let mut frames = Vec::with_capacity(num_frames);

    for i in 0..num_frames {
        let start = i * HOP_SIZE;
        let mut buf: Vec<Complex32> = (0..FFT_SIZE)
            .map(|j| {
                let s = if start + j < samples.len() {
                    samples[start + j]
                } else {
                    0.0
                };
                Complex32::new(s * window[j], 0.0)
            })
            .collect();

        fft.process(&mut buf);

        // Keep only positive frequencies (0..=N/2)
        frames.push(buf[..NUM_BINS].to_vec());
    }

    frames
}

/// Inverse STFT with overlap-add synthesis.
pub(super) fn istft(frames: &[Vec<Complex32>], window: &[f32], target_len: usize) -> Vec<f32> {
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(FFT_SIZE);

    let output_len = if frames.is_empty() {
        0
    } else {
        (frames.len() - 1) * HOP_SIZE + FFT_SIZE
    };
    let mut output = vec![0.0f32; output_len];
    let mut window_sum = vec![0.0f32; output_len];

    for (i, frame) in frames.iter().enumerate() {
        // Reconstruct full symmetric spectrum
        let mut buf: Vec<Complex32> = Vec::with_capacity(FFT_SIZE);
        buf.extend_from_slice(frame);
        // Mirror conjugate for bins N/2+1..N-1
        for j in (1..NUM_BINS - 1).rev() {
            buf.push(frame[j].conj());
        }

        ifft.process(&mut buf);

        let start = i * HOP_SIZE;
        let scale = 1.0 / FFT_SIZE as f32;
        for j in 0..FFT_SIZE {
            if start + j < output_len {
                output[start + j] += buf[j].re * scale * window[j];
                window_sum[start + j] += window[j] * window[j];
            }
        }
    }

    // Normalize by window sum (avoid division by zero)
    for i in 0..output_len {
        if window_sum[i] > 1e-8 {
            output[i] /= window_sum[i];
        }
    }

    // Truncate to original length
    output.truncate(target_len);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window_endpoints() {
        let w = hann_window(512);
        assert_eq!(w.len(), 512);
        assert!(w[0].abs() < 1e-6);
        assert!((w[256] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_stft_istft_roundtrip() {
        let n = 4096;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();

        let window = hann_window(FFT_SIZE);
        let frames = stft(&signal, &window);
        let reconstructed = istft(&frames, &window, signal.len());

        assert_eq!(reconstructed.len(), signal.len());

        let margin = FFT_SIZE;
        for i in margin..signal.len() - margin {
            assert!(
                (reconstructed[i] - signal[i]).abs() < 0.05,
                "Mismatch at sample {}: got {} expected {}",
                i,
                reconstructed[i],
                signal[i]
            );
        }
    }
}
