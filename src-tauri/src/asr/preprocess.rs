//! Audio preprocessing: high-pass filter, peak normalization, silence trimming.

use super::vad::SileroVad;

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
/// Skips if peak already > 0.1 (signal is fine) or < 0.001 (silent).
pub fn peak_normalize(samples: &mut [f32]) {
    let peak = samples.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    if peak < 0.001 || peak > 0.1 {
        return;
    }
    let gain = 0.95 / peak;
    for s in samples.iter_mut() {
        *s *= gain;
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
    fn test_normalize_skips_loud_signal() {
        let mut buf = vec![0.5f32; 100];
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
}
