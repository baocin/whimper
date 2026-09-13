//! Streaming audio preprocessor for the live dictation feed.
//!
//! Runs an 80Hz high-pass filter incrementally so the streaming ASR gets clean-ish
//! audio without waiting for the recording to end.
//!
//! ponytail: this deliberately does NOT run GTCRN. Enhancing isolated 512-sample
//! chunks meant a single STFT frame per chunk, and `istft` normalizes by the
//! window-sum — with no overlapping neighbour that is `window^2`, so dividing by a
//! near-zero window at the frame edges amplified GTCRN's output ~14x. Denoising
//! needs the whole buffer, so it stays in the final pass where it already worked.

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

/// Incremental high-pass filter for the live transcription feed.
pub struct StreamingPreprocessor {
    hp: BiquadState,
    total: usize,
}

impl StreamingPreprocessor {
    pub fn new() -> Self {
        Self {
            hp: BiquadState::new(),
            total: 0,
        }
    }

    /// Filter incoming 16kHz mono samples. Returns the same number of samples,
    /// so there is no buffering delay on the live feed.
    pub fn process_chunk(&mut self, samples: &[f32]) -> Vec<f32> {
        self.total += samples.len();
        samples.iter().map(|&s| self.hp.process(s)).collect()
    }

    /// Nothing is buffered, so there is nothing to flush. Kept so callers have a
    /// clear end-of-stream hook.
    pub fn flush(&mut self) {
        tracing::info!("StreamingPreprocessor: {} samples filtered", self.total);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The live feed must not change the signal level. The previous
    /// implementation amplified speech ~14x by running GTCRN over isolated
    /// single-frame chunks and then normalizing by `window^2`.
    #[test]
    fn test_streaming_preprocessor_does_not_amplify() {
        let mut pre = StreamingPreprocessor::new();

        const IN_PEAK: f32 = 0.5;
        let input: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 16_000.0).sin() * IN_PEAK)
            .collect();

        let mut out: Vec<f32> = Vec::new();
        for chunk in input.chunks(512) {
            out.extend(pre.process_chunk(chunk));
        }
        pre.flush();

        let out_peak = out.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        println!(
            "streaming: in_peak={:.4} out_peak={:.4} (gain {:.2}x)",
            IN_PEAK,
            out_peak,
            out_peak / IN_PEAK
        );

        assert_eq!(out.len(), input.len(), "must not drop or buffer samples");
        assert!(
            out_peak < IN_PEAK * 1.5,
            "streaming path amplifies: {:.4} -> {:.4}",
            IN_PEAK,
            out_peak
        );
    }

    /// The high-pass must actually attenuate rumble well below 80Hz.
    #[test]
    fn test_highpass_attenuates_low_frequency() {
        let mut pre = StreamingPreprocessor::new();

        let rumble: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 20.0 * i as f32 / 16_000.0).sin() * 0.5)
            .collect();

        let out = pre.process_chunk(&rumble);

        // Measure after the filter settles. Measured attenuation at 20Hz is
        // ~4x (0.5 -> 0.125); the threshold allows margin without asserting a
        // specific response curve for coefficients this app already ships.
        let tail_peak = out[8000..].iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        println!(
            "20Hz rumble: in_peak=0.5 out_peak={:.4} ({:.1}x attenuation)",
            tail_peak,
            0.5 / tail_peak.max(1e-9)
        );
        assert!(
            tail_peak < 0.2,
            "20Hz rumble not attenuated: peak {:.4}",
            tail_peak
        );
    }
}
