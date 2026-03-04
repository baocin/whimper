//! Audio pipeline: record all audio in memory, batch-transcribe on stop
//!
//! Mic callback accumulates all samples into a shared buffer.
//! When recording stops, `finalize()` returns the raw audio for transcription.

use anyhow::Result;
use std::sync::{Arc, Mutex};

use crate::audio::microphone::{self, MicStream};

/// Handle to a running audio pipeline.
///
/// Holds the mic stream and audio accumulator.
/// Call `finalize()` to stop recording and retrieve the audio buffer.
pub struct PipelineHandle {
    mic_stream: MicStream,
    accumulator: Arc<Mutex<Vec<f32>>>,
}

// MicStream is Send (unsafe impl in microphone.rs), and everything else is Arc/Mutex
unsafe impl Send for PipelineHandle {}

impl PipelineHandle {
    /// Stop mic capture without retrieving audio (used for cancel).
    pub fn stop_mic(&self) {
        self.mic_stream.stop();
    }

    /// Stop mic and return the accumulated audio buffer (16kHz mono f32).
    pub fn finalize(self) -> Vec<f32> {
        self.mic_stream.stop();
        // Flush remaining resampler samples before draining the accumulator
        self.mic_stream.flush_resampler();
        self.accumulator
            .lock()
            .map(|mut acc| acc.drain(..).collect())
            .unwrap_or_default()
    }
}

impl Drop for PipelineHandle {
    fn drop(&mut self) {
        self.mic_stream.stop();
    }
}

/// Start recording: capture mic audio into memory.
///
/// Returns a `PipelineHandle` that must be kept alive for recording to continue.
pub fn start_pipeline() -> Result<PipelineHandle> {
    let accumulator = Arc::new(Mutex::new(Vec::<f32>::new()));
    let acc_clone = Arc::clone(&accumulator);

    let stream = microphone::start_capture(Box::new(move |samples: &[f32]| {
        if let Ok(mut acc) = acc_clone.lock() {
            acc.extend_from_slice(samples);
        }
    }))?;

    Ok(PipelineHandle {
        mic_stream: stream,
        accumulator,
    })
}
