//! Audio pipeline: record all audio in memory, batch-transcribe on stop
//!
//! Mic callback accumulates all samples into a shared buffer.
//! When recording stops, `finalize()` returns the raw audio for transcription.

use anyhow::Result;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate::audio::microphone::{self, MicStream};
use crate::continuous::AudioSink;

/// 300ms of 16kHz audio
const PREROLL_CAPACITY: usize = 4800;

/// Ring buffer holding the last 300ms of background mic audio.
pub struct PreRollBuffer {
    buf: Mutex<VecDeque<f32>>,
}

impl PreRollBuffer {
    pub fn new() -> Self {
        Self {
            buf: Mutex::new(VecDeque::with_capacity(PREROLL_CAPACITY)),
        }
    }

    /// Push samples into the ring buffer, evicting oldest if over capacity.
    pub fn push(&self, samples: &[f32]) {
        match self.buf.lock() {
            Ok(mut buf) => {
                buf.extend(samples);
                let overflow = buf.len().saturating_sub(PREROLL_CAPACITY);
                if overflow > 0 {
                    buf.drain(..overflow);
                }
            }
            Err(e) => tracing::error!("pre-roll buffer lock poisoned on push: {}", e),
        }
    }

    /// Drain all samples from the buffer, returning them as a Vec.
    pub fn drain(&self) -> Vec<f32> {
        match self.buf.lock() {
            Ok(mut buf) => buf.drain(..).collect(),
            Err(e) => {
                tracing::error!("pre-roll buffer lock poisoned on drain: {}", e);
                Vec::new()
            }
        }
    }
}

/// Handle to a running audio pipeline.
///
/// Holds the mic stream and audio accumulator.
/// Call `finalize()` to stop recording and retrieve the audio buffer.
pub struct PipelineHandle {
    mic_stream: MicStream,
    accumulator: Arc<Mutex<Vec<f32>>>,
}

// Safety: `PipelineHandle` has exactly two fields:
//   - `mic_stream: MicStream`, which is independently declared `Send` (see the
//     `unsafe impl Send for MicStream` in microphone.rs; cpal's stream handle is
//     safe to move across threads on the platforms we target).
//   - `accumulator: Arc<Mutex<Vec<f32>>>`, which is `Send` unconditionally.
// A type whose every field is `Send` is itself sound to send across threads, so
// this `unsafe impl` only restates what the fields already guarantee — it exists
// solely because the auto-derive is blocked by `MicStream`'s `!Send` inner type.
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
        match self.accumulator.lock() {
            Ok(mut acc) => acc.drain(..).collect(),
            Err(e) => {
                tracing::error!("accumulator lock poisoned on finalize: {}", e);
                Vec::new()
            }
        }
    }
}

impl Drop for PipelineHandle {
    fn drop(&mut self) {
        self.mic_stream.stop();
    }
}

/// Start a background pre-roll mic that fills the given ring buffer.
/// If `sink` is provided, samples are also pushed to the continuous sink
/// (so continuous mode always has audio flowing without a second mic device).
pub fn start_preroll(
    buffer: Arc<PreRollBuffer>,
    sink: Option<Arc<AudioSink>>,
) -> Result<PipelineHandle> {
    let accumulator = Arc::new(Mutex::new(Vec::<f32>::new()));

    let stream = microphone::start_capture(Box::new(move |samples: &[f32]| {
        buffer.push(samples);
        if let Some(ref s) = sink {
            s.push(samples);
        }
    }))?;

    tracing::info!(
        "Pre-roll mic started ({}ms buffer)",
        PREROLL_CAPACITY * 1000 / 16000
    );

    Ok(PipelineHandle {
        mic_stream: stream,
        accumulator,
    })
}

/// Start recording: capture mic audio into memory.
///
/// If `preroll` is provided, its contents are prepended to the accumulator.
pub fn start_pipeline(preroll: Option<&PreRollBuffer>) -> Result<PipelineHandle> {
    let accumulator = Arc::new(Mutex::new(Vec::<f32>::new()));

    // Prepend pre-roll audio
    if let Some(pr) = preroll {
        let pre = pr.drain();
        if !pre.is_empty() {
            tracing::info!(
                "Prepending {} pre-roll samples ({:.0}ms)",
                pre.len(),
                pre.len() as f64 / 16.0
            );
            if let Ok(mut acc) = accumulator.lock() {
                *acc = pre;
            }
        }
    }

    let acc_clone = Arc::clone(&accumulator);

    let stream =
        microphone::start_capture(Box::new(move |samples: &[f32]| match acc_clone.lock() {
            Ok(mut acc) => acc.extend_from_slice(samples),
            Err(e) => tracing::error!("recording accumulator lock poisoned: {}", e),
        }))?;

    Ok(PipelineHandle {
        mic_stream: stream,
        accumulator,
    })
}
