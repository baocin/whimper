//! Audio pipeline: record audio with optional streaming preprocessing.
//!
//! Two modes:
//! - Batch mode: mic callback accumulates raw samples; transcribe() does all preprocessing.
//! - Streaming mode: a processing thread runs highpass + VAD-gated GTCRN incrementally
//!   during recording, so transcribe() receives pre-enhanced audio.

use anyhow::Result;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate::asr::{GtcrnEnhancer, SileroVad, StreamingPreprocessor};
use crate::audio::microphone::{self, MicStream};

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
        if let Ok(mut buf) = self.buf.lock() {
            buf.extend(samples);
            let overflow = buf.len().saturating_sub(PREROLL_CAPACITY);
            if overflow > 0 {
                buf.drain(..overflow);
            }
        }
    }

    /// Drain all samples from the buffer, returning them as a Vec.
    pub fn drain(&self) -> Vec<f32> {
        self.buf
            .lock()
            .map(|mut buf| buf.drain(..).collect())
            .unwrap_or_default()
    }
}

/// Handle to a running audio pipeline.
///
/// Holds the mic stream and audio accumulator.
/// In streaming mode, also holds the processing thread and channel sender.
/// Call `finalize()` to stop recording and retrieve the audio buffer.
pub struct PipelineHandle {
    mic_stream: Option<MicStream>,
    accumulator: Arc<Mutex<Vec<f32>>>,
    /// Processing thread for streaming mode (None in batch mode)
    processing_thread: Option<std::thread::JoinHandle<()>>,
    /// Channel sender for streaming mode — dropping this signals the processing thread to flush
    tx: Option<std::sync::mpsc::SyncSender<Vec<f32>>>,
}

// MicStream is Send (unsafe impl in microphone.rs), and everything else is Arc/Mutex
unsafe impl Send for PipelineHandle {}

impl PipelineHandle {
    /// Stop mic capture without retrieving audio (used for cancel).
    pub fn stop_mic(&self) {
        if let Some(ref mic) = self.mic_stream {
            mic.stop();
        }
    }

    /// Stop mic and return the accumulated audio buffer (16kHz mono f32).
    /// In streaming mode, also signals the processing thread to flush and waits for it.
    pub fn finalize(mut self) -> Vec<f32> {
        if let Some(ref mic) = self.mic_stream {
            mic.stop();
            mic.flush_resampler();
        }

        // Drop the stored sender
        drop(self.tx.take());
        // Drop the mic stream — this drops the callback closure containing the cloned sender.
        // Both senders must be gone before rx.recv() will return Err.
        drop(self.mic_stream.take());

        // Now all senders are dropped — processing thread's rx.recv() returns Err
        if let Some(thread) = self.processing_thread.take() {
            if let Err(e) = thread.join() {
                tracing::error!("Processing thread panicked: {:?}", e);
            }
        }

        self.accumulator
            .lock()
            .map(|mut acc| acc.drain(..).collect())
            .unwrap_or_default()
    }

    /// Returns true if this pipeline uses streaming preprocessing.
    pub fn is_streaming(&self) -> bool {
        self.processing_thread.is_some()
    }
}

impl Drop for PipelineHandle {
    fn drop(&mut self) {
        if let Some(ref mic) = self.mic_stream {
            mic.stop();
        }
        // Clean up streaming resources on drop (e.g. cancel path)
        drop(self.tx.take());
        drop(self.mic_stream.take());
        if let Some(thread) = self.processing_thread.take() {
            let _ = thread.join();
        }
    }
}

/// Start a background pre-roll mic that fills the given ring buffer.
pub fn start_preroll(buffer: Arc<PreRollBuffer>) -> Result<PipelineHandle> {
    let accumulator = Arc::new(Mutex::new(Vec::<f32>::new()));

    let stream = microphone::start_capture(Box::new(move |samples: &[f32]| {
        buffer.push(samples);
    }))?;

    tracing::info!("Pre-roll mic started ({}ms buffer)", PREROLL_CAPACITY * 1000 / 16000);

    Ok(PipelineHandle {
        mic_stream: Some(stream),
        accumulator,
        processing_thread: None,
        tx: None,
    })
}

/// Start recording: capture mic audio into memory (batch mode, no streaming preprocessing).
///
/// If `preroll` is provided, its contents are prepended to the accumulator.
pub fn start_pipeline(preroll: Option<&PreRollBuffer>) -> Result<PipelineHandle> {
    let accumulator = Arc::new(Mutex::new(Vec::<f32>::new()));

    // Prepend pre-roll audio
    if let Some(pr) = preroll {
        let pre = pr.drain();
        if !pre.is_empty() {
            tracing::info!("Prepending {} pre-roll samples ({:.0}ms)", pre.len(), pre.len() as f64 / 16.0);
            if let Ok(mut acc) = accumulator.lock() {
                *acc = pre;
            }
        }
    }

    let acc_clone = Arc::clone(&accumulator);

    let stream = microphone::start_capture(Box::new(move |samples: &[f32]| {
        if let Ok(mut acc) = acc_clone.lock() {
            acc.extend_from_slice(samples);
        }
    }))?;

    Ok(PipelineHandle {
        mic_stream: Some(stream),
        accumulator,
        processing_thread: None,
        tx: None,
    })
}

/// Start recording with streaming preprocessing: highpass + VAD-gated GTCRN
/// run incrementally on a processing thread during recording.
///
/// The mic callback sends raw samples through a channel to the processing thread,
/// which runs StreamingPreprocessor and writes enhanced samples to the accumulator.
/// `finalize()` returns pre-enhanced audio.
pub fn start_pipeline_streaming(
    preroll: Option<&PreRollBuffer>,
    vad: Arc<Mutex<SileroVad>>,
    gtcrn: Arc<Mutex<GtcrnEnhancer>>,
) -> Result<PipelineHandle> {
    let accumulator = Arc::new(Mutex::new(Vec::<f32>::new()));

    // Prepend pre-roll audio (will be processed through streaming preprocessor)
    let preroll_samples = preroll.map(|pr| pr.drain()).unwrap_or_default();

    // Channel for mic callback → processing thread
    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<f32>>(64);

    let acc_for_thread = Arc::clone(&accumulator);
    let processing_thread = std::thread::Builder::new()
        .name("streaming-preprocess".into())
        .spawn(move || {
            let mut preprocessor = StreamingPreprocessor::new();

            // Process pre-roll samples first
            if !preroll_samples.is_empty() {
                tracing::info!(
                    "Streaming: processing {} pre-roll samples ({:.0}ms)",
                    preroll_samples.len(),
                    preroll_samples.len() as f64 / 16.0
                );
                let enhanced = preprocessor.process_chunk(&preroll_samples, &vad, &gtcrn);
                if !enhanced.is_empty() {
                    if let Ok(mut acc) = acc_for_thread.lock() {
                        acc.extend_from_slice(&enhanced);
                    }
                }
            }

            // Process incoming mic chunks
            while let Ok(samples) = rx.recv() {
                let enhanced = preprocessor.process_chunk(&samples, &vad, &gtcrn);
                if !enhanced.is_empty() {
                    if let Ok(mut acc) = acc_for_thread.lock() {
                        acc.extend_from_slice(&enhanced);
                    }
                }
            }

            // Channel closed (sender dropped in finalize) — flush remaining
            let flushed = preprocessor.flush(&vad, &gtcrn);
            if !flushed.is_empty() {
                if let Ok(mut acc) = acc_for_thread.lock() {
                    acc.extend_from_slice(&flushed);
                }
            }
        })?;

    let tx_for_mic = tx.clone();
    let stream = microphone::start_capture(Box::new(move |samples: &[f32]| {
        // Non-blocking send — if the processing thread falls behind, we drop samples
        // rather than blocking the audio thread
        let _ = tx_for_mic.try_send(samples.to_vec());
    }))?;

    Ok(PipelineHandle {
        mic_stream: Some(stream),
        accumulator,
        processing_thread: Some(processing_thread),
        tx: Some(tx),
    })
}
