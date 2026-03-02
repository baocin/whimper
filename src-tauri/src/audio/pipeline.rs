//! Audio pipeline: mic → resample → VAD → ASR → transcript events
//!
//! Audio callback accumulates 10s chunks and sends them via mpsc channel
//! to a dedicated processing thread. The processing thread runs VAD first,
//! then ASR only on speech chunks, keeping the audio callback lightweight.
//!
//! When recording stops, `PipelineHandle::finalize()` drains the accumulator
//! and sends a Flush sentinel so tail audio is never lost.

use anyhow::Result;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter};

use crate::asr::{SileroVad, VoxtralAsr};
use crate::audio::microphone::{self, MicStream};

/// 10 seconds at 16kHz — better text/pad ratio than 5s chunks
const CHUNK_DURATION_SAMPLES: usize = 160_000;

/// Minimum tail length worth processing (0.5s at 16kHz)
const MIN_TAIL_SAMPLES: usize = 8_000;

#[derive(Clone, serde::Serialize)]
pub struct TranscriptUpdateEvent {
    pub text: String,
    pub is_final: bool,
}

enum PipelineMsg {
    Chunk(Vec<f32>),
    /// Drain remaining audio, process it, send final transcript back
    Flush {
        tail: Vec<f32>,
        reply: mpsc::SyncSender<String>,
    },
}

/// Handle to a running audio pipeline.
///
/// Holds the mic stream, accumulator, and channel sender.
/// Call `finalize()` to flush tail audio and get the final transcript,
/// or just drop to cancel without processing remaining audio.
pub struct PipelineHandle {
    mic_stream: MicStream,
    accumulator: Arc<Mutex<Vec<f32>>>,
    tx: Option<mpsc::Sender<PipelineMsg>>,
    asr: Arc<Mutex<Option<VoxtralAsr>>>,
}

// MicStream is Send (unsafe impl in microphone.rs), and everything else is Arc/Mutex
unsafe impl Send for PipelineHandle {}

impl PipelineHandle {
    /// Stop mic capture without processing tail audio.
    /// Used for cancel — just stops the mic, drops the channel.
    pub fn stop_mic(&self) {
        self.mic_stream.stop();
    }

    /// Flush remaining audio, wait for processing thread to finish,
    /// return the final accumulated transcript.
    pub fn finalize(mut self) -> String {
        self.mic_stream.stop();

        // Drain whatever's left in the accumulator
        let tail = self
            .accumulator
            .lock()
            .map(|mut acc| acc.drain(..).collect::<Vec<_>>())
            .unwrap_or_default();

        // Send flush with a reply channel
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(PipelineMsg::Flush { tail, reply: reply_tx });
            // Drop tx so processing thread exits after handling Flush
            drop(tx);
        }

        // Wait for final transcript (timeout after 30s to avoid hanging forever)
        match reply_rx.recv_timeout(std::time::Duration::from_secs(30)) {
            Ok(transcript) => transcript,
            Err(_) => {
                tracing::warn!("Timed out waiting for final transcript, falling back to current");
                // Fallback: read current transcript directly
                if let Ok(guard) = self.asr.lock() {
                    if let Some(ref asr) = *guard {
                        let t = asr.current_transcript();
                        asr.reset();
                        return t;
                    }
                }
                String::new()
            }
        }
    }
}

impl Drop for PipelineHandle {
    fn drop(&mut self) {
        self.mic_stream.stop();
        // Drop tx (if still held) to signal processing thread to exit
        self.tx.take();
    }
}

/// Process a single audio chunk: VAD → normalize → ASR → emit event
fn process_chunk(
    chunk: &[f32],
    vad: &Arc<Mutex<Option<SileroVad>>>,
    asr: &Arc<Mutex<Option<VoxtralAsr>>>,
    app: &AppHandle,
) {
    // Run VAD to check for speech
    let has_speech = if let Ok(mut vad_guard) = vad.lock() {
        if let Some(ref mut vad) = *vad_guard {
            match vad.is_speech_realtime(chunk) {
                Ok(speech) => {
                    if speech {
                        tracing::info!(samples = chunk.len(), "VAD: speech detected");
                    } else {
                        tracing::info!(samples = chunk.len(), "VAD: no speech, skipping ASR");
                    }
                    speech
                }
                Err(e) => {
                    tracing::warn!("VAD error, running ASR anyway: {}", e);
                    true // Fall through to ASR on VAD error
                }
            }
        } else {
            true // No VAD loaded, run ASR on everything
        }
    } else {
        true // Lock poisoned, run ASR anyway
    };

    if !has_speech {
        return;
    }

    // Peak-normalize chunk
    let peak = chunk.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    let rms = (chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32).sqrt();
    tracing::info!(
        samples = chunk.len(),
        rms = format_args!("{:.6}", rms),
        peak = format_args!("{:.6}", peak),
        "Audio chunk pre-normalization"
    );

    const MIN_PEAK: f32 = 0.001;
    let normalized = if peak > MIN_PEAK {
        let gain = 1.0 / peak;
        tracing::info!(gain = format_args!("{:.2}x", gain), "Applying gain correction");
        chunk
            .iter()
            .map(|&s| (s * gain).clamp(-1.0, 1.0))
            .collect::<Vec<f32>>()
    } else {
        tracing::warn!(
            "Audio too quiet (peak {:.6}), skipping normalization",
            peak
        );
        chunk.to_vec()
    };

    // Run ASR
    if let Ok(asr_guard) = asr.lock() {
        if let Some(ref asr) = *asr_guard {
            match asr.process_audio_chunk(&normalized) {
                Ok(text) => {
                    tracing::info!(
                        "Transcript update ({} chars): {:?}",
                        text.len(),
                        &text[..text.len().min(80)]
                    );
                    let _ = app.emit(
                        "transcript-update",
                        TranscriptUpdateEvent {
                            text,
                            is_final: false,
                        },
                    );
                }
                Err(e) => {
                    tracing::warn!("ASR chunk error: {}", e);
                }
            }
        }
    }
}

/// Start the audio pipeline: capture mic → chunk → VAD → ASR → emit events
///
/// Returns a `PipelineHandle` that must be kept alive for recording to continue.
/// Call `handle.finalize()` to flush tail audio and get the final transcript.
pub fn start_pipeline(
    app_handle: AppHandle,
    asr: Arc<Mutex<Option<VoxtralAsr>>>,
    vad: Arc<Mutex<Option<SileroVad>>>,
) -> Result<PipelineHandle> {
    let (tx, rx) = mpsc::channel::<PipelineMsg>();

    // ── Processing thread ────────────────────────────────────────────
    let asr_clone = Arc::clone(&asr);
    let vad_clone = Arc::clone(&vad);
    let app_clone = app_handle.clone();

    std::thread::Builder::new()
        .name("asr-pipeline".into())
        .spawn(move || {
            while let Ok(msg) = rx.recv() {
                match msg {
                    PipelineMsg::Chunk(chunk) => {
                        process_chunk(&chunk, &vad_clone, &asr_clone, &app_clone);
                    }
                    PipelineMsg::Flush { tail, reply } => {
                        // Process tail audio if non-trivial (>0.5s)
                        if tail.len() > MIN_TAIL_SAMPLES {
                            tracing::info!(
                                samples = tail.len(),
                                duration_s = tail.len() as f64 / 16000.0,
                                "Processing tail audio"
                            );
                            process_chunk(&tail, &vad_clone, &asr_clone, &app_clone);
                        } else if !tail.is_empty() {
                            tracing::info!(
                                samples = tail.len(),
                                "Tail audio too short, skipping"
                            );
                        }

                        // Reset VAD state for next session
                        if let Ok(mut g) = vad_clone.lock() {
                            if let Some(v) = g.as_mut() {
                                v.reset_state();
                            }
                        }

                        // Send back final transcript and reset ASR
                        let transcript = if let Ok(guard) = asr_clone.lock() {
                            if let Some(ref asr) = *guard {
                                let t = asr.current_transcript();
                                asr.reset();
                                t
                            } else {
                                String::new()
                            }
                        } else {
                            String::new()
                        };

                        let _ = reply.send(transcript);
                        break; // Exit processing thread
                    }
                }
            }
            tracing::info!("ASR processing thread exiting");
        })?;

    // ── Audio callback (CPAL thread — lightweight) ───────────────────
    let sample_accumulator = Arc::new(Mutex::new(Vec::<f32>::new()));
    let acc_clone = Arc::clone(&sample_accumulator);
    let tx_clone = tx.clone();

    let stream = microphone::start_capture(Box::new(move |samples: &[f32]| {
        let maybe_chunk = {
            let mut acc = match acc_clone.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            acc.extend_from_slice(samples);
            if acc.len() < CHUNK_DURATION_SAMPLES {
                return;
            }
            Some(acc.drain(..CHUNK_DURATION_SAMPLES).collect::<Vec<f32>>())
        };

        if let Some(chunk) = maybe_chunk {
            if tx_clone.send(PipelineMsg::Chunk(chunk)).is_err() {
                tracing::warn!("Processing thread gone, dropping audio chunk");
            }
        }
    }))?;

    Ok(PipelineHandle {
        mic_stream: stream,
        accumulator: sample_accumulator,
        tx: Some(tx),
        asr,
    })
}
