//! Live dictation: type text into the focused app while the user speaks,
//! then reconcile against the higher-quality full-buffer transcript on stop.
//!
//! Nemotron is RNNT and append-only — it never retracts a token mid-stream, so
//! nothing is deleted while speaking. The one place text can change is at the end:
//! the final pass runs GTCRN, spectral gate and VAD trim over the whole buffer,
//! which streaming cannot do, so it may disagree with what was typed live.
//! `finish` repairs that by backspacing only the divergent tail.

use crate::asr::NemotronAsr;
use crate::paste;
use std::sync::{Arc, Mutex};

/// 560ms at 16kHz — the model's streaming chunk size.
///
/// ponytail: the crate recomputes mel over its whole buffer on every call, so
/// feeding it the preprocessor's small batches directly would be quadratic.
const CHUNK_SAMPLES: usize = 8960;

/// Tracks what has actually been typed into the target app during one utterance.
///
/// ponytail: no focus tracking. Keystrokes go wherever focus happens to be —
/// switching apps mid-dictation types (and backspaces) into the new one.
pub struct LiveTypist {
    asr: Arc<Mutex<Option<NemotronAsr>>>,
    pending: Vec<f32>,
    typed: String,
    aborted: bool,
}

impl LiveTypist {
    pub fn new(asr: Arc<Mutex<Option<NemotronAsr>>>) -> Self {
        if let Ok(guard) = asr.lock() {
            if let Some(a) = guard.as_ref() {
                let _ = a.reset_stream();
            }
        }

        Self {
            asr,
            pending: Vec::with_capacity(CHUNK_SAMPLES * 2),
            typed: String::new(),
            aborted: false,
        }
    }

    /// Text typed into the target app so far.
    pub fn typed(&self) -> &str {
        &self.typed
    }

    /// True once typing has been abandoned (focus moved away).
    pub fn aborted(&self) -> bool {
        self.aborted
    }

    /// Feed enhanced audio. Transcribes and types whenever a full chunk is buffered.
    pub fn feed(&mut self, samples: &[f32]) {
        if self.aborted {
            return;
        }

        self.pending.extend_from_slice(samples);

        while self.pending.len() >= CHUNK_SAMPLES {
            let chunk: Vec<f32> = self.pending.drain(..CHUNK_SAMPLES).collect();
            self.emit(&chunk);
            if self.aborted {
                return;
            }
        }
    }

    /// Transcribe one chunk and type whatever it emitted.
    fn emit(&mut self, chunk: &[f32]) {
        let text = {
            let guard = match self.asr.lock() {
                Ok(g) => g,
                Err(e) => {
                    tracing::error!("ASR lock poisoned during dictation: {}", e);
                    return;
                }
            };
            let Some(asr) = guard.as_ref() else { return };

            match asr.transcribe_chunk(chunk) {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!("Streaming chunk failed: {}", e);
                    return;
                }
            }
        };

        tracing::info!("stream chunk emitted {:?}", text);

        // The model emits leading-space-delimited tokens, so the first chunk of an
        // utterance arrives as " Testing". The final pass returns "Testing ...", and
        // that one-character difference puts the common prefix at zero — forcing a
        // full delete-and-retype of otherwise correct text. Drop it at the start.
        let text = if self.typed.is_empty() {
            text.trim_start().to_string()
        } else {
            text
        };

        if text.is_empty() {
            return;
        }

        if let Err(e) = paste::type_text(&text) {
            tracing::error!("Live typing failed: {}", e);
            self.aborted = true;
            return;
        }

        self.typed.push_str(&text);
        tracing::info!("typed {:?} (total {} chars)", text, self.typed.chars().count());
    }

    /// Flush any partial chunk so the tail of the utterance is typed too.
    pub fn flush(&mut self) {
        if self.aborted || self.pending.is_empty() {
            return;
        }
        let chunk: Vec<f32> = std::mem::take(&mut self.pending);
        self.emit(&chunk);
    }

    /// Reconcile what was typed against the final full-buffer transcript.
    ///
    /// Returns true if the target app now holds `final_text`.
    pub fn finish(&mut self, final_text: &str) -> bool {
        let (to_delete, to_type) = paste::reconcile(&self.typed, final_text);

        if to_delete == 0 && to_type.is_empty() {
            tracing::info!("Live transcript already matched final pass — no correction");
            return true;
        }

        tracing::info!(
            "Reconciling: backspace {} char(s), type {:?}",
            to_delete,
            &to_type[..to_type.len().min(60)]
        );

        if let Err(e) = paste::press_backspace(to_delete) {
            tracing::error!("Backspace failed: {}", e);
            return false;
        }

        if let Err(e) = paste::type_text(&to_type) {
            tracing::error!("Correction typing failed: {}", e);
            return false;
        }

        self.typed = final_text.to_string();
        true
    }
}
