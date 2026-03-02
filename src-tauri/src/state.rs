use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::asr::{SileroVad, VoxtralAsr};
use crate::audio::pipeline::PipelineHandle;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum ModelStatus {
    #[serde(rename = "not_downloaded")]
    NotDownloaded,
    #[serde(rename = "downloading")]
    Downloading {
        progress: f64,
        speed_mbps: f64,
    },
    #[serde(rename = "downloaded")]
    Downloaded,
    #[serde(rename = "loading")]
    Loading,
    #[serde(rename = "ready")]
    Ready,
    #[serde(rename = "error")]
    Error { message: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordingState {
    Idle,
    Listening,
}

pub struct AppState {
    pub model_status: Mutex<ModelStatus>,
    pub recording_state: Mutex<RecordingState>,
    pub asr: Arc<std::sync::Mutex<Option<VoxtralAsr>>>,
    pub vad: Arc<std::sync::Mutex<Option<SileroVad>>>,
    pub cancel_download: AtomicBool,
    /// PID of the app that was frontmost before we opened the overlay
    pub previous_app_pid: Mutex<Option<i32>>,
    /// Active audio pipeline — held here so it doesn't get dropped while recording
    pub mic_stream: std::sync::Mutex<Option<PipelineHandle>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            model_status: Mutex::new(ModelStatus::NotDownloaded),
            recording_state: Mutex::new(RecordingState::Idle),
            asr: Arc::new(std::sync::Mutex::new(None)),
            vad: Arc::new(std::sync::Mutex::new(None)),
            cancel_download: AtomicBool::new(false),
            previous_app_pid: Mutex::new(None),
            mic_stream: std::sync::Mutex::new(None),
        }
    }

    pub fn is_download_cancelled(&self) -> bool {
        self.cancel_download.load(Ordering::SeqCst)
    }

    pub fn set_cancel_download(&self, cancel: bool) {
        self.cancel_download.store(cancel, Ordering::SeqCst);
    }
}

/// Return the model directory path: ~/.whimper/models/voxtral-mini-4b/
pub fn model_dir() -> std::path::PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
    home.join(".whimper").join("models").join("voxtral-mini-4b")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_status_serialization() {
        let status = ModelStatus::Downloading {
            progress: 0.5,
            speed_mbps: 10.2,
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("downloading"));
        assert!(json.contains("0.5"));
    }

    #[test]
    fn test_model_dir_path() {
        let dir = model_dir();
        assert!(dir.ends_with(".whimper/models/voxtral-mini-4b"));
    }

    #[test]
    fn test_app_state_cancel_download() {
        let state = AppState::new();
        assert!(!state.is_download_cancelled());
        state.set_cancel_download(true);
        assert!(state.is_download_cancelled());
    }
}
