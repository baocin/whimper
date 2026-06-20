use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::asr::ParakeetAsr;
use crate::audio::pipeline::{PipelineHandle, PreRollBuffer};

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

/// Whether the global hotkey can work, and if not, why — surfaced to the UI.
///
/// Lives here (not in the Linux-only `input` module) so the Tauri command and
/// `AppState` field compile on macOS, where it stays `Available` (the macOS
/// hotkey goes through the global-shortcut plugin, not evdev).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HotkeyStatus {
    /// At least one keyboard is readable; the listener is running.
    Available,
    /// User IS in the `input` group, but this session hasn't picked it up and
    /// self-heal didn't apply. Logging out and back in fixes it.
    NeedsRelogin,
    /// User is NOT in the `input` group. They must be added, then re-login.
    NeedsGroupAdd,
    /// No keyboard device was found at all.
    NoKeyboard,
}

impl Default for HotkeyStatus {
    fn default() -> Self {
        HotkeyStatus::Available
    }
}

pub struct AppState {
    pub model_status: Mutex<ModelStatus>,
    pub recording_state: Mutex<RecordingState>,
    pub asr: Arc<std::sync::Mutex<Option<ParakeetAsr>>>,
    pub cancel_download: AtomicBool,
    /// PID of the app that was frontmost before we opened the overlay
    pub previous_app_pid: Mutex<Option<i32>>,
    /// Active audio pipeline — held here so it doesn't get dropped while recording
    pub mic_stream: std::sync::Mutex<Option<PipelineHandle>>,
    /// Pre-roll ring buffer: holds last 300ms of background mic audio
    pub preroll_buffer: Arc<PreRollBuffer>,
    /// Handle to the background pre-roll mic stream
    pub preroll_mic: std::sync::Mutex<Option<PipelineHandle>>,
    /// Whether the global hotkey is usable (Linux: evdev keyboard readable).
    pub hotkey_status: std::sync::Mutex<HotkeyStatus>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            model_status: Mutex::new(ModelStatus::NotDownloaded),
            recording_state: Mutex::new(RecordingState::Idle),
            asr: Arc::new(std::sync::Mutex::new(None)),
            cancel_download: AtomicBool::new(false),
            previous_app_pid: Mutex::new(None),
            mic_stream: std::sync::Mutex::new(None),
            preroll_buffer: Arc::new(PreRollBuffer::new()),
            preroll_mic: std::sync::Mutex::new(None),
            hotkey_status: std::sync::Mutex::new(HotkeyStatus::default()),
        }
    }

    pub fn is_download_cancelled(&self) -> bool {
        self.cancel_download.load(Ordering::SeqCst)
    }

    pub fn set_cancel_download(&self, cancel: bool) {
        self.cancel_download.store(cancel, Ordering::SeqCst);
    }
}

/// Return the whimper data directory: ~/.whimper
///
/// Base for the model cache and the transcript log. Falls back to a relative
/// `.whimper` if the home directory can't be resolved (better than panicking).
pub fn whimper_dir() -> std::path::PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
    home.join(".whimper")
}

/// Return the model directory path: ~/.whimper/models/parakeet-tdt/
pub fn model_dir() -> std::path::PathBuf {
    whimper_dir().join("models").join("parakeet-tdt")
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
    fn test_hotkey_status_serialization() {
        assert_eq!(
            serde_json::to_string(&HotkeyStatus::Available).unwrap(),
            "\"available\""
        );
        assert_eq!(
            serde_json::to_string(&HotkeyStatus::NeedsGroupAdd).unwrap(),
            "\"needs_group_add\""
        );
        assert_eq!(
            serde_json::to_string(&HotkeyStatus::NeedsRelogin).unwrap(),
            "\"needs_relogin\""
        );
        assert_eq!(
            serde_json::to_string(&HotkeyStatus::NoKeyboard).unwrap(),
            "\"no_keyboard\""
        );
        assert_eq!(HotkeyStatus::default(), HotkeyStatus::Available);
    }

    #[test]
    fn test_model_dir_path() {
        let dir = model_dir();
        assert!(dir.ends_with(".whimper/models/parakeet-tdt"));
    }

    #[test]
    fn test_app_state_cancel_download() {
        let state = AppState::new();
        assert!(!state.is_download_cancelled());
        state.set_cancel_download(true);
        assert!(state.is_download_cancelled());
    }
}
