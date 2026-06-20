mod asr;
mod audio;
mod download;
#[cfg(target_os = "linux")]
mod input;
mod paste;
mod state;
mod transcript;

use state::{AppState, ModelStatus, RecordingState};
use std::sync::Arc;
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_global_shortcut::ShortcutState;

// ─��� Tauri Commands ────────────────────────────────────────────────────────

#[tauri::command]
async fn check_model_status(
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<ModelStatus, String> {
    if download::is_model_downloaded() {
        if let Ok(guard) = state.asr.lock() {
            if guard.is_some() {
                return Ok(ModelStatus::Ready);
            }
        }
        return Ok(ModelStatus::Downloaded);
    }
    Ok(ModelStatus::NotDownloaded)
}

#[tauri::command]
async fn start_download(
    app_handle: AppHandle,
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<(), String> {
    state.set_cancel_download(false);
    let state_ref = state.inner().clone();
    let app = app_handle.clone();

    tokio::spawn(async move {
        if let Err(e) = download::download_model(app.clone(), &state_ref).await {
            tracing::error!("Download failed: {}", e);
            let _ = app.emit("download-error", e.to_string());
        }
    });

    Ok(())
}

#[tauri::command]
async fn cancel_download(state: tauri::State<'_, Arc<AppState>>) -> Result<(), String> {
    state.set_cancel_download(true);
    Ok(())
}

#[tauri::command]
async fn load_model(state: tauri::State<'_, Arc<AppState>>) -> Result<(), String> {
    let model_dir = state::model_dir();
    let dir_str = model_dir.to_string_lossy().to_string();

    *state.model_status.lock().await = ModelStatus::Loading;

    let asr_arc = state.asr.clone();

    tokio::task::spawn_blocking(move || {
        let asr = asr::ParakeetAsr::new();
        asr.load_model(&dir_str).map_err(|e| e.to_string())?;
        if let Ok(mut guard) = asr_arc.lock() {
            *guard = Some(asr);
        }
        Ok::<(), String>(())
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e: String| e)?;

    *state.model_status.lock().await = ModelStatus::Ready;

    // Start background pre-roll mic
    start_preroll_mic(&state);

    Ok(())
}

#[tauri::command]
async fn cancel_recording(
    app_handle: AppHandle,
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<(), String> {
    let mut recording = state.recording_state.lock().await;
    if *recording == RecordingState::Listening {
        *recording = RecordingState::Idle;

        // Stop mic stream (don't process audio on cancel)
        if let Ok(mut guard) = state.mic_stream.lock() {
            if let Some(handle) = guard.take() {
                handle.stop_mic();
            }
        }

        // Close overlay window
        if let Some(window) = app_handle.get_webview_window("overlay") {
            let _ = window.close();
        }

        // Restart pre-roll mic
        start_preroll_mic(&state);
    }
    Ok(())
}

#[tauri::command]
async fn hide_main_window(app_handle: AppHandle) -> Result<(), String> {
    if let Some(window) = app_handle.get_webview_window("main") {
        window.hide().map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
async fn check_hotkey_status(
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<state::HotkeyStatus, String> {
    Ok(state.hotkey_status.lock().map(|g| *g).unwrap_or_default())
}

// ── Pre-roll Mic ─────────────────────────────────────────────────────────

/// Start (or restart) the background pre-roll mic that fills the ring buffer.
fn start_preroll_mic(state: &Arc<AppState>) {
    let buffer = Arc::clone(&state.preroll_buffer);
    match audio::pipeline::start_preroll(buffer) {
        Ok(handle) => {
            if let Ok(mut guard) = state.preroll_mic.lock() {
                *guard = Some(handle);
            }
        }
        Err(e) => {
            tracing::error!("Failed to start pre-roll mic: {}", e);
        }
    }
}

/// Recover the recording state machine to Idle after a failed start: reset
/// state, close the overlay, and resume the background pre-roll mic. Used when a
/// start-up stage fails mid-way so a transient error can't wedge Listening.
async fn recover_to_idle(app: &AppHandle, state: &Arc<AppState>) {
    *state.recording_state.lock().await = RecordingState::Idle;
    if let Some(window) = app.get_webview_window("overlay") {
        let _ = window.close();
    }
    start_preroll_mic(state);
}

// ── Global Hotkey Handler ────────────────────────────────────────────────

fn handle_hotkey(app_handle: &AppHandle, state: &Arc<AppState>) {
    let app = app_handle.clone();
    let state = state.clone();

    tauri::async_runtime::spawn(async move {
        let mut recording = state.recording_state.lock().await;

        match *recording {
            RecordingState::Idle => {
                // Check model is ready
                match state.asr.lock() {
                    Ok(guard) => {
                        if guard.is_none() {
                            tracing::warn!("Model not loaded, ignoring hotkey");
                            return;
                        }
                    }
                    Err(e) => {
                        tracing::error!("ASR lock poisoned, ignoring hotkey: {}", e);
                        return;
                    }
                }

                // Save frontmost app PID before we steal focus
                if let Some(pid) = paste::get_frontmost_app_pid() {
                    *state.previous_app_pid.lock().await = Some(pid);
                }

                *recording = RecordingState::Listening;
                tracing::info!("Recording started");
                drop(recording);

                // Create overlay window
                let overlay = tauri::WebviewWindowBuilder::new(
                    &app,
                    "overlay",
                    tauri::WebviewUrl::App("/overlay".into()),
                )
                // Title is invisible (decorations off) but lets Hyprland match a
                // no-focus window rule so the overlay doesn't steal keyboard focus.
                .title("whimper-overlay")
                .inner_size(400.0, 80.0)
                .decorations(false)
                .transparent(true)
                .always_on_top(true)
                .focused(false)
                .resizable(false)
                .build();

                if let Ok(window) = overlay {
                    if let Ok(monitor) = window.current_monitor() {
                        if let Some(m) = monitor {
                            let screen_width: f64 = m.size().width as f64;
                            let x = (screen_width / 2.0 - 200.0) as i32;
                            let _ = window.set_position(tauri::Position::Physical(
                                tauri::PhysicalPosition::new(x, 100),
                            ));
                        }
                    }
                }

                // Re-activate previous app so overlay doesn't steal keyboard focus
                if let Some(pid) = *state.previous_app_pid.lock().await {
                    let _ = paste::activate_app(pid);
                }

                // Stop pre-roll mic before starting recording (avoid two mic streams)
                match state.preroll_mic.lock() {
                    Ok(mut guard) => {
                        if let Some(handle) = guard.take() {
                            handle.stop_mic();
                            tracing::info!("Pre-roll mic stopped for recording");
                        }
                    }
                    Err(e) => tracing::error!("preroll_mic lock poisoned: {}", e),
                }

                // Start audio pipeline with pre-roll audio prepended
                match audio::pipeline::start_pipeline(Some(&state.preroll_buffer)) {
                    Ok(handle) => {
                        // Store the stream, then drop the (non-Send) std mutex
                        // guard *before* any await. `stored` carries the outcome
                        // out of the guard's scope.
                        let stored = match state.mic_stream.lock() {
                            Ok(mut guard) => {
                                *guard = Some(handle);
                                true
                            }
                            Err(e) => {
                                tracing::error!("mic_stream lock poisoned: {}", e);
                                false
                            }
                        };
                        if !stored {
                            recover_to_idle(&app, &state).await;
                        }
                    }
                    Err(e) => {
                        // Pipeline failed to start: recover the state machine to
                        // Idle so a transient mic error doesn't wedge Listening
                        // forever, close the overlay, and resume pre-roll.
                        tracing::error!("Failed to start audio pipeline: {}", e);
                        recover_to_idle(&app, &state).await;
                    }
                }
            }
            RecordingState::Listening => {
                *recording = RecordingState::Idle;
                drop(recording);

                // Finalize pipeline: stop mic, get audio buffer
                let audio_samples = match state.mic_stream.lock() {
                    Ok(mut guard) => guard.take().map(|h| h.finalize()).unwrap_or_default(),
                    Err(e) => {
                        tracing::error!("mic_stream lock poisoned on stop: {}", e);
                        Vec::new()
                    }
                };

                tracing::info!(
                    "Recording stopped, {} samples ({:.1}s)",
                    audio_samples.len(),
                    audio_samples.len() as f64 / 16000.0
                );

                // Restart pre-roll mic now that recording is done
                start_preroll_mic(&state);

                let _ = app.emit("recording-state", "processing");

                // Fallback duration if transcribe never runs (empty audio / error).
                let fallback_duration_ms = (audio_samples.len() as f64 / 16.0) as u64;

                // Batch-transcribe on blocking thread. Capture the full result
                // (text + timings) so the durable log gets untruncated text and RTF.
                let asr_arc = state.asr.clone();
                let (transcript, processing_time_ms, audio_duration_ms) = if !audio_samples
                    .is_empty()
                {
                    match tokio::task::spawn_blocking(move || {
                        let guard = asr_arc.lock().map_err(|e| format!("ASR lock: {}", e))?;
                        let asr = guard
                            .as_ref()
                            .ok_or_else(|| "Model not loaded".to_string())?;
                        asr.transcribe(&audio_samples).map_err(|e| e.to_string())
                    })
                    .await
                    {
                        Ok(Ok(r)) => (r.text, r.processing_time_ms, r.audio_duration_ms),
                        Ok(Err(e)) => {
                            tracing::error!("Transcription failed: {}", e);
                            (String::new(), 0, fallback_duration_ms)
                        }
                        Err(e) => {
                            tracing::error!("Transcription task panicked: {}", e);
                            (String::new(), 0, fallback_duration_ms)
                        }
                    }
                } else {
                    (String::new(), 0, fallback_duration_ms)
                };

                // tracing stays truncated for readability; the JSONL log below
                // holds the full text — don't double-truncate.
                tracing::info!(
                    "Transcript ({} chars): {:?}",
                    transcript.len(),
                    &transcript[..transcript.len().min(120)]
                );

                // Close overlay
                if let Some(window) = app.get_webview_window("overlay") {
                    let _ = window.close();
                }

                // Decide paste using the same classification we persist, so the
                // logged flags can never disagree with the behaviour.
                let (empty, hallucination) = transcript::classify_flags(&transcript);
                let mut pasted = false;
                if !empty && !hallucination {
                    let prev_pid = *state.previous_app_pid.lock().await;

                    // Small delay for overlay to close
                    tokio::time::sleep(std::time::Duration::from_millis(150)).await;

                    match paste::paste_text(&transcript, prev_pid) {
                        Ok(_) => {
                            tracing::info!("Paste succeeded");
                            pasted = true;
                        }
                        Err(e) => tracing::error!("Paste failed: {}", e),
                    }
                }

                // Durable log of EVERY attempt (empty / hallucination / skipped
                // paste included). Append happens after paste so it never adds
                // latency to or blocks the paste path; a write error is logged
                // inside `append` and is non-fatal.
                let record = transcript::TranscriptRecord::new(
                    transcript,
                    audio_duration_ms,
                    processing_time_ms,
                    pasted,
                );
                transcript::append(&record);
            }
        }
    });
}

// ── App Entry Point ──────────────────────────────────────────────────────

pub fn run() {
    // NVIDIA + Wayland: WebKitGTK's DMABUF renderer crashes the GTK backend with
    // "Error 71 (Protocol error) dispatching to Wayland display". Disabling it
    // forces a software/GL path that renders correctly. Must be set before the
    // Tauri/WebKit runtime initializes.
    #[cfg(target_os = "linux")]
    if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
        std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
    }

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("whimper=info".parse().unwrap()),
        )
        .init();

    // Self-heal keyboard access (Linux): if we can't read the keyboard because
    // the session never picked up the `input` group, but the user IS a member,
    // re-exec under `sg input` (no sudo) so the hotkey works without re-login.
    // The env guard prevents an infinite loop; if self-heal can't apply, we fall
    // through and the UI surfaces the problem instead.
    #[cfg(target_os = "linux")]
    if std::env::var_os("WHIMPER_INPUT_REEXEC").is_none()
        && input::probe_keyboard_access() == input::Probe::PermissionDenied
        && input::user_in_input_group()
    {
        let e = input::reexec_with_input_group();
        tracing::error!("self-heal re-exec failed, continuing without it: {}", e);
    }

    let app_state = Arc::new(AppState::new());

    let state_for_shortcut = app_state.clone();

    tauri::Builder::default()
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(move |app, _shortcut, event| {
                    if event.state() == ShortcutState::Pressed {
                        handle_hotkey(app, &state_for_shortcut);
                    }
                })
                .build(),
        )
        .manage(app_state.clone())
        .invoke_handler(tauri::generate_handler![
            check_model_status,
            start_download,
            cancel_download,
            load_model,
            cancel_recording,
            hide_main_window,
            check_hotkey_status,
        ])
        .setup(move |app| {
            // Hotkey: macOS uses the Tauri global-shortcut plugin. On Wayland
            // that can't grab global keys, so Linux reads evdev directly.
            #[cfg(target_os = "macos")]
            {
                use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut};
                match "Alt+Space".parse::<Shortcut>() {
                    Ok(shortcut) => {
                        if let Err(e) = app.global_shortcut().register(shortcut) {
                            tracing::error!("Failed to register Alt+Space shortcut: {}", e);
                        }
                    }
                    Err(e) => tracing::error!("Failed to parse Alt+Space shortcut: {}", e),
                }
            }

            #[cfg(target_os = "linux")]
            {
                let app_for_hotkey = app.handle().clone();
                let state_for_hotkey = app_state.clone();
                let hotkey_status = input::start_evdev_listener(move || {
                    handle_hotkey(&app_for_hotkey, &state_for_hotkey);
                });
                if let Ok(mut g) = app_state.hotkey_status.lock() {
                    *g = hotkey_status;
                }
                if hotkey_status != state::HotkeyStatus::Available {
                    // Live-notify the UI (it also queries on mount).
                    let _ = app.handle().emit("hotkey-status", hotkey_status);
                }
            }

            // Auto-load model if already downloaded
            let app_handle2 = app.handle().clone();
            let state_for_load = app_state.clone();
            if download::is_model_downloaded() {
                tauri::async_runtime::spawn(async move {
                    let model_dir = state::model_dir();
                    let dir_str = model_dir.to_string_lossy().to_string();
                    let asr_arc = state_for_load.asr.clone();

                    let result = tokio::task::spawn_blocking(move || {
                        let asr = asr::ParakeetAsr::new();
                        asr.load_model(&dir_str)?;
                        Ok::<_, anyhow::Error>(asr)
                    })
                    .await;

                    match result {
                        Ok(Ok(asr)) => {
                            if let Ok(mut guard) = asr_arc.lock() {
                                *guard = Some(asr);
                            }
                            *state_for_load.model_status.lock().await = ModelStatus::Ready;
                            let _ = app_handle2.emit("model-ready", ());
                            tracing::info!("Parakeet TDT model auto-loaded successfully");

                            // Start background pre-roll mic
                            start_preroll_mic(&state_for_load);
                        }
                        Ok(Err(e)) => {
                            tracing::error!("Failed to auto-load model: {}", e);
                        }
                        Err(e) => {
                            tracing::error!("Model loading task panicked: {}", e);
                        }
                    }
                });
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .unwrap_or_else(|e| {
            // Don't panic on a runtime failure of the Tauri event loop; surface
            // it through tracing so it lands in logs instead of an abort.
            tracing::error!("fatal: whimper runtime error: {}", e);
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tauri_conf_is_valid_json() {
        let conf_str = include_str!("../tauri.conf.json");
        let parsed: serde_json::Value = serde_json::from_str(conf_str)
            .expect("tauri.conf.json is not valid JSON");

        assert!(parsed.get("productName").is_some(), "missing productName");
        assert!(parsed.get("identifier").is_some(), "missing identifier");
        assert!(parsed.get("app").is_some(), "missing app");
    }

    #[test]
    fn test_plugins_config_has_no_unit_type_violations() {
        let conf_str = include_str!("../tauri.conf.json");
        let parsed: serde_json::Value = serde_json::from_str(conf_str).unwrap();

        if let Some(plugins) = parsed.get("plugins") {
            if let Some(gs) = plugins.get("global-shortcut") {
                assert!(
                    gs.is_null(),
                    "plugins.global-shortcut must be null (unit type), not {:?}",
                    gs
                );
            }
        }
    }

    #[test]
    fn test_app_state_initial_values() {
        let state = AppState::new();
        assert!(!state.is_download_cancelled());
        assert!(state.asr.lock().unwrap().is_none());
    }
}
