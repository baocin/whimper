mod asr;
mod audio;
mod download;
mod paste;
mod state;

use state::{AppState, ModelStatus, RecordingState};
use std::sync::Arc;
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};

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

// ── Global Hotkey Handler ────────────────────────────────────────────────

fn handle_hotkey(app_handle: &AppHandle, state: &Arc<AppState>) {
    let app = app_handle.clone();
    let state = state.clone();

    tauri::async_runtime::spawn(async move {
        let mut recording = state.recording_state.lock().await;

        match *recording {
            RecordingState::Idle => {
                // Check model is ready
                if let Ok(guard) = state.asr.lock() {
                    if guard.is_none() {
                        tracing::warn!("Model not loaded, ignoring hotkey");
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
                .title("")
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

                // Start audio pipeline (record-only, no ASR/VAD needed)
                match audio::pipeline::start_pipeline() {
                    Ok(handle) => {
                        if let Ok(mut guard) = state.mic_stream.lock() {
                            *guard = Some(handle);
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to start audio pipeline: {}", e);
                    }
                }
            }
            RecordingState::Listening => {
                *recording = RecordingState::Idle;
                drop(recording);

                // Finalize pipeline: stop mic, get audio buffer
                let audio_samples = if let Ok(mut guard) = state.mic_stream.lock() {
                    guard.take().map(|h| h.finalize()).unwrap_or_default()
                } else {
                    Vec::new()
                };

                tracing::info!(
                    "Recording stopped, {} samples ({:.1}s)",
                    audio_samples.len(),
                    audio_samples.len() as f64 / 16000.0
                );

                let _ = app.emit("recording-state", "processing");

                // Batch-transcribe on blocking thread
                let asr_arc = state.asr.clone();
                let transcript = if !audio_samples.is_empty() {
                    match tokio::task::spawn_blocking(move || {
                        let guard = asr_arc.lock().map_err(|e| format!("ASR lock: {}", e))?;
                        let asr = guard
                            .as_ref()
                            .ok_or_else(|| "Model not loaded".to_string())?;
                        asr.transcribe(&audio_samples)
                            .map(|r| r.text)
                            .map_err(|e| e.to_string())
                    })
                    .await
                    {
                        Ok(Ok(text)) => text,
                        Ok(Err(e)) => {
                            tracing::error!("Transcription failed: {}", e);
                            String::new()
                        }
                        Err(e) => {
                            tracing::error!("Transcription task panicked: {}", e);
                            String::new()
                        }
                    }
                } else {
                    String::new()
                };

                tracing::info!(
                    "Transcript ({} chars): {:?}",
                    transcript.len(),
                    &transcript[..transcript.len().min(120)]
                );

                // Close overlay
                if let Some(window) = app.get_webview_window("overlay") {
                    let _ = window.close();
                }

                // Paste if we have text (and it's not a hallucination)
                if !transcript.trim().is_empty()
                    && !asr::ParakeetAsr::is_hallucination(&transcript)
                {
                    let prev_pid = *state.previous_app_pid.lock().await;

                    // Small delay for overlay to close
                    tokio::time::sleep(std::time::Duration::from_millis(150)).await;

                    match paste::paste_text(&transcript, prev_pid) {
                        Ok(_) => tracing::info!("Paste succeeded"),
                        Err(e) => tracing::error!("Paste failed: {}", e),
                    }
                }
            }
        }
    });
}

// ── App Entry Point ──────────────────────────────────────────────────────

pub fn run() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("whimper=info".parse().unwrap()),
        )
        .init();

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
        ])
        .setup(move |app| {
            let shortcut: Shortcut = "Alt+Space".parse().unwrap();
            app.global_shortcut().register(shortcut)?;

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
        .expect("error while running whimper");
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
