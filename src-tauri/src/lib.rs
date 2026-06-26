mod asr;
mod audio;
mod continuous;
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

// ── Tauri Commands ────────────────────────────────────────────────────────

#[tauri::command]
async fn check_model_status(state: tauri::State<'_, Arc<AppState>>) -> Result<ModelStatus, String> {
    // Clone the client out of the lock so we can await without holding it
    let client = {
        let guard = state.asr.lock().map_err(|e| e.to_string())?;
        guard.clone()
    };

    match client {
        Some(c) => match c.is_ready().await {
            Ok(true) => Ok(ModelStatus::Ready),
            Ok(false) => Ok(ModelStatus::Loading),
            Err(e) => Ok(ModelStatus::Error {
                message: format!("ASR server check failed: {}", e),
            }),
        },
        None => Ok(ModelStatus::NotDownloaded),
    }
}

#[tauri::command]
async fn cancel_download(state: tauri::State<'_, Arc<AppState>>) -> Result<(), String> {
    state.set_cancel_download(true);
    Ok(())
}

#[tauri::command]
async fn load_model(state: tauri::State<'_, Arc<AppState>>) -> Result<(), String> {
    *state.model_status.lock().await = ModelStatus::Loading;

    // Create the HTTP client and test server readiness
    let url = state.asr_server_url.clone();
    let client = asr::HttpAsrClient::new(url);

    // Wait for server to be ready (up to 30 seconds)
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let mut last_ready = false;
    while std::time::Instant::now() < deadline {
        match client.is_ready().await {
            Ok(true) => {
                last_ready = true;
                break;
            }
            Ok(false) => {
                tracing::info!("Waiting for ASR server to be ready...");
            }
            Err(e) => {
                tracing::warn!("ASR server not reachable yet: {}", e);
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
    }

    if !last_ready {
        return Err("ASR server did not become ready within 30 seconds. Is the model-server-asr container running?".into());
    }

    // Store client
    {
        let mut guard = state.asr.lock().map_err(|e| e.to_string())?;
        *guard = Some(client);
    }

    *state.model_status.lock().await = ModelStatus::Ready;
    tracing::info!("ASR server is ready via HTTP client");

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
async fn show_main_window(app_handle: AppHandle) -> Result<(), String> {
    if let Some(window) = app_handle.get_webview_window("main") {
        window.show().map_err(|e| e.to_string())?;
        window.set_focus().map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
async fn check_hotkey_status(
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<state::HotkeyStatus, String> {
    Ok(state.hotkey_status.lock().map(|g| *g).unwrap_or_default())
}

// ── Continuous Listening Commands ─────────────────────────────────────────

#[tauri::command]
async fn start_continuous(
    app_handle: AppHandle,
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<(), String> {
    // Guard: don't start if hotkey recording is in progress
    if *state.recording_state.lock().await == RecordingState::Listening {
        return Err("Cannot start continuous: hotkey recording in progress".into());
    }

    let client = {
        let guard = state.asr.lock().map_err(|e| e.to_string())?;
        guard
            .clone()
            .ok_or_else(|| "ASR client not initialized".to_string())?
    };

    let sink = state.continuous_sink.clone();
    let pid = *state.previous_app_pid.lock().await;

    let _handle = continuous::start(client, sink, Some(app_handle), pid);
    *state.continuous_handle.lock().map_err(|e| e.to_string())? = Some(_handle);
    *state.continuous_active.lock().await = true;

    tracing::info!("Continuous listening started");
    Ok(())
}

#[tauri::command]
async fn stop_continuous(state: tauri::State<'_, Arc<AppState>>) -> Result<(), String> {
    if let Ok(mut handle) = state.continuous_handle.lock() {
        if let Some(h) = handle.take() {
            h.stop();
        }
    }
    *state.continuous_active.lock().await = false;
    tracing::info!("Continuous listening stopped");
    Ok(())
}

#[tauri::command]
async fn is_continuous_active(state: tauri::State<'_, Arc<AppState>>) -> Result<bool, String> {
    Ok(*state.continuous_active.lock().await)
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
                // Guard: if continuous mode is active, ignore hotkey
                if *state.continuous_active.lock().await {
                    tracing::info!("hotkey ignored: continuous mode active");
                    return;
                }

                // Check ASR server is ready (not blocking — quick clone + drop)
                let client_ready = {
                    let guard = state.asr.lock();
                    match guard {
                        Ok(g) => g.is_some(),
                        Err(_) => {
                            tracing::error!("ASR lock poisoned, ignoring hotkey");
                            return;
                        }
                    }
                };

                if !client_ready {
                    tracing::warn!("ASR client not initialized, ignoring hotkey");
                    return;
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

                // Transcribe via HTTP — clone the client out of the std::sync::Mutex
                // so we don't hold the lock across an await point.
                let client_clone = {
                    let guard = state.asr.lock();
                    match guard {
                        Ok(g) => g.clone(),
                        Err(e) => {
                            tracing::error!("ASR lock poisoned: {}", e);
                            None
                        }
                    }
                };

                let (transcript, processing_time_ms, audio_duration_ms) =
                    if let Some(client) = client_clone {
                        if !audio_samples.is_empty() {
                            // Convert float samples to i16 WAV bytes
                            let wav_bytes = audio_samples_to_wav(&audio_samples);

                            match client.transcribe(&wav_bytes, "whimper_recording.wav").await {
                                Ok(r) => (r.text, r.processing_time_ms, r.audio_duration_ms),
                                Err(e) => {
                                    tracing::error!("HTTP transcription failed: {}", e);
                                    (String::new(), 0, fallback_duration_ms)
                                }
                            }
                        } else {
                            (String::new(), 0, fallback_duration_ms)
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

/// Convert Vec<f32> audio samples to a WAV byte buffer (mono, 16kHz, 16-bit PCM).
fn audio_samples_to_wav(samples: &[f32]) -> Vec<u8> {
    use hound::WavWriter;
    use std::io::Cursor;

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = WavWriter::new(&mut cursor, spec).expect("Failed to create WAV writer");
        for &s in samples {
            let sample = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            writer.write_sample(sample).ok();
        }
        writer.finalize().ok();
    }
    cursor.into_inner()
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

    // ASR server URL: Docker internal or localhost for dev
    let asr_server_url =
        std::env::var("WHIMPER_ASR_URL").unwrap_or_else(|_| "http://localhost:9360".to_string());

    let app_state = Arc::new(AppState::new(asr_server_url));

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
            load_model,
            cancel_download,
            cancel_recording,
            hide_main_window,
            show_main_window,
            check_hotkey_status,
            start_continuous,
            stop_continuous,
            is_continuous_active,
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

            // Auto-connect to ASR server at startup
            let app_handle2 = app.handle().clone();
            let state_for_load = app_state.clone();
            let asr_url = app_state.asr_server_url.clone();

            tauri::async_runtime::spawn(async move {
                tracing::info!("Connecting to ASR server at {}", asr_url);
                let client = asr::HttpAsrClient::new(asr_url);

                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
                let mut connected = false;
                while std::time::Instant::now() < deadline {
                    match client.is_ready().await {
                        Ok(true) => {
                            connected = true;
                            break;
                        }
                        _ => {
                            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                        }
                    }
                }

                if connected {
                    if let Ok(mut guard) = state_for_load.asr.lock() {
                        *guard = Some(client);
                    }
                    *state_for_load.model_status.lock().await = ModelStatus::Ready;
                    let _ = app_handle2.emit("model-ready", ());
                    tracing::info!("Auto-connected to ASR server successfully");

                    // Start background pre-roll mic
                    start_preroll_mic(&state_for_load);
                } else {
                    tracing::warn!("ASR server not available at startup (will retry on demand)");
                }
            });

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
        let parsed: serde_json::Value =
            serde_json::from_str(conf_str).expect("tauri.conf.json is not valid JSON");

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
        let state = AppState::new("http://localhost:9360".into());
        assert!(!state.is_download_cancelled());
        assert!(state.asr.lock().unwrap().is_none());
    }
}
