mod asr;
mod audio;
mod dictation;
mod download;
mod paste;
mod state;

use state::{AppState, HealthStatus, ModelStatus, RecordingState};
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
async fn check_health() -> Result<HealthStatus, String> {
    Ok(HealthStatus {
        microphone_available: audio::microphone::is_microphone_available(),
        accessibility_granted: paste::check_accessibility(),
        model_files_present: download::is_model_downloaded(),
    })
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
        let asr = asr::NemotronAsr::new();
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

/// Shared cancel logic used by both the Tauri command and the ESC global shortcut.
async fn do_cancel_recording(app_handle: &AppHandle, state: &Arc<AppState>) {
    let mut recording = state.recording_state.lock().await;
    if *recording == RecordingState::Listening {
        *recording = RecordingState::Idle;
        drop(recording);

        // Unregister ESC shortcut
        unregister_escape(app_handle);

        // Stop mic stream (don't process audio on cancel)
        if let Ok(mut guard) = state.mic_stream.lock() {
            if let Some(handle) = guard.take() {
                handle.stop_mic();
            }
        }

        // Erase anything already typed — cancel means "leave no trace".
        // finish("") reconciles to empty, and keeps the focus guard.
        let typist = state.live_typist.lock().ok().and_then(|mut g| g.take());
        if let Some(t) = typist {
            if let Ok(mut t) = t.lock() {
                let n = t.typed().chars().count();
                if n > 0 && t.finish("") {
                    tracing::info!("Cancelled: erased {} typed char(s)", n);
                }
            }
        }

        // Hide overlay window
        if let Some(window) = app_handle.get_webview_window("overlay") {
            let _ = window.hide();
        }

        // Restart pre-roll mic
        start_preroll_mic(state);

        tracing::info!("Recording cancelled");
    }
}

#[tauri::command]
async fn cancel_recording(
    app_handle: AppHandle,
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<(), String> {
    do_cancel_recording(&app_handle, &state).await;
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
async fn cancel_file_transcription(state: tauri::State<'_, Arc<AppState>>) -> Result<(), String> {
    state.set_cancel_file_transcription(true);
    Ok(())
}

#[tauri::command]
async fn pick_and_transcribe_file(
    state: tauri::State<'_, Arc<AppState>>,
    app: AppHandle,
) -> Result<Option<String>, String> {
    // Check model is loaded before opening the picker
    {
        let guard = state.asr.lock().map_err(|e| format!("ASR lock: {}", e))?;
        if guard.is_none() {
            return Err("Model is not loaded yet".to_string());
        }
    }

    // Reset cancellation flag before starting
    state.set_cancel_file_transcription(false);

    // Open native file picker
    let handle = rfd::AsyncFileDialog::new()
        .set_title("Choose a media file to transcribe")
        .add_filter(
            "Audio/Video",
            &["mov", "mp4", "m4a", "mp3", "wav", "aac", "flac", "ogg"],
        )
        .pick_file()
        .await;

    let picked = match handle {
        Some(f) => f,
        None => return Ok(None), // user cancelled
    };

    let path = picked.path().to_string_lossy().to_string();
    tracing::info!("Transcribing file: {}", path);

    let _ = app.emit("recording-state", "processing");

    // ponytail: Nemotron is cache-aware streaming, so it has no positional-encoding
    // ceiling the way Parakeet TDT did. Chunking stays only to bound peak memory and
    // to keep the progress bar moving. Raise it if long-file throughput matters.
    const CHUNK_SAMPLES: usize = 5 * 60 * 16_000; // 4,800,000 samples = 300s

    let asr_arc = state.asr.clone();
    let state_arc = state.inner().clone();
    let app_arc = app.clone();

    let transcript = tokio::task::spawn_blocking(move || {
        let samples = audio::file_decoder::decode_audio_file(&path)?;
        tracing::info!(
            "Decoded {} samples ({:.1}s) from file",
            samples.len(),
            samples.len() as f64 / 16000.0
        );

        let guard = asr_arc.lock().map_err(|e| format!("ASR lock: {}", e))?;
        let asr = guard
            .as_ref()
            .ok_or_else(|| "Model not loaded".to_string())?;

        let total_chunks = (samples.len() + CHUNK_SAMPLES - 1) / CHUNK_SAMPLES;
        let mut parts: Vec<String> = Vec::with_capacity(total_chunks);

        for (i, chunk) in samples.chunks(CHUNK_SAMPLES).enumerate() {
            if state_arc.is_file_transcription_cancelled() {
                tracing::info!("File transcription cancelled at chunk {}/{}", i, total_chunks);
                break;
            }

            let _ = app_arc.emit(
                "file-transcribe-progress",
                serde_json::json!({"current": i, "total": total_chunks}),
            );

            tracing::info!(
                "Transcribing chunk {}/{} ({:.1}s)",
                i + 1,
                total_chunks,
                chunk.len() as f64 / 16000.0
            );

            match asr.transcribe(chunk, false) {
                Ok(result) if !result.text.is_empty() => parts.push(result.text),
                Ok(_) => tracing::debug!("Chunk {}/{} produced no speech", i + 1, total_chunks),
                Err(e) => tracing::warn!("Chunk {}/{} failed: {}", i + 1, total_chunks, e),
            }
        }

        let _ = app_arc.emit(
            "file-transcribe-progress",
            serde_json::json!({"current": total_chunks, "total": total_chunks}),
        );

        Ok::<String, String>(parts.join(" "))
    })
    .await
    .map_err(|e| format!("Task panicked: {}", e))??;

    let _ = app.emit("recording-state", "idle");

    tracing::info!(
        "File transcript ({} chars): {:?}",
        transcript.len(),
        &transcript[..transcript.len().min(120)]
    );

    Ok(Some(transcript))
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

// ── ESC Shortcut Management ─────────────────────────────────────────────

static ESC_SHORTCUT: std::sync::LazyLock<Shortcut> =
    std::sync::LazyLock::new(|| "Escape".parse().unwrap());

fn register_escape(app_handle: &AppHandle) {
    if let Err(e) = app_handle.global_shortcut().register(*ESC_SHORTCUT) {
        tracing::warn!("Failed to register ESC shortcut: {}", e);
    }
}

fn unregister_escape(app_handle: &AppHandle) {
    if let Err(e) = app_handle.global_shortcut().unregister(*ESC_SHORTCUT) {
        tracing::warn!("Failed to unregister ESC shortcut: {}", e);
    }
}

// ── Global Hotkey Handlers ───────────────────────────────────────────────

/// How long to wait for the mic to deliver its first samples before giving up.
/// A healthy device fires well inside this; the cap only catches a dead mic.
const MIC_START_POLLS: u64 = 100;
const MIC_START_POLL_INTERVAL_MS: u64 = 10;

/// Stream-only dictation: keep whatever was typed live and skip the full-buffer
/// finalization pass, so Option+Space stops instantly instead of waiting seconds
/// for a re-transcription.
///
/// Defaults ON. Set `WHIMPER_FINALIZE=1` to restore the finalization + reconcile
/// pass, which is also the only thing that runs Silero VAD and GTCRN — in
/// stream-only mode the audio gets a high-pass filter and nothing else.
fn stream_only() -> bool {
    !matches!(std::env::var("WHIMPER_FINALIZE").as_deref(), Ok("1"))
}

fn handle_escape(app_handle: &AppHandle, state: &Arc<AppState>) {
    let app = app_handle.clone();
    let state = state.clone();
    tauri::async_runtime::spawn(async move {
        do_cancel_recording(&app, &state).await;
    });
}

fn handle_hotkey(app_handle: &AppHandle, state: &Arc<AppState>) {
    let app = app_handle.clone();
    let state = state.clone();

    tauri::async_runtime::spawn(async move {
        let mut recording = state.recording_state.lock().await;

        match *recording {
            RecordingState::Idle => {
                // Nothing is announced to the user until the model is loaded AND the
                // microphone is actually delivering samples. Everything below is setup;
                // the overlay only appears once both are true.

                // The model must be loaded, not merely downloaded.
                let model_ready = state
                    .asr
                    .lock()
                    .map(|g| g.as_ref().is_some_and(|a| a.is_loaded()))
                    .unwrap_or(false);
                if !model_ready {
                    tracing::warn!("Model not loaded, ignoring hotkey");
                    return;
                }

                // Save frontmost app PID before we steal focus
                if let Some(pid) = paste::get_frontmost_app_pid() {
                    *state.previous_app_pid.lock().await = Some(pid);
                }
                let guard_pid = *state.previous_app_pid.lock().await;

                // Stop pre-roll mic before starting recording (avoid two mic streams)
                if let Ok(mut guard) = state.preroll_mic.lock() {
                    if let Some(handle) = guard.take() {
                        handle.stop_mic();
                        tracing::info!("Pre-roll mic stopped for recording");
                    }
                }

                // Live dictation: stream preprocessing during recording and type
                // text into the target app as the model emits it.
                let typist = Arc::new(std::sync::Mutex::new(dictation::LiveTypist::new(
                    state.asr.clone(),
                )));
                let cb_typist = Arc::clone(&typist);
                let on_chunk: Box<dyn FnMut(&[f32]) + Send> = Box::new(move |samples: &[f32]| {
                    if let Ok(mut t) = cb_typist.lock() {
                        t.feed(samples);
                    }
                });

                // Only buffer raw audio when a finalization pass will actually consume it.
                let handle = match audio::pipeline::start_pipeline_streaming(
                    Some(&state.preroll_buffer),
                    !stream_only(),
                    Some(on_chunk),
                ) {
                    Ok(h) => h,
                    Err(e) => {
                        tracing::error!("Failed to start audio pipeline: {}", e);
                        start_preroll_mic(&state);
                        return;
                    }
                };

                // `stream.play()` returning Ok does not mean the device is delivering
                // audio — it can succeed and stay silent if the mic is busy or muted.
                // Wait for real samples before claiming to be listening.
                let mut live = false;
                for _ in 0..MIC_START_POLLS {
                    if handle.has_audio() {
                        live = true;
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(
                        MIC_START_POLL_INTERVAL_MS,
                    ))
                    .await;
                }

                if !live {
                    tracing::error!(
                        "Microphone produced no audio within {}ms — not starting",
                        MIC_START_POLLS * MIC_START_POLL_INTERVAL_MS
                    );
                    handle.stop_mic();
                    drop(handle);
                    let _ = app.emit("mic-error", "Microphone produced no audio");
                    start_preroll_mic(&state);
                    return;
                }

                // Mic is genuinely live — commit and announce.
                if let Ok(mut guard) = state.mic_stream.lock() {
                    *guard = Some(handle);
                }
                if let Ok(mut guard) = state.live_typist.lock() {
                    *guard = Some(typist);
                }

                *recording = RecordingState::Listening;
                tracing::info!("Recording started (mic confirmed live)");
                drop(recording);

                register_escape(&app);

                let _ = app.emit("recording-state", "listening");
                if let Some(window) = app.get_webview_window("overlay") {
                    let _ = window.show();
                }

                // Re-activate previous app so overlay doesn't steal keyboard focus
                if let Some(pid) = guard_pid {
                    let _ = paste::activate_app(pid);
                }
            }
            RecordingState::Listening => {
                *recording = RecordingState::Idle;
                drop(recording);

                // Unregister ESC shortcut
                unregister_escape(&app);

                // Finalize pipeline: stop mic, get audio buffer. This joins the
                // processing thread, so the typist is no longer being fed after it.
                let audio_samples = if let Ok(mut guard) = state.mic_stream.lock() {
                    guard.take().map(|h| h.finalize()).unwrap_or_default()
                } else {
                    Vec::new()
                };

                // Type the tail of the utterance that never filled a full chunk.
                let typist = state.live_typist.lock().ok().and_then(|mut g| g.take());
                if let Some(ref t) = typist {
                    if let Ok(mut t) = t.lock() {
                        t.flush();
                    }
                }

                tracing::info!(
                    "Recording stopped, {} samples ({:.1}s)",
                    audio_samples.len(),
                    audio_samples.len() as f64 / 16000.0
                );

                // Restart pre-roll mic now that recording is done
                start_preroll_mic(&state);

                let stream_only = stream_only();

                if stream_only {
                    // Stop must feel instant: there is no finalization pass to wait
                    // for, so drop the overlay now instead of showing "Processing...".
                    if let Some(window) = app.get_webview_window("overlay") {
                        let _ = window.hide();
                    }
                    let _ = app.emit("recording-state", "idle");
                } else {
                    let _ = app.emit("recording-state", "processing");
                }

                // Batch-transcribe on blocking thread
                let asr_arc = state.asr.clone();
                let transcript = if stream_only {
                    typist
                        .as_ref()
                        .and_then(|t| t.lock().ok().map(|t| t.typed().to_string()))
                        .unwrap_or_default()
                } else if !audio_samples.is_empty() {
                    match tokio::task::spawn_blocking(move || {
                        let guard = asr_arc.lock().map_err(|e| format!("ASR lock: {}", e))?;
                        let asr = guard
                            .as_ref()
                            .ok_or_else(|| "Model not loaded".to_string())?;
                        // The accumulator holds raw audio — run the full batch pipeline.
                        asr.transcribe(&audio_samples, false)
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

                // Append transcript to log file (in app data dir to avoid
                // triggering cargo-tauri dev file watcher)
                if !transcript.trim().is_empty() {
                    use std::io::Write;
                    if let Ok(data_dir) = app.path().app_data_dir() {
                        let _ = std::fs::create_dir_all(&data_dir);
                        let log_path = data_dir.join("log.txt");
                        if let Ok(mut f) = std::fs::OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open(&log_path)
                        {
                            let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
                            let _ = writeln!(f, "[{}] {}", timestamp, transcript.trim());
                        }
                    }
                }

                if stream_only {
                    // The text is already in the target app — nothing to reconcile.
                    tracing::info!("Stream-only: finalization skipped");
                    return;
                }

                // Hide overlay
                if let Some(window) = app.get_webview_window("overlay") {
                    let _ = window.hide();
                }

                // Reconcile what was typed live against the higher-quality final pass.
                let keep = !transcript.trim().is_empty()
                    && !asr::NemotronAsr::is_hallucination(&transcript);
                let final_text = if keep { transcript.as_str() } else { "" };

                match typist {
                    Some(t) => {
                        // Small delay for overlay to close before touching the target app
                        tokio::time::sleep(std::time::Duration::from_millis(150)).await;

                        if let Ok(mut t) = t.lock() {
                            if t.aborted() {
                                tracing::warn!(
                                    "Live typing was aborted (focus moved); leaving {} typed char(s) as-is",
                                    t.typed().chars().count()
                                );
                            } else if t.finish(final_text) {
                                tracing::info!("Dictation reconciled");
                            } else {
                                tracing::warn!("Reconciliation skipped");
                            }
                        }
                    }
                    None if keep => {
                        // No dictation session (shouldn't happen) — fall back to pasting.
                        let prev_pid = *state.previous_app_pid.lock().await;
                        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
                        match paste::paste_text(&transcript, prev_pid) {
                            Ok(_) => tracing::info!("Paste succeeded"),
                            Err(e) => tracing::error!("Paste failed: {}", e),
                        }
                    }
                    None => {}
                }
            }
        }
    });
}

// ── App Entry Point ──────────────────────────────────────────────────────

pub fn run() {
    // RUST_LOG wins when set. Previously an added `whimper=info` directive
    // silently overrode it, so RUST_LOG=whimper=debug produced no debug output.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("whimper=info")),
        )
        .init();

    let app_state = Arc::new(AppState::new());

    let state_for_shortcut = app_state.clone();
    let alt_space: Shortcut = "Alt+Space".parse().unwrap();

    tauri::Builder::default()
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(move |app, shortcut, event| {
                    if event.state() == ShortcutState::Pressed {
                        if *shortcut == alt_space {
                            handle_hotkey(app, &state_for_shortcut);
                        } else if *shortcut == *ESC_SHORTCUT {
                            handle_escape(app, &state_for_shortcut);
                        }
                    }
                })
                .build(),
        )
        .manage(app_state.clone())
        .invoke_handler(tauri::generate_handler![
            check_model_status,
            check_health,
            start_download,
            cancel_download,
            load_model,
            cancel_recording,
            hide_main_window,
            show_main_window,
            pick_and_transcribe_file,
            cancel_file_transcription,
        ])
        .setup(move |app| {
            app.global_shortcut().register(alt_space)?;

            // Create overlay window once (hidden) — avoids Vite HMR reloads
            // from repeated webview creation/destruction
            let overlay = tauri::WebviewWindowBuilder::new(
                app,
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
            .visible(false)
            .build()?;

            if let Ok(Some(m)) = overlay.current_monitor() {
                let screen_width = m.size().width as f64;
                let x = (screen_width / 2.0 - 200.0) as i32;
                let _ = overlay.set_position(tauri::Position::Physical(
                    tauri::PhysicalPosition::new(x, 100),
                ));
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
                        let asr = asr::NemotronAsr::new();
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
                            tracing::info!("Nemotron model auto-loaded successfully");

                            // Start background pre-roll mic
                            start_preroll_mic(&state_for_load);
                        }
                        Ok(Err(e)) => {
                            tracing::error!("Failed to auto-load model: {}", e);
                            let _ = app_handle2.emit("model-load-error", e.to_string());
                        }
                        Err(e) => {
                            tracing::error!("Model loading task panicked: {}", e);
                            let _ = app_handle2.emit("model-load-error", e.to_string());
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
