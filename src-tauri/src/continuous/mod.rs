//! Continuous listening: always-on mic, silence-gated chunks, "paste" keyword.

use std::sync::Arc;
use tauri::{AppHandle, Emitter, Manager};
use tokio::sync::Notify;

use crate::asr::HttpAsrClient;
use crate::paste;
use crate::transcript;

const CHUNK_INTERVAL_SECS: f64 = 3.0;
const CHUNK_SAMPLES: usize = (16000.0 * CHUNK_INTERVAL_SECS) as usize;
const SILENCE_THRESHOLD: f32 = 0.005;
const SILENT_CHUNK_LIMIT: u32 = 20;
const REPORT_INTERVAL_SECS: u64 = 10;

const TRIGGER_WORDS: &[&str] = &[
    "paste", "paced", "based", "baste", "waist", "waste", "taste", "pasta", "pacing", "racing",
    "basing", "placing", "pasted", "paces", "pace",
];

fn has_trigger(text: &str) -> bool {
    let lower = text.to_lowercase();
    TRIGGER_WORDS.iter().any(|&w| lower.contains(w))
}

pub struct AudioSink {
    buf: std::sync::Mutex<Vec<f32>>,
    pub(crate) notify: Notify,
}

impl AudioSink {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            buf: std::sync::Mutex::new(Vec::new()),
            notify: Notify::new(),
        })
    }

    pub fn push(&self, samples: &[f32]) {
        if let Ok(mut b) = self.buf.lock() {
            b.extend_from_slice(samples);
        }
        self.notify.notify_one();
    }
}

/// Open the overlay window with a paste-confirmation flash.
////// ponytail: reuses the same webview pattern as hotkey overlay; auto-dismisses
/// after 1.5s. If the window can't be built (e.g. in tests), the error is logged
/// and discarded — non-fatal.
fn show_paste_feedback(app: &AppHandle, mode: &str, label: &str) {
    use tauri::Manager;
    let url = format!("/overlay?mode={}&label={}", mode, urlencoding(label));
    let overlay = tauri::WebviewWindowBuilder::new(
        app,
        "continuous-feedback",
        tauri::WebviewUrl::App(url.into()),
    )
    .title("whimper-overlay")
    .inner_size(400.0, 80.0)
    .decorations(false)
    .transparent(true)
    .always_on_top(true)
    .focused(false)
    .resizable(false)
    .build();

    if let Ok(ref window) = overlay {
        if let Ok(Some(monitor)) = window.current_monitor() {
            let x = (monitor.size().width as f64 / 2.0 - 200.0) as i32;
            let _ = window.set_position(tauri::Position::Physical(tauri::PhysicalPosition::new(
                x, 100,
            )));
        }
        let h = app.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
            if let Some(w) = h.get_webview_window("continuous-feedback") {
                let _ = w.close();
            }
        });
    }
}

/// URL-encode a simple string (just enough for label text).
fn urlencoding(s: &str) -> String {
    s.replace('%', "%25")
        .replace(' ', "%20")
        .replace('&', "%26")
        .replace('?', "%3F")
}

pub fn start(
    asr: HttpAsrClient,
    sink: Arc<AudioSink>,
    app: Option<AppHandle>,
    previous_pid: Option<i32>,
) -> ContinuousHandle {
    let stop = Arc::new(Notify::new());
    let stop_clone = stop.clone();

    if let Some(ref a) = app {
        let _ = a.emit("continuous-state", "listening");
    }

    tokio::spawn(async move {
        let mut utterance_buf: Vec<f32> = Vec::new();
        let mut utterance_text = String::new();
        let mut silent_count: u32 = 0;
        let mut in_silence = false;
        let mut last_log = std::time::Instant::now();

        loop {
            tokio::select! {
                _ = stop_clone.notified() => {
                    tracing::info!("continuous: stopped");
                    if let Some(ref a) = app {
                        let _ = a.emit("continuous-state", "stopped");
                    }
                    break;
                }
                _ = sink.notify.notified() => {}
            }

            let samples = {
                let mut buf = match sink.buf.lock() {
                    Ok(b) => b,
                    Err(_) => break,
                };
                if buf.len() < CHUNK_SAMPLES {
                    continue;
                }
                buf.drain(..CHUNK_SAMPLES).collect::<Vec<f32>>()
            };

            let rms = if samples.is_empty() {
                0.0
            } else {
                let sum: f32 = samples.iter().map(|s| s * s).sum();
                (sum / samples.len() as f32).sqrt()
            };

            utterance_buf.extend_from_slice(&samples);

            if rms < SILENCE_THRESHOLD {
                silent_count += 1;
                if !in_silence && silent_count >= SILENT_CHUNK_LIMIT {
                    tracing::info!(
                        "continuous: silence gap ({} silent chunks), resetting",
                        silent_count
                    );
                    utterance_buf.clear();
                    utterance_text.clear();
                    in_silence = true;
                }
            } else {
                in_silence = false;
                silent_count = 0;

                if utterance_buf.len() >= CHUNK_SAMPLES {
                    let chunk: Vec<f32> = utterance_buf.drain(..CHUNK_SAMPLES).collect();
                    let text = transcribe_chunk(&asr, &chunk).await;
                    if !text.is_empty() {
                        if has_trigger(&text) {
                            let full = if utterance_text.is_empty() {
                                text.clone()
                            } else {
                                format!("{}. {}", utterance_text, text)
                            };
                            let word_count = full.split_whitespace().count();
                            tracing::info!(
                                "continuous: paste trigger → pasting {} chars",
                                full.len()
                            );
                            let _ = paste::paste_text(&full, previous_pid);
                            utterance_text.clear();
                            if let Some(ref a) = app {
                                let _ = a.emit(
                                    "continuous-pasted",
                                    serde_json::json!({
                                        "chars": full.len(),
                                        "words": word_count,
                                    }),
                                );
                                show_paste_feedback(
                                    a,
                                    "continuous",
                                    &format!("Pasted {} words", word_count),
                                );
                            }
                        } else {
                            if !utterance_text.is_empty() {
                                utterance_text.push(' ');
                            }
                            utterance_text.push_str(&text);
                        }
                    }
                }
            }

            if last_log.elapsed().as_secs() >= REPORT_INTERVAL_SECS {
                tracing::debug!(
                    "continuous: buf={} samples, silent={}, text_len={}",
                    utterance_buf.len(),
                    silent_count,
                    utterance_text.len(),
                );
                last_log = std::time::Instant::now();
            }
        }
    });

    ContinuousHandle { stop }
}

async fn transcribe_chunk(asr: &HttpAsrClient, samples: &[f32]) -> String {
    if samples.is_empty() {
        return String::new();
    }
    let wav = audio_to_wav(samples);
    match asr.transcribe(&wav, "chunk.wav").await {
        Ok(r) => {
            let t = r.text.trim().to_string();
            let (empty, hall) = transcript::classify_flags(&t);
            if empty || hall { String::new() } else { t }
        }
        Err(e) => {
            tracing::warn!("continuous: ASR error: {e}");
            String::new()
        }
    }
}

fn audio_to_wav(samples: &[f32]) -> Vec<u8> {
    use hound::{SampleFormat, WavSpec, WavWriter};
    use std::io::Cursor;
    let spec = WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut c = Cursor::new(Vec::new());
    {
        let mut w = WavWriter::new(&mut c, spec).unwrap();
        for &s in samples {
            w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
                .ok();
        }
        w.finalize().ok();
    }
    c.into_inner()
}

pub struct ContinuousHandle {
    stop: Arc<Notify>,
}

impl ContinuousHandle {
    pub fn stop(&self) {
        self.stop.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_trigger_variants() {
        assert!(has_trigger("paste"));
        assert!(has_trigger("please paste that"));
        assert!(has_trigger("based on that"));
        assert!(has_trigger("waste of time"));
        assert!(has_trigger("taste test"));
        assert!(has_trigger("paced up and down"));
        assert!(!has_trigger("hello world"));
        assert!(!has_trigger("copy that"));
        assert!(!has_trigger(""));
    }

    #[test]
    fn test_silence_limit_is_60s() {
        assert_eq!(SILENT_CHUNK_LIMIT * (CHUNK_INTERVAL_SECS as u32), 60);
    }

    #[test]
    fn test_audio_to_wav_roundtrip() {
        let input = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let wav = audio_to_wav(&input);
        assert!(wav.len() > 44, "should have WAV header + data");
        assert_eq!(&wav[..4], b"RIFF");
    }

    fn read_recording(path: &str) -> Option<Vec<f32>> {
        let out = std::process::Command::new("ffmpeg")
            .args([
                "-y",
                "-i",
                path,
                "-f",
                "f32le",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-loglevel",
                "error",
                "pipe:1",
            ])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        Some(
            out.stdout
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    }

    #[test]
    #[ignore = "needs ASR server at localhost:9360"]
    fn test_replay_file_realtime() {
        let paths = [
            "test_recordings/Connecting Cables and Managing Power Grounds.mp3",
            "../test_recordings/Connecting Cables and Managing Power Grounds.mp3",
            "/home/aoi/code/whimper/test_recordings/Connecting Cables and Managing Power Grounds.mp3",
        ];
        let audio = paths
            .iter()
            .find_map(|p| read_recording(p))
            .expect("place an MP3 in test_recordings/");
        let audio_duration = audio.len() as f64 / 16000.0;
        eprintln!("Loaded {:.1}s of audio", audio_duration);

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let client = HttpAsrClient::new("http://localhost:9360".to_string());
            let sink = AudioSink::new();
            let _handle = start(client, sink.clone(), None, None);

            let chunk_size = 1600;
            let chunk_dur = std::time::Duration::from_millis(100);
            let total = audio.len() / chunk_size;
            let start = std::time::Instant::now();
            for i in 0..total {
                let lo = i * chunk_size;
                let hi = (lo + chunk_size).min(audio.len());
                sink.push(&audio[lo..hi]);
                let expected = chunk_dur * (i as u32 + 1);
                if start.elapsed() < expected {
                    tokio::time::sleep(expected - start.elapsed()).await;
                }
                if i % (total / 10).max(1) == 0 {
                    eprintln!("  {:.0}%", i as f64 / total as f64 * 100.0);
                }
            }
            let wall = start.elapsed().as_secs_f64();
            let ratio = wall / audio_duration;
            eprintln!(
                "Done: {:.1}s audio, {:.1}s wall (ratio={:.2})",
                audio_duration, wall, ratio
            );
            assert!(ratio < 1.1, "pipeline too slow: {:.2}x realtime", ratio);
        });
    }
}
