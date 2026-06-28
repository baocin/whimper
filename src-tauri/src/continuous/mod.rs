//! Continuous listening: always-on mic, silence-gated chunks, "paste" keyword.

use std::io::Write;
use std::sync::Arc;
use tauri::{AppHandle, Emitter, Manager};
use tokio::sync::Notify;

use crate::asr::HttpAsrClient;
use crate::paste;
use crate::transcript;

/// Path to the rolling continuous audio capture file. Overwritten each run.
const CAPTURE_WAV: &str = "continuous.wav";

// ponytail: 300ms chunks — GPU ASR finishes in ~100ms, no reason to wait 3s
const CHUNK_INTERVAL_SECS: f64 = 0.3;
const CHUNK_SAMPLES: usize = (16000.0 * CHUNK_INTERVAL_SECS) as usize; // 4800
const SILENCE_THRESHOLD: f32 = 0.005;
// ponytail: 300ms chunks × 200 = 60s silence gap
const SILENT_CHUNK_LIMIT: u32 = 200;
const REPORT_INTERVAL_SECS: u64 = 10;

const TRIGGER_WORDS: &[&str] = &[
    "paste", "paced", "based", "baste", "waist", "waste", "taste", "pasta", "pacing", "racing",
    "basing", "placing", "pasted", "paces", "pace", "haste", "pasteur",
];

fn has_trigger(text: &str) -> bool {
    let lower = text.to_lowercase();
    TRIGGER_WORDS
        .iter()
        .any(|&w| word_boundary_match(&lower, w))
}

/// Check if `word` appears as a whole word (not substring) in `text`.
fn word_boundary_match(text: &str, word: &str) -> bool {
    let word_len = word.len();
    let text_len = text.len();
    if word_len > text_len {
        return false;
    }
    if text == word {
        return true;
    }
    // Check start
    if text.starts_with(word) && is_word_boundary_char(text.as_bytes()[word_len]) {
        return true;
    }
    // Check end
    if text.ends_with(word) && is_word_boundary_char(text.as_bytes()[text_len - word_len - 1]) {
        return true;
    }
    // Check middle
    if let Some(pos) = text[1..text_len.saturating_sub(1)].find(word) {
        let real_pos = pos + 1;
        if is_word_boundary_char(text.as_bytes()[real_pos - 1])
            && is_word_boundary_char(text.as_bytes()[real_pos + word_len])
        {
            return true;
        }
    }
    false
}

fn is_word_boundary_char(c: u8) -> bool {
    !c.is_ascii_alphanumeric()
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

/// Open a WAV file for writing, truncating any previous content.
fn open_wav(path: &std::path::Path) -> Option<hound::WavWriter<std::io::BufWriter<std::fs::File>>> {
    use hound::{SampleFormat, WavSpec, WavWriter};
    use std::io::BufWriter;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok()?;
    }
    let spec = WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let file = std::fs::File::create(path).ok()?;
    WavWriter::new(BufWriter::new(file), spec).ok()
}

/// Write f32 samples to an open WAV writer (no-op if writer is None).
fn write_wav_samples(
    writer: &mut Option<hound::WavWriter<std::io::BufWriter<std::fs::File>>>,
    samples: &[f32],
) {
    let Some(ref mut w) = writer else { return };
    for &s in samples {
        let _ = w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16);
    }
}

/// Process a transcription result: paste on trigger, accumulate otherwise.
/// Shared between the chunk path and the flush-on-silence path.
async fn handle_transcription(
    _asr: &HttpAsrClient,
    app: &Option<AppHandle>,
    utterance_text: &mut String,
    previous_pid: Option<i32>,
    text: String,
) {
    if text.is_empty() {
        return;
    }
    if has_trigger(&text) {
        let full = if utterance_text.is_empty() {
            text
        } else {
            format!("{}. {}", utterance_text, text)
        };
        let word_count = full.split_whitespace().count();
        tracing::info!("continuous: paste trigger → pasting {} chars", full.len());
        let _ = paste::paste_text(&full, previous_pid);
        // ponytail: log to transcripts.jsonl for diagnostics
        let rec = transcript::TranscriptRecord::new(full.clone(), 300, 0, true);
        transcript::append(&rec);
        utterance_text.clear();
        if let Some(ref a) = app {
            let _ = a.emit(
                "continuous-pasted",
                serde_json::json!({"chars": full.len(), "words": word_count}),
            );
            show_paste_feedback(a, "continuous", &format!("Pasted {} words", word_count));
        }
    } else {
        if !utterance_text.is_empty() {
            utterance_text.push(' ');
        }
        utterance_text.push_str(&text);
    }
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

        // ponytail: open rolling WAV for diagnostic capture
        let wav_path = crate::state::whimper_dir().join(CAPTURE_WAV);
        let mut wav_writer = open_wav(&wav_path);

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

            // ponytail: write to diagnostic WAV
            write_wav_samples(&mut wav_writer, &samples);

            // ponytail: info so user sees audio flow by default
            tracing::info!(
                "continuous: rms={rms:.5} silent={silent_count} buf={}",
                utterance_buf.len()
            );

            utterance_buf.extend_from_slice(&samples);

            if rms < SILENCE_THRESHOLD {
                silent_count += 1;
                // Flush partial buffer on first silence after audio (catches short utterances)
                if silent_count == 1 && !utterance_buf.is_empty() && !in_silence {
                    let chunk: Vec<f32> = utterance_buf.drain(..).collect();
                    let text = transcribe_chunk(&asr, &chunk).await;
                    handle_transcription(&asr, &app, &mut utterance_text, previous_pid, text).await;
                }
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
                    handle_transcription(&asr, &app, &mut utterance_text, previous_pid, text).await;
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
        // Whole word matches
        assert!(has_trigger("paste"));
        assert!(has_trigger("please paste that"));
        assert!(has_trigger("paste it now"));
        assert!(has_trigger("paste."));
        assert!(has_trigger("I said paste,"));
        // Phonetic variants (whole word)
        assert!(has_trigger("based on that"));
        assert!(has_trigger("a waste of time"));
        assert!(has_trigger("taste test"));
        assert!(has_trigger("paced up and down"));
        // False positives prevented by word boundary match
        assert!(!has_trigger("pasteurize")); // substring of "pasteurize"
        assert!(!has_trigger("space")); // "pace" is substring
        assert!(!has_trigger("wasted")); // "waste" + 'd' — word-internal
        assert!(!has_trigger("wasting"));
        assert!(!has_trigger("tastes good")); // "taste" + 's'
        assert!(!has_trigger("basement")); // "base" + 'ment'
        assert!(!has_trigger("hello world"));
        assert!(!has_trigger("copy that"));
        assert!(!has_trigger(""));
    }

    #[test]
    fn test_word_boundary_match() {
        assert!(word_boundary_match("paste", "paste"));
        assert!(word_boundary_match("a paste", "paste"));
        assert!(word_boundary_match("paste it", "paste"));
        assert!(word_boundary_match("a paste.", "paste"));
        assert!(!word_boundary_match("pasteurize", "paste"));
        assert!(!word_boundary_match("wasted", "waste"));
        assert!(!word_boundary_match("", "paste"));
        assert!(!word_boundary_match("x", "paste"));
    }

    #[test]
    fn test_silence_limit_is_60s() {
        let gap_secs = SILENT_CHUNK_LIMIT as f64 * CHUNK_INTERVAL_SECS;
        let gap_secs = (gap_secs * 100.0).round() / 100.0; // 2-decimal precision
        assert!(
            (gap_secs - 60.0).abs() < 0.1,
            "{} chunks × {}s = {}s ≠ 60s",
            SILENT_CHUNK_LIMIT,
            CHUNK_INTERVAL_SECS,
            gap_secs
        );
    }

    #[test]
    fn test_audio_to_wav_roundtrip() {
        let input = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let wav = audio_to_wav(&input);
        assert!(wav.len() > 44, "should have WAV header + data");
        assert_eq!(&wav[..4], b"RIFF");
    }

    #[test]
    fn test_audio_sink_push_drain() {
        let sink = AudioSink::new();
        sink.push(&[0.1f32; 16000]); // 1s of audio
        let buf = sink.buf.lock().unwrap();
        assert_eq!(buf.len(), 16000);
    }

    #[test]
    fn test_urlencoding_basics() {
        assert_eq!(urlencoding("Pasted 3 words"), "Pasted%203%20words");
        assert_eq!(urlencoding("a&b?c%"), "a%26b%3Fc%25");
        assert_eq!(urlencoding("hello"), "hello");
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
