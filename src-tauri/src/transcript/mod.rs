//! Durable transcript logging.
//!
//! Every transcription attempt — including empty / VAD-trimmed-to-nothing,
//! hallucination-filtered, and skipped-paste results — is appended as one JSON
//! object per line to `~/.whimper/transcripts.jsonl`. This is the only place
//! the *full* transcript text is persisted; the `tracing` log lines stay
//! truncated for readability.
//!
//! Design constraints (see goal):
//! - Append-only with `O_APPEND` and an explicit `flush` per write, so a crash
//!   never loses an already-written line and concurrent writers don't interleave
//!   partial lines.
//! - File mode `0600` — this is private speech content. We set the mode both at
//!   creation time and after open, so a pre-existing looser file is tightened.
//! - A write failure logs an error and returns; it must never panic or block the
//!   paste path. Callers ignore the result.
//! - Opt-out: setting `WHIMPER_NO_TRANSCRIPT_LOG=1` (any value) disables writes.

use serde::Serialize;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Env var that, when set to any value, disables transcript logging.
pub const OPT_OUT_ENV: &str = "WHIMPER_NO_TRANSCRIPT_LOG";

/// One persisted transcription attempt.
#[derive(Debug, Clone, Serialize)]
pub struct TranscriptRecord {
    /// UTC timestamp, RFC 3339 (e.g. `2026-06-20T06:23:01Z`).
    pub ts_rfc3339: String,
    /// Full, untruncated transcript text.
    pub text: String,
    /// Duration of the captured audio in milliseconds.
    pub audio_duration_ms: u64,
    /// Wall-clock transcription time in milliseconds.
    pub processing_time_ms: u64,
    /// Real-time factor: processing_time / audio_duration (lower is faster).
    pub rtf: f32,
    /// Whether the text was actually pasted into the focused app.
    pub pasted: bool,
    /// Whether the text matched a known hallucination pattern (skip-paste).
    pub hallucination: bool,
    /// Whether the result was empty / trimmed to nothing (skip-paste).
    pub empty: bool,
}

impl TranscriptRecord {
    /// Build a record, deriving `empty`, `hallucination`, and `rtf` from inputs.
    /// `pasted` is supplied by the caller (it reflects the real paste outcome).
    pub fn new(
        text: String,
        audio_duration_ms: u64,
        processing_time_ms: u64,
        pasted: bool,
    ) -> Self {
        let (empty, hallucination) = classify_flags(&text);
        let rtf = if audio_duration_ms > 0 {
            processing_time_ms as f32 / audio_duration_ms as f32
        } else {
            0.0
        };
        Self {
            ts_rfc3339: now_rfc3339(),
            text,
            audio_duration_ms,
            processing_time_ms,
            rtf,
            pasted,
            hallucination,
            empty,
        }
    }
}

/// Classify a transcript into `(empty, hallucination)` flags.
///
/// Single source of truth shared by the record builder and the paste-decision
/// at the call site, so the logged flags can never disagree with the behaviour.
/// `empty` takes precedence: an empty string is reported as empty, not as a
/// hallucination, even though `is_hallucination("")` is also true.
pub fn classify_flags(text: &str) -> (bool, bool) {
    let empty = text.trim().is_empty();
    let hallucination = !empty && crate::asr::ParakeetAsr::is_hallucination(text);
    (empty, hallucination)
}

/// Resolve the transcript log path: `~/.whimper/transcripts.jsonl`.
pub fn transcript_path() -> PathBuf {
    crate::state::whimper_dir().join("transcripts.jsonl")
}

/// Append a record to the transcript log. Error-safe: logs and returns on any
/// failure. Honors the `WHIMPER_NO_TRANSCRIPT_LOG` opt-out.
pub fn append(record: &TranscriptRecord) {
    if std::env::var_os(OPT_OUT_ENV).is_some() {
        tracing::debug!("transcript logging disabled via {}", OPT_OUT_ENV);
        return;
    }
    let path = transcript_path();
    if let Err(e) = append_to(&path, record) {
        tracing::error!("transcript: failed to write {}: {}", path.display(), e);
    }
}

/// Core append, parameterized by path for testability. Creates the parent dir
/// if missing, opens `O_APPEND|O_CREAT` with mode 0600, writes one JSON line,
/// and flushes.
fn append_to(path: &Path, record: &TranscriptRecord) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut line = serde_json::to_string(record)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    line.push('\n');

    let mut opts = OpenOptions::new();
    opts.append(true).create(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }

    let mut f = opts.open(path)?;

    // Tighten perms even if the file pre-existed with a looser mode.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = f.set_permissions(std::fs::Permissions::from_mode(0o600));
    }

    f.write_all(line.as_bytes())?;
    f.flush()?;
    Ok(())
}

/// Current time as an RFC 3339 UTC string, computed from `SystemTime` without
/// pulling in a date library. Uses Howard Hinnant's `civil_from_days` algorithm.
fn now_rfc3339() -> String {
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format_rfc3339(dur.as_secs() as i64)
}

/// Format a Unix timestamp (seconds, UTC) as `YYYY-MM-DDTHH:MM:SSZ`.
fn format_rfc3339(unix_secs: i64) -> String {
    let days = unix_secs.div_euclid(86_400);
    let secs_of_day = unix_secs.rem_euclid(86_400);
    let (h, m, s) = (
        secs_of_day / 3600,
        (secs_of_day % 3600) / 60,
        secs_of_day % 60,
    );
    let (y, mon, d) = civil_from_days(days);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y, mon, d, h, m, s
    )
}

/// Convert days-since-1970-01-01 to a `(year, month, day)` civil date (UTC).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as i64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn test_format_rfc3339_epoch() {
        assert_eq!(format_rfc3339(0), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn test_format_rfc3339_known() {
        // 2026-06-20T06:23:01Z == 1781936581 unix seconds (verified via `date -u`)
        assert_eq!(format_rfc3339(1_781_936_581), "2026-06-20T06:23:01Z");
    }

    #[test]
    fn test_record_serialization_has_all_fields() {
        let rec = TranscriptRecord::new("hello world".to_string(), 1000, 250, true);
        let json = serde_json::to_string(&rec).unwrap();
        for key in [
            "ts_rfc3339",
            "text",
            "audio_duration_ms",
            "processing_time_ms",
            "rtf",
            "pasted",
            "hallucination",
            "empty",
        ] {
            assert!(json.contains(key), "missing field {} in {}", key, json);
        }
        assert!(json.contains("hello world"));
        assert!(!rec.empty);
        assert!(!rec.hallucination);
        // rtf = 250/1000
        assert!((rec.rtf - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_classify_flags_empty() {
        assert_eq!(classify_flags(""), (true, false));
        assert_eq!(classify_flags("   "), (true, false));
    }

    #[test]
    fn test_classify_flags_hallucination() {
        assert_eq!(classify_flags("Thank you"), (false, true));
        assert_eq!(classify_flags("[music]"), (false, true));
    }

    #[test]
    fn test_classify_flags_real_text() {
        assert_eq!(classify_flags("I ordered the salmon"), (false, false));
    }

    #[test]
    fn test_record_flags_for_empty_and_halluc() {
        let empty = TranscriptRecord::new(String::new(), 500, 30, false);
        assert!(empty.empty);
        assert!(!empty.hallucination);

        let halluc = TranscriptRecord::new("thanks for watching".to_string(), 800, 40, false);
        assert!(!halluc.empty);
        assert!(halluc.hallucination);
    }

    #[test]
    fn test_append_to_creates_and_appends() {
        let dir = std::env::temp_dir().join(format!("whimper_test_{}", std::process::id()));
        let path = dir.join("transcripts.jsonl");
        let _ = std::fs::remove_file(&path);

        let r1 = TranscriptRecord::new("first line".to_string(), 1000, 100, true);
        let r2 = TranscriptRecord::new("second line".to_string(), 2000, 200, false);
        append_to(&path, &r1).unwrap();
        append_to(&path, &r2).unwrap();

        let mut content = String::new();
        std::fs::File::open(&path)
            .unwrap()
            .read_to_string(&mut content)
            .unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2, "expected two appended lines");
        assert!(lines[0].contains("first line"));
        assert!(lines[1].contains("second line"));
        // Each line must be valid standalone JSON.
        for l in &lines {
            let _: serde_json::Value = serde_json::from_str(l).unwrap();
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn test_append_to_sets_0600() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("whimper_perm_{}", std::process::id()));
        let path = dir.join("transcripts.jsonl");
        let _ = std::fs::remove_file(&path);

        let r = TranscriptRecord::new("perm check".to_string(), 100, 10, false);
        append_to(&path, &r).unwrap();

        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600, "file mode should be 0600, got {:o}", mode & 0o777);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
