//! UniSE speech enhancement client.
//! ponytail: single function, POST WAV to /enhance?return_wav=true, get enhanced WAV back.
//! Skipped: health check, connection pool, streaming. Add when latency matters.

use anyhow::{Result, anyhow};

const DEFAULT_UNISE_URL: &str = "http://localhost:9363";

/// Enhance audio by sending it through UniSE noise reduction.
/// Returns enhanced WAV bytes, or the original input if UNISE_URL is unset or the
/// server is unreachable (ponytail: best-effort, silence on failure).
pub async fn enhance(wav_bytes: &[u8]) -> Vec<u8> {
    let url = match std::env::var("WHIMPER_UNISE_URL") {
        Ok(s) if !s.is_empty() => s,
        _ => return wav_bytes.to_vec(), // ponytail: skip if unset
    };

    let client = reqwest::Client::new();
    let resp = match client
        .post(format!("{}/enhance", url))
        .multipart(
            reqwest::multipart::Form::new()
                .part(
                    "file",
                    reqwest::multipart::Part::bytes(wav_bytes.to_vec()).file_name("audio.wav"),
                )
                .text("return_wav", "true"),
        )
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("unise: connection failed: {e}");
            return wav_bytes.to_vec();
        }
    };

    if !resp.status().is_success() {
        tracing::warn!("unise: server returned {}", resp.status());
        return wav_bytes.to_vec();
    }

    let body = match resp.bytes().await {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!("unise: read failed: {e}");
            return wav_bytes.to_vec();
        }
    };

    if body.len() < 44 || &body[..4] != b"RIFF" {
        tracing::warn!("unise: response is not a WAV ({} bytes)", body.len());
        return wav_bytes.to_vec();
    }

    tracing::info!(
        "unise: enhanced {} bytes -> {} bytes",
        wav_bytes.len(),
        body.len()
    );
    body.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhance_skip_when_unset() {
        // WHIMPER_UNISE_URL is not set → enhance returns the input unchanged.
        let rt = tokio::runtime::Runtime::new().unwrap();
        std::env::remove_var("WHIMPER_UNISE_URL");
        let input = vec![0u8, 1, 2, 3];
        let result = rt.block_on(enhance(&input));
        assert_eq!(
            result, input,
            "should return input unchanged when URL is unset"
        );
    }

    #[test]
    fn test_enhance_noop_on_bad_url() {
        // WHIMPER_UNISE_URL set to unreachable → returns input unchanged.
        let rt = tokio::runtime::Runtime::new().unwrap();
        std::env::set_var("WHIMPER_UNISE_URL", "http://127.0.0.1:1");
        let input = b"test wav data".to_vec();
        let result = rt.block_on(enhance(&input));
        assert_eq!(result, input, "should return input on connection failure");
    }
}
