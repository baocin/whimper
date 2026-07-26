//! Speaker verification: check if audio matches "me" via Wespeaker + Titanet.
//! ponytail: HTTP POST WAV bytes, get 192-dim embedding back, cosine compare.

use anyhow::{Result, anyhow};
use std::sync::OnceLock;

/// Target speaker embedding (wespeaker 192 floats). Loaded from ~/.whimper/me_speaker.json
/// or WHIMPER_ME_SPEAKER env var.
static ME_WESPEAKER: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();

fn init_me_embedding() {
    let json_str = match std::env::var("WHIMPER_ME_SPEAKER") {
        Ok(s) => s,
        // ponytail: fall back to ~/.whimper/me_speaker.json
        Err(_) => {
            match std::fs::read_to_string(crate::state::whimper_dir().join("me_speaker.json")) {
                Ok(s) => s,
                Err(_) => return,
            }
        }
    };
    let Ok(val) = serde_json::from_str::<serde_json::Value>(&json_str) else {
        return;
    };
    // ponytail: wespeaker key; also accept bare JSON array for simple format
    let arr = val
        .get("wespeaker")
        .or_else(|| val.as_array().map(|_| &val))
        .and_then(|v| v.as_array());
    if let Some(arr) = arr {
        let _ = ME_WESPEAKER.set(
            arr.iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect(),
        );
    }
}

/// Get embedding via HTTP POST (async — call from existing tokio context).
pub(crate) async fn embedding_from(url: &str, wav_bytes: &[u8], model: &str) -> Result<Vec<f32>> {
    let b64 = base64_encode(wav_bytes);
    let body = serde_json::json!({"input": b64, "model": model}).to_string();
    let resp = reqwest::Client::new()
        .post(&format!("{}/v1/embeddings", url))
        .header("Content-Type", "application/json")
        .body(body)
        .send()
        .await
        .map_err(|e| anyhow!("{model} request: {e}"))?;
    let resp_body = resp
        .text()
        .await
        .map_err(|e| anyhow!("{model} body: {e}"))?;
    let result: serde_json::Value =
        serde_json::from_str(&resp_body).map_err(|e| anyhow!("{model} parse: {e}"))?;
    let arr = result["data"][0]["embedding"]
        .as_array()
        .ok_or_else(|| anyhow!("{model} missing data[0].embedding: {resp_body}"))?;
    arr.iter()
        .map(|v| {
            v.as_f64()
                .map(|f| f as f32)
                .ok_or_else(|| anyhow!("{model} non-float in embedding"))
        })
        .collect()
}

fn base64_encode(data: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(data)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum();
    let nb: f32 = b.iter().map(|x| x * x).sum();
    dot / ((na * nb).sqrt() + 1e-10)
}

/// Returns true if the speaker in `wav_bytes` matches the saved "me" embedding.
/// Uses wespeaker only — titanet was too noisy on live mic chunks.
/// threshold: 0.5 separates you from MOSS hallucination (0.09) cleanly.
pub async fn is_me(wav_bytes: &[u8], wespeaker_url: &str, _titanet_url: &str) -> bool {
    init_me_embedding();
    let ws_target = match ME_WESPEAKER.get() {
        Some(e) => e.clone(),
        _ => {
            tracing::info!("speaker: no wespeaker target, allowing");
            return true;
        }
    };
    let ws_url =
        std::env::var("WHIMPER_WESPEAKER_URL").unwrap_or_else(|_| wespeaker_url.to_string());

    let ws_emb = match embedding_from(&ws_url, wav_bytes, "wespeaker").await {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!("speaker: wespeaker: {e}");
            return true;
        }
    };
    let ws_sim = cosine_similarity(&ws_emb, &ws_target);
    tracing::info!(
        "speaker: wespeaker_sim={ws_sim:.3} distance={:.3}",
        1.0 - ws_sim
    );
    ws_sim > 0.5
}
