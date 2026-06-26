/// HTTP-based ASR client for the model-server-asr Docker container.
///
/// Replaces the local ONNX Runtime inference with HTTP POST calls to the
/// FastAPI ASR server at model-server-asr:9360 (or localhost:9360 for ad-hoc).
///
/// The server expects multipart form data with a 'file' field containing
/// audio (any format; server converts to WAV internally).
///
/// Response JSON:
///   {
///     "text": "...",
///     "words": [{ "text": "...", "start": ms, "end": ms }, ...],
///     "processing_time_ms": ...,
///     "audio_duration_ms": ...,
///     "rtf": ...,
///     "token_count": ...
///   }
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrResult {
    pub text: String,
    pub words: Vec<AsrWord>,
    pub processing_time_ms: u64,
    pub audio_duration_ms: u64,
    pub rtf: f64,
    pub token_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrWord {
    pub text: String,
    pub start: u64,
    pub end: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerHealth {
    pub status: String,
    pub model: String,
    pub vram_used_mb: i64,
    pub vram_total_mb: i64,
}

/// HTTP-based ASR client.
///
/// Wraps the FastAPI model-server-asr endpoint. All inference runs on the
/// remote GPU container -- this client only sends audio and parses responses.
#[derive(Clone)]
pub struct HttpAsrClient {
    base_url: String,
    client: reqwest::Client,
}

impl HttpAsrClient {
    pub fn new(base_url: String) -> Self {
        Self {
            base_url,
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(600)) // 10 min for long audio
                .build()
                .expect("Failed to build HTTP ASR client"),
        }
    }

    /// Check if the ASR server is healthy and ready.
    pub async fn health(&self) -> Result<ServerHealth, AsrError> {
        let url = format!("{}/health", self.base_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| AsrError::Connection(e.to_string()))?;

        if resp.status() == 503 {
            return Err(AsrError::ServerNotReady("Model still loading".into()));
        }

        if !resp.status().is_success() {
            return Err(AsrError::Server(format!(
                "Health check returned {}",
                resp.status()
            )));
        }

        let health: ServerHealth = resp
            .json()
            .await
            .map_err(|e| AsrError::Parse(e.to_string()))?;

        Ok(health)
    }

    /// Check if the server is ready (model loaded).
    pub async fn is_ready(&self) -> Result<bool, AsrError> {
        let url = format!("{}/ready", self.base_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| AsrError::Connection(e.to_string()))?;

        if !resp.status().is_success() {
            return Ok(false);
        }

        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| AsrError::Parse(e.to_string()))?;
        Ok(body.get("ready").and_then(|v| v.as_bool()).unwrap_or(false))
    }

    /// Transcribe audio data (raw bytes, any format).
    ///
    /// The server will convert to WAV and run VAD + Parakeet TDT 0.6B.
    pub async fn transcribe(
        &self,
        audio_data: &[u8],
        filename: &str,
    ) -> Result<AsrResult, AsrError> {
        let url = format!("{}/transcribe", self.base_url);

        let resp = self
            .client
            .post(&url)
            .multipart(
                reqwest::multipart::Form::new().part(
                    "file",
                    reqwest::multipart::Part::bytes(audio_data.to_vec())
                        .file_name(filename.to_string()),
                ),
            )
            .send()
            .await
            .map_err(|e| AsrError::Connection(e.to_string()))?;

        if resp.status() == 503 {
            return Err(AsrError::ServerNotReady("Model not loaded".into()));
        }

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(AsrError::Server(format!("{}: {}", status, body)));
        }

        let result: AsrResult = resp
            .json()
            .await
            .map_err(|e| AsrError::Parse(e.to_string()))?;

        Ok(result)
    }

    /// Transcribe audio data with a custom filename.
    pub async fn transcribe_with_name(
        &self,
        audio_data: &[u8],
        filename: &str,
    ) -> Result<AsrResult, AsrError> {
        self.transcribe(audio_data, filename).await
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AsrError {
    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Server error: {0}")]
    Server(String),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Server not ready: {0}")]
    ServerNotReady(String),
}
