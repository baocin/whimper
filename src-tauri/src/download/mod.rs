//! Model download with R2 CDN primary and HuggingFace fallback
//!
//! Downloads voxtral-q4.gguf (~2.51 GB) and tekken.json (~14.9 MB)
//! to ~/.whimper/models/voxtral-mini-4b/

use anyhow::{Context, Result};
use futures_util::StreamExt;
use std::path::Path;
use tauri::{AppHandle, Emitter};
use tokio::io::AsyncWriteExt;

use crate::state;

const R2_BASE: &str = "https://public.mydatatimeline.com/models/voxtral-mini-4b";
const HF_BASE: &str = "https://huggingface.co/TrevorJS/voxtral-mini-realtime-gguf/resolve/main";

const MODEL_FILES: &[(&str, u64)] = &[
    ("voxtral-q4.gguf", 2_510_000_000), // ~2.51 GB
    ("tekken.json", 14_900_000),          // ~14.9 MB
];

#[derive(Clone, serde::Serialize)]
pub struct DownloadProgress {
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub speed_mbps: f64,
}

/// Check if model files exist on disk
pub fn is_model_downloaded() -> bool {
    let dir = state::model_dir();
    MODEL_FILES
        .iter()
        .all(|(name, _)| dir.join(name).exists())
}

/// Download all model files
pub async fn download_model(app_handle: AppHandle, state: &state::AppState) -> Result<()> {
    let dir = state::model_dir();
    tokio::fs::create_dir_all(&dir)
        .await
        .context("Failed to create model directory")?;

    let total_bytes: u64 = MODEL_FILES.iter().map(|(_, size)| size).sum();
    let mut downloaded_total: u64 = 0;

    let client = reqwest::Client::new();

    for (filename, _expected_size) in MODEL_FILES {
        let dest = dir.join(filename);

        // Skip if already downloaded
        if dest.exists() {
            let metadata = tokio::fs::metadata(&dest).await?;
            if metadata.len() > 0 {
                downloaded_total += metadata.len();
                continue;
            }
        }

        // Try R2 first, fall back to HuggingFace
        let r2_url = format!("{}/{}", R2_BASE, filename);
        let hf_url = format!("{}/{}", HF_BASE, filename);

        let result = download_file(
            &client,
            &r2_url,
            &dest,
            &app_handle,
            state,
            downloaded_total,
            total_bytes,
        )
        .await;

        if result.is_err() {
            tracing::warn!("R2 download failed for {}, trying HuggingFace", filename);
            download_file(
                &client,
                &hf_url,
                &dest,
                &app_handle,
                state,
                downloaded_total,
                total_bytes,
            )
            .await
            .context(format!("Failed to download {} from both sources", filename))?;
        }

        // Update total downloaded
        let metadata = tokio::fs::metadata(&dest).await?;
        downloaded_total += metadata.len();
    }

    let _ = app_handle.emit("download-complete", ());
    Ok(())
}

async fn download_file(
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    app_handle: &AppHandle,
    state: &state::AppState,
    base_downloaded: u64,
    total_bytes: u64,
) -> Result<()> {
    tracing::info!("Downloading {} → {:?}", url, dest);

    // Check for partial download (resume support)
    let mut file_downloaded: u64 = 0;
    let tmp_path = dest.with_extension("part");

    if tmp_path.exists() {
        file_downloaded = tokio::fs::metadata(&tmp_path).await?.len();
    }

    let mut request = client.get(url);
    if file_downloaded > 0 {
        request = request.header("Range", format!("bytes={}-", file_downloaded));
    }

    let response = request.send().await?.error_for_status()?;
    let _content_length = response.content_length().unwrap_or(0);

    let mut file = if file_downloaded > 0 {
        tokio::fs::OpenOptions::new()
            .append(true)
            .open(&tmp_path)
            .await?
    } else {
        tokio::fs::File::create(&tmp_path).await?
    };

    let mut stream = response.bytes_stream();
    let mut chunk_downloaded = file_downloaded;
    let start = std::time::Instant::now();
    let mut last_emit = std::time::Instant::now();

    while let Some(chunk) = stream.next().await {
        if state.is_download_cancelled() {
            let _ = tokio::fs::remove_file(&tmp_path).await;
            return Err(anyhow::anyhow!("Download cancelled"));
        }

        let chunk = chunk.context("Stream error")?;
        file.write_all(&chunk).await?;
        chunk_downloaded += chunk.len() as u64;

        // Emit progress at most 10 times per second
        if last_emit.elapsed().as_millis() >= 100 {
            let elapsed = start.elapsed().as_secs_f64();
            let speed_mbps = if elapsed > 0.0 {
                (chunk_downloaded - file_downloaded) as f64 / elapsed / 1_000_000.0
            } else {
                0.0
            };

            let _ = app_handle.emit(
                "download-progress",
                DownloadProgress {
                    downloaded_bytes: base_downloaded + chunk_downloaded,
                    total_bytes,
                    speed_mbps,
                },
            );
            last_emit = std::time::Instant::now();
        }
    }

    file.flush().await?;
    drop(file);

    // Move from .part to final destination
    tokio::fs::rename(&tmp_path, dest).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_files_defined() {
        assert_eq!(MODEL_FILES.len(), 2);
        assert_eq!(MODEL_FILES[0].0, "voxtral-q4.gguf");
        assert_eq!(MODEL_FILES[1].0, "tekken.json");
    }

    #[test]
    fn test_is_model_downloaded_false() {
        // Model shouldn't be downloaded in test environment
        // (unless running on a dev machine that has it)
        let dir = state::model_dir();
        if !dir.exists() {
            assert!(!is_model_downloaded());
        }
    }
}
