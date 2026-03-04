//! Model download with R2 CDN
//!
//! Downloads Parakeet TDT INT8 model files (~662 MB total)
//! to ~/.whimper/models/parakeet-tdt/

use anyhow::{Context, Result};
use futures_util::StreamExt;
use std::path::Path;
use tauri::{AppHandle, Emitter};
use tokio::io::AsyncWriteExt;

use crate::state;

const R2_BASE: &str = "https://public.mydatatimeline.com/models/parakeet-tdt-int8";

const MODEL_FILES: &[(&str, u64)] = &[
    ("encoder.int8.onnx", 652_000_000), // ~652 MB
    ("decoder.int8.onnx", 8_000_000),   // ~8 MB
    ("joiner.int8.onnx", 2_000_000),    // ~2 MB
    ("tokens.txt", 10_000),             // ~10 KB
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

        let url = format!("{}/{}", R2_BASE, filename);

        download_file(
            &client,
            &url,
            &dest,
            &app_handle,
            state,
            downloaded_total,
            total_bytes,
        )
        .await
        .context(format!("Failed to download {}", filename))?;

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

    tokio::fs::rename(&tmp_path, dest).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_files_defined() {
        assert_eq!(MODEL_FILES.len(), 4);
        assert_eq!(MODEL_FILES[0].0, "encoder.int8.onnx");
        assert_eq!(MODEL_FILES[3].0, "tokens.txt");
    }

    #[test]
    fn test_is_model_downloaded_false() {
        let dir = state::model_dir();
        if !dir.exists() {
            assert!(!is_model_downloaded());
        }
    }
}
