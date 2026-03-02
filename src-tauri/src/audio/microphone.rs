//! Streaming microphone capture
//!
//! Ported from timeline_app's microphone.rs. Adapted from fixed-duration
//! recording to streaming callback model for real-time transcription.

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rubato::Resampler;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

const TARGET_SAMPLE_RATE: u32 = 16000;
const RESAMPLER_CHUNK_SIZE: usize = 1024;

/// Callback invoked with 16kHz mono f32 samples as they arrive
pub type AudioChunkCallback = Box<dyn Fn(&[f32]) + Send + Sync>;

/// A running microphone capture stream
pub struct MicStream {
    _stream: cpal::Stream,
    is_active: Arc<AtomicBool>,
}

// Safety: cpal::Stream is not Send by default, but on macOS (CoreAudio) the underlying
// AudioUnit handle is safe to move across threads. We need Send so MicStream can be
// stored in AppState (which is shared across async tasks).
unsafe impl Send for MicStream {}

impl MicStream {
    /// Stop the microphone stream
    pub fn stop(&self) {
        self.is_active.store(false, Ordering::SeqCst);
    }
}

/// Check if a microphone is available
pub fn is_microphone_available() -> bool {
    cpal::default_host().default_input_device().is_some()
}

/// Start streaming microphone capture. Calls `on_chunk` with 16kHz mono samples.
pub fn start_capture(on_chunk: AudioChunkCallback) -> Result<MicStream> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("No input device available")?;

    let supported_config = device
        .default_input_config()
        .context("Failed to get default input config")?;

    let sample_rate = supported_config.sample_rate().0;
    let channels = supported_config.channels() as usize;

    tracing::info!(
        "Mic capture: {}Hz, {} channels, {:?}",
        sample_rate,
        channels,
        supported_config.sample_format()
    );

    let resample_ratio = TARGET_SAMPLE_RATE as f64 / sample_rate as f64;
    let needs_resample = (resample_ratio - 1.0).abs() > 0.001;

    let resampler = if needs_resample {
        Some(Arc::new(Mutex::new(
            rubato::FftFixedIn::<f32>::new(
                sample_rate as usize,
                TARGET_SAMPLE_RATE as usize,
                RESAMPLER_CHUNK_SIZE,
                1,
                1,
            )
            .context("Failed to create resampler")?,
        )))
    } else {
        None
    };

    let is_active = Arc::new(AtomicBool::new(true));
    let accumulator = Arc::new(Mutex::new(Vec::<f32>::new()));
    let on_chunk = Arc::new(on_chunk);

    let is_active_clone = Arc::clone(&is_active);
    let resampler_clone = resampler.clone();
    let accumulator_clone = Arc::clone(&accumulator);
    let on_chunk_clone = Arc::clone(&on_chunk);

    let stream = match supported_config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &supported_config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if !is_active_clone.load(Ordering::SeqCst) {
                    return;
                }
                process_and_deliver(data, channels, &resampler_clone, &accumulator_clone, &on_chunk_clone);
            },
            |err| tracing::error!("Audio stream error: {}", err),
            None,
        )?,
        cpal::SampleFormat::I16 => {
            let is_active2 = Arc::clone(&is_active);
            let resampler2 = resampler.clone();
            let accumulator2 = Arc::clone(&accumulator);
            let on_chunk2 = Arc::clone(&on_chunk);

            device.build_input_stream(
                &supported_config.into(),
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    if !is_active2.load(Ordering::SeqCst) {
                        return;
                    }
                    let float_data: Vec<f32> =
                        data.iter().map(|&s| s as f32 / 32768.0).collect();
                    process_and_deliver(&float_data, channels, &resampler2, &accumulator2, &on_chunk2);
                },
                |err| tracing::error!("Audio stream error: {}", err),
                None,
            )?
        }
        format => {
            return Err(anyhow::anyhow!("Unsupported sample format: {:?}", format));
        }
    };

    stream.play().context("Failed to start audio stream")?;

    Ok(MicStream {
        _stream: stream,
        is_active,
    })
}

/// Convert to mono, resample if needed, and deliver via callback
fn process_and_deliver(
    data: &[f32],
    channels: usize,
    resampler: &Option<Arc<Mutex<rubato::FftFixedIn<f32>>>>,
    accumulator: &Arc<Mutex<Vec<f32>>>,
    on_chunk: &Arc<AudioChunkCallback>,
) {
    // Convert to mono
    let mono: Vec<f32> = if channels > 1 {
        data.chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        data.to_vec()
    };

    if let Some(ref rs) = resampler {
        let mut acc = match accumulator.lock() {
            Ok(guard) => guard,
            Err(_) => return,
        };
        acc.extend(mono);

        while acc.len() >= RESAMPLER_CHUNK_SIZE {
            let chunk: Vec<f32> = acc.drain(..RESAMPLER_CHUNK_SIZE).collect();
            let input = vec![chunk];

            if let Ok(mut guard) = rs.lock() {
                if let Ok(output) = guard.process(&input, None) {
                    if let Some(resampled) = output.into_iter().next() {
                        on_chunk(&resampled);
                    }
                }
            }
        }
    } else {
        on_chunk(&mono);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_microphone_check() {
        let _available = is_microphone_available();
    }

    #[test]
    fn test_process_mono_no_resample() {
        let received = Arc::new(Mutex::new(Vec::<f32>::new()));
        let received_clone = Arc::clone(&received);
        let cb: AudioChunkCallback = Box::new(move |samples| {
            received_clone.lock().unwrap().extend_from_slice(samples);
        });
        let cb = Arc::new(cb);
        let acc = Arc::new(Mutex::new(Vec::new()));

        let input = vec![0.1f32, 0.2, 0.3];
        process_and_deliver(&input, 1, &None, &acc, &cb);

        let result = received.lock().unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_process_stereo_to_mono() {
        let received = Arc::new(Mutex::new(Vec::<f32>::new()));
        let received_clone = Arc::clone(&received);
        let cb: AudioChunkCallback = Box::new(move |samples| {
            received_clone.lock().unwrap().extend_from_slice(samples);
        });
        let cb = Arc::new(cb);
        let acc = Arc::new(Mutex::new(Vec::new()));

        // Stereo: L=0.2, R=0.4 → mono = 0.3
        let input = vec![0.2f32, 0.4];
        process_and_deliver(&input, 2, &None, &acc, &cb);

        let result = received.lock().unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 0.3).abs() < 1e-6);
    }
}
