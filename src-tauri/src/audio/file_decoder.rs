//! Decode an audio or video file to 16kHz mono f32 samples.
//!
//! Uses `symphonia` for demuxing/decoding and `rubato` for resampling,
//! matching the same pipeline used by `microphone.rs`.

use rubato::Resampler;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

const TARGET_SAMPLE_RATE: u32 = 16000;
const RESAMPLER_CHUNK_SIZE: usize = 1024;

/// Decode the audio track of a media file and return 16kHz mono f32 samples.
///
/// Supports `.mov`, `.mp4`, `.m4a` (isomp4 + aac), `.mp3`, and `.wav`.
/// Returns an error string if the file cannot be opened or decoded.
pub fn decode_audio_file(path: &str) -> Result<Vec<f32>, String> {
    // Open the file
    let file = std::fs::File::open(path)
        .map_err(|e| format!("Cannot open file: {e}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Probe the file format
    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(path).extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| format!("Unsupported file format: {e}"))?;

    let mut format = probed.format;

    // Find the first audio track
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or_else(|| "No audio track found in file".to_string())?;

    let track_id = track.id;
    let codec_params = track.codec_params.clone();

    let native_sample_rate = codec_params.sample_rate
        .ok_or_else(|| "Audio track has no sample rate".to_string())?;
    let channels = codec_params.channels
        .map(|c| c.count())
        .unwrap_or(1);

    // Create decoder
    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| format!("Cannot create decoder: {e}"))?;

    // Set up resampler if needed
    let needs_resample = native_sample_rate != TARGET_SAMPLE_RATE;
    let mut resampler: Option<rubato::FftFixedIn<f32>> = if needs_resample {
        Some(
            rubato::FftFixedIn::<f32>::new(
                native_sample_rate as usize,
                TARGET_SAMPLE_RATE as usize,
                RESAMPLER_CHUNK_SIZE,
                1,
                1, // mono output
            )
            .map_err(|e| format!("Cannot create resampler: {e}"))?,
        )
    } else {
        None
    };

    let mut mono_samples: Vec<f32> = Vec::new();
    let mut resample_input_buf: Vec<f32> = Vec::new();

    // Decode all packets
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(symphonia::core::errors::Error::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(e) => return Err(format!("Decode error: {e}")),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::IoError(_))
            | Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(format!("Decode error: {e}")),
        };

        // Convert to interleaved f32
        let interleaved = audio_buf_to_f32(&decoded);

        // Mix down to mono
        let frame_count = interleaved.len() / channels;
        let mut mono_chunk: Vec<f32> = Vec::with_capacity(frame_count);
        for frame in 0..frame_count {
            let sum: f32 = (0..channels)
                .map(|ch| interleaved[frame * channels + ch])
                .sum();
            mono_chunk.push(sum / channels as f32);
        }

        if let Some(ref mut rs) = resampler {
            // Feed into resampler accumulator
            resample_input_buf.extend_from_slice(&mono_chunk);

            // Process full chunks
            while resample_input_buf.len() >= RESAMPLER_CHUNK_SIZE {
                let chunk: Vec<f32> = resample_input_buf.drain(..RESAMPLER_CHUNK_SIZE).collect();
                let input = vec![chunk];
                match rs.process(&input, None) {
                    Ok(output) => {
                        if let Some(resampled) = output.into_iter().next() {
                            mono_samples.extend_from_slice(&resampled);
                        }
                    }
                    Err(e) => return Err(format!("Resample error: {e}")),
                }
            }
        } else {
            mono_samples.extend_from_slice(&mono_chunk);
        }
    }

    // Flush remaining samples through the resampler
    if let Some(ref mut rs) = resampler {
        let remaining = resample_input_buf.len();
        if remaining > 0 {
            let ratio = TARGET_SAMPLE_RATE as f64 / native_sample_rate as f64;
            let valid_output = (remaining as f64 * ratio).ceil() as usize;

            resample_input_buf.resize(RESAMPLER_CHUNK_SIZE, 0.0);
            let input = vec![resample_input_buf.clone()];
            if let Ok(output) = rs.process(&input, None) {
                if let Some(resampled) = output.into_iter().next() {
                    let valid = valid_output.min(resampled.len());
                    mono_samples.extend_from_slice(&resampled[..valid]);
                }
            }
        }
    }

    if mono_samples.is_empty() {
        return Err("No audio samples decoded from file".to_string());
    }

    Ok(mono_samples)
}

/// Convert a symphonia `AudioBufferRef` to interleaved f32 samples.
fn audio_buf_to_f32(buf: &AudioBufferRef<'_>) -> Vec<f32> {
    match buf {
        AudioBufferRef::F32(b) => {
            let channels = b.spec().channels.count();
            let frames = b.frames();
            let mut out = vec![0.0f32; frames * channels];
            for ch in 0..channels {
                let plane = b.chan(ch);
                for (i, &s) in plane.iter().enumerate() {
                    out[i * channels + ch] = s;
                }
            }
            out
        }
        AudioBufferRef::F64(b) => {
            let channels = b.spec().channels.count();
            let frames = b.frames();
            let mut out = vec![0.0f32; frames * channels];
            for ch in 0..channels {
                let plane = b.chan(ch);
                for (i, &s) in plane.iter().enumerate() {
                    out[i * channels + ch] = s as f32;
                }
            }
            out
        }
        AudioBufferRef::S16(b) => {
            let channels = b.spec().channels.count();
            let frames = b.frames();
            let mut out = vec![0.0f32; frames * channels];
            for ch in 0..channels {
                let plane = b.chan(ch);
                for (i, &s) in plane.iter().enumerate() {
                    out[i * channels + ch] = s as f32 / 32768.0;
                }
            }
            out
        }
        AudioBufferRef::S32(b) => {
            let channels = b.spec().channels.count();
            let frames = b.frames();
            let mut out = vec![0.0f32; frames * channels];
            for ch in 0..channels {
                let plane = b.chan(ch);
                for (i, &s) in plane.iter().enumerate() {
                    out[i * channels + ch] = s as f32 / 2_147_483_648.0;
                }
            }
            out
        }
        AudioBufferRef::U8(b) => {
            let channels = b.spec().channels.count();
            let frames = b.frames();
            let mut out = vec![0.0f32; frames * channels];
            for ch in 0..channels {
                let plane = b.chan(ch);
                for (i, &s) in plane.iter().enumerate() {
                    out[i * channels + ch] = (s as f32 - 128.0) / 128.0;
                }
            }
            out
        }
        AudioBufferRef::S24(b) => {
            let channels = b.spec().channels.count();
            let frames = b.frames();
            let mut out = vec![0.0f32; frames * channels];
            for ch in 0..channels {
                let plane = b.chan(ch);
                for (i, &s) in plane.iter().enumerate() {
                    out[i * channels + ch] = s.0 as f32 / 8_388_608.0;
                }
            }
            out
        }
        _ => Vec::new(),
    }
}
