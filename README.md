# Whimper

Local, privacy-first voice-to-text transcription for macOS. An open-source alternative to SuperWhisper that runs entirely on-device.

Seriously got tired of not owning the software that most of my thoughts were streaming through.

## macOS downloads

Get the app from [GitHub Releases](https://github.com/baocin/whimper/releases/latest).
Choose `aarch64` for Apple Silicon or `x64` / `x86_64` for Intel. Releases include
a disk image and an app archive, plus SHA-256 checksums. The builds are not
Developer ID signed or notarized; macOS may require approval through System
Settings → Privacy & Security → Open Anyway.

Completed dictation is appended to
`~/Library/Application Support/com.whimper.app/log.txt`. This file contains your
transcribed text and stays on your Mac.

## How it works

1. **First launch** — downloads the Nemotron Speech Streaming EN 0.6B ASR model, INT4 (~662 MB) to `~/.whimper/models/`
2. **Background** — the app hides after model load, a pre-roll mic keeps a 300ms audio buffer ready
3. **Option+Space** — starts recording with a translucent overlay; pre-roll captures the start of your speech. The overlay only appears once the model is loaded *and* the mic is confirmed delivering audio
4. **While you speak** — text is typed into the focused app as the model emits it, in 560ms chunks. Keystrokes are synthesized directly, so your clipboard is never touched
5. **Option+Space again** — stops instantly. The streamed text is what you keep; there is no re-transcription to wait for
6. **ESC** — cancels recording and erases everything typed during that utterance

Set `WHIMPER_FINALIZE=1` to trade that instant stop for accuracy: on stop it re-transcribes the whole recording through the full pipeline (GTCRN, spectral gate, VAD trim), then backspaces and retypes wherever that disagrees with the live text. Costs roughly a second or two per utterance, and is the only mode that adds end-of-utterance punctuation.

Note: keystrokes go wherever focus is. Switching apps mid-dictation types — and backspaces — into the new one.

All speech processing happens locally via ONNX Runtime. No network calls after the initial model download.

## Requirements

- macOS (Apple Silicon or Intel)
- Microphone permission
- Accessibility permission (for synthesized keystrokes via CGEvent)

Note: replacing the app bundle can invalidate its Accessibility grant. If dictation
produces no text at all, re-check it under System Settings → Privacy & Security.

## Tech stack

| Layer | Technology |
|-------|-----------|
| Framework | Tauri 2 |
| Backend | Rust |
| Frontend | React 18 + TypeScript + Vite |
| Styling | Tailwind CSS |
| ASR | NVIDIA Nemotron Speech Streaming EN 0.6B (ONNX, INT4) via `parakeet-rs` |
| Audio | cpal + rubato (16kHz resampling) |
| Hotkey | tauri-plugin-global-shortcut |
| Typing | core-graphics CGEvent (unicode keystrokes + backspace); NSPasteboard only as fallback |

## The ASR model

Nemotron Speech Streaming EN 0.6B is a cache-aware FastConformer encoder (24 layers) with an RNNT decoder. It emits punctuation and capitalization natively, and being cache-aware it has no fixed positional-encoding ceiling — long files stream without the chunk-size gymnastics the previous Parakeet TDT model needed.

The ONNX sessions, mel extraction, SentencePiece vocabulary and streaming cache are all owned by the [`parakeet-rs`](https://crates.io/crates/parakeet-rs) crate. Weights are the INT4 export from [`lokkju/nemotron-speech-streaming-en-0.6b-int4`](https://huggingface.co/lokkju/nemotron-speech-streaming-en-0.6b-int4) — same download size as the old model. INT8 (~892 MB) and FP32 (~2.5 GB) exports exist if accuracy matters more than disk.

### Audio preprocessing (`src-tauri/src/asr/preprocess.rs`)

The live stream gets an 80Hz high-pass filter and nothing else — denoising needs
the whole buffer, so it cannot run incrementally. Everything below therefore runs
only in `WHIMPER_FINALIZE=1` mode, and on file transcription:

- **80Hz Butterworth high-pass filter** — removes desk vibration, AC hum, and other low-frequency noise
- **VAD-gated GTCRN noise reduction** — only processes frames Silero VAD marks as speech
- **Spectral gate** — cleans residual GTCRN artifacts using the noise floor from non-speech frames
- **RMS normalization**
- **Silence trimming via Silero VAD** — trims leading/trailing silence using 512-sample chunks with 20ms padding to preserve word boundaries

### Audio capture (`src-tauri/src/audio/`)

- **FFT-based resampling** to 16kHz via rubato
- **Resampler flush on stop** — zero-pads the final chunk and proportionally trims output so trailing audio isn't lost
- **300ms pre-roll ring buffer** — continuously captures mic input so speech before the hotkey press isn't clipped

### Hallucination filtering (`src-tauri/src/asr/nemotron.rs`)

- Filters known spurious outputs ("thank you", "thanks for watching", "[music]", etc.) that the model hallucinates on silence or noise
- Rejects transcripts shorter than 3 characters

## Development

```bash
# Install frontend dependencies
bun install

# Run in development mode
bun tauri dev

# Build release
bun tauri build

# Run tests
bun test                          # frontend (vitest)
cd src-tauri && cargo test        # backend (rust)
```

Pushing a `v*` tag builds and tests both macOS architectures, then publishes the
downloads to GitHub Releases after both builds succeed. Branch pushes and pull
requests build downloadable Actions artifacts without publishing a release.

## Architecture

```
src/                  # React frontend (overlay UI, download screen)
src-tauri/
  src/
    audio/
      microphone.rs   # cpal mic capture, mono conversion, resampling
      pipeline.rs     # Pre-roll ring buffer, recording accumulator
    asr/
      nemotron.rs     # Nemotron streaming ASR (via parakeet-rs)
      preprocess.rs   # Highpass, spectral gate, RMS normalize, silence trim
      gtcrn.rs        # GTCRN speech enhancement
      vad.rs          # Silero VAD (voice activity detection)
    dictation.rs      # Live typing during recording + reconcile against final pass
    download/         # Model download with progress + resume
    paste/            # Keystroke injection (CGEvent), pasteboard + Cmd+V fallback
    state.rs          # Shared app state
    lib.rs            # Tauri commands, hotkey handler, app setup
```

## License

MIT
