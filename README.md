# Whimper

Local, privacy-first voice-to-text transcription for macOS. An open-source alternative to SuperWhisper that runs entirely on-device.

## How it works

1. **First launch** — downloads the Parakeet TDT 0.6B ASR model (~250 MB) to `~/.whimper/models/`
2. **Background** — the app hides after model load, a pre-roll mic keeps a 300ms audio buffer ready
3. **Option+Space** — starts recording with a translucent overlay; pre-roll captures the start of your speech
4. **Option+Space again** — stops recording, transcribes, and pastes the text into the previously focused app
5. **ESC** — cancels recording without pasting

All speech processing happens locally via ONNX Runtime. No network calls after the initial model download.

## Requirements

- macOS (Apple Silicon or Intel)
- Microphone permission
- Accessibility permission (for paste via simulated Cmd+V)

## Tech stack

| Layer | Technology |
|-------|-----------|
| Framework | Tauri 2 |
| Backend | Rust |
| Frontend | React 18 + TypeScript + Vite |
| Styling | Tailwind CSS |
| ASR | NVIDIA Parakeet TDT 0.6B (ONNX) |
| Audio | cpal + rubato (16kHz resampling) |
| Hotkey | tauri-plugin-global-shortcut |
| Paste | cocoa + objc + core-graphics (NSPasteboard + CGEvent) |

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

## Architecture

```
src/                  # React frontend (overlay UI, download screen)
src-tauri/
  src/
    audio/
      microphone.rs   # cpal mic capture, mono conversion, resampling
      pipeline.rs     # Pre-roll ring buffer, recording accumulator
    asr/
      parakeet.rs     # Parakeet TDT ONNX inference
      preprocess.rs   # Mel spectrogram feature extraction
      vad.rs          # Silero VAD (voice activity detection)
    download/         # Model download with progress + resume
    paste/            # macOS pasteboard + simulated Cmd+V
    state.rs          # Shared app state
    lib.rs            # Tauri commands, hotkey handler, app setup
```

## License

MIT
