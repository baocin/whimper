# Whimper

Local, privacy-first voice-to-text transcription for macOS. An open-source alternative to SuperWhisper that runs entirely on-device.

Seriously got tired of not owning the software that most of my thoughts were streaming through.

## How it works

1. **First launch** — downloads the Parakeet TDT 0.6B ASR model (~662 MB) to `~/.whimper/models/`
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

## Making Parakeet TDT work

Getting good transcription from the base Parakeet TDT 0.6B ONNX model required several non-obvious fixes. Here's what mattered.

### Audio preprocessing (`src-tauri/src/asr/preprocess.rs`)

- **80Hz Butterworth high-pass filter** — removes desk vibration, AC hum, and other low-frequency noise before feature extraction
- **Peak normalization** — amplifies quiet signals to 0.95 peak; skips near-silence (peak < 0.001) to avoid amplifying noise
- **Silence trimming via Silero VAD** — trims leading/trailing silence using 512-sample chunks with 20ms padding to preserve word boundaries

### Mel spectrogram features (`src-tauri/src/asr/mel_features.rs`)

- 0.97 preemphasis, 25ms Hann window, 10ms hop, 512-pt FFT, 128 mel bins
- **Slaney-normalized mel filterbank** — this is critical; must match the NeMo training pipeline exactly or accuracy degrades significantly
- **Log energy with ε = 2⁻²⁴** — matches NeMo's epsilon for numerical stability
- Per-band mean-variance normalization

### TDT decoder fixes (`src-tauri/src/asr/parakeet.rs`)

- **Cache decoder output** — only call the decoder on non-blank token emission, not every frame (massive speedup)
- **Single-token decoder input** — pass only the last emitted token to the decoder, not the full hypothesis
- **Duration off-by-one fix** — `max(argmax, 1)` prevents zero-frame skips that cause the decode loop to stall

### Audio capture (`src-tauri/src/audio/`)

- **FFT-based resampling** to 16kHz via rubato
- **Resampler flush on stop** — zero-pads the final chunk and proportionally trims output so trailing audio isn't lost
- **300ms pre-roll ring buffer** — continuously captures mic input so speech before the hotkey press isn't clipped

### Hallucination filtering (`src-tauri/src/asr/parakeet.rs`)

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
