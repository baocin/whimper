# Whimper

**Default ASR:** MOSS (Parakeet TDT 0.6B) via HTTP on this machine's Tailscale
IP (`100.76.212.98:9364`). Override with `WHIMPER_ASR_URL`. The original
local-ONNX path is preserved for macOS — the release build from this branch
(`msp/continuous`) uses it.

**Continuous mode:** set `WHIMPER_CONTINUOUS=1` to auto-start always-on
listening. Silence-gated 300ms chunks, trigger word "paste" (and phonetic
variants) to paste accumulated speech.

Local, privacy-first voice-to-text transcription for macOS and Linux. An open-source alternative to SuperWhisper that runs entirely on-device.

Seriously got tired of not owning the software that most of my thoughts were streaming through.

## How it works

1. **First launch** — downloads the Parakeet TDT 0.6B ASR model (~662 MB) to `~/.whimper/models/`
2. **Background** — the app hides after model load, a pre-roll mic keeps a 300ms audio buffer ready
3. **Hotkey** (Option+Space on macOS, Alt+Space on Linux) — starts recording with a translucent overlay; pre-roll captures the start of your speech
4. **Hotkey again** — stops recording, transcribes, and pastes the text into the focused app (Cmd+V on macOS, Ctrl+V on Linux)
5. **ESC** — cancels recording without pasting

All speech processing happens locally via ONNX Runtime. No network calls after the initial model download.

## Transcript log

Every transcription attempt is appended to `~/.whimper/transcripts.jsonl` — one JSON
object per line, holding the **full, untruncated** text plus timing. This includes
empty / silence-trimmed results and hallucination-filtered results (which are not
pasted), so the log is a complete record of what the model produced.

Each line has these fields:

| field | meaning |
|-------|---------|
| `ts_rfc3339` | UTC timestamp, RFC 3339 |
| `text` | full transcript text (untruncated) |
| `audio_duration_ms` | length of captured audio |
| `processing_time_ms` | wall-clock transcription time |
| `rtf` | real-time factor (`processing / audio`; lower is faster) |
| `pasted` | whether the text was pasted into the focused app |
| `hallucination` | matched a known hallucination pattern (skip-paste) |
| `empty` | empty / trimmed to nothing (skip-paste) |

The file is created with mode `0600` (owner-only) because it contains private
speech content. Writes are append-only (`O_APPEND`) and flushed per line; a write
failure is logged and never blocks or crashes the paste path.

**Opt out:** set `WHIMPER_NO_TRANSCRIPT_LOG=1` to disable logging entirely. Nothing
is written while it's set. Delete `~/.whimper/transcripts.jsonl` at any time to clear
the history.

## Requirements

### macOS
- macOS (Apple Silicon or Intel)
- Microphone permission
- Accessibility permission (for paste via simulated Cmd+V)

### Linux
- A microphone (cpal uses ALSA; routes through PipeWire on most modern setups)
- Membership in the `input` group, so the app can read the keyboard for the global hotkey:
  ```bash
  sudo usermod -aG input $USER   # then log out and back in
  ```
  Wayland doesn't let apps grab global shortcuts, so the Alt+Space hotkey is detected by reading `/dev/input/event*` directly via evdev.
  - **Group not active yet?** If you're already in the `input` group but your *current login session* started before you were added (supplementary groups are fixed at login), whimper can't read the keyboard. On startup it detects this and **self-heals** by re-launching itself under the group via `sg input` (no sudo, no re-login needed). If you're not a member at all, the app stays visible and shows an in-window warning with the exact fix instead of failing silently.
- **Wayland (Hyprland/Sway/wlroots):** key injection uses the `zwp_virtual_keyboard` protocol, which these compositors support out of the box. To stop the recording overlay from stealing keyboard focus (which would send the paste to the overlay instead of your app), add these rules to your Hyprland config:
  ```
  windowrulev2 = noinitialfocus, title:^(whimper-overlay)$
  windowrulev2 = nofocus, title:^(whimper-overlay)$
  ```
- Build-time system packages (Arch): `sudo pacman -S webkit2gtk-4.1 gtk3 librsvg alsa-lib base-devel`
- Speech inference runs on CPU (ONNX Runtime CPU provider); a GPU is not used.
- **NVIDIA + Wayland:** WebKitGTK's DMABUF renderer crashes the GTK backend (`Error 71 (Protocol error) dispatching to Wayland display`). whimper sets `WEBKIT_DISABLE_DMABUF_RENDERER=1` automatically at startup on Linux, so no manual workaround is needed. (Don't run the app with `sudo` — root has no Wayland auth and GTK won't initialize; the `input` group above is what grants hotkey access.)

> Note: the Alt+Space keystroke isn't consumed — it still reaches the focused app. Most apps ignore Alt+Space, but pick a different chord if it conflicts.

## Tech stack

| Layer | Technology |
|-------|-----------|
| Framework | Tauri 2 |
| Backend | Rust |
| Frontend | React 18 + TypeScript + Vite |
| Styling | Tailwind CSS |
| ASR | NVIDIA Parakeet TDT 0.6B (ONNX) |
| Audio | cpal + rubato (16kHz resampling) |
| Hotkey | macOS: tauri-plugin-global-shortcut · Linux: evdev (`/dev/input`) |
| Paste | macOS: cocoa + core-graphics (NSPasteboard + CGEvent) · Linux: arboard + enigo (clipboard + Ctrl+V) |

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

# Build release (binary + bundles in src-tauri/target/release/)
bun tauri build

# On Linux, building the AppImage needs extract-and-run (linuxdeploy and
# appimagetool are AppImages and FUSE-mounting them often fails):
APPIMAGE_EXTRACT_AND_RUN=1 bun tauri build

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
    paste/            # Clipboard + simulated paste (macOS Cmd+V / Linux Ctrl+V)
    input/            # Linux-only: evdev global-hotkey listener
    state.rs          # Shared app state
    lib.rs            # Tauri commands, hotkey handler, app setup
```

## License

MIT
