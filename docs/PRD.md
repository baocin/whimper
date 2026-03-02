# Whimper — Product Requirements Document

## Overview

Whimper is a macOS desktop application that provides local, privacy-first voice-to-text transcription using the Voxtral-Mini-4B-Realtime model. It replaces SuperWhisper with an open-source alternative that runs entirely on-device after initial model download.

## Problem

SuperWhisper is paid and closed-source. Users who want fast, accurate, local dictation on macOS have limited open-source options — especially with state-of-the-art models like Voxtral that offer real-time streaming ASR.

## Solution

A lightweight Tauri 2 app (Rust + React) that:
1. Downloads the Voxtral-Mini-4B GGUF model (~2.51 GB) on first launch
2. Runs the model locally on GPU via Burn/wgpu
3. Listens for a global hotkey (Option+Space) to start/stop dictation
4. Shows a minimal translucent overlay with live transcript
5. Pastes the transcript into the previously focused application

## User Journey

1. **First Launch**: App detects missing model → shows download progress screen
2. **Model Ready**: Model loads on GPU (~3-5s) → main window hides, app lives in background
3. **Dictation Start**: User presses Option+Space from any app → small translucent overlay appears at top-center showing "Listening..."
4. **Speaking**: User speaks → real-time transcript streams into overlay
5. **Dictation End (paste)**: User presses Option+Space again → overlay closes, transcript is pasted into the previously focused app via pasteboard + simulated Cmd+V
6. **Dictation Cancel**: User presses ESC → overlay closes, recording discarded, nothing pasted

## Functional Requirements

### FR-1: Model Management
- Download Voxtral-Mini-4B GGUF (~2.51 GB) and tokenizer (tekken.json ~14.9 MB) on first launch
- Primary source: R2 CDN (https://public.mydatatimeline.com/models/voxtral-mini-4b/)
- Fallback: HuggingFace (https://huggingface.co/TrevorJS/voxtral-mini-realtime-gguf/resolve/main/)
- Show download progress with percentage, speed, and ETA
- Support HTTP Range headers for download resume
- Store models in `~/.whimper/models/voxtral-mini-4b/`
- User can cancel and resume downloads

### FR-2: Audio Capture
- Capture microphone input using the default input device (cpal)
- Resample to 16kHz mono (Voxtral requirement) using rubato
- Stream audio chunks to ASR engine in real-time
- Handle mic permission denial gracefully

### FR-3: Speech Recognition
- Load Voxtral-Mini-4B-Realtime GGUF model via Burn/wgpu
- Process streaming audio chunks and emit partial transcripts
- Support continuous dictation (no fixed duration limit)
- Reset state between dictation sessions

### FR-4: Global Hotkey
- Register Option+Space (Alt+Space) as global shortcut via tauri-plugin-global-shortcut
- Toggle behavior: first press starts recording, second press stops and pastes
- ESC key dismisses overlay and cancels recording (no paste)
- Hotkey works from any application

### FR-5: Overlay Window
- Frameless, transparent, always-on-top window
- Size: 400x120px, positioned top-center, 100px from top of screen
- Non-focus-stealing (does not take focus from current app)
- Shows: pulsing mic indicator + live streaming transcript
- Dark translucent background with white text

### FR-6: Paste Mechanism
- Save reference to frontmost application before showing overlay (NSWorkspace)
- On dictation end: write transcript to NSPasteboard
- Reactivate the previously focused application
- Simulate Cmd+V keystroke via CGEvent to paste
- Requires Accessibility permission (AXIsProcessTrusted)

## Non-Functional Requirements

- **Privacy**: All processing is local. No network calls after model download.
- **Performance**: Transcript latency ≤ 500ms from speech. Model load ≤ 5s on Apple Silicon.
- **Storage**: ~2.6 GB for model files in `~/.whimper/`
- **Platform**: macOS only (Apple Silicon + Intel with Metal/wgpu support)
- **Accessibility**: Prompt user for microphone and accessibility permissions on first use

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | Tauri 2.x |
| Backend | Rust |
| Frontend | React 18 + TypeScript + Vite |
| Styling | Tailwind CSS |
| ASR Model | Voxtral-Mini-4B-Realtime (Q4 GGUF) |
| ML Runtime | Burn + wgpu |
| Audio | cpal + rubato |
| Hotkey | tauri-plugin-global-shortcut |
| Paste | cocoa + objc + core-graphics crates |
| Unit Tests | Vitest (frontend), cargo test (Rust) |
| E2E Tests | tauri-driver (WebDriver) |

## Testing Strategy

- **TDD**: Tests written before implementation at each phase
- **Rust unit tests**: Model loading, audio pipeline, paste mechanism
- **Frontend unit tests**: Component rendering, hook behavior (Vitest)
- **E2E tests**: Window lifecycle, hotkey simulation, mock audio transcription flow (tauri-driver)
- **Manual testing**: Real microphone input, cross-app paste verification

## Success Criteria

1. App launches and downloads model with progress indication
2. Option+Space opens overlay from any application
3. Speech is transcribed in real-time with ≤ 500ms latency
4. Option+Space again pastes transcript into the previously focused app
5. ESC cancels recording without pasting
6. All automated tests pass
