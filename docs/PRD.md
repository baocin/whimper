# Whimper — Product Requirements Document

## Overview

Whimper is a macOS desktop application that provides local, privacy-first voice-to-text transcription using the NVIDIA Parakeet TDT 0.6B model. It replaces SuperWhisper with an open-source alternative that runs entirely on-device after initial model download.

## Problem

SuperWhisper is paid and closed-source. Users who want fast, accurate, local dictation on macOS have limited open-source options.

## Solution

A lightweight Tauri 2 app (Rust + React) that:
1. Downloads the Parakeet TDT 0.6B ONNX model (~250 MB) on first launch
2. Runs the model locally via ONNX Runtime
3. Listens for a global hotkey (Option+Space) to start/stop dictation
4. Shows a minimal translucent overlay while recording
5. Batch-transcribes and pastes the transcript into the previously focused application

## User Journey

1. **First Launch**: App detects missing model → shows download progress screen
2. **Model Ready**: Model loads (~3-5s) → main window hides, background pre-roll mic starts
3. **Dictation Start**: User presses Option+Space from any app → 300ms of pre-roll audio captured, small translucent overlay appears at top-center showing "Listening..."
4. **Speaking**: User speaks → audio accumulates in memory
5. **Dictation End (paste)**: User presses Option+Space again → audio is batch-transcribed, overlay closes, transcript is pasted into the previously focused app via pasteboard + simulated Cmd+V
6. **Dictation Cancel**: User presses ESC → overlay closes, recording discarded, nothing pasted

## Functional Requirements

### FR-1: Model Management
- Download NVIDIA Parakeet TDT 0.6B ONNX model (~250 MB) on first launch
- Show download progress with percentage, speed, and ETA
- Support HTTP Range headers for download resume
- Store models in `~/.whimper/models/parakeet-tdt/`
- User can cancel and resume downloads

### FR-2: Audio Capture
- Capture microphone input using the default input device (cpal)
- Resample to 16kHz mono using rubato
- Pre-roll ring buffer (300ms) captures speech before hotkey press
- Background mic runs continuously after model load; paused during active recording
- Handle mic permission denial gracefully

### FR-3: Speech Recognition
- Load Parakeet TDT 0.6B ONNX model via ONNX Runtime
- Batch-transcribe accumulated audio on recording stop
- Mel spectrogram feature extraction (80-bin, 400-sample FFT, 160-sample hop)
- TDT decoder with token and duration heads
- Hallucination detection filters out model artifacts
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
- Shows: pulsing mic indicator while recording
- Dark translucent background with white text

### FR-6: Paste Mechanism
- Save reference to frontmost application before showing overlay (NSWorkspace)
- On dictation end: write transcript to NSPasteboard
- Reactivate the previously focused application
- Simulate Cmd+V keystroke via CGEvent to paste
- Requires Accessibility permission (AXIsProcessTrusted)

## Non-Functional Requirements

- **Privacy**: All processing is local. No network calls after model download.
- **Performance**: Transcription completes within seconds of recording stop. Model load ≤ 5s on Apple Silicon.
- **Storage**: ~250 MB for model files in `~/.whimper/`
- **Platform**: macOS only (Apple Silicon + Intel with Metal/wgpu support)
- **Accessibility**: Prompt user for microphone and accessibility permissions on first use

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | Tauri 2.x |
| Backend | Rust |
| Frontend | React 18 + TypeScript + Vite |
| Styling | Tailwind CSS |
| ASR Model | NVIDIA Parakeet TDT 0.6B (ONNX) |
| ML Runtime | ONNX Runtime |
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
3. Speech is transcribed accurately after recording stops
4. Option+Space again pastes transcript into the previously focused app
5. ESC cancels recording without pasting
6. All automated tests pass
