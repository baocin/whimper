/**
 * E2E test: Transcription flow with mock audio
 *
 * Uses tauri-driver to verify:
 * - Download screen → model loaded transition
 * - Mock audio input produces transcript in overlay
 * - Transcript is pasted on hotkey toggle
 */
import { describe, it, expect } from 'vitest';

describe('E2E: Transcription', () => {
  it.todo('download screen shows when model is missing');
  it.todo('transcript appears in overlay with mock audio');
  it.todo('transcript is cleared after paste');
});
