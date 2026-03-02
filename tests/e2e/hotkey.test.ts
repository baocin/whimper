/**
 * E2E test: Global hotkey + overlay lifecycle
 *
 * Uses tauri-driver (WebDriver) to verify:
 * - App launches correctly
 * - Overlay window appears/disappears
 *
 * Note: Actual global hotkey simulation requires OS-level tools.
 * These tests focus on window lifecycle via Tauri commands.
 */
import { describe, it, expect } from 'vitest';

describe('E2E: Hotkey + Overlay', () => {
  it.todo('app launches and shows download screen');
  it.todo('overlay window can be created and destroyed');
  it.todo('ESC closes the overlay window');
});
