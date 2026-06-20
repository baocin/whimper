import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';

const invokeMock = vi.fn();
vi.mock('@tauri-apps/api/core', () => ({
  invoke: (...args: unknown[]) => invokeMock(...args),
}));
vi.mock('@tauri-apps/api/event', () => ({
  listen: vi.fn(() => Promise.resolve(() => {})),
}));

import App from '../../src/App';

describe('MainWindow hotkey gating', () => {
  beforeEach(() => {
    invokeMock.mockReset();
  });

  it('does NOT hide the window when the hotkey is unavailable, even if the model is ready', async () => {
    invokeMock.mockImplementation((cmd: string) => {
      if (cmd === 'check_model_status') return Promise.resolve({ kind: 'ready' });
      if (cmd === 'check_hotkey_status') return Promise.resolve('needs_group_add');
      return Promise.resolve();
    });

    render(<App />);

    // Warning rendered => hotkey state applied and the hide effect has run.
    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument());
    expect(invokeMock).not.toHaveBeenCalledWith('hide_main_window');
  });

  it('hides the window when the model is ready AND the hotkey is available', async () => {
    invokeMock.mockImplementation((cmd: string) => {
      if (cmd === 'check_model_status') return Promise.resolve({ kind: 'ready' });
      if (cmd === 'check_hotkey_status') return Promise.resolve('available');
      return Promise.resolve();
    });

    render(<App />);

    await waitFor(() => expect(invokeMock).toHaveBeenCalledWith('hide_main_window'));
  });
});
