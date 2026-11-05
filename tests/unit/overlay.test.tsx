import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import Overlay from '../../src/components/Overlay';

// Mock Tauri APIs
vi.mock('@tauri-apps/api/core', () => ({
  invoke: vi.fn(() => Promise.resolve()),
}));

vi.mock('@tauri-apps/api/event', () => ({
  listen: vi.fn(() => Promise.resolve(() => {})),
}));

describe('Overlay', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders listening state by default', () => {
    render(<Overlay />);
    expect(screen.getByText('Listening...')).toBeInTheDocument();
  });

  it('shows processing state from the backend', async () => {
    const { listen } = await import('@tauri-apps/api/event');
    render(<Overlay />);
    act(() => {
      for (const [event, callback] of vi.mocked(listen).mock.calls) {
        if (event === 'recording-state') {
          callback({ event, id: 1, payload: 'processing' });
        }
      }
    });
    expect(screen.getByText('Processing...')).toBeInTheDocument();
  });

  it('has a pulsing mic indicator', () => {
    const { container } = render(<Overlay />);
    const pulse = container.querySelector('.animate-pulse');
    expect(pulse).toBeInTheDocument();
  });
});
