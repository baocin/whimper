import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
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

  it('calls cancel_recording on ESC key', async () => {
    const { invoke } = await import('@tauri-apps/api/core');
    render(<Overlay />);
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(invoke).toHaveBeenCalledWith('cancel_recording');
  });

  it('has a pulsing mic indicator', () => {
    const { container } = render(<Overlay />);
    const pulse = container.querySelector('.animate-pulse');
    expect(pulse).toBeInTheDocument();
  });
});
