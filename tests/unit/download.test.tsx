import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import DownloadScreen from '../../src/components/DownloadScreen';
import type { ModelStatus } from '../../src/types';

vi.mock('@tauri-apps/api/core', () => ({
  invoke: vi.fn(() => Promise.resolve()),
}));

vi.mock('@tauri-apps/api/event', () => ({
  listen: vi.fn(() => Promise.resolve(() => {})),
}));

describe('DownloadScreen', () => {
  const mockOnStatusChange = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows download button when not downloaded', () => {
    const status: ModelStatus = { kind: 'not_downloaded' };
    render(<DownloadScreen modelStatus={status} onStatusChange={mockOnStatusChange} />);
    expect(screen.getByText('Download Model')).toBeInTheDocument();
    expect(screen.getByText(/2.5 GB/)).toBeInTheDocument();
  });

  it('shows progress bar when downloading', () => {
    const status: ModelStatus = { kind: 'downloading', progress: 0.5, speed_mbps: 10.2 };
    render(<DownloadScreen modelStatus={status} onStatusChange={mockOnStatusChange} />);
    expect(screen.getByText('50%')).toBeInTheDocument();
    expect(screen.getByText('10.2 MB/s')).toBeInTheDocument();
    expect(screen.getByText('Cancel')).toBeInTheDocument();
  });

  it('shows loading spinner when model is loading', () => {
    const status: ModelStatus = { kind: 'loading' };
    render(<DownloadScreen modelStatus={status} onStatusChange={mockOnStatusChange} />);
    expect(screen.getByText('Loading model on GPU...')).toBeInTheDocument();
  });

  it('shows error with retry button', () => {
    const status: ModelStatus = { kind: 'error', message: 'Network error' };
    render(<DownloadScreen modelStatus={status} onStatusChange={mockOnStatusChange} />);
    expect(screen.getByText('Network error')).toBeInTheDocument();
    expect(screen.getByText('Retry Download')).toBeInTheDocument();
  });

  it('calls invoke on download button click', async () => {
    const status: ModelStatus = { kind: 'not_downloaded' };
    render(<DownloadScreen modelStatus={status} onStatusChange={mockOnStatusChange} />);
    fireEvent.click(screen.getByText('Download Model'));
    expect(mockOnStatusChange).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'downloading' }),
    );
  });
});
