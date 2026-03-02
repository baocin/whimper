import { useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import type { ModelStatus, DownloadProgress } from '../types';

export function useDownload(onStatusChange: (status: ModelStatus) => void) {
  useEffect(() => {
    const unlisten = listen<DownloadProgress>('download-progress', (event) => {
      const { downloaded_bytes, total_bytes, speed_mbps } = event.payload;
      const progress = total_bytes > 0 ? downloaded_bytes / total_bytes : 0;
      onStatusChange({ kind: 'downloading', progress, speed_mbps });
    });

    const unlistenComplete = listen('download-complete', () => {
      onStatusChange({ kind: 'downloaded' });
      // Auto-load model after download
      invoke('load_model')
        .then(() => onStatusChange({ kind: 'ready' }))
        .catch((err) => onStatusChange({ kind: 'error', message: String(err) }));
    });

    const unlistenError = listen<string>('download-error', (event) => {
      onStatusChange({ kind: 'error', message: event.payload });
    });

    return () => {
      unlisten.then((f) => f());
      unlistenComplete.then((f) => f());
      unlistenError.then((f) => f());
    };
  }, [onStatusChange]);

  const startDownload = () => {
    onStatusChange({ kind: 'downloading', progress: 0, speed_mbps: 0 });
    invoke('start_download').catch((err) =>
      onStatusChange({ kind: 'error', message: String(err) }),
    );
  };

  const cancelDownload = () => {
    invoke('cancel_download').catch(console.error);
    onStatusChange({ kind: 'not_downloaded' });
  };

  return { startDownload, cancelDownload };
}
