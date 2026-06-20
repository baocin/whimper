export type ModelStatus =
  | { kind: 'not_downloaded' }
  | { kind: 'downloading'; progress: number; speed_mbps: number }
  | { kind: 'downloaded' }
  | { kind: 'loading' }
  | { kind: 'ready' }
  | { kind: 'error'; message: string };

export type RecordingState = 'idle' | 'listening' | 'processing';

export type HotkeyStatus =
  | 'available'
  | 'needs_relogin'
  | 'needs_group_add'
  | 'no_keyboard';

export interface TranscriptUpdate {
  text: string;
  is_final: boolean;
}

export interface DownloadProgress {
  downloaded_bytes: number;
  total_bytes: number;
  speed_mbps: number;
}
