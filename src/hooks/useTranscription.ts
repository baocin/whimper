import { useEffect, useState } from 'react';
import { listen } from '@tauri-apps/api/event';
import type { TranscriptUpdate } from '../types';

export function useTranscription() {
  const [transcript, setTranscript] = useState('');

  useEffect(() => {
    const unlistenTranscript = listen<TranscriptUpdate>('transcript-update', (event) => {
      setTranscript(event.payload.text);
    });
    const unlistenReset = listen<string>('recording-state', (event) => {
      if (event.payload === 'listening') {
        setTranscript('');
      }
    });

    return () => {
      unlistenTranscript.then((f) => f());
      unlistenReset.then((f) => f());
    };
  }, []);

  return { transcript };
}
