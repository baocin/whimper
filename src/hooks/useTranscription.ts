import { useEffect, useState } from 'react';
import { listen } from '@tauri-apps/api/event';
import type { TranscriptUpdate } from '../types';

export function useTranscription() {
  const [transcript, setTranscript] = useState('');

  useEffect(() => {
    const unlisten = listen<TranscriptUpdate>('transcript-update', (event) => {
      setTranscript(event.payload.text);
    });

    return () => {
      unlisten.then((f) => f());
    };
  }, []);

  return { transcript };
}
