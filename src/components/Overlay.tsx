import { useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { useTranscription } from '../hooks/useTranscription';

export default function Overlay() {
  const { transcript } = useTranscription();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        invoke('cancel_recording');
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <div className="w-full h-full bg-black/80 backdrop-blur-md rounded-2xl flex items-center px-5 gap-3">
      <div className="flex-shrink-0">
        <div className="w-4 h-4 rounded-full bg-red-500 animate-pulse" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-white text-sm truncate">
          {transcript || 'Listening...'}
        </p>
      </div>
    </div>
  );
}
