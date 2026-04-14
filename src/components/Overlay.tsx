import { useEffect, useState } from 'react';
import { listen } from '@tauri-apps/api/event';
import { useTranscription } from '../hooks/useTranscription';

export default function Overlay() {
  const { transcript } = useTranscription();
  const [processing, setProcessing] = useState(false);

  useEffect(() => {
    const unlisten = listen<string>('recording-state', (event) => {
      if (event.payload === 'processing') {
        setProcessing(true);
      } else if (event.payload === 'listening') {
        setProcessing(false);
      }
    });
    return () => { unlisten.then((f) => f()); };
  }, []);

  const dotColor = processing ? 'bg-amber-400' : 'bg-red-500';
  const label = processing ? 'Processing...' : (transcript || 'Listening...');

  return (
    <div className="w-full h-full bg-black/80 backdrop-blur-md rounded-2xl flex items-center px-5 gap-3">
      <div className="flex-shrink-0">
        <div className={`w-4 h-4 rounded-full ${dotColor} animate-pulse`} />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-white text-sm truncate">
          {label}
        </p>
      </div>
    </div>
  );
}
