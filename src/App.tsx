import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import DownloadScreen from './components/DownloadScreen';
import Overlay from './components/Overlay';
import type { ModelStatus } from './types';

function App() {
  // Overlay window loads at /overlay — render the overlay component directly
  if (window.location.pathname === '/overlay') {
    return <Overlay />;
  }

  return <MainWindow />;
}

function MainWindow() {
  const [modelStatus, setModelStatus] = useState<ModelStatus>({ kind: 'not_downloaded' });

  useEffect(() => {
    invoke<ModelStatus>('check_model_status').then((status) => {
      setModelStatus(status);
      if (status.kind === 'ready') {
        invoke('hide_main_window');
      }
    });
  }, []);

  useEffect(() => {
    if (modelStatus.kind === 'ready') {
      invoke('hide_main_window');
    }
  }, [modelStatus]);

  // Listen for backend auto-load completion
  useEffect(() => {
    const unlisten = listen('model-ready', () => {
      invoke('hide_main_window');
    });
    return () => { unlisten.then(fn => fn()); };
  }, []);

  return <DownloadScreen modelStatus={modelStatus} onStatusChange={setModelStatus} />;
}

export default App;
