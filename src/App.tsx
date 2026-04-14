import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import DownloadScreen from './components/DownloadScreen';
import Overlay from './components/Overlay';
import type { HealthStatus, ModelStatus } from './types';

function App() {
  // Overlay window loads at /overlay — render the overlay component directly
  if (window.location.pathname === '/overlay') {
    return <Overlay />;
  }

  return <MainWindow />;
}

function MainWindow() {
  const [modelStatus, setModelStatus] = useState<ModelStatus>({ kind: 'not_downloaded' });
  const [healthWarnings, setHealthWarnings] = useState<string[]>([]);

  useEffect(() => {
    Promise.all([
      invoke<HealthStatus>('check_health'),
      invoke<ModelStatus>('check_model_status'),
    ]).then(([health, status]) => {
      const warnings: string[] = [];
      if (!health.microphone_available) {
        warnings.push('No microphone detected. Check System Settings → Privacy & Security → Microphone.');
      }
      if (!health.accessibility_granted) {
        warnings.push('Accessibility access required for pasting. Check System Settings → Privacy & Security → Accessibility.');
      }
      setHealthWarnings(warnings);
      setModelStatus(status);
    });
  }, []);

  // Listen for backend auto-load completion
  useEffect(() => {
    const unlisten = listen('model-ready', () => {
      setModelStatus({ kind: 'ready' });
    });
    return () => { unlisten.then(fn => fn()); };
  }, []);

  // Listen for model load errors
  useEffect(() => {
    const unlisten = listen<string>('model-load-error', (event) => {
      setHealthWarnings(prev => [...prev, `Model failed to load: ${event.payload}`]);
    });
    return () => { unlisten.then(fn => fn()); };
  }, []);

  const dismissWarnings = () => {
    setHealthWarnings([]);
  };

  return (
    <DownloadScreen
      modelStatus={modelStatus}
      onStatusChange={setModelStatus}
      healthWarnings={healthWarnings}
      onDismissWarnings={dismissWarnings}
    />
  );
}

export default App;
