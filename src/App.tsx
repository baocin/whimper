import { useCallback, useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import DownloadScreen from './components/DownloadScreen';
import HotkeyWarning from './components/HotkeyWarning';
import Overlay from './components/Overlay';
import type { HotkeyStatus, ModelStatus } from './types';

function App() {
  // Overlay window loads at /overlay — render the overlay component directly
  if (window.location.pathname === '/overlay') {
    return <Overlay />;
  }

  return <MainWindow />;
}

function MainWindow() {
  const [modelStatus, setModelStatus] = useState<ModelStatus>({ kind: 'not_downloaded' });
  // null = not yet known; until we know, don't hide (avoids hiding before we
  // learn the hotkey is unavailable).
  const [hotkeyStatus, setHotkeyStatus] = useState<HotkeyStatus | null>(null);

  const recheckHotkey = useCallback(() => {
    invoke<HotkeyStatus>('check_hotkey_status').then(setHotkeyStatus).catch(() => {});
  }, []);

  // Initial status fetches.
  useEffect(() => {
    invoke<ModelStatus>('check_model_status').then(setModelStatus).catch(() => {});
    recheckHotkey();
  }, [recheckHotkey]);

  // Backend events: auto-load completion + live hotkey status.
  useEffect(() => {
    const unModel = listen('model-ready', () => setModelStatus({ kind: 'ready' }));
    const unHotkey = listen<HotkeyStatus>('hotkey-status', (e) => setHotkeyStatus(e.payload));
    return () => {
      unModel.then((fn) => fn());
      unHotkey.then((fn) => fn());
    };
  }, []);

  // Only hide the window once the model is ready AND the hotkey actually works.
  // If the hotkey is unavailable we keep the window up so the warning is visible.
  useEffect(() => {
    if (modelStatus.kind === 'ready' && hotkeyStatus === 'available') {
      invoke('hide_main_window');
    }
  }, [modelStatus, hotkeyStatus]);

  const banner =
    hotkeyStatus && hotkeyStatus !== 'available' ? (
      <HotkeyWarning status={hotkeyStatus} onRecheck={recheckHotkey} />
    ) : null;

  return (
    <DownloadScreen modelStatus={modelStatus} onStatusChange={setModelStatus} banner={banner} />
  );
}

export default App;
