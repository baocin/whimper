import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import HotkeyWarning from "./components/HotkeyWarning";
import Overlay from "./components/Overlay";
import type { ContinuousPastedEvent, HotkeyStatus } from "./types";

function App() {
  if (window.location.pathname === "/overlay") {
    return <Overlay />;
  }

  return <MainWindow />;
}

function MainWindow() {
  const [hotkeyStatus, setHotkeyStatus] = useState<HotkeyStatus | null>(null);
  const [continuousActive, setContinuousActive] = useState(false);
  const [lastPaste, setLastPaste] = useState<ContinuousPastedEvent | null>(
    null,
  );

  const recheckHotkey = useCallback(() => {
    invoke<HotkeyStatus>("check_hotkey_status")
      .then(setHotkeyStatus)
      .catch(() => {});
  }, []);

  useEffect(() => {
    invoke<boolean>("is_continuous_active")
      .then(setContinuousActive)
      .catch(() => {});
    recheckHotkey();
  }, [recheckHotkey]);

  useEffect(() => {
    const unHotkey = listen<HotkeyStatus>("hotkey-status", (e) =>
      setHotkeyStatus(e.payload),
    );
    const unContState = listen<string>("continuous-state", (e) => {
      setContinuousActive(e.payload === "listening");
    });
    const unContPaste = listen<ContinuousPastedEvent>(
      "continuous-pasted",
      (e) => {
        setLastPaste(e.payload);
        setTimeout(() => setLastPaste(null), 3000);
      },
    );
    return () => {
      unHotkey.then((fn) => fn());
      unContState.then((fn) => fn());
      unContPaste.then((fn) => fn());
    };
  }, []);

  useEffect(() => {
    if (hotkeyStatus === "available" && !continuousActive) {
      invoke("hide_main_window");
    }
  }, [hotkeyStatus, continuousActive]);

  const toggleContinuous = () => {
    if (continuousActive) {
      invoke("stop_continuous").catch(console.error);
    } else {
      invoke("start_continuous").catch(console.error);
    }
  };

  const banner =
    hotkeyStatus && hotkeyStatus !== "available" ? (
      <HotkeyWarning status={hotkeyStatus} onRecheck={recheckHotkey} />
    ) : null;

  return (
    <div className="min-h-screen bg-zinc-900 flex flex-col items-center justify-center p-6">
      <h1 className="text-2xl font-bold text-white mb-1">Whimper</h1>
      <p className="text-gray-400 mb-6 text-sm">
        Local voice-to-text transcription
      </p>
      {banner}

      <div className="flex flex-col items-center gap-3">
        <button
          onClick={toggleContinuous}
          className={`rounded-lg px-6 py-2 text-sm font-medium transition-colors ${
            continuousActive
              ? "bg-green-600 hover:bg-green-700 text-white"
              : "bg-zinc-700 hover:bg-zinc-600 text-zinc-200"
          }`}
        >
          {continuousActive ? "● Continuous (active)" : "○ Continuous (off)"}
        </button>
        <p className="text-xs text-zinc-500 max-w-xs text-center">
          {continuousActive
            ? 'Always-on mic. Say "paste" when done speaking to insert text.'
            : "Press Alt+Space to record, press again to stop and paste."}
        </p>
        {lastPaste && (
          <p className="text-xs text-green-400 animate-pulse">
            Pasted {lastPaste.words} words ({lastPaste.chars} chars)
          </p>
        )}
      </div>
    </div>
  );
}

export default App;
