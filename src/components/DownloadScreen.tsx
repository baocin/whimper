import { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { useDownload } from '../hooks/useDownload';
import type { ModelStatus } from '../types';

interface Props {
  modelStatus: ModelStatus;
  onStatusChange: (status: ModelStatus) => void;
  healthWarnings: string[];
  onDismissWarnings: () => void;
}

export default function DownloadScreen({ modelStatus, onStatusChange, healthWarnings, onDismissWarnings }: Props) {
  const { startDownload, cancelDownload } = useDownload(onStatusChange);
  const [fileTranscript, setFileTranscript] = useState<string | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [isTranscribingFile, setIsTranscribingFile] = useState(false);
  const [fileProgress, setFileProgress] = useState<{ current: number; total: number } | null>(null);
  const [copied, setCopied] = useState(false);

  async function handleTranscribeFile() {
    setFileTranscript(null);
    setFileError(null);
    setCopied(false);
    setFileProgress(null);
    setIsTranscribingFile(true);

    const unlisten = await listen<{ current: number; total: number }>(
      'file-transcribe-progress',
      (event) => setFileProgress(event.payload),
    );

    try {
      const result = await invoke<string | null>('pick_and_transcribe_file');
      if (result !== null) {
        setFileTranscript(result);
      }
    } catch (e) {
      setFileError(String(e));
    } finally {
      unlisten();
      setIsTranscribingFile(false);
      setFileProgress(null);
    }
  }

  async function handleCancelFile() {
    await invoke('cancel_file_transcription');
  }

  async function handleCopy() {
    if (!fileTranscript) return;
    await navigator.clipboard.writeText(fileTranscript);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  const progressPercent =
    modelStatus.kind === 'downloading' ? Math.round(modelStatus.progress * 100) : 0;

  return (
    <div className="flex flex-col items-center justify-center h-screen bg-gray-900 text-white p-8">
      <h1 className="text-2xl font-bold mb-2">Whimper</h1>
      <p className="text-gray-400 mb-8 text-sm">Local voice-to-text transcription</p>

      {healthWarnings.length > 0 && (
        <div className="w-full max-w-md mb-6 space-y-2">
          {healthWarnings.map((warning, i) => (
            <div key={i} className="bg-yellow-900/50 border border-yellow-600 rounded-lg px-4 py-3 text-yellow-200 text-sm">
              {warning}
            </div>
          ))}
          <button
            onClick={onDismissWarnings}
            className="text-xs text-gray-500 hover:text-gray-400 transition-colors"
          >
            Dismiss
          </button>
        </div>
      )}

      {modelStatus.kind === 'not_downloaded' && (
        <div className="text-center">
          <p className="mb-4 text-gray-300">
            Nemotron streaming speech model required (~662 MB download)
          </p>
          <button
            onClick={startDownload}
            className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-medium transition-colors"
          >
            Download Model
          </button>
        </div>
      )}

      {modelStatus.kind === 'downloading' && (
        <div className="w-full max-w-md">
          <div className="flex justify-between text-sm text-gray-400 mb-1">
            <span>Downloading...</span>
            <span>{progressPercent}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div
              className="bg-blue-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>{modelStatus.speed_mbps.toFixed(1)} MB/s</span>
            <button onClick={cancelDownload} className="text-red-400 hover:text-red-300">
              Cancel
            </button>
          </div>
        </div>
      )}

      {modelStatus.kind === 'downloaded' && (
        <div className="text-center">
          <p className="text-gray-300 mb-4">Model downloaded. Loading...</p>
          <div className="animate-spin w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full mx-auto" />
        </div>
      )}

      {modelStatus.kind === 'loading' && (
        <div className="text-center">
          <p className="text-gray-300 mb-4">Loading model...</p>
          <div className="animate-spin w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full mx-auto" />
        </div>
      )}

      {modelStatus.kind === 'error' && (
        <div className="text-center">
          <p className="text-red-400 mb-4">{modelStatus.message}</p>
          <button
            onClick={startDownload}
            className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-medium transition-colors"
          >
            Retry Download
          </button>
        </div>
      )}

      {modelStatus.kind === 'ready' && (
        <div className="w-full max-w-md space-y-4">
          <div className="text-center space-y-1">
            <p className="text-green-400 text-sm font-medium">Model ready</p>
            <p className="text-gray-400 text-sm">
              Press <kbd className="px-1.5 py-0.5 bg-gray-700 rounded text-xs font-mono">Alt+Space</kbd> to record
            </p>
          </div>

          <div className="border-t border-gray-700 pt-4 space-y-3">
            <button
              onClick={handleTranscribeFile}
              disabled={isTranscribingFile}
              className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-sm font-medium transition-colors"
            >
              {isTranscribingFile ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="animate-spin w-3 h-3 border border-gray-300 border-t-transparent rounded-full inline-block" />
                  {fileProgress && fileProgress.total > 1
                    ? `Transcribing chunk ${fileProgress.current + 1} of ${fileProgress.total}...`
                    : 'Transcribing...'}
                </span>
              ) : (
                'Transcribe File...'
              )}
            </button>

            {isTranscribingFile && fileProgress && fileProgress.total > 1 && (
              <div className="space-y-1">
                <div className="w-full bg-gray-700 rounded-full h-1.5">
                  <div
                    className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                    style={{ width: `${Math.round((fileProgress.current / fileProgress.total) * 100)}%` }}
                  />
                </div>
                <button
                  onClick={handleCancelFile}
                  className="w-full text-xs text-red-400 hover:text-red-300 transition-colors"
                >
                  Cancel
                </button>
              </div>
            )}

            {fileError && (
              <p className="text-red-400 text-xs text-center">{fileError}</p>
            )}

            {fileTranscript !== null && (
              <div className="space-y-2">
                <div className="bg-gray-800 rounded-lg p-3 text-sm text-gray-200 max-h-48 overflow-y-auto whitespace-pre-wrap break-words">
                  {fileTranscript || <span className="text-gray-500 italic">No speech detected</span>}
                </div>
                {fileTranscript && (
                  <button
                    onClick={handleCopy}
                    className="w-full px-3 py-1.5 bg-blue-600 hover:bg-blue-700 rounded text-xs font-medium transition-colors"
                  >
                    {copied ? 'Copied!' : 'Copy to Clipboard'}
                  </button>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
