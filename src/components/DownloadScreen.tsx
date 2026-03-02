import { invoke } from '@tauri-apps/api/core';
import { useDownload } from '../hooks/useDownload';
import type { ModelStatus } from '../types';

interface Props {
  modelStatus: ModelStatus;
  onStatusChange: (status: ModelStatus) => void;
}

export default function DownloadScreen({ modelStatus, onStatusChange }: Props) {
  const { startDownload, cancelDownload } = useDownload(onStatusChange);

  const progressPercent =
    modelStatus.kind === 'downloading' ? Math.round(modelStatus.progress * 100) : 0;

  return (
    <div className="flex flex-col items-center justify-center h-screen bg-gray-900 text-white p-8">
      <h1 className="text-2xl font-bold mb-2">Whimper</h1>
      <p className="text-gray-400 mb-8 text-sm">Local voice-to-text transcription</p>

      {modelStatus.kind === 'not_downloaded' && (
        <div className="text-center">
          <p className="mb-4 text-gray-300">
            Voxtral-Mini-4B model required (~2.5 GB download)
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
          <p className="text-gray-300 mb-4">Loading model on GPU...</p>
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
    </div>
  );
}
