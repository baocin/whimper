import { useState } from 'react';
import type { HotkeyStatus } from '../types';

interface Props {
  status: HotkeyStatus;
  onRecheck?: () => void;
}

interface Content {
  title: string;
  body: string;
  command?: string;
  hint?: string;
}

function contentFor(status: HotkeyStatus): Content | null {
  switch (status) {
    case 'needs_group_add':
      return {
        title: 'Global hotkey unavailable',
        body: "whimper can't read your keyboard. Add yourself to the input group, then log out and back in:",
        command: 'sudo usermod -aG input $USER',
        hint: 'A logout/login (or reboot) is required for the change to take effect.',
      };
    case 'needs_relogin':
      return {
        title: 'Global hotkey unavailable',
        body: "Keyboard access isn't active in this session. Log out and back in (or reboot) to enable the Alt+Space hotkey.",
        hint: "You're already in the input group — the session just needs to pick it up.",
      };
    case 'no_keyboard':
      return {
        title: 'Global hotkey unavailable',
        body: 'No keyboard device was found in /dev/input. The Alt+Space hotkey will not work.',
      };
    case 'available':
      return null;
  }
}

export default function HotkeyWarning({ status, onRecheck }: Props) {
  const [copied, setCopied] = useState(false);
  const content = contentFor(status);
  if (!content) return null;

  const copy = () => {
    if (!content.command) return;
    navigator.clipboard?.writeText(content.command).then(
      () => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      },
      () => {},
    );
  };

  return (
    <div
      role="alert"
      className="w-full max-w-md mb-6 rounded-lg border border-amber-500/40 bg-amber-500/10 p-4 text-left"
    >
      <p className="font-semibold text-amber-300 mb-1">⚠ {content.title}</p>
      <p className="text-sm text-amber-100/90">{content.body}</p>

      {content.command && (
        <div className="mt-3 flex items-center gap-2">
          <code className="flex-1 rounded bg-black/40 px-2 py-1 text-xs text-amber-100 font-mono break-all">
            {content.command}
          </code>
          <button
            onClick={copy}
            className="shrink-0 rounded bg-amber-600/80 hover:bg-amber-600 px-2 py-1 text-xs font-medium text-white transition-colors"
          >
            {copied ? 'Copied' : 'Copy'}
          </button>
        </div>
      )}

      {content.hint && <p className="mt-2 text-xs text-amber-200/70">{content.hint}</p>}

      {onRecheck && (
        <button
          onClick={onRecheck}
          className="mt-3 text-xs text-amber-300 hover:text-amber-200 underline"
        >
          Re-check
        </button>
      )}
    </div>
  );
}
