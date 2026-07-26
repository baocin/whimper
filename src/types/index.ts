export type ModelStatus =
  | { kind: "ready" }
  | { kind: "error"; message: string };

export type RecordingState = "idle" | "listening";

export type HotkeyStatus =
  | "available"
  | "needs_relogin"
  | "needs_group_add"
  | "no_keyboard";

export type ContinuousState = "listening" | "stopped";

export interface TranscriptUpdate {
  text: string;
  is_final: boolean;
}

export interface ContinuousPastedEvent {
  chars: number;
  words: number;
}

export interface SpeakerStatus {
  has_embedding: boolean;
}
