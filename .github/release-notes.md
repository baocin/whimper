Local Nemotron Speech Streaming EN 0.6B (INT4) dictation for macOS.

| Mac | Download |
|---|---|
| Apple Silicon (M1 and newer) | `aarch64.dmg` or `aarch64.app.tar.gz` |
| Intel | `x64.dmg` or `x86_64.app.tar.gz` |

Open the disk image and drag Whimper into Applications, or extract the app archive.
The first launch downloads the model (~662 MB). Grant Microphone and Accessibility
access in System Settings → Privacy & Security, then use Option+Space to start or
stop dictation. ESC cancels and erases the current utterance.

These builds are not Developer ID signed or notarized. macOS may require approval
in System Settings → Privacy & Security → Open Anyway after the first launch attempt.
Replacing an older app may require granting Accessibility access again.

Transcripts are stored locally in
`~/Library/Application Support/com.whimper.app/log.txt`. No models, recordings,
transcripts, or user configuration are included in these downloads.

`SHA256SUMS` contains checksums for the release downloads.
