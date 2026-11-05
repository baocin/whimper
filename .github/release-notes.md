Local Nemotron Speech Streaming EN 0.6B (INT4) dictation for macOS.

| Mac | Download |
|---|---|
| Apple Silicon (M1 and newer) | `aarch64.dmg` or `aarch64.app.tar.gz` |

Intel Macs are not supported by this release: the current ONNX Runtime dependency
has no prebuilt Intel macOS binaries.

Open the disk image and drag Whimper into Applications, or extract the app archive.
The first launch downloads the model (~662 MB). Grant Microphone and Accessibility
access in System Settings → Privacy & Security, then use Option+Space to start or
stop dictation. ESC cancels and erases the current utterance.

These builds are not Developer ID signed or notarized. macOS may require approval
in System Settings → Privacy & Security → Open Anyway after the first launch attempt.
Replacing an older app may require granting Accessibility access again.

Transcripts are stored locally in
`~/Library/Application Support/com.whimper.app/log.txt`. No downloaded ASR weights,
recordings, transcripts, or user configuration are included in these downloads.
Small VAD and noise-reduction models are embedded in the executable.

`SHA256SUMS` contains checksums for the release downloads.
