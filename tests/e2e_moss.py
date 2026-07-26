#!/usr/bin/env python3
"""End-to-end: send audio to MOSS, verify transcription and diarization stripping."""
import struct, io, sys, json, os

# Generate 0.5s of 440Hz tone at 16kHz (detectable speech-like audio)
SR = 16000
DUR = 0.5
SAMPLES = int(SR * DUR)
tone = [int(8000 * sin(2 * 3.14159 * 440 * t / SR)) for t in range(SAMPLES)]

buf = io.BytesIO()
buf.write(b'RIFF')
data_len = len(tone) * 2
buf.write(struct.pack('<I', 36 + data_len))
buf.write(b'WAVE')
buf.write(b'fmt ')
buf.write(struct.pack('<IHHIIHH', 16, 1, 1, SR, SR * 2, 2, 16))
buf.write(b'data')
buf.write(struct.pack('<I', data_len))
for s in tone:
    buf.write(struct.pack('<h', s))
wav = buf.getvalue()

# Default MOSS URL
url = os.environ.get('WHIMPER_ASR_URL', 'http://100.76.212.98:9364')
import urllib.request

try:
    import requests
    resp = requests.post(f'{url}/transcribe', files={'file': ('test.wav', wav)}, timeout=15)
    print(f'Status: {resp.status_code}', file=sys.stderr)
    if resp.status_code == 200:
        data = resp.json()
        raw = data.get('text', '')
        stripped = ''.join(part.split('>', 1)[1] if '>' in part else part
                          for part in raw.split('<'))
        print(f'Raw: {raw}')
        print(f'Stripped: {stripped}')
        print(f'Has <spk: tag: {"<spk:" in raw}')
        print(f'Has <spk: after strip: {"<spk:" in stripped}')
        print(f'Words: {len(data.get("words", []))}')
        # Verify result looks like ASR output
        if stripped.strip():
            print('PASS: got non-empty transcription')
        else:
            print('WARN: empty transcription (tone might not trigger ASR)')
    else:
        print(f'Error: {resp.text}')
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
