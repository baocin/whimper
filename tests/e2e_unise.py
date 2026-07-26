#!/usr/bin/env python3
"""E2E: UniSE return_wav works — send tone, verify enhanced WAV back, measure RMS."""

import struct
import io
import math
import json
import sys
import os
import requests

SR = 16000
S = int(SR * 0.1)
tone = [int(8000 * math.sin(2 * 3.14159 * 440 * t / SR)) for t in range(S)]

buf = io.BytesIO()
buf.write(b"RIFF")
buf.write(struct.pack("<I", 36 + S * 2))
buf.write(b"WAVE")
buf.write(b"fmt ")
buf.write(struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16))
buf.write(b"data")
buf.write(struct.pack("<I", S * 2))
for s in tone:
    buf.write(struct.pack("<h", s))
wav = buf.getvalue()

url = os.environ.get("UNISE_URL", "http://localhost:9363")
resp = requests.post(
    f"{url}/enhance",
    files={"file": ("test.wav", wav)},
    data={"return_wav": "true"},
    timeout=30,
)

out = {"pass": False}

if resp.status_code != 200:
    out["error"] = f"status {resp.status_code}: {resp.text[:200]}"
    print(json.dumps(out))
    sys.exit(1)

if resp.content[:4] != b"RIFF":
    out["error"] = f"not a WAV: {resp.content[:20].hex()}"
    print(json.dumps(out))
    sys.exit(1)

# Parse WAV data chunk
pos = 12
while pos < len(resp.content) - 8:
    ck = resp.content[pos : pos + 4]
    sz = struct.unpack("<I", resp.content[pos + 4 : pos + 8])[0]
    if ck == b"data":
        raw = struct.unpack(f"<{sz // 2}h", resp.content[pos + 8 : pos + 8 + sz])
        break
    pos += 8 + sz

raw_f = [x / 32768.0 for x in raw]
bef_f = [x / 32768.0 for x in tone]
bef_rms = (sum(x * x for x in bef_f) / len(bef_f)) ** 0.5
aft_rms = (sum(x * x for x in raw_f) / len(raw_f)) ** 0.5
db = 10 * math.log10(max(aft_rms, 1e-10) / max(bef_rms, 1e-10))

out.update(
    {
        "pass": True,
        "in_samples": S,
        "out_samples": len(raw),
        "before_rms": round(bef_rms, 5),
        "after_rms": round(aft_rms, 5),
        "gain_db": round(db, 2),
    }
)
print(json.dumps(out))
