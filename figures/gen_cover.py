#!/usr/bin/env python3
"""Generate textbook cover image using Gemini 3.1 Flash."""

import os
import json
import base64
import urllib.request

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-image-preview:generateContent?key={API_KEY}"

prompt = (
    "Professional academic textbook cover image. "
    "Theme: intersection of artificial intelligence, power engineering, "
    "battery technology, and data science. "
    "Include abstract geometric elements suggesting neural networks, "
    "power transmission towers, battery cells, and flowing data streams. "
    "Color palette: deep navy (#2C3E50), teal (#1B7A8A), amber accents (#D4984A). "
    "Style: clean, minimalist, sophisticated engineering aesthetic. "
    "Vertical orientation (portrait, taller than wide). "
    "NO text or letters in the image at all."
)

payload = {
    "contents": [{"parts": [{"text": prompt}]}],
    "generationConfig": {
        "responseModalities": ["TEXT", "IMAGE"],
    },
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(
    URL,
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST",
)

print("Calling Gemini API...")
with urllib.request.urlopen(req, timeout=120) as resp:
    result = json.loads(resp.read().decode("utf-8"))

# Extract image from response
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
for candidate in result.get("candidates", []):
    for part in candidate.get("content", {}).get("parts", []):
        if "inlineData" in part:
            img_data = base64.b64decode(part["inlineData"]["data"])
            out_path = os.path.join(OUT_DIR, "cover.png")
            with open(out_path, "wb") as f:
                f.write(img_data)
            print(f"Saved cover image: {out_path} ({len(img_data)} bytes)")
            break
    else:
        continue
    break
else:
    print("No image found in response!")
    print(json.dumps(result, indent=2, ensure_ascii=False)[:2000])
