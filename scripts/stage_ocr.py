#!/usr/bin/env python3
"""Explicit online provisioning of pinned OCR language data. Never called by generate."""

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "87416418657359cb625c412a48b6e1d6d41c29bd"
FILES = {
    "eng": "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2",
    "ind": "69786901da87ab8766c1ea7fbb10b28f2110c14da3f6c8f2735df131fba95d88",
    "jav": "46107b6bed3bf2d9617e5e192e52524e368242543fead1a7175c363dddd400f2",
    "sun": "fa939213b005b3477a1032b7222f63cd0b0ac486f83fd9554caf653d6e6c6ffe",
}


def main():
    target = ROOT / "models/tesseract"
    target.mkdir(parents=True, exist_ok=True)
    for language, expected in FILES.items():
        path = target / f"{language}.traineddata"
        source = f"https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/{REVISION}/{path.name}"
        if not path.exists():
            with urllib.request.urlopen(source, timeout=60) as response:
                data = response.read(15_000_001)
            if len(data) > 15_000_000 or hashlib.sha256(data).hexdigest() != expected:
                raise SystemExit(f"Downloaded data failed size/hash validation: {language}")
            path.write_bytes(data)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"Existing language data checksum mismatch: {path}")
        print(f"Verified {language}: {path}")
    (target / "provenance.json").write_text(
        json.dumps(
            {
                "source": "https://github.com/tesseract-ocr/tessdata_fast",
                "revision": REVISION,
                "license": "Apache-2.0",
                "sha256": FILES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Keep models/ outside Git; copy it separately for offline OCR.")


if __name__ == "__main__":
    main()
