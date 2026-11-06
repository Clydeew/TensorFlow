#!/usr/bin/env python3
"""Run the project environment from any working directory."""

import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
if not python.is_file():
    raise SystemExit("Run python scripts/setup.py first")
raise SystemExit(subprocess.call([str(python), "-m", "tensorforge.cli", *sys.argv[1:]], cwd=root))
