#!/usr/bin/env python3
"""Portable standard-library bootstrap; installs only in this project."""

import argparse
import os
import shutil
import subprocess
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Set up TensorFlow without modifying system Python")
    parser.add_argument("--extras", choices=["core", "ocr", "all"], default="core")
    parser.add_argument("--offline", action="store_true", help="Use only pre-populated uv caches")
    args = parser.parse_args()
    binary = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" else ""
    uv = shutil.which("uv")
    if not uv:
        existing = ROOT / ".venv" / binary / ("uv" + suffix)
        if existing.is_file():
            uv = str(existing)
    if not uv:
        bootstrap = ROOT / ".bootstrap-uv"
        executable = bootstrap / binary / ("uv" + suffix)
        if not executable.exists():
            if args.offline:
                parser.error("uv is not installed; install it while online or transfer the matching wheel")
            venv.EnvBuilder(with_pip=True).create(bootstrap)
            python = bootstrap / binary / ("python" + suffix)
            subprocess.run([str(python), "-m", "pip", "install", "uv==0.13.7"], check=True)
        uv = str(executable)
    command = [uv, "sync", "--locked", "--python", "3.12"]
    if args.offline:
        command.append("--offline")
    if args.extras == "ocr":
        command += ["--extra", "ocr"]
    elif args.extras == "all":
        command.append("--all-extras")
    subprocess.run(command, cwd=ROOT, check=True)
    python = ROOT / ".venv" / binary / ("python" + suffix)
    subprocess.run([str(python), "-m", "tensorforge.cli", "doctor"], cwd=ROOT, check=True)
    print("Ready. Start with: python scripts/run.py ui")


if __name__ == "__main__":
    main()
