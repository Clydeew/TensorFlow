#!/usr/bin/env bash
# Linux bootstrap. No sudo and no changes to system Python.
set -euo pipefail
cd -- "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [[ "$(uname -s)" != "Linux" ]]; then
  printf '%s\n' 'Use setup-macos.sh on macOS or setup-windows.cmd on Windows.' >&2
  exit 1
fi
if command -v python3 >/dev/null 2>&1; then
  exec python3 scripts/setup.py "$@"
elif command -v python >/dev/null 2>&1; then
  exec python scripts/setup.py "$@"
elif command -v uv >/dev/null 2>&1; then
  exec uv run --no-project --python 3.12 python scripts/setup.py "$@"
else
  printf '%s\n' 'Install Python 3 (with venv support) or uv first. See docs/SETUP.md.' >&2
  exit 1
fi
