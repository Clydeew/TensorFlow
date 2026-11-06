#!/usr/bin/env bash
# macOS Intel/Apple Silicon bootstrap using the same Python 3.12 lock.
set -euo pipefail
cd -- "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [[ "$(uname -s)" != "Darwin" ]]; then
  printf '%s\n' 'Use setup-linux.sh on Linux or setup-windows.cmd on Windows.' >&2
  exit 1
fi
if command -v python3 >/dev/null 2>&1; then
  exec python3 scripts/setup.py "$@"
elif command -v uv >/dev/null 2>&1; then
  exec uv run --no-project --python 3.12 python scripts/setup.py "$@"
else
  printf '%s\n' 'Install Python 3 from python.org or uv first. See docs/SETUP.md.' >&2
  exit 1
fi
