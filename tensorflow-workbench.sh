#!/usr/bin/env bash
# CLI launcher for Linux and macOS, usable from any working directory.
set -euo pipefail
cd -- "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [[ ! -x .venv/bin/python ]]; then
  printf '%s\n' 'Run the setup script for your operating system first.' >&2
  exit 1
fi
exec .venv/bin/python -m tensorforge.cli "$@"
