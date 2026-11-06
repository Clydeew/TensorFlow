@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup-windows.cmd first. 1>&2
    exit /b 1
)
".venv\Scripts\python.exe" -m tensorforge.cli %*
