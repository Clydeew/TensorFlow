@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 scripts\setup.py %*
    exit /b
)
where python >nul 2>nul
if %errorlevel%==0 (
    python scripts\setup.py %*
    exit /b
)
where uv >nul 2>nul
if %errorlevel%==0 (
    uv run --no-project --python 3.12 python scripts\setup.py %*
    exit /b
)
echo Install Python 3 or uv first. See docs\SETUP.md. 1>&2
exit /b 1
