# Optional native PowerShell launcher; the .cmd alternative needs no script policy changes.
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (-not (Test-Path '.venv\Scripts\python.exe')) {
        throw 'Run setup-windows.ps1 or setup-windows.cmd first.'
    }
    & '.venv\Scripts\python.exe' -m tensorforge.cli @args
    $Code = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $Code
