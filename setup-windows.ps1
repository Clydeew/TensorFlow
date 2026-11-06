# Windows PowerShell setup; does not change execution policy or system Python.
[CmdletBinding()]
param(
    [ValidateSet('core', 'ocr', 'all')][string]$Extras = 'core',
    [switch]$Offline
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $SetupArgs = @('scripts/setup.py', '--extras', $Extras)
    if ($Offline) { $SetupArgs += '--offline' }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 @SetupArgs
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python @SetupArgs
    } elseif (Get-Command uv -ErrorAction SilentlyContinue) {
        & uv run --no-project --python 3.12 python @SetupArgs
    } else {
        throw 'Install Python 3 from python.org (enable its launcher) or uv first. See docs/SETUP.md.'
    }
    if ($LASTEXITCODE -ne 0) { throw "Setup failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
