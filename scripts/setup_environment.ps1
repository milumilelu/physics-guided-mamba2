param(
    [Parameter(Mandatory = $true)]
    [string]$Python,
    [string]$IndexUrl = "https://pypi.org/simple"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}

$version = & $Python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
if ($LASTEXITCODE -ne 0 -or $version -ne "3.12.13") {
    throw "Python 3.12.13 is required; got '$version' from $Python"
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    & $Python -m venv (Join-Path $repoRoot ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv" }
}

& $venvPython -m pip install --upgrade "pip==26.2.1" --index-url $IndexUrl
if ($LASTEXITCODE -ne 0) { throw "Failed to install the pinned pip version" }

& $venvPython -m pip install -r (Join-Path $repoRoot "requirements.txt") --index-url $IndexUrl
if ($LASTEXITCODE -ne 0) { throw "Failed to install the core environment" }

& $venvPython (Join-Path $repoRoot "scripts\verify_environment.py")
if ($LASTEXITCODE -ne 0) { throw "Environment verification failed" }
