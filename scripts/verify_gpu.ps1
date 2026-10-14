# Local GPU check. GitHub-hosted runners have no NVIDIA GPU, and this script
# is not a CI job. CI only compiles dllib_tests and the custom binaries.
param(
    [switch]$Short
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ($env:GITHUB_ACTIONS) {
    Write-Error "Refusing to run under GitHub Actions. Hosted runners have no NVIDIA GPU."
    exit 1
}

$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) {
    Write-Error "nvidia-smi was not found. This script is for a local machine with a GPU, not CI."
    exit 1
}
& nvidia-smi | Out-Host
if ($LASTEXITCODE -ne 0) {
    Write-Error "nvidia-smi failed. This script is for a local machine with a GPU, not CI."
    exit 1
}

$candidates = @(
    "build\tests\dllib_tests.exe",
    "build\tests\dllib_tests",
    "build_release\tests\dllib_tests.exe",
    "build_release\tests\dllib_tests",
    "build_debug\tests\dllib_tests.exe",
    "build_debug\tests\dllib_tests"
)
$testBin = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $testBin) {
    Write-Error "dllib_tests was not found. Build first with scripts/dev.ps1."
    exit 1
}

Write-Host "Running $testBin"
& $testBin
exit $LASTEXITCODE
