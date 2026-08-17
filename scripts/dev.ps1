# Windows launcher for scripts/dev.sh.
# Loads the VS Developer environment, then runs Git Bash (not WSL `bash.exe`).
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$vcvars = $null
if (Test-Path $vswhere)
{
    $vsInstall = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if ($vsInstall)
    {
        $candidate = Join-Path $vsInstall "VC\Auxiliary\Build\vcvars64.bat"
        if (Test-Path $candidate)
        {
            $vcvars = $candidate
        }
    }
}
if (-not $vcvars)
{
    $fallback = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
    if (Test-Path $fallback)
    {
        $vcvars = $fallback
    }
}
if (-not $vcvars)
{
    throw "vcvars64.bat not found. Install Visual Studio C++ tools or open a Developer Command Prompt."
}

$gitBash = "C:\Program Files\Git\bin\bash.exe"
if (-not (Test-Path $gitBash))
{
    throw "Git Bash not found at $gitBash. Do not use WSL bash.exe from System32."
}

$devSh = (Join-Path $PSScriptRoot "dev.sh") -replace "\\", "/"
cmd.exe /c "call `"$vcvars`" && `"$gitBash`" `"$devSh`""
if ($LASTEXITCODE -ne 0)
{
    exit $LASTEXITCODE
}
