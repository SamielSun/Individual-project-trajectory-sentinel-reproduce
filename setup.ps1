# Environment bootstrap for this fork.
#
#   pwsh -File setup.ps1
#
# The upstream sources are already in this repository (this is a fork of
# agent-trajectory-sentinel at 1b3e07f), so there is nothing to clone: this
# script only builds `.venv` and installs the pinned study stack.
#
# Why an explicit transitive list instead of `pip install -r
# requirements-core.lock.txt`: on a Windows box behind a proxy, pip's resolver
# stalled for 15+ minutes on the pinned set (no bytes fetched) while the wheels
# themselves download at ~0.5 MB/s. Installing the top-level pins with
# --no-deps and naming the closure explicitly is the workaround; the pinned
# versions are unchanged and the resolved transitive versions are recorded in
# out/environment.json. Details: report section 6, entries 1-3.

param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$venv = Join-Path $root ".venv"
$py = Join-Path $venv "Scripts\python.exe"

if ($Force -and (Test-Path $venv)) {
    Remove-Item -Recurse -Force $venv
}

if (-not (Test-Path $py)) {
    Write-Host "[setup] creating .venv ..."
    $base = (Get-Command python).Source
    & $base -m venv $venv
}

Write-Host "[setup] installing the pinned study stack ..."
& $py -m pip install --no-cache-dir --progress-bar off -q --upgrade pip
& $py -m pip install --no-cache-dir --progress-bar off -q --no-deps `
    "numpy==2.4.6" "scipy==1.17.1" "scikit-learn==1.9.0" "pandas==3.0.3" `
    "matplotlib==3.11.0" "pillow==12.2.0" "httpx==0.28.1" "pytest==9.1.0"

Write-Host "[setup] installing the transitive closure (unpinned, recorded) ..."
& $py -m pip install --no-cache-dir --progress-bar off -q --no-deps `
    joblib threadpoolctl cloudpickle narwhals python-dateutil tzdata six `
    contourpy cycler fonttools kiwisolver packaging pyparsing pygments `
    anyio certifi httpcore idna sniffio iniconfig pluggy colorama tqdm

Write-Host "[setup] verifying imports ..."
& $py -c "import numpy, scipy, sklearn, pandas, matplotlib, httpx, pytest; print('ok', numpy.__version__, scipy.__version__, pandas.__version__)"

$env:PYTHONPATH = $root
& $py -m repro.env
Write-Host ""
Write-Host "[setup] done. Next:"
Write-Host "  `$env:PYTHONPATH = (Get-Location)"
Write-Host "  .\.venv\Scripts\python.exe -m repro.run_all      # 13 arms, ~2.5 min each"
Write-Host "  .\.venv\Scripts\python.exe -m repro.collect      # tables + report/numbers.tex"
Write-Host "  .\.venv\Scripts\python.exe -m repro.figures      # report figures"
