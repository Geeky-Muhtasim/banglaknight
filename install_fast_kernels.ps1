# Optional: fast Gated DeltaNet kernels for Qwen3.5 on Windows (flash-linear-attention on Triton for Windows).
# 24 of Qwen3.5's 32 layers are Gated DeltaNet; without these kernels transformers uses a slow PyTorch fallback.
# Run from PowerShell (after setup_llm.ps1):   powershell -ExecutionPolicy Bypass -File .\install_fast_kernels.ps1
#
# Installs into .venv-llm, then check_fast_kernels.py compares the fast kernel with the PyTorch reference on the GPU.
# If it fails to run or the numbers differ, the kernel package is removed again and training keeps the fallback.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$py = ".\.venv-llm\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "Run setup_llm.ps1 first." -ForegroundColor Red; exit 1 }
$env:PYTHONUTF8 = "1"

# triton-windows 3.3.x matches torch 2.7; fla-core provides the `fla` package that transformers picks up.
& $py -m pip install "triton-windows==3.3.1.post21" "fla-core==0.5.2" "einops==0.8.2"
if ($LASTEXITCODE -ne 0) { Write-Host "Install failed; nothing changed for training (fallback stays)." -ForegroundColor Red; exit 1 }

Write-Host "`nChecking the fast kernel against the PyTorch reference (first run compiles kernels, can take a few minutes)..." -ForegroundColor Cyan
& $py check_fast_kernels.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Fast kernel not usable here. Removing it so training uses the (slower) PyTorch fallback." -ForegroundColor Yellow
    & $py -m pip uninstall -y fla-core
    exit 1
}
"--extra-index-url https://download.pytorch.org/whl/cu126" | Out-File -Encoding ascii requirements-llm.lock.txt
& $py -m pip freeze | Out-File -Encoding ascii -Append requirements-llm.lock.txt
Write-Host "`nFast kernels active; requirements-llm.lock.txt updated. Re-measure speed with:" -ForegroundColor Green
Write-Host "  powershell -ExecutionPolicy Bypass -File .\smoke_llm.ps1 -Only rows"
