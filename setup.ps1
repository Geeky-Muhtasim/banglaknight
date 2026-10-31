# One-time setup for BanglaKnight on Windows + NVIDIA GPU.
# Run from PowerShell:   powershell -ExecutionPolicy Bypass -File .\setup.ps1
#
# - Creates .venv with Python 3.11 (or 3.12)
# - Installs the CUDA build of PyTorch + pinned libraries
# - Fails loudly if PyTorch cannot see the GPU
# - First run writes requirements.lock.txt (exact versions of everything);
#   later runs / other machines install from that lock for a reproducible env.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
function Fail($msg) { Write-Host "FAILED: $msg" -ForegroundColor Red; exit 1 }
function Check($what) { if ($LASTEXITCODE -ne 0) { Fail $what } }

# 1. GPU driver
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
} else { Fail "nvidia-smi not found. Install the NVIDIA driver first." }

# 2. venv with a supported Python
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    $made = $false
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in @("3.11", "3.12")) {
            py -$v --version *> $null
            if ($LASTEXITCODE -eq 0) { Write-Host "Creating venv with Python $v"; py -$v -m venv .venv; $made = ($LASTEXITCODE -eq 0); break }
        }
    }
    if (-not $made) { Fail "Python 3.11 or 3.12 not found via the 'py' launcher. Install Python 3.11 from python.org (tick 'Add to PATH' and 'py launcher')." }
}
$py = ".\.venv\Scripts\python.exe"
& $py --version
& $py -m pip install --upgrade "pip==25.2"; Check "pip upgrade"

# 3. Remove a CPU-only torch if one slipped in earlier
$cpuTorch = & $py -c "import importlib.util,sys; import torch; print('cpu' if '+cu' not in torch.__version__ else 'gpu')" 2>$null
if ($cpuTorch -eq "cpu") { Write-Host "Removing CPU-only torch"; & $py -m pip uninstall -y torch }

# 4. Install (lock file if present, otherwise pinned direct deps then freeze)
if (Test-Path "requirements.lock.txt") {
    Write-Host "Installing from requirements.lock.txt (exact environment)"
    & $py -m pip install -r requirements.lock.txt; Check "install from lock"
} else {
    Write-Host "Installing from requirements.txt"
    & $py -m pip install -r requirements.txt; Check "install from requirements.txt"
    "--extra-index-url https://download.pytorch.org/whl/cu126" | Out-File -Encoding ascii requirements.lock.txt
    & $py -m pip freeze | Out-File -Encoding ascii -Append requirements.lock.txt
    Write-Host "Wrote requirements.lock.txt"
}

# 5. Verify the GPU is actually used
& $py -c @"
import torch, transformers
print('torch', torch.__version__, '| CUDA build', torch.version.cuda, '| transformers', transformers.__version__)
assert torch.cuda.is_available(), 'PyTorch cannot see the GPU'
print('GPU:', torch.cuda.get_device_name(0), '| bf16:', torch.cuda.is_bf16_supported())
x = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16); y = x @ x; torch.cuda.synchronize()
print('GPU matmul OK')
"@
Check "GPU check (torch.cuda.is_available() is False). Check the NVIDIA driver is >= 560."
Write-Host "`nSetup done. Next: powershell -ExecutionPolicy Bypass -File .\smoke_test.ps1" -ForegroundColor Green
