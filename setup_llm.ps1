# One-time setup of the LLM environment (.venv-llm) for the Qwen3.5-9B teacher.
# Run from PowerShell:   powershell -ExecutionPolicy Bypass -File .\setup_llm.ps1
# Optional:              -ModelPath "H:\Abliterate\Qwen3.5-9B"
#
# - Separate from .venv (encoders stay on transformers 4.57.6)
# - CUDA build of PyTorch, transformers 5.20.2 (Qwen3.5 support), peft, bitsandbytes
# - First run writes requirements-llm.lock.txt; later runs install from it
# - Ends with a GPU check and a tokenizer + architecture check on the local model
param([string]$ModelPath = "H:\Abliterate\Qwen3.5-9B")
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
function Fail($msg) { Write-Host "FAILED: $msg" -ForegroundColor Red; exit 1 }
function Check($what) { if ($LASTEXITCODE -ne 0) { Fail $what } }

if (-not (Test-Path "$ModelPath\config.json")) { Fail "model not found at $ModelPath (pass -ModelPath)" }

if (-not (Test-Path ".venv-llm\Scripts\python.exe")) {
    $made = $false
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in @("3.12", "3.11")) {
            py -$v --version *> $null
            if ($LASTEXITCODE -eq 0) { Write-Host "Creating .venv-llm with Python $v"; py -$v -m venv .venv-llm; $made = ($LASTEXITCODE -eq 0); break }
        }
    }
    if (-not $made) { Fail "Python 3.12 or 3.11 not found via the 'py' launcher (transformers 5.x needs Python >= 3.11)." }
}
$py = ".\.venv-llm\Scripts\python.exe"
& $py --version
& $py -m pip install --upgrade "pip==25.2"; Check "pip upgrade"

if (Test-Path "requirements-llm.lock.txt") {
    Write-Host "Installing from requirements-llm.lock.txt (exact environment)"
    & $py -m pip install -r requirements-llm.lock.txt; Check "install from lock"
} else {
    Write-Host "Installing from requirements-llm.txt"
    & $py -m pip install -r requirements-llm.txt; Check "install from requirements-llm.txt"
    "--extra-index-url https://download.pytorch.org/whl/cu126" | Out-File -Encoding ascii requirements-llm.lock.txt
    & $py -m pip freeze | Out-File -Encoding ascii -Append requirements-llm.lock.txt
    Write-Host "Wrote requirements-llm.lock.txt"
}

$env:BK_MODEL = $ModelPath
& $py -c @"
import os, torch, transformers, peft
print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)
assert torch.cuda.is_available(), 'PyTorch cannot see the GPU'
free, total = torch.cuda.mem_get_info()
print('GPU:', torch.cuda.get_device_name(0), '| free %.1f of %.1f GB' % (free / 2**30, total / 2**30))
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
p = os.environ['BK_MODEL']
cfg = AutoConfig.from_pretrained(p); t = cfg.text_config
print('model_type', cfg.model_type, '| layers', t.num_hidden_layers, '| hidden', t.hidden_size,
      '| full-attention layers', [i + 1 for i, x in enumerate(t.layer_types) if x == 'full_attention'])
tok = AutoTokenizer.from_pretrained(p)
print('tokenizer OK, vocab', len(tok), '| pad', tok.pad_token)
try:
    import bitsandbytes; print('bitsandbytes', bitsandbytes.__version__, '(4-bit fallback available)')
except Exception as e:
    print('bitsandbytes not usable:', e, '(only needed for --load_4bit)')
"@
Check "LLM environment check"
Write-Host "`nLLM setup done. Next: powershell -ExecutionPolicy Bypass -File .\smoke_llm.ps1" -ForegroundColor Green
