# Small feasibility run: checks the pipeline end to end and measures speed + VRAM.
# Run from PowerShell:   powershell -ExecutionPolicy Bypass -File .\smoke_test.ps1
# Note: "Continue" so that harmless warnings printed to stderr (e.g. by transformers) don't abort the run.
# Output is not piped so the tqdm progress bars render live; each script writes its own log file
# (runs/<id>/train.log, runs/smoke_sub.log).
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "FAILED: .venv not found. Run setup.ps1 first." -ForegroundColor Red; exit 1 }
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
if (-not $env:HF_HOME) { $env:HF_HOME = "H:\hf_cache" }
New-Item -ItemType Directory -Force -Path runs, logs | Out-Null

function Step($title, [string[]]$cmd, $log) {
    Write-Host "`n=== $title ===" -ForegroundColor Cyan
    & $py @cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED ($title). See $log" -ForegroundColor Red; exit 1 }
}

Step "1/3 lightweight: mDeBERTa-v3-base, 800 prompts, 1 epoch" `
    @("train_encoder.py","--model","microsoft/mdeberta-v3-base","--train","data/train.jsonl","--fold","0",
      "--max_prompts","800","--epochs","1","--lr","3e-5","--llrd","1.0","--out","runs/smoke_mdeberta") runs/smoke_mdeberta/train.log

Step "2/3 large: XLM-R large, 800 prompts, 1 epoch (speed + VRAM)" `
    @("train_encoder.py","--model","xlm-roberta-large","--train","data/train.jsonl","--fold","0",
      "--max_prompts","800","--epochs","1","--out","runs/smoke_xlmrL") runs/smoke_xlmrL/train.log

Step "3/3 decision layer on smoke logits" `
    @("decide.py","--val","runs/smoke_xlmrL/val_logits.csv","--test","runs/smoke_xlmrL/val_logits.csv",
      "--out","runs/smoke_sub") runs/smoke_sub.log

Write-Host "`nSmoke test finished. Logs: runs\smoke_mdeberta\train.log, runs\smoke_xlmrL\train.log, runs\smoke_sub.log" -ForegroundColor Green
