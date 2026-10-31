# L0: feasibility test of the Qwen3.5-9B classifier on the RTX 4090.
# Run from PowerShell:   powershell -ExecutionPolicy Bypass -File .\smoke_llm.ps1
# Optional:              -ModelPath "H:\Abliterate\Qwen3.5-9B"   -Only rows|joint
#
# Each part: 300 training prompts, 30 optimizer steps, evaluation on 200 dev prompts before and after.
# Reports peak VRAM, tokens/s and an estimate of minutes per full epoch. Scores after 30 steps are NOT meaningful;
# the "before training" score shows whether the label-word head init gives a zero-shot start.
param([string]$ModelPath = "H:\Abliterate\Qwen3.5-9B", [string]$Only = "")
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$py = ".\.venv-llm\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "Run setup_llm.ps1 first." -ForegroundColor Red; exit 1 }
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force -Path runs, logs | Out-Null

function Step($title, [string[]]$cmd, $log) {
    Write-Host "`n=== $title ===" -ForegroundColor Cyan
    & $py @cmd 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED ($title). See $log. If it is out of memory: rerun that part with --load_4bit, or a smaller --micro_bs." -ForegroundColor Red
    }
}

$common = @("train_llm.py", "--model", $ModelPath, "--train", "data/train.jsonl", "--dev", "data/dev.jsonl",
            "--max_prompts", "300", "--max_val_prompts", "200", "--max_steps", "30", "--eval_at_start")

if ($Only -ne "joint") {
    Step "L0a rows mode, bf16 LoRA, micro_bs 8 x accum 4" `
        ($common + @("--mode", "rows", "--micro_bs", "8", "--grad_accum", "4", "--out", "runs/L0a_rows_smoke")) logs/L0a_rows_smoke.log
}
if ($Only -ne "rows") {
    Step "L0b joint mode (3 views per input), bf16 LoRA, micro_bs 4 x accum 8" `
        ($common + @("--mode", "joint", "--micro_bs", "4", "--grad_accum", "8", "--out", "runs/L0b_joint_smoke")) logs/L0b_joint_smoke.log
}
Write-Host "`nL0 finished. Key lines (peak VRAM, tok/s, epoch estimate) are at the end of each log in .\logs" -ForegroundColor Green
