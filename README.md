# BanglaKnight (BLP-2027 Shared Task 1)

Score: **Defense Score = mean macro-F1 over the 3 forms x (1 - USR)**,
where USR = share of unsafe prompts (Offensive/Harmful/Severe) predicted Safe.

## Setup (once, PowerShell)
```powershell
cd H:\BanglaKnight
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```
Then activate the venv in any new terminal: `.\.venv\Scripts\Activate.ps1`

## Feasibility test (~10 min on a 4090)
```powershell
powershell -ExecutionPolicy Bypass -File .\smoke_test.ps1
```
Prints speed, peak VRAM and a Defense Score on a small validation slice. The scores
from this test are meaningless (tiny data, 1 epoch); it only checks the pipeline.

## Data
Put the official files in `data\`: `train.jsonl`, `dev.jsonl`, `devtest.jsonl` (later `test.jsonl`).

## Files
| File | What it does |
|---|---|
| `bk_common.py` | loading, Defense Score, logit-file helpers |
| `tfidf_baseline.py` | CPU char+word TF-IDF baseline (~0.71 DS in grouped CV) |
| `train_encoder.py` | GPU fine-tuning with cross-form consistency loss |
| `decide.py` | ensemble, prompt-level averaging, Safe-bias tuning, CodaBench zip |

## Typical workflow
```powershell
# 1. a real model: train on train, pick best step on dev, predict the blind dev-test
python train_encoder.py --model xlm-roberta-large --train data/train.jsonl --dev data/dev.jsonl --predict data/devtest.jsonl --out runs/xlmrL_s42

# 2. cheap CPU member
python tfidf_baseline.py --train data/train.jsonl --predict data/dev.jsonl data/devtest.jsonl --out runs/tfidf

# 3. ensemble + submission (same model order in --val and --test)
python decide.py --val runs/xlmrL_s42/val_logits.csv runs/tfidf/dev_logits.csv --test runs/xlmrL_s42/devtest_logits.csv runs/tfidf/devtest_logits.csv --out submissions/sub1
```
Upload `submissions\sub1.zip` to CodaBench.

## Useful flags (`train_encoder.py`)
- `--cons_w 0` turns off the consistency loss (ablation for the paper)
- `--joint` one input per prompt with all three forms concatenated
- `--fold k --folds 5` cross-validation instead of `--dev`
- `--grad_ckpt` if you run out of VRAM; `--max_prompts N` for quick tests

## Candidate models
`xlm-roberta-large` (560M), `google/muril-large-cased` (506M, trained on transliterated Indic text),
`csebuetnlp/banglabert_large`, `microsoft/mdeberta-v3-base` (276M, the lightweight target).
