# CLAUDE.md: BanglaKnight / TriGuard

Context and working rules for any Claude agent in this repo. Read this first, then `INSTRUCTIONS.md` (the step-by-step plan) and `EXPERIMENTS.md` (what has been done).

## Project in one paragraph
Muhtasim Ibteda Shochcho is competing in **BanglaKnight, BLP-2027 Shared Task 1** (Adversarial Prompt Classification, NAACL 2027 workshop). Each Bangla prompt appears in three forms (Bangla script `_ba`, romanized `_ro`, code-mixed `_co`) with one shared label: 0 Safe, 1 Offensive, 2 Harmful, 3 Severe. Goals: **1st place on the test leaderboard** and a **best-paper-quality system paper**. The method is called **TriGuard**; the full design is in `APPROACH.md` (diagram in `docs/architecture.png`). The user wants the final system to be **lightweight and powerful**: strong teachers for the leaderboard, a small distilled student for the paper story.

## Metric (never forget this)
`Defense Score = mean(macro-F1 per form over bangla, romanized, codemixed) x (1 - USR)`, where USR = share of truly unsafe rows (label 1 to 3) predicted 0. Implemented in `bk_common.defense_score`. Always report DS, macro-F1, USR, **Safe FPR** (share of Safe prompts flagged unsafe) and per-class recall, plus the three per-form F1s at row level. Shifting decisions away from Safe trades Safe FPR for USR; never report a DS gain without its Safe FPR.

## Key dates (AoE)
- Dev phase (practice): until Jan 7, 2027
- Test data release: Jan 8, 2027 | CodaBench registration deadline: Jan 17 | **System submission deadline: Jan 24** (latest *finished* submission counts)
- Results: Feb 3 | **Paper deadline: Feb 15** (4 pages, ACL template, OpenReview)

## Environment
- Windows, PowerShell, NVIDIA **RTX 4090 (24 GB)**, bf16 supported.
- Repo root: `H:\BanglaKnight`. Virtual env: `.venv` (Python 3.11 or 3.12), created by `setup.ps1`.
- Run Python as `.\.venv\Scripts\python.exe` (or activate with `.\.venv\Scripts\Activate.ps1`).
- Before running anything set `$env:PYTHONUTF8 = "1"` (Bangla text). Recommended: `$env:HF_HOME = "H:\hf_cache"` so model downloads do not fill drive C:.
- Dependencies are pinned in `requirements.txt`; the exact environment is frozen in `requirements.lock.txt`. PyTorch must be the CUDA build (`+cu126`). To add a package: pin it in `requirements.txt`, install it, then regenerate the lock (see `setup.ps1`). Never `pip install torch` without the CUDA index.
- **LLM teacher** (Qwen3.5-9B, original release, at `H:\Abliterate\Qwen3.5-9B`; the abliterated copy lives elsewhere and is not used): needs transformers 5.x, so it runs in a separate venv `.venv-llm` (`setup_llm.ps1`, `requirements-llm.txt`, `requirements-llm.lock.txt`). Never upgrade transformers in `.venv`. Design: `docs/LLM_DESIGN.md`.
- If the agent's shell is a sandbox that cannot reach the GPU, do not try to train there. Write the command, ask the user to run it in PowerShell, then read the logs from `logs/` and `runs/`.

## Repo map
| Path | Purpose |
|---|---|
| `bk_common.py` | loading (jsonl/csv), Defense Score, logit-file I/O |
| `tfidf_baseline.py` | CPU baseline, cheap ensemble member |
| `train_encoder.py` | GPU encoder fine-tuning with cross-form consistency loss |
| `decide.py` | ensemble, prompt-level pooling, Defense-Score bias tuning, CodaBench zip |
| `train_llm.py` | Qwen3.5-9B LoRA classifier (run with `.venv-llm`); `setup_llm.ps1`, `smoke_llm.ps1`, `requirements-llm.txt` |
| `decision_lab.py` | compares ensemble / pooling / decision strategies (repeated grouped CV + paired bootstrap) |
| `docs/LLM_DESIGN.md` | Qwen3.5-9B classifier design (approved plan for Phase 3) |
| `setup.ps1`, `smoke_test.ps1` | environment setup, feasibility test |
| `data/` | official files only: `train.jsonl`, `dev.jsonl`, `devtest.jsonl`, later `test.jsonl` |
| `runs/<run_id>/` | one folder per run: `val_logits.csv`, `<split>_logits.csv`, `summary.json`, `train.log` (all console lines + every 50th step) |
| `submissions/` | CodaBench zips + `_config.json` describing how each was made |
| `logs/` | console logs |
| `APPROACH.md`, `INSTRUCTIONS.md`, `EXPERIMENTS.md` | plan, playbook, run log |

Logit files are CSV: `id,prompt_id,form,label,l0,l1,l2,l3` (label empty for blind sets). Every model must produce this format so `decide.py` can ensemble it.

## How to work with the user
1. **Find the current step**: open `INSTRUCTIONS.md`, take the first unchecked step whose prerequisites are done. Tell the user which step you are on and what you will do, in one or two sentences.
2. **One step at a time.** Finish it, check its "Done when" criteria, tick the box, log results in `EXPERIMENTS.md`, then propose the next step.
3. **Ask before**: runs expected to take more than ~1 hour, any CodaBench submission, changing pinned dependencies, deleting anything in `runs/` or `submissions/`, or changing the plan in `APPROACH.md`.
4. The user prefers ideas and designs **brainstormed and reviewed before they are built**. For new components (new script, new loss, new model family) propose the design briefly first.
5. Report results as numbers in a small table (DS, macro-F1, USR, Safe FPR, per-class recall, per-form F1, minutes, VRAM). Compare against the best previous run. Say plainly when something did not help.
6. **No em dashes** in any prose written for the user (docs, paper text, messages). Use commas, colons or parentheses.

## Rules for valid results
- **Never** train or tune on `devtest` or `test` labels (they have none) and never hand-label blind data.
- Select checkpoints and tune biases on `dev` or on out-of-fold (OOF) logits, never on the leaderboard. Use the leaderboard only as a sanity check.
- Keep prompt groups intact: all three forms of a `prompt_id` go in the same split or fold (`GroupKFold` by `prompt_id`).
- Fix seeds (`--seed`); report mean over seeds when comparing methods for the paper.
- **Noise level**: on dev (2,049 prompts) the DS standard deviation is about 1.0 point; on OOF train+dev (about 10,000 prompts) about 0.5. Do not adopt a change (ensemble mode, pooling, bias, loss) unless `decision_lab.py`-style paired bootstrap gives P(better) >= 0.9, preferably on OOF.
- Anything learned on labelled predictions (weights, temperatures, biases, gates) is fit on training folds and scored on held-out folds, never on the same examples.
- Unknown competition rules (external data, LLM use, closed APIs, submission limits): treat as **not allowed until the organizers confirm**. Record the answer in `EXPERIMENTS.md` under "Rules".
- Every submission gets a `_config.json` (written by `decide.py`) and a row in `EXPERIMENTS.md`.

## Coding conventions
- Python 3.11, argparse CLIs, UTF-8 everywhere (`encoding="utf-8"`), Unicode NFC normalisation via `bk_common.load_rows`.
- Windows: DataLoader `num_workers=0` (or plain Python batching as in `train_encoder.py`), no bash-only syntax in scripts, PowerShell for launchers.
- GPU: bf16 autocast, TF32 on, gradient checkpointing (`--grad_ckpt`) if VRAM is tight.
- New models write logits through `bk_common.write_logits`. New runs write `summary.json` with args, best score, minutes, peak VRAM, params.
- Keep scripts small and composable. Do not hide metric logic in notebooks.

## Useful commands
```powershell
$env:PYTHONUTF8 = "1"; $env:HF_HOME = "H:\hf_cache"
.\.venv\Scripts\Activate.ps1
python train_encoder.py --model xlm-roberta-large --train data/train.jsonl --dev data/dev.jsonl --predict data/devtest.jsonl --out runs/E2_xlmrL_cons_s42
python tfidf_baseline.py --train data/train.jsonl --predict data/dev.jsonl data/devtest.jsonl --out runs/B0_tfidf
python decide.py --val runs/E2_xlmrL_cons_s42/val_logits.csv --test runs/E2_xlmrL_cons_s42/devtest_logits.csv --blind data/devtest.jsonl --out submissions/S01_E2
```

## References
- Task site: https://banglaknight.github.io/ (data, rules, leaderboard, paper pages)
- CodaBench: https://www.codabench.org/competitions/18411/ (forum: https://www.codabench.org/forums/18099/)
- Dataset paper to cite: BanglaShield (Kabir et al., 2026), arXiv 2610.24288
- Related: BanglaVeilGuard, arXiv 2608.21880
