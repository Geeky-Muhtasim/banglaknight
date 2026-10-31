# INSTRUCTIONS.md: TriGuard step-by-step playbook

How to use this file:
- Work top to bottom. Each step has **Goal**, **Do**, **Done when**, **Record**.
- Tick the box (`[x]`) only when "Done when" is met, and log the run in `EXPERIMENTS.md`.
- Run IDs follow the tables in `APPROACH.md` (B0, E1, E2, ...). Output folder: `runs/<ID>_<model>_<variant>_s<seed>`.
- All commands are PowerShell from `H:\BanglaKnight` with the venv active and `$env:PYTHONUTF8 = "1"`.

Current phase is the first one with an unticked step.

---

## Phase 0: Environment (target: Nov 1 to 3)

- [x] **0.1 Setup**
  - Do: `powershell -ExecutionPolicy Bypass -File .\setup.ps1`
  - Done when: it prints `GPU: NVIDIA GeForce RTX 4090` and `GPU matmul OK`, and `requirements.lock.txt` exists.
- [x] **0.2 Smoke test**
  - Do: `powershell -ExecutionPolicy Bypass -File .\smoke_test.ps1`
  - Done when: both small trainings finish, `runs/smoke_sub.zip` exists. Note minutes and peak VRAM for mDeBERTa-base and XLM-R large.
  - Record: speed and VRAM in `EXPERIMENTS.md` (these set the compute budget for later phases).
- [ ] **0.3 Register and ask the rules**
  - Do: register the team on CodaBench (one account per team). Post on the forum asking: external data allowed? LLMs / closed APIs allowed? submission limits per day and per phase? will `prompt_id` be present in the test file?
  - Done when: questions posted. Record answers under "Rules" in `EXPERIMENTS.md` as they arrive.

## Phase 1: Data and baselines (target: Nov 3 to 10)

- [x] **1.1 Get the official files**
  - Do: download dev and dev-test from https://banglaknight.github.io/data/ into `data/` (official names: `dev.jsonl`, `devtest.jsonl`). Also download the starter notebook.
  - Done when: `python -c "from bk_common import load_rows; [print(p, len(load_rows('data/'+p))) for p in ['train.jsonl','dev.jsonl','devtest.jsonl']]"` prints sizes.
- [x] **1.2 Sanity-check the blind file**
  - Do: confirm `devtest` has `prompt_id` (or `_ba/_ro/_co` suffixes) and exactly 3 rows per prompt; check dev label distribution vs train.
  - Done when: findings written to `EXPERIMENTS.md` ("Data notes"). If prompt pooling is impossible, flag it to the user immediately (the plan changes).
- [x] **1.3 B0: TF-IDF baseline on dev**
  - Do: `python tfidf_baseline.py --train data/train.jsonl --predict data/dev.jsonl data/devtest.jsonl --out runs/B0_tfidf`
  - Then: `python decide.py --val runs/B0_tfidf/dev_logits.csv --test runs/B0_tfidf/devtest_logits.csv --blind data/devtest.jsonl --out submissions/S00_B0`
  - Done when: dev DS logged (expect about 0.70).
- [ ] **1.4 B1: starter BanglaBERT** (organizers' notebook, unchanged)
  - Done when: its dev Defense Score is logged. This is the "official baseline" row in the paper.
- [x] **1.5 First CodaBench submission** (ask the user first)
  - Do: upload `submissions/S00b_B0_nobias.zip` to the dev phase (no-bias version, see Decisions in `EXPERIMENTS.md`).
  - Done when: leaderboard score recorded; compare with local dev DS to measure the dev vs dev-test gap.

## Phase 2: Strong single encoders (target: Nov 10 to 30)

**Deprioritised 2026-11-01**: encoders lose about twice as much as Qwen from dev to devtest (see `EXPERIMENTS.md` Decisions); only run E4 to E6 if GPU time is free after the LLM steps.

Defaults: `--epochs 4 --lr 1e-5 --llrd 0.9 --bs_prompts 8 --max_len 256`. One seed (42) first; add seeds 7 and 1234 only for the final candidates.

- [x] **2.1 E1: XLM-R large without consistency** (`--cons_w 0`)
  - `python train_encoder.py --model xlm-roberta-large --train data/train.jsonl --dev data/dev.jsonl --predict data/devtest.jsonl --cons_w 0 --out runs/E1_xlmrL_nocons_s42`
- [x] **2.2 E2: XLM-R large with consistency** (`--cons_w 1.0`)
  - Done when: E1 vs E2 compared on DS and on per-form F1 gap (max form F1 minus min form F1). This is the core ablation of the paper.
- [x] **2.2b Decision-layer study on dev** (done 2026-11-01, see `APPROACH.md` C2/C4 and `EXPERIMENTS.md`): no variant beats the original by more than dev noise (about 1 pt); stacking clearly worse. Re-run on OOF in step 4.3.
- [ ] ~~2.3 Tune the consistency weight~~ **Dropped**: E2 showed no pooled gain; keep `--cons_w 0` as default for new encoders. (Optional later: one `--cons_w 0.3` run for the paper ablation.)
- [ ] **2.4 E4: MuRIL large** (`google/muril-large-cased`), `--cons_w 0`. Romanized specialist candidate: check row-level romanized F1 against E1 (0.724).
- [ ] **2.5 E5: BanglaBERT large, Bangla rows only** (`csebuetnlp/banglabert_large`). Needs a `--forms bangla` filter in `train_encoder.py` (small change, propose first). Bangla-script specialist for the ensemble.
- [ ] **2.6 E6: class-weighted CE** on XLM-R large (training-time alternative to the Safe bias). Needs a `--class_weights` flag (propose first). Compare USR and Safe FPR with E1 + Safe bias.
- [ ] **2.7 E3: joint input** (`--joint`), lower priority: the LLM tests the same idea more strongly (L2).
- [ ] **2.8 Light candidates** for the student baseline: `microsoft/mdeberta-v3-base` (`--lr 3e-5 --llrd 1.0`), `xlm-roberta-base`.
- [ ] **2.9 Submit the best single model or small ensemble** to the dev phase (ask first).
  - Done when: `EXPERIMENTS.md` has every run with DS, macro-F1, USR, Safe FPR, per-class recall, per-form F1, minutes, VRAM.
  - Decision point with the user: which encoders go forward as teachers.

If a run diverges (DS stuck near a single-class score): lower `--lr` to 7e-6, raise `--warmup` to 0.2, or try another seed.

## Phase 3: LLM teacher, Qwen3.5-9B (target: Nov 3 to 30)

Design: `docs/LLM_DESIGN.md` (reviewed by the user 2026-11-01). Model: original Qwen3.5-9B at `H:\Abliterate\Qwen3.5-9B` (the abliterated copy is elsewhere and is not used for L1).

- [x] **3.1 Environment**: `powershell -ExecutionPolicy Bypass -File .\setup_llm.ps1` creates `.venv-llm` from `requirements-llm.txt` (torch 2.7.1+cu126, transformers 5.20.2, peft 0.21.2, accelerate 1.15.0, bitsandbytes 0.50.2) and freezes `requirements-llm.lock.txt`. The encoder `.venv` stays on transformers 4.57.6.
  - Done when: it prints the GPU, `model_type qwen3_5 | layers 32`, full-attention layers `[4, 8, ..., 32]`, and `tokenizer OK`.
- [x] **3.2 `train_llm.py` implemented** (2026-11-01) per `docs/LLM_DESIGN.md`, tested end to end on CPU with a tiny random Qwen3.5 checkpoint in the real file layout: weights load exactly under the key remapping (0 mismatches), LoRA on all 10 module types, rows/joint modes, view dropout, `--keep_layers`, `--unsafe_w`, `--no_label_init`, `--fold`, single-view eval, adapter + head save, logit files. Not yet run on GPU; `--load_4bit` is untested (bitsandbytes needs CUDA).
  - Label words for the head init: " Safe", " Offensive", " Harmful" (first token " Harm"), " Dangerous" for the Severe class (" Severe" splits into " Se" + "vere"). The instruction text uses the same words.
  - Inputs are clipped per form to a token budget so the instruction and the final "Risk:" always survive (rows: 253 tokens per form, 1.2% of rows clipped; joint: 221 tokens per form).
- [x] **3.3 L0 smoke test**: `powershell -ExecutionPolicy Bypass -File .\smoke_llm.ps1` (rows part, then joint part; `-Only rows` or `-Only joint` to run one).
  - Record from `logs/L0a_rows_smoke.log` and `logs/L0b_joint_smoke.log`: model load time, peak VRAM (alloc and reserved), tok/s, "min per full epoch" estimate, and the step-0 score (zero-shot head check).
  - If a part runs out of memory: rerun it with `--micro_bs 4 --grad_accum 8` (rows) or `--micro_bs 2 --grad_accum 16` (joint); last resort `--load_4bit`.
  - If tok/s is very low: the Gated DeltaNet layers run on the PyTorch fallback on Windows. Options: `--keep_layers 24`, or WSL2 where `flash-linear-attention` and `causal-conv1d` install. Discuss with the user before switching.
  - Then propose L1 settings (epochs, micro-batch) from the measured speed and ask before starting.
  - **L0 result 2026-11-01**: rows mode OK (load 16 s, peak 17.7 GB alloc / 19.9 GB reserved, ~1,100 tok/s, ~48 min per training epoch; DS 0.597 on 200 dev prompts after only 30 steps; step-0 zero-shot DS 0.18, mostly predicts Severe). Joint mode stalled in its step-0 eval: a fixed eval batch of 32 joint inputs exceeded 24 GB and Windows spilled into shared GPU memory instead of raising out-of-memory. **Fixed** in `train_llm.py`: token-budget batches (`--eval_tokens 8192`, `--train_tokens 4096`), PyTorch memory cap (`--vram_cap_gb 22`, so overflow now raises an error), cache clearing around evals. Re-run: `smoke_llm.ps1 -Only joint`.
- [x] **3.3b Optional fast kernels** (done 2026-11-01: check passed, rel. diff 0.0044, x3.6 on the op, ~1.4x end to end): `powershell -ExecutionPolicy Bypass -File .\install_fast_kernels.ps1` installs Triton for Windows + flash-linear-attention into `.venv-llm` and runs `check_fast_kernels.py` (numerical match vs the PyTorch reference on the GPU + speed-up). If the check fails it uninstalls itself. If it passes, re-run `smoke_llm.ps1 -Only rows` and compare tok/s with the 1,100 above.
  - Extra safety (recommended): NVIDIA Control Panel > Manage 3D settings > Program Settings > add `.venv-llm\Scripts\python.exe` > **CUDA - Sysmem Fallback Policy = Prefer No Sysmem Fallback**. Then running out of VRAM always gives an error instead of a silent slowdown.
- [x] **3.4 (run first) L2 joint input + view dropout 0.3** (done 2026-11-01: dev 0.7925 alone, 0.8031 in ENS1; LB 0.7525, rank 7), 2 epochs, train to dev, predict devtest, single-view dev logits. About 1.5 to 2 hours. Ask before starting.
  - `.\.venv-llm\Scripts\python.exe train_llm.py --model H:/Abliterate/Qwen3.5-9B --train data/train.jsonl --dev data/dev.jsonl --predict data/devtest.jsonl --mode joint --view_dropout 0.3 --micro_bs 4 --grad_accum 8 --epochs 2 --evals_per_epoch 2 --eval_single_views --out runs/L2_qwen35_joint_s42`
  - Then: log DS, macro-F1, USR, Safe FPR, per-class recall, single-view table; submit L2 alone and L2 + E1/E2/B0 to the dev phase (ask first).
- [x] **3.4-sub Submit `S04_L2_safe06.zip` and `S04_L2_safe10.zip`** (2026-11-01: 0.7656 / **0.7664**, current best; use `--safe_bias 1.0` for single-LLM submissions until OOF tuning).
- [ ] **3.4a L3 = L2 recipe with 3 epochs and view dropout 0.5** (about 80 to 85 min; ask before starting). Reason: L2 was best at its last step, and its single-view scores show it leans on the Bangla view (romanized-only 0.70).
  - `.\.venv-llm\Scripts\python.exe train_llm.py --model H:/Abliterate/Qwen3.5-9B --train data/train.jsonl --dev data/dev.jsonl --predict data/devtest.jsonl --mode joint --view_dropout 0.5 --micro_bs 4 --grad_accum 8 --epochs 3 --evals_per_epoch 2 --eval_single_views --out runs/L3_qwen35_joint_vd05_ep3_s42`
  - Compare with L2: DS, Safe FPR, USR, and the three single-view scores. Then rebuild the ensemble with `decide.py --calibrate --no_bias`.
- [ ] **3.4b L1 rows mode** (second; ablation and ensemble diversity), 2 epochs, train to dev, predict devtest. Ask before starting (multi-hour run).
  - `.\.venv-llm\Scripts\python.exe train_llm.py --model H:/Abliterate/Qwen3.5-9B --train data/train.jsonl --dev data/dev.jsonl --predict data/devtest.jsonl --mode rows --eval_at_start --out runs/L1_qwen35_rows_s42`
- [ ] **3.5 Compare L2 vs L1** on DS and on romanized-only / Bangla-only inference (missing-view robustness).
- [ ] **3.6 L3 unsafe-mass penalty** (`--unsafe_w 0.5`) on the better of L1/L2: report USR and Safe FPR.
- [ ] **3.7 L4 `--keep_layers 24`**: speed, VRAM, DS vs 32 layers.
  - Done when: L runs logged with the full metric set and a confusion matrix; submit the best L (+ E1/E2/B0 ensemble) to the dev phase (ask first).

## Phase 4: Cross-validation, ensemble, decision layer (target: Dec 1 to 20)

- [ ] **4.1 5-fold CV over train+dev** for each chosen teacher: `--train data/train.jsonl data/dev.jsonl --fold k --folds 5 --predict data/devtest.jsonl`, k = 0..4. About 10,000 labelled prompts of OOF logits: halves the noise of decisions compared with dev alone.
- [ ] **4.2 Merge OOF logits** per model into one file; average the 5 fold predictions for devtest (agent writes `merge_oof.py`).
- [ ] **4.3 Decision-layer comparison on OOF**: `python decision_lab.py --val NAME=runs/.../oof_logits.csv ...` Compares geometric vs arithmetic vs NLL-weighted vs calibrated ensembling; mean vs confidence-weighted pooling (+ missing-view table); argmax vs Safe-only bias vs 4-class bias vs threshold.
  - Adopt a change only if its paired-bootstrap P(better) >= 0.9 on OOF. Otherwise keep: calibrated geometric ensemble, mean pooling, argmax.
  - Report for the chosen configuration: DS, macro-F1, USR, Safe FPR, per-class recall. If a bias is adopted, state its Safe FPR cost explicitly.
- [ ] **4.4 Port the winner into `decide.py`** (temperature scaling, ensemble mode flag, pooling mode flag, Safe-only bias), with frozen parameters saved in the submission `_config.json`.
- [ ] **4.5 Submit the ensemble** to the dev phase (ask first).

## Phase 5: Lightweight student (target: Dec 20 to Jan 5)

- [ ] **5.1 Design review of `distill.py`** (propose first): `L = CE(y, P_s(x_v)) + alpha * T^2 * KL(P_t^T(prompt) || P_s^T(x_v)) + beta * L_consistency`, with **cross-view targets** (each single-view row learns the teacher's pooled three-view distribution). Teacher targets come from the frozen ensemble's OOF probabilities on train+dev. Grid: T in {1, 2, 4}, alpha in {0.5, 1}, beta in {0, 0.5}.
- [ ] **5.2 Student comparison** with the best (T, alpha, beta): mDeBERTa-v3-base, XLM-R base, distilbert-base-multilingual-cased, Multilingual-MiniLM-L12-H384, BanglaBERT base. Report DS (pooled and per single view), macro-F1, USR, Safe FPR.
- [ ] **5.3 Vocabulary trimming** for the best 1 or 2 students: keep token ids seen in train/dev/devtest(/test) text + specials; verify identical logits on dev before and after; report new parameter count.
- [ ] **5.4 ONNX benchmark** (`bench_onnx.py`): export, dynamic INT8 quantisation, then measure on the real artifact: DS FP32 vs INT8, file size, peak RAM, CPU latency p50/p95 at batch 1, throughput at batch 32, max length 256; state the CPU model.
  - Done when: teacher vs student table (DS, size, RAM, latency, throughput) is in `EXPERIMENTS.md`.
- [ ] **5.5 (only if organizers allow transductive use of blind text)**: add devtest/test texts with teacher targets to distillation; compare.

## Phase 6: Freeze for the test phase (target: Jan 1 to 7)

- [ ] **6.1 Write `final.ps1`**: one command that retrains the chosen models on train+dev (fixed epochs from CV, no early stopping), predicts `data/test.jsonl`, runs `decide.py` with the frozen bias/weights, and writes `submissions/FINAL.zip`.
- [ ] **6.2 Dry run** `final.ps1` with `devtest.jsonl` in place of `test.jsonl`; the resulting zip must score as expected on the dev leaderboard.
- [ ] **6.3 Time it.** It must finish comfortably inside the test window.

## Phase 7: Test phase protocol (Jan 8 to 24)

- [ ] **7.1** Download `test` into `data/test.jsonl`; repeat the checks of step 1.2.
- [ ] **7.2** Run `final.ps1`. Check row count, ids, label distribution vs dev predictions.
- [ ] **7.3** Submit early (ask first). Keep a backup zip from a single strong model.
- [ ] **7.4** Final submission at least 48 hours before Jan 24, 11:59 PM AoE. Remember: the latest *finished* submission is ranked.

## Phase 8: Analysis and paper (Jan 25 to Feb 15)

- [ ] **8.1 Ablation table** (with bootstrap CIs): pooling on/off and missing-view table, consistency on/off, ensemble mode, bias on/off (with Safe FPR), LLM rows vs joint, 32 vs 24 layers, cross-view vs plain distillation, teacher vs student.
- [ ] **8.2 Per-form analysis**: F1 by form and USR by form, before and after consistency.
- [ ] **8.3 Representation analysis**: layer-wise similarity (e.g. CKA or cosine) between the three forms' representations, in XLM-R and in Qwen3.5-9B (where in depth do the scripts converge?).
- [ ] **8.4 Error analysis**: Harmful vs Severe confusions with a few anonymised examples (no harmful details reproduced in the paper).
- [ ] **8.5 Write the paper** in the ACL template (4 pages + references): intro, data, method (diagram from `docs/`), results, analysis, limitations, ethics. Run ACL PubCheck. Create OpenReview accounts for all authors early (Gmail accounts can take up to two weeks).
- [ ] **8.6 Submit** on OpenReview before Feb 15, 11:59 PM AoE.

---

## When something breaks
- `CUDA available: False`: rerun `setup.ps1`; check `nvidia-smi` driver is 560 or newer.
- Out of memory: add `--grad_ckpt`, or lower `--bs_prompts` and raise `--grad_accum` to keep the effective batch.
- Tokenizer errors for DeBERTa: make sure `sentencepiece` and `protobuf` from the lock are installed.
- Bangla shows as `???` in logs: set `$env:PYTHONUTF8 = "1"`.
