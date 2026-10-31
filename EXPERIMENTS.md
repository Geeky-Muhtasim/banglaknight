# EXPERIMENTS.md: run log

Append newest entries at the bottom of each table. Scores are on the split named in "Eval on".
DS = Defense Score, F1 = mean macro-F1 over forms, bn/ro/cm = per-form macro-F1.

## Rules (answers from the organizers)
| Question | Answer | Source / date |
|---|---|---|
| External data allowed? | unknown | |
| LLMs / closed APIs allowed? | unknown | |
| Submission limits (dev / test phase)? | Dev: 10/day, 300 total. Test: 3/day, 10 total. Daily reset 00:00 UTC. Failed submissions do not count. Latest *finished* submission is shown (dev) / ranked (test); test scores hidden until Feb 3 | task site, Submission page, 2026-11-01 |
| `prompt_id` in test file? | yes: data page lists `id, prompt_id, form, text` for all files, and `prompt_id` always links the 3 rows of a prompt (verified in devtest) | task site, Data page, 2026-11-01 |
| Blind-set rules | no hand labelling of blind prompts, no searching for them or their labels in other sources; predictions must come from the automatic system | task site, Data page, 2026-11-01 |
| Submission format | zip with a single top-level `predictions.csv`, header `id,label`, one row per blind id (no missing, extra or repeated), labels 0 to 3. Submit as "Yourself" (not an organization), same team name and affiliation every time | task site, Submission page, 2026-11-01 |

## Data notes
- train: 8,196 prompts x 3 forms = 24,588 rows; labels identical across forms; Safe 34.1%, Offensive 15.3%, Harmful 26.3%, Severe 24.3%.
- code-mixed rows: ~23% contain Devanagari (Hindi), ~15% Arabic script (Urdu).
- dev: 2,049 prompts x 3 = 6,147 rows; Safe 34.1%, Offensive 15.3%, Harmful 26.3%, Severe 24.4% (same distribution as train).
- devtest (blind, official name `devtest.jsonl`): 5,000 prompts x 3 = 15,000 rows; has `prompt_id`, no `label` field.
- All three splits: exactly 3 rows per prompt (one per form), id = prompt_id + `_ba/_ro/_co`, no duplicate ids, no empty texts, labels shared across forms (train, dev).
- No leakage: 0 shared prompt_ids between splits; 1 dev row and 0 devtest rows have an exact (lower-cased) text match in train; 0 devtest/dev text matches.
- Mean characters per row: about 140 to 175, romanized longest. Prompt pooling (`group_mean` over prompt_id) is possible on devtest, so the plan stands.
- Possible devtest shift: B0 predicts Safe for 17.1% of devtest prompts vs 27.8% of dev prompts (true dev Safe 34.1%). E1 shows the same (19.4% vs 29.8%), so devtest probably has fewer Safe prompts than dev; the first leaderboard score will tell.
- Token lengths (train+dev+devtest): XLM-R p99 = 216 to 238 tokens, 0.4 to 0.7% of rows over 256; mDeBERTa p99 = 245 to 274, 0.7 to 1.3% over 256. `--max_len 256` is fine. Joint input (3 forms) can exceed 512 for the longest prompts.

## Compute (from smoke test)
| Model | Params | Rows/s train | Peak VRAM | Notes |
|---|---|---|---|---|
| mdeberta-v3-base | 279M | ~170 | 9.6 GB | 800 prompts x 1 epoch (100 steps) in 0.26 min; bs 8 prompts = 24 rows, max_len 256, bf16 |
| xlm-roberta-large | 560M | ~230 | 13.0 GB | smoke: 0.29 min. Full run (E1, 4 epochs, 8 evals incl. devtest prediction): 12.0 min; ~230 rows/s between evals |
| Qwen3.5-9B rows, PyTorch fallback (L0a) | 7.94B (+40.1M trainable) | ~12.8 rows/s steady | 17.7 GB alloc, 19.9 GB reserved | micro_bs 8 x accum 4; ~25 s per 10 steps (320 rows) -> ~32 min per epoch steady state (the script's 48 min estimate includes warm-up) |
| Qwen3.5-9B joint, PyTorch fallback (L0b) | same | ~920 to 1,080 tok/s | 18.2 GB alloc, 20.9 GB reserved | micro_bs 4 x accum 8 (32 prompts per step), token-budget batches + 22 GB cap: no spill. Script estimate 32 min per epoch |
| Qwen3.5-9B rows, **fast kernels** (L0a re-run) | same | ~17.8 rows/s steady | 17.1 GB alloc, 17.6 GB reserved | triton-windows 3.3.1.post21 + fla-core 0.5.2: kernel check rel. diff 0.0044 (output) / 0.0045 (grad), x3.6 on the DeltaNet op. Steady ~18 s per 10 steps (vs ~25) -> ~23 min per epoch; first 10 steps 87 s are Triton compile/autotune, which inflates the script's 59 min estimate |

Rows/s read from the 50-step timestamps in `runs/smoke_*/train.log` (includes CUDA warm-up and one 600-row eval), so treat as rough.
Env (2026-11-01): Python 3.12.8 venv, torch 2.7.1+cu126, transformers 4.57.6, lock written to `requirements.lock.txt`.
Smoke DS values (0.00 mDeBERTa, 0.05 XLM-R, 0.44 after bias) are meaningless: 100 steps, mostly still predicting Safe. Pipeline incl. `runs/smoke_sub.zip` works.

## Runs
| ID | Date | Model | Key flags | Seed | Eval on | DS | F1 | USR | bn | ro | cm | Min | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0-cv | 2026-11-01 | TF-IDF + LR | char 2-5 + word 1-2, pooled, bias | - | 5-fold CV train | 0.719 | 0.726 | 0.009 | | | | 3 | pooling +2.0 pts, bias +0.9 pts |
| B0 | 2026-11-01 | TF-IDF + LR | same, fit on train, pooled, no bias | - | dev | 0.7016 | 0.7280 | 0.0363 | 0.728 | 0.728 | 0.728 | 0.8 | row-level DS 0.6812 (bn 0.723, ro 0.712, cm 0.704); pooling +2.0 pts again |
| B0+bias | 2026-11-01 | TF-IDF + LR | bias tuned on full dev [-0.95,-0.7,0,0] | - | dev (in-sample) | 0.7102 | 0.7177 | 0.0104 | | | | | in-sample only; held-out half gain -0.14 pts (5 splits, seeds 0-4) and -0.86 pts (seeds 100-104), so not used |
| E1 | 2026-11-01 | xlm-roberta-large | cons_w 0, ep 4, lr 1e-5, llrd 0.9, bs 8 prompts, len 256 | 42 | dev | 0.7713 | 0.7960 | 0.0311 | 0.796 | 0.796 | 0.796 | 12.0 | best at step 3584 (ep 3.5) of 8 dev evals, so slightly optimistic; 13.0 GB. Row-level: DS 0.7329, F1 bn 0.791 / ro 0.724 / cm 0.789 (gap 0.067), all 3 forms agree on 76.2% of prompts (B0 77.4%). Pooling +3.8 pts. Main error: Harmful vs Severe (88+80 of 1037). +7.0 pts DS over B0 |
| E2 | 2026-11-01 | xlm-roberta-large | cons_w 1.0, rest as E1 | 42 | dev | 0.7716 | 0.8006 | 0.0363 | 0.801 | 0.801 | 0.801 | 12.9 | best at last eval (step 4096). Pooled DS vs E1 +0.0004 (paired bootstrap 95% CI -0.0097 to +0.0109): no gain. Row-level DS 0.7371 (E1 0.7329), row F1 bn 0.797 / ro 0.730 / cm 0.790, gap 0.067 (unchanged). Forms agree 78.7% (E1 76.2%), mean pairwise TV 0.124 (E1 0.145). Row DS higher than E1 at 7 of 8 matched evals. E1+E2 logit average: 0.7719 |

| L0a | 2026-11-01 | Qwen3.5-9B LoRA r16, rows | 300 prompts, 30 steps, label-word init | 42 | 200 dev prompts | 0.5971 (fallback) / 0.6499 (fast-kernel re-run) | 0.630 / 0.692 | 0.053 / 0.060 | | | | 2 | smoke only. Step 0 (zero-shot head) DS 0.18, predicts mostly Severe, so the label-word init gives no useful zero-shot start; SafeFPR 0.22 / 0.21 |
| L0b | 2026-11-01 | Qwen3.5-9B LoRA r16, joint (3 views, view dropout 0.3) | 300 prompts, 30 steps | 42 | 200 dev prompts | **0.7463** | 0.758 | 0.015 | | | | 3.6 | smoke only, but far ahead of rows mode at the same 30 steps (0.65); recall S/O/H/Sv 0.76/0.72/0.80/0.72, SafeFPR 0.24. Noise on 200 prompts is about +-3 pts |

| L2 | 2026-11-01 | Qwen3.5-9B LoRA r16, joint, view dropout 0.3 | 2 epochs (514 steps), micro_bs 4 x accum 8, lr 2e-4, fast kernels | 42 | dev | **0.7925** | 0.8281 | 0.0429 | | | | 46 train (52 total) | best at the LAST step; evals 0.685 / 0.766 / 0.761 / 0.791 / 0.7925, so not converged and noisy mid-run (step 384: Severe recall fell to 0.65). Safe FPR 0.105 (XLM-R ~0.18) but USR 0.043: leans Safe. Recall S/O/H/Sv 0.90/0.86/0.78/0.78. Single view: Bangla-only 0.7912, code-mixed-only 0.7664, romanized-only 0.6986, so the joint model mostly relies on the Bangla view. vs E1: L2 right & E1 wrong 8.4% of prompts, E1 right & L2 wrong 4.8%. devtest 3.7 min, 17.6 GB |
| ENS1 | 2026-11-01 | L2 + E1 + E2 + B0, log-prob mean, pooled | equal weights, no bias | - | dev | 0.8031 | 0.8238 | 0.0252 | | | | | Safe FPR 0.150. decision_lab (5x5 CV): calibrated log-prob +0.26 (P 0.86); calibrated + Safe bias +0.52 (P 0.91) in CV, but decide.py's 5 held-out halves give the Safe bias -0.03, so not adopted; NLL-weighted -0.74 (P 0.21); stacking -2.2. Missing views with the ensemble: romanized-only 0.780 (was 0.726 without L2) |
| ENS1-cal | 2026-11-01 | same, per-model temperature calibration (T = 0.93 / 1.12 / 1.01 / 1.25) | no bias | - | dev | 0.8048 | 0.8256 | 0.0252 | | | | | Safe FPR 0.146; candidate for the next submission (decide.py --calibrate) |

## Submissions
| ID | Date | Phase | Built from | Local DS | Leaderboard DS | Notes |
|---|---|---|---|---|---|---|
| S00_B0 | 2026-11-01 | dev | B0, pooled, bias | 0.7102 (in-sample) | not submitted | devtest pred dist 11.5/22.8/33.3/32.4 % |
| S00b_B0_nobias | 2026-11-01 | dev | B0, pooled, no bias | 0.7016 | 0.6386 (F1 0.6704, USR 0.0475), rank 17 | dev to devtest gap -6.3 pts DS (F1 -5.8, USR +1.1). devtest pred dist 17.1/26.4/28.3/28.2 % (dev pred: 27.8/18.7/25.7/27.7 %) |
| S01_E1_nobias | 2026-11-01 | dev | E1, pooled, no bias (dev bias held-out gain -0.50 pts over 5 splits) | 0.7713 | 0.7061 (F1 0.737, USR 0.0419), rank 13 | gap -6.5 pts DS (F1 -5.9, USR +1.1), same as B0. Gain over B0: +6.8 LB vs +7.0 dev, so dev ranks models reliably. devtest pred dist 19.4/24.6/29.1/26.9 % |
| S02_L2 | 2026-11-01 | dev | L2 alone, pooled, no bias | 0.7925 | **0.7622 (F1 0.7948, USR 0.0411)** | dev to devtest gap only -3.0 |
| S03_L2_E1_E2_B0 | 2026-11-01 | dev | L2+E1+E2+B0 equal weights, no bias | 0.8031 | 0.7525 (F1 0.7761, USR 0.0304), rank 7 | gap -5.1: the encoders drag the ensemble down on devtest |
| S04_L2_safe06 | 2026-11-01 | dev | L2 alone + fixed Safe bias 0.6 | 0.8024 (Safe FPR 0.129) | 0.7656 (F1 0.7896, USR 0.0304) | +0.34 over S02; USR 4.1% -> 3.0%, F1 -0.5 |
| S04_L2_safe10 | 2026-11-01 | dev | L2 alone + fixed Safe bias 1.0 | 0.8045 (Safe FPR 0.140) | **0.7664 (F1 0.7846, USR 0.0232), current best** | +0.42 over S02; USR -> 2.3%, F1 -1.0. Bias direction confirmed on devtest, but the gain flattens between 0.6 and 1.0: do not tune the bias further on the leaderboard |

## Decisions
| Date | Decision | Why |
|---|---|---|
| 2026-11-01 | Method = TriGuard (see APPROACH.md) | baseline evidence for pooling + bias; code-mixed needs multilingual encoders |
| 2026-11-01 | Consistency loss at cons_w 1.0 does not improve pooled DS (single seed); keep testing in 2.3 before it carries the paper | E2 vs E1: forms agree more (+2.5 pts) and row-level DS +0.4, but pooling at inference already averages the forms, and the romanized gap (0.067) did not shrink. Needs cons_w sweep and seeds |
| 2026-11-01 | Use local dev DS to compare models; expect leaderboard DS about 6.5 pts lower | B0 and E1 lose the same 6.3 to 6.5 pts on devtest and keep their dev ordering (+7.0 dev vs +6.8 LB), so the shift is in the data, not the model |
| 2026-11-01 | Do not tune the bias on dev alone; use it only when the held-out-half gain is positive (decide.py now repeats the check over 5 random halves and warns) | on B0 the dev-tuned bias lowered held-out DS; 2,049 dev prompts are too few. Retry on train+dev OOF logits (step 4.4) |
| 2026-11-01 | LLM teacher = original Qwen3.5-9B (`H:\Abliterate\Qwen3.5-9B`), separate `.venv-llm` with transformers 5.20.2 | 14-pt gap to #1 is capability; transformers 4.57.6 cannot load qwen3_5; keep encoder env stable |
| 2026-11-01 | Decision layer: keep current pipeline until OOF study; candidates are calibrated geometric ensemble and NLL-weighted arithmetic ensemble; stacking rejected | dev study below: no change beats dev noise (SD 1.0 pt) except stacking, which is clearly worse |
| 2026-11-01 | Every result reports DS, macro-F1, USR, Safe FPR, per-class recall | review: a DS gain via more false alarms must be visible |
| 2026-11-01 | **Dev overrates the encoders under the devtest shift**: dev-to-devtest gap is -6.3 (B0), -6.5 (E1), -5.1 (ENS1) but only -3.0 for Qwen L2. Qwen alone beats the ensemble on the leaderboard (0.7622 vs 0.7525) although dev says the opposite. From now on: LLM-first; ensemble members and weights are judged with this shift in mind (encoders at most a small weight, checked on the leaderboard with few submissions); further encoder runs (E4 to E6) deprioritised | S02 vs S03 |
| 2026-11-01 | Default decision for single-LLM submissions: Safe bias in 0.6 to 1.0 (use 1.0 until OOF says otherwise) | S04: both biases beat no bias on devtest (+0.34 / +0.42), plateau between them |
| 2026-11-01 | Next LLM run L3 = 3 epochs + view dropout 0.5 (same as L2 otherwise) | L2 best at its last step (under-trained); single-view results show over-reliance on the Bangla view (romanized-only 0.70) |
| 2026-11-01 | decide.py gets --calibrate (adopted) and --safe_bias_only (available, not adopted: held-out gain -0.03 on ENS1) | calibration is consistent and free; bias gains do not survive held-out checks on dev |
| 2026-11-01 | Run joint mode first (L2 before L1); keep fast kernels | L0b joint reached 0.746 vs 0.650 for rows after the same 30 steps and is cheaper per epoch; fla kernels match the reference and give ~1.4x end-to-end |
| 2026-11-01 | Drop consistency-weight sweep (old 2.3); distillation uses cross-view teacher targets | E2: no pooled gain; cross-view targets carry the tri-view benefit into a single-view student |

## Analysis 2026-11-01 (after E2): where the remaining points are
Leaderboard (dev phase, Nov 1): #1 84.48, #3 82.43, #5 79.65, #10 73.50. Our best LB: E1 70.61. Gap to #1: about 14 pts, so the bottleneck is model capability, not the decision layer or the loss.

Findings (dev logits of E1, E2, B0; devtest logits of E2):
- Consistency loss: no pooled gain (E2 vs E1 +0.0003). Pooling at inference already does the averaging. Keep as a paper ablation, do not invest further in loss variants for now.
- Ensembling more of the same helps little (E1+E2 pooled 0.7719 vs 0.7716). Diversity helps: E1+E2 + 0.5 x TF-IDF = 0.7790. Honest check (tune weight/pooling/bias on half of dev, score other half, 200 splits): mean gain +0.65 pts, positive in 89% of splits.
- Probability-mean pooling beats log-prob-mean pooling slightly (0.7736 vs 0.7719).
- A single Safe-only bias (subtract 0.2 to 0.4 from the Safe logit) is robust: +0.3 pts on dev and +1.0 pts in a simulation of devtest-like class priors. The 4-parameter bias overfits (as found before).
- Devtest differs from dev in source mix, not just labels: native "bhai/bro" style prompts 0.2% vs 1.2% on dev, Latin-heavy Bangla 0.7% vs 4.5%, shorter, fewer questions; slightly further from train (char-tfidf nearest-neighbour similarity median 0.322 vs 0.337); E2 less confident (mean max-prob 0.741 vs 0.776) and forms disagree more (26.3% vs 21.3%).
- EM label-shift estimate for devtest: Safe about 14 to 19%, Offensive about 25 to 30% (dev: 34% / 15%). But simulating pure label shift on dev predicts only a small drop, so most of the 6.5-pt dev to devtest gap is harder / different prompts, which a stronger model must close.
- Dev errors (E1+E2 pooled, prompts): Harmful<->Severe 165, Safe->unsafe 129, unsafe->Safe 43. Per-class F1: Safe 0.869, Offensive 0.828, Harmful 0.750, Severe 0.743.

## Decision-layer study 2026-11-01 (dev, E1 + E2 + B0, `decision_lab.py`, 5 x 5-fold grouped CV, learned parts fit on training folds)
Safe FPR = share of Safe prompts flagged unsafe. Paired bootstrap (2,000 resamples of prompts) vs the original pipeline (geometric ensemble, mean pooling, argmax).

| Variant | DS | macro-F1 | USR | Safe FPR | diff (pts) | 95% CI | P(better) |
|---|---|---|---|---|---|---|---|
| original: log-prob mean, mean pooling, argmax | 0.7755 | 0.7974 | 0.027 | 0.196 | | | |
| arithmetic prob mean | 0.7757 | 0.7975 | 0.027 | 0.199 | +0.03 | -0.5, +0.6 | 0.53 |
| NLL-weighted prob (w E1/E2/B0 = 0.3/0.5/0.2) | 0.7803 | 0.8033 | 0.029 | 0.185 | +0.36 | -0.5, +1.3 | 0.78 |
| calibrated log-prob (T = 1.12/1.01/1.25) | 0.7785 | 0.7992 | 0.026 | 0.195 | +0.30 | -0.1, +0.7 | 0.92 |
| Safe-only bias | 0.7797 | 0.7956 | 0.020 | 0.215 | +0.44 | -0.3, +1.2 | 0.86 |
| 4-class bias | 0.7762 | 0.7951 | 0.024 | 0.209 | | | |
| Safe threshold | 0.7706 | 0.7906 | 0.025 | 0.220 | | | |
| confidence-weighted pooling | 0.7766 | 0.7991 | 0.028 | 0.196 | | | |
| learned per-form weights | 0.7686 | 0.7948 | 0.033 | 0.200 | | | |
| stacking meta-classifier | 0.7476 | 0.7989 | 0.064 | 0.139 | -2.60 | -4.0, -1.3 | 0.00 |

Missing views (mean pooling over available views): all 3 0.776 | bn+cm 0.770 | bn+ro 0.769 | ro+cm 0.764 | bn only 0.777 | cm only 0.767 | ro only 0.726. Bangla-only vs 3-view pooling: +0.16 (CI -1.0, +1.3): pooling lowers USR, Bangla-only has higher F1; not resolved on dev.
Per-class recall (original): Safe 0.80, Offensive 0.87, Harmful 0.77, Severe 0.78. DS bootstrap SD on dev: 1.0 pt.
