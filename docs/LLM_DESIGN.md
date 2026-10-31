# Design: Qwen3.5-9B safety classifier (L1) for TriGuard

Status: **reviewed 2026-11-01**; implementation is Phase 3 of `INSTRUCTIONS.md`. Checkpoint confirmed as the **original** Qwen3.5-9B release (the abliterated copy is in a different directory and is not used for L1).

## 1. What the checkpoint actually is (`H:\Abliterate\Qwen3.5-9B`)
- `Qwen3_5ForConditionalGeneration`, model_type `qwen3_5`, 19.3 GB in bf16 (about 9.65B params incl. vision).
- Text model: 32 layers, hidden 4096, MLP 12288. **Hybrid attention**: 24 Gated DeltaNet (linear attention) layers + 8 full-attention layers (every 4th). Vocab 248,320 (embedding alone is about 1B params).
- Extra parts we do not need for classification: vision tower (27 blocks, about 0.4B), `lm_head` (about 1B, untied), MTP head.
- **transformers 4.57.6 (current venv) cannot load it.** Support exists in transformers 5.x, which has `Qwen3_5ForSequenceClassification` and `Qwen3_5TextForSequenceClassification`.
- Gated DeltaNet uses optional fast kernels (`flash-linear-attention`, `causal-conv1d`). Without them transformers falls back to a pure PyTorch implementation: works on Windows, slower.
- Confirmed: original Qwen release (not abliterated).

## 2. Environment: separate venv, existing one untouched
`.venv-llm` with torch 2.7.1+cu126 (same as now) plus:
`transformers==5.20.2` (2026-10-19, has Qwen3.5 incl. sequence classification), `peft==0.21.2`, `accelerate==1.15.0`, `bitsandbytes==0.50.2` (only for the 4-bit fallback). Verified to resolve for Windows / Python 3.12. The encoder pipeline keeps `.venv` on transformers 4.57.6 so nothing that works today breaks.

## 3. Fitting on the RTX 4090 (24 GB)
Load **text-only** (`Qwen3_5TextForSequenceClassification`, weights remapped from `model.language_model.*`), dropping vision, `lm_head` and MTP.

| Item | bf16 LoRA (plan A) | 4-bit QLoRA (plan B) |
|---|---|---|
| Frozen text backbone (about 8B) | ~16 GB | ~5.5 GB |
| LoRA r=16 on all linear layers + Adam states | ~0.6 GB | ~0.6 GB |
| Activations (grad checkpointing, 12 rows x <=256 tokens per micro-batch) | ~2.5 to 4 GB | ~2.5 to 4 GB |
| **Total** | **~19 to 21 GB** | **~9 to 11 GB** |

Plan A fits with a small micro-batch plus gradient accumulation (effective 32 rows). If the 4090 also drives your monitors (Windows reserves ~1 GB), or the PyTorch fallback for DeltaNet uses more memory than estimated, switch to plan B with one flag. Length-bucketed batches to avoid padding waste.

Speed (rough): about 3 to 4M tokens per epoch, roughly 1 to 2 hours per epoch with the PyTorch DeltaNet fallback. 2 epochs per run. If too slow: WSL2 (Ubuntu) lets `flash-linear-attention` and `causal-conv1d` install cleanly, a large speed-up for the 24 DeltaNet layers.

## 4. Architecture choices that should actually move the score
1. **Instruction-framed input** (cheap, usually several points for LLM classifiers). Each input is wrapped as:
   `Classify the safety risk of this user prompt as Safe, Offensive, Harmful or Severe.\nPrompt: {text}\nRisk:`
   The last token then summarises the text with the task in view, instead of being an arbitrary end of sentence.
2. **Label-word head initialisation** (cheap, faster and more stable training). The 4-way score head starts from the pretrained `lm_head` rows of the tokens " Safe", " Offensive", " Harmful", " Severe" instead of random weights, so step 0 already behaves like a zero-shot classifier.
3. **Joint multi-view input with view dropout** (the LLM version of TriGuard). One input shows all three forms of a prompt (in random order during training); each form is dropped with probability 0.3 (never all three). This lets attention cross-check the romanized form against the Bangla one before deciding, and turns every training prompt into several training examples. At test time: the joint input with all three forms, ensembled with per-row predictions that are pooled. Ablation: per-row mode only (like E1).
4. **Loss**: CE with label smoothing 0.05, plus an optional **unsafe-mass penalty** `lambda_u * -log(1 - p_Safe)` on unsafe rows, which targets the USR multiplier directly. No consistency term (it did not help in E2), no R-Drop (doubles cost on a 9B model).
5. **Layer truncation ablation** (lightweight and interpretable). Safety signals often peak in middle layers. Option `--keep_layers k` keeps only the first k of 32 layers (must end on a full-attention layer: 16, 20, 24, 28 or 32). Fewer layers = less VRAM and faster, and if 24 layers match 32 it is a nice finding for the paper (and close to the layer-pruning work).

## 5. Output contract
Same as every other model: `runs/L1_.../val_logits.csv` and `devtest_logits.csv` via `bk_common.write_logits`, plus `summary.json`. LoRA adapter saved in `runs/L1_.../adapter/`. `decide.py` ensembles it with E1/E2/B0 unchanged.

## 6. Optional paper extra: original vs abliterated
L1 uses the original weights. The abliterated copy (refusal direction removed) exists locally, so one extra run with identical settings would answer a small interpretability question: does removing the refusal direction hurt cross-script harm detection? Only if time allows after L1 to L4.

## 7. How L-runs plug into the rest of TriGuard
- Output logits go through the same decision layer (calibration, ensembling, view pooling) chosen in Phase 4 with `decision_lab.py`; no LLM-specific decision rule.
- In joint mode the model produces one distribution per prompt; it is written to all three rows, so pooling leaves it unchanged and it ensembles with row-mode models at prompt level.
- The best L run is the main teacher for cross-view distillation (APPROACH C5).
- Report for every L run: DS, macro-F1, USR, Safe FPR, per-class recall, plus single-view results (Bangla only, romanized only, code-mixed only) for the missing-view table.

## 8. Run plan
| ID | Setting | Purpose |
|---|---|---|
| L0 | 300 prompts, 30 steps, plan A | VRAM, speed, loss decreases |
| L1 | rows mode, instruction input, label-word init, 2 epochs, train -> dev | main LLM result |
| L2 | joint + view dropout | multi-view LLM |
| L3 | L1 or L2 + unsafe-mass penalty | metric-aligned loss |
| L4 | best of above with `--keep_layers 24` | lightweight variant |

Then ensemble with E1/E2/B0 and move to 5-fold train+dev for the final system.
