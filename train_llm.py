"""Qwen3.5-9B LoRA safety classifier for BanglaKnight (TriGuard teacher L1 to L4).

Design: docs/LLM_DESIGN.md. Runs in .venv-llm (transformers 5.x), not in .venv.

What it does
- Loads only the text part of the Qwen3.5 checkpoint (vision tower, lm_head and MTP head are skipped).
- LoRA on all linear layers of the full-attention, Gated DeltaNet and MLP blocks; a small fp32 score head
  reads the last real token. The head is initialised from the lm_head rows of the label words
  " Safe", " Offensive", " Harmful", " Dangerous" (zero-shot start; "Dangerous" = the Severe class).
- Instruction-framed input. Two modes:
    rows  : one example per row (form), like the encoders; pooled over forms later by decide.py
    joint : one example per prompt with all three forms; random order + view dropout while training
- Loss: CE (label smoothing) + optional unsafe-mass penalty  unsafe_w * -log(1 - p_Safe)  on unsafe examples.
- --keep_layers k keeps only the first k decoder layers (layer-truncation ablation).
- --load_4bit for QLoRA if bf16 does not fit.
- Writes val_logits.csv / <split>_logits.csv (bk_common format), summary.json, train.log, best adapter + head.

Examples (PowerShell, .venv-llm active):
  python train_llm.py --model H:/Abliterate/Qwen3.5-9B --train data/train.jsonl --dev data/dev.jsonl --max_prompts 300 --max_steps 30 --out runs/L0_smoke
  python train_llm.py --model H:/Abliterate/Qwen3.5-9B --train data/train.jsonl --dev data/dev.jsonl --predict data/devtest.jsonl --mode rows --out runs/L1_qwen35_rows_s42
"""
import argparse
import json
import math
import os
import random
import re
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import recall_score

from bk_common import FORMS, defense_score, group_mean, load_rows, log_softmax, make_logger, write_logits

# Head-init anchors: first token of each word. " Severe" splits into " Se"+"vere" in the Qwen tokenizer,
# so the severe class is named "Dangerous" (single token); " Harmful" -> " Harm" (first token, meaningful).
LABEL_WORDS = [" Safe", " Offensive", " Harmful", " Dangerous"]
FORM_TAG = {"bangla": "Bangla script", "romanized": "Romanized", "codemixed": "Code-mixed"}
TASK = ("Classify the safety risk of this user prompt as Safe, Offensive (insults, no request for harm), "
        "Harmful (asks for help with low or medium risk harm) or Dangerous (asks for help with severe, high-risk harm).")
ROW_TEMPLATE = TASK + "\nPrompt: {text}\nRisk:"
JOINT_HEAD = TASK + " The same prompt is shown in up to three written forms.\n"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",            # full attention
                "in_proj_qkv", "in_proj_z", "out_proj",            # Gated DeltaNet
                "gate_proj", "up_proj", "down_proj"]               # MLP


# ----------------------------------------------------------------------------- model
def load_backbone(path, keep_layers, load_4bit, grad_ckpt, log):
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    cfg = AutoConfig.from_pretrained(path)
    tcfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
    if keep_layers:
        assert tcfg.layer_types[keep_layers - 1] == "full_attention", \
            f"--keep_layers must end on a full-attention layer, valid: {[i + 1 for i, t in enumerate(tcfg.layer_types) if t == 'full_attention']}"
        tcfg.num_hidden_layers = keep_layers
        tcfg.layer_types = tcfg.layer_types[:keep_layers]
    kw = dict(config=tcfg, dtype=torch.bfloat16, key_mapping={r"^model\.language_model\.": ""})
    if torch.cuda.is_available():
        kw["device_map"] = {"": 0}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16,
                                                       bnb_4bit_use_double_quant=True)
    t0 = time.time()
    model = Qwen3_5TextModel.from_pretrained(path, **kw)
    log(f"loaded text backbone: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params, "
        f"{tcfg.num_hidden_layers} layers, {time.time() - t0:.0f}s")
    n_loaded = len(model.layers)
    assert n_loaded == tcfg.num_hidden_layers, f"expected {tcfg.num_hidden_layers} layers, got {n_loaded}"
    if load_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=grad_ckpt,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    elif grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    return model, tcfg


def label_word_rows(path, tok, hidden):
    """Rows of the pretrained lm_head for the first token of each label word (fp32, [4, hidden])."""
    from safetensors import safe_open
    ids = [tok(w, add_special_tokens=False)["input_ids"][0] for w in LABEL_WORDS]
    idx_file = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx_file):
        with open(idx_file, encoding="utf-8") as f:
            shard = json.load(f)["weight_map"].get("lm_head.weight")
        files = [os.path.join(path, shard)] if shard else []
    else:
        files = [os.path.join(path, "model.safetensors")]
    for fp in files:
        with safe_open(fp, framework="pt") as f:
            if "lm_head.weight" in f.keys():
                W = f.get_slice("lm_head.weight")
                return torch.stack([W[i:i + 1][0] for i in ids]).float(), ids
    return None, ids


class SafetyClassifier(nn.Module):
    def __init__(self, backbone, hidden, init_rows=None):
        super().__init__()
        self.backbone = backbone
        self.score = nn.Linear(hidden, 4, bias=True)
        nn.init.zeros_(self.score.bias)
        if init_rows is not None:
            with torch.no_grad():
                self.score.weight.copy_(init_rows)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        last = attention_mask.sum(1) - 1                      # right padding: last real token
        pooled = h[torch.arange(h.size(0), device=h.device), last]
        return self.score(pooled.float())


# ----------------------------------------------------------------------------- data
def prompt_groups(rows):
    d = OrderedDict()
    for r in rows:
        d.setdefault(r["prompt_id"], {})[r["form"]] = r
    return list(d.values())


def joint_text(group, forms):
    body = "".join(f"[{FORM_TAG.get(f, f)}] {group[f]['text']}\n" for f in forms)
    return JOINT_HEAD + body + "Risk:"


def make_examples(rows, mode, train, view_dropout, rng):
    """Returns list of (text, label, row_indices_to_write)."""
    index = {r["id"]: i for i, r in enumerate(rows)}
    if mode == "rows":
        return [(ROW_TEMPLATE.format(text=r["text"]), r["label"], [i]) for i, r in enumerate(rows)]
    out = []
    for g in prompt_groups(rows):
        forms = [f for f in FORMS if f in g]
        if train:
            forms = forms[:]
            rng.shuffle(forms)
            keep = [f for f in forms if rng.random() >= view_dropout]
            forms = keep or [rng.choice(forms)]
        label = next(iter(g.values()))["label"]
        out.append((joint_text(g, forms), label, [index[r["id"]] for r in g.values()]))
    return out


def clip_texts(rows, tok, mode, max_len):
    """Clip each row's text to a token budget so the instruction and the final "Risk:" always survive
    (plain right-truncation would cut off the end of the prompt, which is what the head reads)."""
    overhead = len(tok(JOINT_HEAD + "Risk:" if mode == "joint" else ROW_TEMPLATE.format(text=""))["input_ids"]) + (40 if mode == "joint" else 12)
    cap = (max_len - overhead) // (3 if mode == "joint" else 1)
    n = 0
    for r in rows:
        ids = tok(r["text"], add_special_tokens=False)["input_ids"]
        if len(ids) > cap:
            r["text"] = tok.decode(ids[:cap]); n += 1
    return n, cap


def batches(examples, tok, max_len, bs, shuffle, rng, max_tokens=0):
    """Length-bucketed batches. A batch holds at most `bs` examples and, if max_tokens > 0, at most
    max_tokens padded tokens (n_examples x longest example), so long joint inputs get smaller batches."""
    order = list(range(len(examples)))
    if shuffle:
        rng.shuffle(order)
    lens = [min(max_len, len(ids)) for ids in tok([e[0] for e in examples])["input_ids"]] if examples else []
    chunk = bs * 50
    out = []
    for s in range(0, len(order), chunk):
        part = sorted(order[s:s + chunk], key=lambda i: lens[i])
        cur = []
        for i in part:
            if cur and (len(cur) >= bs or (max_tokens and (len(cur) + 1) * lens[i] > max_tokens)):
                out.append(cur); cur = []
            cur.append(i)
        if cur:
            out.append(cur)
    if shuffle:
        rng.shuffle(out)
    for b in out:
        enc = tok([examples[i][0] for i in b], truncation=True, max_length=max_len, padding=True,
                  padding_side="right", return_tensors="pt")
        y = torch.tensor([-100 if examples[i][1] is None else examples[i][1] for i in b])
        yield b, enc, y


# ----------------------------------------------------------------------------- eval
def full_metrics(rows, L):
    y = np.array([r["label"] for r in rows]); forms = [r["form"] for r in rows]
    agg_lp = group_mean(log_softmax(L), [r["prompt_id"] for r in rows])
    p = agg_lp.argmax(1)
    m = defense_score(y, p, forms)
    m["safe_fpr"] = float(np.mean(p[y == 0] != 0)) if (y == 0).any() else 0.0
    rec = recall_score(y, p, average=None, labels=[0, 1, 2, 3], zero_division=0)
    m.update({f"recall_{c}": float(r) for c, r in zip(["safe", "off", "harm", "sev"], rec)})
    raw = defense_score(y, log_softmax(L).argmax(1), forms)
    m["row_ds"] = raw["ds"]
    m.update({f"row_f1_{k[3:]}": v for k, v in raw.items() if k.startswith("f1_")})
    return m


@torch.no_grad()
def predict(model, tok, rows, mode, max_len, bs, device, only_form=None, max_tokens=0):
    model.eval()
    use = [r for r in rows if only_form is None or r["form"] == only_form] if mode == "joint" else rows
    ex = make_examples(use, mode, False, 0.0, random.Random(0))
    index = {r["id"]: i for i, r in enumerate(rows)}
    remap = [index[r["id"]] for r in use]
    L = np.zeros((len(rows), 4), dtype=np.float32)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for b, enc, _ in batches(ex, tok, max_len, bs, False, None, max_tokens):
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = model(enc["input_ids"], enc["attention_mask"]).float().cpu().numpy()
        for i, l in zip(b, lg):
            for ri in ex[i][2]:
                L[remap[ri]] = l
    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return L


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local path of the original Qwen3.5-9B")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--dev", nargs="*", default=[])
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--predict", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["rows", "joint"], default="rows")
    ap.add_argument("--view_dropout", type=float, default=0.3, help="joint mode, training only")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max_steps", type=int, default=0, help="stop after N optimizer steps (smoke tests)")
    ap.add_argument("--lr", type=float, default=2e-4, help="LoRA learning rate")
    ap.add_argument("--head_lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--micro_bs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--eval_bs", type=int, default=32)
    ap.add_argument("--eval_tokens", type=int, default=8192, help="max padded tokens per eval batch")
    ap.add_argument("--train_tokens", type=int, default=4096, help="max padded tokens per training micro-batch")
    ap.add_argument("--vram_cap_gb", type=float, default=22.0,
                    help="hard cap for PyTorch GPU memory: raise out-of-memory instead of Windows silently spilling into shared memory")
    ap.add_argument("--max_len", type=int, default=0, help="0 = 320 (rows) / 768 (joint)")
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--unsafe_w", type=float, default=0.0, help="weight of -log(1 - p_Safe) on unsafe examples")
    ap.add_argument("--no_label_init", action="store_true", help="random score head (ablation)")
    ap.add_argument("--keep_layers", type=int, default=0, help="keep only the first k layers (16/20/24/28/32)")
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--no_grad_ckpt", action="store_true")
    ap.add_argument("--evals_per_epoch", type=int, default=2)
    ap.add_argument("--eval_at_start", action="store_true", help="evaluate before training (zero-shot head check)")
    ap.add_argument("--eval_single_views", action="store_true", help="joint mode: also write one-view-only dev logits")
    ap.add_argument("--max_prompts", type=int, default=0)
    ap.add_argument("--max_val_prompts", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow_cpu", action="store_true")
    a = ap.parse_args()
    if not a.max_len:
        a.max_len = 320 if a.mode == "rows" else 768
    os.makedirs(a.out, exist_ok=True)
    log = make_logger(os.path.join(a.out, "train.log"))
    log("args:", json.dumps(vars(a)))
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rng = random.Random(a.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not a.allow_cpu:
        raise SystemExit("No CUDA GPU visible. Use .venv-llm (CUDA torch) or --allow_cpu for a tiny test.")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        total = torch.cuda.get_device_properties(0).total_memory
        frac = min(0.98, a.vram_cap_gb * 2**30 / total)
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        free, _ = torch.cuda.mem_get_info()
        log(f"GPU: {torch.cuda.get_device_name(0)} | free {free / 2**30:.1f} of {total / 2**30:.1f} GB | PyTorch capped at {frac * total / 2**30:.1f} GB")
        try:
            import fla  # noqa: F401
            log("fast Gated DeltaNet kernels: flash-linear-attention", getattr(fla, "__version__", ""))
        except Exception:
            log("fast Gated DeltaNet kernels: not installed (PyTorch fallback; see install_fast_kernels.ps1)")

    # ---------------- data
    rows = [r for p in a.train for r in load_rows(p)]
    val_rows = [r for p in a.dev for r in load_rows(p)]
    if a.fold >= 0:
        from sklearn.model_selection import GroupKFold
        pids = np.array([r["prompt_id"] for r in rows])
        _, va = list(GroupKFold(a.folds).split(rows, groups=pids))[a.fold]
        vs = set(pids[va])
        val_rows = [r for r in rows if r["prompt_id"] in vs]
        rows = [r for r in rows if r["prompt_id"] not in vs]
    def subsample(rs, n, seed):
        if not n:
            return rs
        keep = sorted({r["prompt_id"] for r in rs}); random.Random(seed).shuffle(keep); keep = set(keep[:n])
        return [r for r in rs if r["prompt_id"] in keep]
    rows = subsample(rows, a.max_prompts, a.seed)
    val_rows = subsample(val_rows, a.max_val_prompts or (max(200, a.max_prompts // 2) if a.max_prompts else 0), a.seed)
    log(f"train rows={len(rows)} prompts={len({r['prompt_id'] for r in rows})} | val rows={len(val_rows)} | mode={a.mode}")

    # ---------------- model
    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    lens = [len(tok(ex[0])["input_ids"]) for ex in make_examples(rows[:3000], a.mode, False, 0, rng)]
    log(f"token length before clipping (sample): p50={int(np.percentile(lens, 50))} p95={int(np.percentile(lens, 95))} "
        f"max={max(lens)} | over {a.max_len}: {np.mean(np.array(lens) > a.max_len) * 100:.1f}%")
    for name, rs in [("train", rows), ("val", val_rows)]:
        n, cap = clip_texts(rs, tok, a.mode, a.max_len)
        log(f"clipped {n} {name} rows to {cap} tokens per form")

    backbone, tcfg = load_backbone(a.model, a.keep_layers, a.load_4bit, not a.no_grad_ckpt, log)
    if device.type == "cpu":
        backbone = backbone.float()
    lcfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                      target_modules=LORA_TARGETS, bias="none")
    backbone = get_peft_model(backbone, lcfg)
    init_rows, label_ids = (None, None) if a.no_label_init else label_word_rows(a.model, tok, tcfg.hidden_size)
    log(f"label-word head init: {'off' if init_rows is None else 'on'} (token ids {label_ids})")
    model = SafetyClassifier(backbone, tcfg.hidden_size, init_rows).to(device)
    model.score.float()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"trainable params: {n_tr / 1e6:.1f}M (LoRA + head)")

    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("score.")]
    opt = torch.optim.AdamW([{"params": lora_params, "lr": a.lr, "weight_decay": a.wd},
                             {"params": model.score.parameters(), "lr": a.head_lr, "weight_decay": 0.0}],
                            fused=device.type == "cuda")
    n_train_ex = len(make_examples(rows, a.mode, False, 0, rng))
    n_micro = sum(1 for _ in batches(make_examples(rows, a.mode, False, 0, rng), tok, a.max_len, a.micro_bs,
                                     False, None, a.train_tokens))
    steps_per_epoch = math.ceil(n_micro / a.grad_accum)
    total = a.max_steps or int(steps_per_epoch * a.epochs)
    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(a.warmup * total)), total)
    eval_every = total if a.max_steps else max(1, steps_per_epoch // a.evals_per_epoch)
    log(f"steps: {total} total ({steps_per_epoch}/epoch), micro_bs {a.micro_bs} x accum {a.grad_accum}, eval every {eval_every}")

    # ---------------- train
    best, best_state, hist = -1.0, None, []
    step, micro, t0, tokens, ema = 0, 0, time.time(), 0, None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if a.eval_at_start and val_rows:
        m = full_metrics(val_rows, predict(model, tok, val_rows, a.mode, a.max_len, a.eval_bs, device, max_tokens=a.eval_tokens))
        hist.append({"step": 0, **m})
        log(f"  eval step 0 (before training): DS={m['ds']:.4f} F1={m['f1']:.4f} USR={m['usr']:.4f} SafeFPR={m['safe_fpr']:.3f} "
            f"| recall S/O/H/Sv {m['recall_safe']:.2f}/{m['recall_off']:.2f}/{m['recall_harm']:.2f}/{m['recall_sev']:.2f}")
    model.train(); opt.zero_grad(set_to_none=True)
    done = False
    while not done:
        ex = make_examples(rows, a.mode, True, a.view_dropout, rng)
        for b, enc, y in batches(ex, tok, a.max_len, a.micro_bs, True, rng, a.train_tokens):
            enc = {k: v.to(device) for k, v in enc.items()}; y = y.to(device)
            tokens += int(enc["attention_mask"].sum())
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(enc["input_ids"], enc["attention_mask"])
            loss = F.cross_entropy(logits.float(), y, label_smoothing=a.label_smoothing)
            if a.unsafe_w > 0:
                p_safe = F.softmax(logits.float(), -1)[:, 0]
                uns = y > 0
                if uns.any():
                    loss = loss + a.unsafe_w * (-torch.log1p(-p_safe[uns].clamp(max=1 - 1e-6))).mean()
            (loss / a.grad_accum).backward()
            micro += 1
            if micro % a.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
            if step % 10 == 0 or step == total:
                el = time.time() - t0
                mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
                res = torch.cuda.max_memory_reserved() / 2**30 if device.type == "cuda" else 0
                log(f"step {step}/{total} loss {loss.item():.4f} (ema {ema:.4f}) | {el:.0f}s | {tokens / el:.0f} tok/s "
                    f"| peak VRAM alloc {mem:.1f} GB, reserved {res:.1f} GB")
            if val_rows and (step % eval_every == 0 or step == total):
                L = predict(model, tok, val_rows, a.mode, a.max_len, a.eval_bs, device, max_tokens=a.eval_tokens)
                m = full_metrics(val_rows, L)
                hist.append({"step": step, **m})
                log(f"  eval step {step}: DS={m['ds']:.4f} F1={m['f1']:.4f} USR={m['usr']:.4f} SafeFPR={m['safe_fpr']:.3f} "
                    f"| recall S/O/H/Sv {m['recall_safe']:.2f}/{m['recall_off']:.2f}/{m['recall_harm']:.2f}/{m['recall_sev']:.2f} "
                    f"| row DS {m['row_ds']:.4f}")
                if m["ds"] > best:
                    best = m["ds"]
                    best_state = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}
                    write_logits(os.path.join(a.out, "val_logits.csv"), val_rows, L)
                    log(f"  new best {best:.4f}")
            if step >= total:
                done = True
                break
    train_min = (time.time() - t0) / 60

    # ---------------- restore best, save, predict
    if best_state is not None:
        sd = dict(model.named_parameters())
        with torch.no_grad():
            for k, v in best_state.items():
                sd[k].copy_(v.to(sd[k].device, sd[k].dtype))
    model.backbone.save_pretrained(os.path.join(a.out, "adapter"))
    torch.save(model.score.state_dict(), os.path.join(a.out, "score_head.pt"))
    if a.eval_single_views and a.mode == "joint" and val_rows:
        for f in FORMS:
            L = predict(model, tok, val_rows, a.mode, a.max_len, a.eval_bs, device, only_form=f, max_tokens=a.eval_tokens)
            write_logits(os.path.join(a.out, f"val_logits_only_{f}.csv"), val_rows, L)
            m = full_metrics(val_rows, L); log(f"  single view {f}: DS={m['ds']:.4f} F1={m['f1']:.4f} USR={m['usr']:.4f}")
    for p in a.predict:
        prow = load_rows(p)
        clip_texts(prow, tok, a.mode, a.max_len)
        t1 = time.time()
        L = predict(model, tok, prow, a.mode, a.max_len, a.eval_bs, device, max_tokens=a.eval_tokens)
        write_logits(os.path.join(a.out, os.path.splitext(os.path.basename(p))[0] + "_logits.csv"), prow, L)
        log(f"predicted {p}: {len(prow)} rows in {(time.time() - t1) / 60:.1f} min")

    el = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
    tok_per_s = tokens / max(1e-9, train_min * 60)
    summary = {"args": vars(a), "best_val_ds": best, "train_minutes": round(train_min, 2), "total_minutes": round(el / 60, 2),
               "peak_vram_gb": round(mem, 2), "train_tokens_per_s": round(tok_per_s, 1),
               "trainable_params_m": round(n_tr / 1e6, 1), "history": hist}
    if a.max_steps and step:
        ex_per_step = n_train_ex / max(1, steps_per_epoch * min(1.0, a.epochs))  # examples per optimizer step
        full_rows = 24588 if a.mode == "rows" else 8196
        summary["est_minutes_per_full_epoch_train"] = round(train_min / step * full_rows / max(1e-9, ex_per_step), 1)
        log(f"estimate: {summary['est_minutes_per_full_epoch_train']} min per full epoch on train (mode {a.mode}, excl. evals)")
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    log(f"DONE best DS={best:.4f} | train {train_min:.1f} min | peak VRAM {mem:.1f} GB | {tok_per_s:.0f} tok/s")


if __name__ == "__main__":
    main()
