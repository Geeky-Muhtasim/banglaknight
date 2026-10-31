"""Fine-tune a multilingual encoder for BanglaKnight with a cross-form consistency loss.

Every batch holds whole prompts (all three written forms together). The loss is
  cross-entropy on every row  +  cons_w * mean KL(p_form || p_prompt_mean)
which pushes the Bangla, romanized and code-mixed versions of a prompt toward one answer.

Examples (PowerShell, from H:\\BanglaKnight with the venv active):
  # small feasibility run
  python train_encoder.py --model microsoft/mdeberta-v3-base --train data/train.jsonl --fold 0 --max_prompts 800 --epochs 1 --out runs/smoke
  # real run: train on train, validate on dev, predict the blind dev-test
  python train_encoder.py --model xlm-roberta-large --train data/train.jsonl --dev data/dev.jsonl --predict data/devtest.jsonl --out runs/xlmrL_s42
  # 5-fold CV over train+dev (gives OOF logits for ensembling)
  python train_encoder.py --model google/muril-large-cased --train data/train.jsonl data/dev.jsonl --fold 0 --folds 5 --predict data/devtest.jsonl --out runs/muril_f0
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
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from bk_common import FORMS, defense_score, group_mean, load_rows, log_softmax, make_logger, write_logits

FORM_ORDER = {f: i for i, f in enumerate(FORMS)}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def prompt_groups(rows):
    d = OrderedDict()
    for i, r in enumerate(rows):
        d.setdefault(r["prompt_id"], []).append(i)
    return [sorted(g, key=lambda i: FORM_ORDER.get(rows[i]["form"], 9)) for g in d.values()]


class Batcher:
    """Turns a list of prompt groups into tokenized tensors."""

    def __init__(self, tok, rows, max_len, joint):
        self.tok, self.rows, self.max_len, self.joint = tok, rows, max_len, joint
        self.sep = f" {tok.sep_token} " if tok.sep_token else " | "

    def __call__(self, groups):
        if self.joint:  # one input per prompt: all three forms concatenated
            texts = [self.sep.join(self.rows[i]["text"] for i in g) for g in groups]
            labels = [self.rows[g[0]]["label"] for g in groups]
            gid = list(range(len(groups)))
        else:
            idx = [i for g in groups for i in g]
            texts = [self.rows[i]["text"] for i in idx]
            labels = [self.rows[i]["label"] for i in idx]
            gid = [k for k, g in enumerate(groups) for _ in g]
        enc = self.tok(texts, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
        y = torch.tensor([-100 if l is None else l for l in labels])
        return enc, y, torch.tensor(gid)


def chunks(groups, bs, shuffle):
    groups = list(groups)
    if shuffle:
        random.shuffle(groups)
    for i in range(0, len(groups), bs):
        yield groups[i:i + bs]


def llrd_param_groups(model, lr, decay, wd):
    """Layer-wise LR decay: top layers keep lr, lower layers get lr*decay^depth."""
    n_layers = model.config.num_hidden_layers
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        m = re.search(r"\.layer\.(\d+)\.", name)
        if m:
            depth = n_layers - int(m.group(1))
        elif "embeddings" in name:
            depth = n_layers + 1
        else:
            depth = 0  # classifier head, pooler
        no_wd = p.ndim == 1 or name.endswith(".bias")
        key = (depth, no_wd)
        if key not in groups:
            groups[key] = {"params": [], "lr": lr * (decay ** depth), "weight_decay": 0.0 if no_wd else wd}
        groups[key]["params"].append(p)
    return list(groups.values())


def consistency_loss(logits, gid):
    """Generalised JS divergence between each form and its prompt's mean distribution."""
    logp = F.log_softmax(logits, -1)
    p = logp.exp()
    G = int(gid.max()) + 1
    cnt = torch.zeros(G, device=logits.device).index_add_(0, gid, torch.ones_like(gid, dtype=p.dtype))
    pbar = torch.zeros(G, p.size(1), device=logits.device, dtype=p.dtype).index_add_(0, gid, p) / cnt[:, None]
    kl = (p * (logp - torch.log(pbar[gid] + 1e-8))).sum(-1)
    multi = cnt[gid] > 1
    return kl[multi].mean() if multi.any() else logits.sum() * 0


@torch.no_grad()
def predict(model, batcher, rows, bs, device, amp_dtype, desc="predict"):
    model.eval()
    groups = prompt_groups(rows)
    out = np.zeros((len(rows), model.config.num_labels), dtype=np.float32)
    for chunk in tqdm(chunks(groups, bs, shuffle=False), total=math.ceil(len(groups) / bs),
                      desc=f"  {desc}", unit="batch", leave=False, dynamic_ncols=True):
        enc, _, _ = batcher(chunk)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            lg = model(**enc).logits.float().cpu().numpy()
        if batcher.joint:
            for g, l in zip(chunk, lg):
                out[g] = l
        else:
            out[[i for g in chunk for i in g]] = lg
    model.train()
    return out


def evaluate(rows, L):
    y = [r["label"] for r in rows]; forms = [r["form"] for r in rows]
    lp = log_softmax(L)
    agg = defense_score(y, group_mean(lp, [r["prompt_id"] for r in rows]).argmax(1), forms)
    raw = defense_score(y, lp.argmax(1), forms)
    return agg, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xlm-roberta-large")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--dev", nargs="*", default=[], help="labelled validation file(s)")
    ap.add_argument("--fold", type=int, default=-1, help="use fold k of a GroupKFold over --train as validation")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--predict", nargs="*", default=[], help="files to write logits for (blind or labelled)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--llrd", type=float, default=0.9, help="layer-wise LR decay (1.0 = off)")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--bs_prompts", type=int, default=8, help="prompts per step (x3 rows unless --joint)")
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--cons_w", type=float, default=1.0, help="weight of the cross-form consistency loss")
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--joint", action="store_true", help="one input per prompt with all three forms concatenated")
    ap.add_argument("--evals_per_epoch", type=int, default=2)
    ap.add_argument("--max_prompts", type=int, default=0, help="subsample training prompts for quick tests")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing to save VRAM")
    ap.add_argument("--allow_cpu", action="store_true", help="permit running without a GPU (tiny tests only)")
    a = ap.parse_args()
    if a.joint and a.max_len == 256:
        a.max_len = 512
    os.makedirs(a.out, exist_ok=True)
    log = make_logger(os.path.join(a.out, "train.log"))
    log("args:", json.dumps(vars(a)))
    set_seed(a.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        log("GPU:", torch.cuda.get_device_name(0), "| bf16:", amp_dtype is not None)
    elif not a.allow_cpu:
        raise SystemExit("No CUDA GPU visible to PyTorch. Run setup.ps1 (installs the CUDA build) "
                         "or pass --allow_cpu for a tiny CPU test.")
    else:
        log("WARNING: running on CPU (only OK for tiny tests)")

    rows = [r for p in a.train for r in load_rows(p)]
    val_rows = [r for p in a.dev for r in load_rows(p)]
    if a.fold >= 0:
        pids = np.array([r["prompt_id"] for r in rows])
        _, va = list(GroupKFold(a.folds).split(rows, groups=pids))[a.fold]
        va_set = set(pids[va])
        val_rows = [r for r in rows if r["prompt_id"] in va_set]
        rows = [r for r in rows if r["prompt_id"] not in va_set]
    if a.max_prompts:
        rng = random.Random(a.seed)
        keep = sorted({r["prompt_id"] for r in rows}); rng.shuffle(keep); keep = set(keep[: a.max_prompts])
        rows = [r for r in rows if r["prompt_id"] in keep]
        vkeep = sorted({r["prompt_id"] for r in val_rows}); rng.shuffle(vkeep); vkeep = set(vkeep[: max(200, a.max_prompts // 4)])
        val_rows = [r for r in val_rows if r["prompt_id"] in vkeep]
    log(f"train rows={len(rows)} prompts={len({r['prompt_id'] for r in rows})} | val rows={len(val_rows)}")

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=4).to(device)
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()
    batcher = Batcher(tok, rows, a.max_len, a.joint)

    groups = prompt_groups(rows)
    steps_per_epoch = math.ceil(len(groups) / a.bs_prompts / a.grad_accum)
    total = int(steps_per_epoch * a.epochs)
    opt = torch.optim.AdamW(llrd_param_groups(model, a.lr, a.llrd, a.wd), lr=a.lr,
                            fused=device.type == "cuda")
    sched = get_cosine_schedule_with_warmup(opt, int(a.warmup * total), total)
    eval_every = max(1, steps_per_epoch // a.evals_per_epoch)

    best, step, micro, t0 = -1.0, 0, 0, time.time()
    hist = []
    model.train()
    done = False
    log(f"steps: {total} total, {steps_per_epoch}/epoch, eval every {eval_every}")
    pbar = tqdm(total=total, desc="train", unit="step", dynamic_ncols=True)
    loss_ema = None
    while not done:
        for chunk in chunks(groups, a.bs_prompts, shuffle=True):
            enc, y, gid = batcher(chunk)
            enc = {k: v.to(device) for k, v in enc.items()}
            y, gid = y.to(device), gid.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(**enc).logits.float()
            loss = F.cross_entropy(logits, y, label_smoothing=a.label_smoothing)
            if a.cons_w > 0 and not a.joint:
                loss = loss + a.cons_w * consistency_loss(logits, gid)
            (loss / a.grad_accum).backward()
            micro += 1
            if micro % a.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            loss_ema = loss.item() if loss_ema is None else 0.95 * loss_ema + 0.05 * loss.item()
            mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
            pbar.update(1)
            if step % 10 == 0 or step == total:
                pbar.set_postfix(ep=f"{step / steps_per_epoch:.2f}", loss=f"{loss_ema:.4f}",
                                 lr=f"{sched.get_last_lr()[0]:.1e}", vram=f"{mem:.1f}G",
                                 best=f"{best:.4f}" if best >= 0 else "-", refresh=False)
            if step % 50 == 0:
                log(f"step {step}/{total} loss {loss.item():.4f} (ema {loss_ema:.4f}) | {time.time()-t0:.0f}s "
                    f"| peak VRAM {mem:.1f} GB", console=False)
            if val_rows and (step % eval_every == 0 or step == total):
                L = predict(model, Batcher(tok, val_rows, a.max_len, a.joint), val_rows, a.bs_prompts * 4, device, amp_dtype, "eval")
                agg, raw = evaluate(val_rows, L)
                hist.append({"step": step, **{f"agg_{k}": v for k, v in agg.items()}, "raw_ds": raw["ds"]})
                log(f"  eval step {step}: DS(agg)={agg['ds']:.4f} F1={agg['f1']:.4f} USR={agg['usr']:.4f} "
                    f"| per-form F1 bn={agg.get('f1_bangla',0):.3f} ro={agg.get('f1_romanized',0):.3f} "
                    f"cm={agg.get('f1_codemixed',0):.3f} | DS(row-level)={raw['ds']:.4f}")
                if agg["ds"] > best:
                    best = agg["ds"]
                    write_logits(os.path.join(a.out, "val_logits.csv"), val_rows, L)
                    for p in a.predict:
                        prow = load_rows(p)
                        name = os.path.splitext(os.path.basename(p))[0]
                        PL = predict(model, Batcher(tok, prow, a.max_len, a.joint), prow, a.bs_prompts * 4, device, amp_dtype, name)
                        write_logits(os.path.join(a.out, name + "_logits.csv"), prow, PL)
                    if a.save_model:
                        model.save_pretrained(os.path.join(a.out, "model")); tok.save_pretrained(os.path.join(a.out, "model"))
                    log(f"  new best {best:.4f}, logits saved")
            if step >= total:
                done = True
                break
    pbar.close()

    if not val_rows:  # no validation: predict with the final weights
        for p in a.predict:
            prow = load_rows(p)
            name = os.path.splitext(os.path.basename(p))[0]
            PL = predict(model, Batcher(tok, prow, a.max_len, a.joint), prow, a.bs_prompts * 4, device, amp_dtype, name)
            write_logits(os.path.join(a.out, name + "_logits.csv"), prow, PL)
        if a.save_model:
            model.save_pretrained(os.path.join(a.out, "model")); tok.save_pretrained(os.path.join(a.out, "model"))

    mins = (time.time() - t0) / 60
    mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
    summary = {"args": vars(a), "best_val_ds": best, "minutes": round(mins, 2), "peak_vram_gb": round(mem, 2),
               "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1), "history": hist}
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    log(f"DONE best DS={best:.4f} | {mins:.1f} min | peak VRAM {mem:.1f} GB | {summary['params_m']}M params")


if __name__ == "__main__":
    main()
