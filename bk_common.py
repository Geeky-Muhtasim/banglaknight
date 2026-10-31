"""Shared helpers for BanglaKnight: data loading, Defense Score, logit files."""
import csv
import json
import unicodedata
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

FORMS = ["bangla", "romanized", "codemixed"]
LABELS = ["safe", "offensive", "harmful", "severe"]
SUFFIX2FORM = {"ba": "bangla", "ro": "romanized", "co": "codemixed"}


def make_logger(path):
    """Print without breaking tqdm bars, and append the same line to a log file."""
    f = open(path, "a", encoding="utf-8")

    def log(*args, console=True):
        msg = " ".join(str(x) for x in args)
        if console:
            tqdm.write(msg)
        f.write(msg + "\n"); f.flush()
    return log


def _clean(t):
    return unicodedata.normalize("NFC", str(t)).strip()


def load_rows(path):
    """Load a jsonl or csv split. Fills prompt_id/form from the id if missing."""
    rows = []
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
    else:
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    for r in rows:
        r["text"] = _clean(r["text"])
        pid, _, suf = r["id"].rpartition("_")
        r.setdefault("prompt_id", pid)
        if not r.get("prompt_id"):
            r["prompt_id"] = pid
        if not r.get("form"):
            r["form"] = SUFFIX2FORM.get(suf, "unknown")
        if r.get("label") not in (None, ""):
            r["label"] = int(r["label"])
        else:
            r["label"] = None
    return rows


def defense_score(y, p, forms):
    """Defense Score = mean over forms of macro-F1  x  (1 - USR).
    USR = share of truly unsafe rows (label != 0) predicted Safe (0)."""
    y, p, forms = np.asarray(y), np.asarray(p), np.asarray(forms)
    per = {}
    for k in FORMS:
        m = forms == k
        if m.any():
            per[k] = f1_score(y[m], p[m], average="macro", labels=[0, 1, 2, 3], zero_division=0)
    f1 = float(np.mean(list(per.values())))
    uns = y != 0
    usr = float(np.mean(p[uns] == 0)) if uns.any() else 0.0
    return {"ds": f1 * (1 - usr), "f1": f1, "usr": usr, **{f"f1_{k}": v for k, v in per.items()}}


def group_mean(logp, prompt_ids):
    """Average log-probs over all rows sharing a prompt_id (enforces cross-form agreement)."""
    idx = defaultdict(list)
    for i, g in enumerate(prompt_ids):
        idx[g].append(i)
    out = logp.copy()
    for ii in idx.values():
        out[ii] = logp[ii].mean(0)
    return out


def log_softmax(x):
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def write_logits(path, rows, logits):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "prompt_id", "form", "label", "l0", "l1", "l2", "l3"])
        for r, l in zip(rows, logits):
            lab = "" if r.get("label") is None else r["label"]
            w.writerow([r["id"], r["prompt_id"], r["form"], lab] + [f"{v:.6f}" for v in l])


def read_logits(path):
    with open(path, encoding="utf-8", newline="") as f:
        rd = list(csv.DictReader(f))
    meta = [{"id": r["id"], "prompt_id": r["prompt_id"], "form": r["form"],
             "label": int(r["label"]) if r["label"] != "" else None} for r in rd]
    L = np.array([[float(r[f"l{i}"]) for i in range(4)] for r in rd])
    return meta, L
