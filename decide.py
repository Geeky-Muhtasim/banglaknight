"""Ensemble logit files, average across the three forms of each prompt, tune per-class
biases to maximise Defense Score on labelled data, and write the CodaBench zip.

Example:
  python decide.py --val runs/xlmr/val_logits.csv runs/tfidf/dev_logits.csv \
                   --test runs/xlmr/devtest_logits.csv runs/tfidf/devtest_logits.csv \
                   --out submissions/sub1
Files passed to --val and --test must be in the same model order.
"""
import argparse
import csv
import json
import os
import zipfile

import numpy as np
from tqdm import tqdm

from bk_common import defense_score, group_mean, load_rows, log_softmax, make_logger, read_logits


def load_ensemble(paths, weights, temps=None):
    """Weighted mean of (temperature-scaled) log-probabilities, aligned by id."""
    metas, Ls = zip(*[read_logits(p) for p in paths])
    ids = [m["id"] for m in metas[0]]
    temps = temps or [1.0] * len(paths)
    acc = np.zeros((len(ids), 4))
    for meta, L, w, t in zip(metas, Ls, weights, temps):
        pos = {m["id"]: i for i, m in enumerate(meta)}
        missing = [i for i in ids if i not in pos]
        assert not missing, f"{len(missing)} ids missing in one file, e.g. {missing[:3]}"
        acc += w * log_softmax(L[[pos[i] for i in ids]] / t)
    return metas[0], acc / sum(weights)


def fit_temperatures(paths):
    """Per-model temperature minimising NLL on the labelled rows (calibration before ensembling)."""
    from scipy.optimize import minimize_scalar
    temps = []
    for p in paths:
        meta, L = read_logits(p)
        y = np.array([m["label"] for m in meta])
        nll = lambda t: -log_softmax(L / t)[np.arange(len(y)), y].mean()
        temps.append(float(minimize_scalar(nll, bounds=(0.2, 5.0), method="bounded").x))
    return temps


def tune_bias(L, y, forms, grid=np.arange(-2.0, 2.01, 0.05), rounds=3, desc="bias", safe_only=False):
    if safe_only:  # one parameter: how much to subtract from the Safe log-prob
        best_b, best = 0.0, defense_score(y, L.argmax(1), forms)["ds"]
        for v in np.arange(0.0, 1.51, 0.05):
            t = L.copy(); t[:, 0] -= v
            s = defense_score(y, t.argmax(1), forms)["ds"]
            if s > best + 1e-9:
                best, best_b = s, v
        b = np.zeros(4); b[0] = -best_b
        return b, best
    b = np.zeros(4)
    best = defense_score(y, L.argmax(1), forms)["ds"]
    pbar = tqdm(total=rounds * 4 * len(grid), desc=desc, unit="try", leave=False, dynamic_ncols=True)
    for _ in range(rounds):
        for c in range(4):
            for v in grid:
                pbar.update(1)
                t = b.copy(); t[c] = v
                s = defense_score(y, (L + t).argmax(1), forms)["ds"]
                if s > best + 1e-9:
                    best, b = s, t
                    pbar.set_postfix(ds=f"{best:.4f}", refresh=False)
    pbar.close()
    return b, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", nargs="+", required=True, help="labelled logit files (dev or OOF)")
    ap.add_argument("--test", nargs="*", default=[], help="blind logit files, same model order as --val")
    ap.add_argument("--weights", nargs="*", type=float)
    ap.add_argument("--no_agg", action="store_true", help="disable prompt-level averaging")
    ap.add_argument("--no_bias", action="store_true")
    ap.add_argument("--calibrate", action="store_true", help="per-model temperature scaling (fit on --val) before ensembling")
    ap.add_argument("--safe_bias_only", action="store_true", help="tune one Safe-only bias instead of 4 class biases")
    ap.add_argument("--safe_bias", type=float, default=None,
                    help="fixed amount subtracted from the Safe log-prob (no tuning); reports its dev effect")
    ap.add_argument("--bias_splits", type=int, default=5, help="random half splits for the honest bias check")
    ap.add_argument("--out", default="submissions/sub")
    ap.add_argument("--blind", help="official blind file (e.g. data/devtest.jsonl): check ids match before zipping")
    a = ap.parse_args()
    w = a.weights or [1.0] * len(a.val)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    log = make_logger(a.out + ".log")
    log("args:", json.dumps(vars(a)))
    scores = {}

    temps = fit_temperatures(a.val) if a.calibrate else None
    if temps:
        log("temperatures:", [round(t, 3) for t in temps])
    meta, L = load_ensemble(a.val, w, temps)
    y = np.array([m["label"] for m in meta]); forms = [m["form"] for m in meta]
    pids = [m["prompt_id"] for m in meta]
    if not a.no_agg:
        L = group_mean(L, pids)
    base = defense_score(y, L.argmax(1), forms)
    pv = L.argmax(1)
    base["safe_fpr"] = float(np.mean(pv[y == 0] != 0))
    scores["val_no_bias"] = base
    log("val (no bias):", {k: round(v, 4) for k, v in base.items()})

    b = np.zeros(4)
    if a.safe_bias is not None:
        b[0] = -a.safe_bias
        scores["val_fixed_safe_bias"] = defense_score(y, (L + b).argmax(1), forms)
        pf = (L + b).argmax(1); scores["val_fixed_safe_bias"]["safe_fpr"] = float(np.mean(pf[y == 0] != 0))
        log(f"fixed Safe bias {a.safe_bias}:", {k: round(v, 4) for k, v in scores["val_fixed_safe_bias"].items()})
    elif not a.no_bias:
        # honest estimate: tune on half the prompts, score on the other half (repeated over random halves)
        up = np.array(sorted(set(pids))); fa = np.array(forms); pa = np.array(pids); gains = []
        for s in range(a.bias_splits):
            u = up.copy(); np.random.RandomState(s).shuffle(u)
            m = np.isin(pa, u[: len(u) // 2])
            bh, _ = tune_bias(L[m], y[m], fa[m], desc=f"bias (half {s + 1}/{a.bias_splits})", safe_only=a.safe_bias_only)
            held = defense_score(y[~m], (L[~m] + bh).argmax(1), fa[~m])["ds"]
            held0 = defense_score(y[~m], L[~m].argmax(1), fa[~m])["ds"]
            gains.append(held - held0)
            log(f"bias honest check {s + 1}/{a.bias_splits}: held-out half DS {held0:.4f} -> {held:.4f}")
        scores["heldout_bias_gain"] = {"per_split": gains, "mean": float(np.mean(gains))}
        log(f"bias held-out gain: mean {np.mean(gains):+.4f} over {len(gains)} splits")
        if np.mean(gains) <= 0:
            log("WARNING: bias does not help on held-out halves; consider --no_bias")
        b, best = tune_bias(L, y, forms, desc="bias (full)", safe_only=a.safe_bias_only)
        log("bias (full val):", np.round(b, 2), "DS ->", round(best, 4))
        scores["val_bias"] = defense_score(y, (L + b).argmax(1), forms)
        pb_ = (L + b).argmax(1)
        scores["val_bias"]["safe_fpr"] = float(np.mean(pb_[y == 0] != 0))
        log("val (bias):", {k: round(v, 4) for k, v in scores["val_bias"].items()})

    if a.test:
        tmeta, TL = load_ensemble(a.test, w, temps)
        if not a.no_agg:
            TL = group_mean(TL, [m["prompt_id"] for m in tmeta])
        pred = (TL + b).argmax(1)
        tids = [m_["id"] for m_ in tmeta]
        assert len(set(tids)) == len(tids), "repeated ids in the test logits"
        if a.blind:
            want = {r["id"] for r in load_rows(a.blind)}
            missing, extra = want - set(tids), set(tids) - want
            assert not missing and not extra, f"id mismatch vs {a.blind}: {len(missing)} missing, {len(extra)} extra"
            log(f"submission check OK: {len(tids)} ids match {a.blind}")
        csv_path = a.out + "_predictions.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            wr = csv.writer(f); wr.writerow(["id", "label"])
            for m_, p in zip(tmeta, pred):
                wr.writerow([m_["id"], int(p)])
        with zipfile.ZipFile(a.out + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.write(csv_path, arcname="predictions.csv")
        dist = np.bincount(pred, minlength=4) / len(pred)
        log(f"wrote {a.out}.zip  rows={len(pred)}  label dist={np.round(dist, 3)}")
        with open(a.out + "_config.json", "w", encoding="utf-8") as f:
            json.dump({"val": a.val, "test": a.test, "weights": w, "temperatures": temps, "bias": b.tolist(), "agg": not a.no_agg,
                       "scores": scores, "test_label_dist": dist.tolist()}, f, indent=1)


if __name__ == "__main__":
    main()
