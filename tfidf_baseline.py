"""CPU-only char+word TF-IDF + logistic regression. A fast sanity baseline and a cheap ensemble member.

Examples:
  # 5-fold grouped out-of-fold logits on train (for analysis / bias tuning)
  python tfidf_baseline.py --train data/train.jsonl --oof --out runs/tfidf
  # fit on train, predict dev and the blind dev-test
  python tfidf_baseline.py --train data/train.jsonl --predict data/dev.jsonl data/devtest.jsonl --out runs/tfidf
"""
import argparse
import json
import os
import time

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from bk_common import defense_score, group_mean, load_rows, make_logger, write_logits


def featurize(train_txt, other_txts, max_feats):
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=max_feats)
    vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"\S+")
    Xtr = hstack([vc.fit_transform(train_txt), vw.fit_transform(train_txt)]).tocsr()
    others = [hstack([vc.transform(t), vw.transform(t)]).tocsr() for t in other_txts]
    return Xtr, others


def fit_predict(tr_rows, others, C, max_feats):
    low = lambda rs: [r["text"].lower() for r in rs]
    Xtr, Xo = featurize(low(tr_rows), [low(o) for o in others], max_feats)
    m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    m.fit(Xtr, [r["label"] for r in tr_rows])
    return [m.predict_log_proba(X) for X in Xo]


def report(name, rows, L, log):
    y = [r["label"] for r in rows]
    forms = [r["form"] for r in rows]
    raw = defense_score(y, L.argmax(1), forms)
    agg = defense_score(y, group_mean(L, [r["prompt_id"] for r in rows]).argmax(1), forms)
    log(f"[{name}] row-level DS={raw['ds']:.4f} | prompt-agg DS={agg['ds']:.4f} F1={agg['f1']:.4f} USR={agg['usr']:.4f} "
        f"| per-form F1 bn={agg.get('f1_bangla',0):.3f} ro={agg.get('f1_romanized',0):.3f} cm={agg.get('f1_codemixed',0):.3f}")
    return {"raw_ds": raw["ds"], **{f"agg_{k}": v for k, v in agg.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--predict", nargs="*", default=[])
    ap.add_argument("--oof", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--C", type=float, default=8.0)
    ap.add_argument("--max_feats", type=int, default=300000)
    ap.add_argument("--max_prompts", type=int, default=0, help="subsample for a quick test")
    ap.add_argument("--out", default="runs/tfidf")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = make_logger(os.path.join(a.out, "train.log"))
    log("args:", json.dumps(vars(a)))
    t0, scores = time.time(), {}

    rows = [r for p in a.train for r in load_rows(p)]
    if a.max_prompts:
        keep = sorted({r["prompt_id"] for r in rows})[: a.max_prompts]
        keep = set(keep)
        rows = [r for r in rows if r["prompt_id"] in keep]

    if a.oof:
        L = np.zeros((len(rows), 4))
        groups = [r["prompt_id"] for r in rows]
        splits = GroupKFold(a.folds).split(rows, groups=groups)
        for k, (tr, te) in enumerate(tqdm(splits, total=a.folds, desc="oof folds", unit="fold", dynamic_ncols=True)):
            (L[te],) = fit_predict([rows[i] for i in tr], [[rows[i] for i in te]], a.C, a.max_feats)
            log(f"fold {k} done | {time.time()-t0:.0f}s")
        write_logits(os.path.join(a.out, "oof_logits.csv"), rows, L)
        scores["oof"] = report("oof", rows, L, log)

    if a.predict:
        log("fitting on all training rows ...")
        pred_sets = [load_rows(p) for p in a.predict]
        outs = fit_predict(rows, pred_sets, a.C, a.max_feats)
        for p, rs, L in zip(a.predict, pred_sets, outs):
            name = os.path.splitext(os.path.basename(p))[0]
            write_logits(os.path.join(a.out, f"{name}_logits.csv"), rs, L)
            if rs and rs[0]["label"] is not None:
                scores[name] = report(name, rs, L, log)
            log("wrote", name)

    mins = (time.time() - t0) / 60
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(a), "scores": scores, "minutes": round(mins, 2)}, f, indent=1)
    log(f"DONE | {mins:.1f} min")


if __name__ == "__main__":
    main()
