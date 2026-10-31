"""Decision-layer lab: compare ensemble / pooling / decision strategies with repeated grouped CV.

Everything learned (ensemble weights, temperatures, view weights, stacking, biases) is fit on the
training folds only and scored on the held-out fold. Reports Defense Score, macro-F1, USR,
Safe false-positive rate (Safe prompts flagged unsafe) and per-class recall.

Usage (labelled logit files that cover the same prompts, e.g. dev logits or OOF logits on train+dev):
  python decision_lab.py --val E1=runs/E1_xlmrL_nocons_s42/val_logits.csv E2=runs/E2_xlmrL_cons_s42/val_logits.csv B0=runs/B0_tfidf/dev_logits.csv
Rule of thumb: adopt a change only if its paired-bootstrap P(better) >= 0.9 (section F).
"""
import argparse, itertools, collections
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import KFold
from bk_common import read_logits, log_softmax

FORMS = ['bangla', 'romanized', 'codemixed']
_ap = argparse.ArgumentParser()
_ap.add_argument("--val", nargs="+", required=True, help="NAME=path to a labelled logit CSV")
_ap.add_argument("--reps", type=int, default=5, help="repeats of 5-fold CV")
_ap.add_argument("--bootstrap", type=int, default=2000)
ARGS = _ap.parse_args()
MODELS = dict(v.split("=", 1) for v in ARGS.val)

# ---- load into X[g, m, v, c] (raw logits) ----
metas = {}; Ls = {}
for k, p in MODELS.items():
    metas[k], Ls[k] = read_logits(p)
ref = metas['E1']
pids = list(dict.fromkeys(m['prompt_id'] for m in ref))
gi = {p: i for i, p in enumerate(pids)}
G, M, V = len(pids), len(MODELS), 3
X = np.zeros((G, M, V, 4)); y = np.zeros(G, int)
for mi, k in enumerate(MODELS):
    for r, l in zip(metas[k], Ls[k]):
        X[gi[r['prompt_id']], mi, FORMS.index(r['form'])] = l
        y[gi[r['prompt_id']]] = r['label']
LP = X - np.log(np.exp(X - X.max(-1, keepdims=True)).sum(-1, keepdims=True)) - X.max(-1, keepdims=True)  # log-probs

def metrics(yt, p):
    f1 = f1_score(yt, p, average='macro', labels=[0, 1, 2, 3], zero_division=0)
    usr = np.mean(p[yt != 0] == 0)
    fpr = np.mean(p[yt == 0] != 0)  # Safe prompts flagged unsafe (over-flagging)
    rec = recall_score(yt, p, average=None, labels=[0, 1, 2, 3], zero_division=0)
    return dict(ds=f1 * (1 - usr), f1=f1, usr=usr, safe_fpr=fpr, r0=rec[0], r1=rec[1], r2=rec[2], r3=rec[3])

def softmax(z): z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)

# ---------- ensemble strategies: return per-view prob tensor P[g, v, c] ----------
def ens_logmean(lp, tr, T=None): return softmax(lp.mean(1))
def ens_probmean(lp, tr, T=None): return np.exp(lp).mean(1)
SIMPLEX = [w for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=M) if abs(sum(w) - 1) < 1e-9]
def fit_temps(lp, tr):
    Ts = []
    for m in range(M):
        z = lp[tr][:, m].reshape(-1, 4); yy = np.repeat(y[tr], V)
        nll = lambda t: -np.log(softmax(z / t)[np.arange(len(yy)), yy] + 1e-12).mean()
        Ts.append(minimize_scalar(nll, bounds=(0.2, 5), method='bounded').x)
    return np.array(Ts)
def ens_cal_logmean(lp, tr, T=None):
    Ts = fit_temps(lp, tr); z = lp / Ts[None, :, None, None]
    return softmax(np.log(softmax(z) + 1e-12).mean(1))
def ens_cal_probmean(lp, tr, T=None):
    Ts = fit_temps(lp, tr); return softmax(lp / Ts[None, :, None, None]).mean(1)
def ens_weighted(lp, tr, T=None):
    P = np.exp(lp); best = None
    for w in SIMPLEX:
        q = np.einsum('gmvc,m->gvc', P[tr], np.array(w)).mean(1)
        s = metrics(y[tr], q.argmax(1))['ds']
        if best is None or s > best[0]: best = (s, w)
    return np.einsum('gmvc,m->gvc', P, np.array(best[1]))
def ens_weighted_nll(lp, tr, T=None):
    P = np.exp(lp); best = None
    for w in SIMPLEX:
        q = np.einsum('gmvc,m->gvc', P[tr], np.array(w)).mean(1)
        s = -np.log(q[np.arange(len(tr)), y[tr]] + 1e-12).mean()
        if best is None or s < best[0]: best = (s, w)
    return np.einsum('gmvc,m->gvc', P, np.array(best[1]))
ENS = {'log-prob mean (current)': ens_logmean, 'prob mean': ens_probmean,
       'weighted prob (DS-fit)': ens_weighted, 'weighted prob (NLL-fit)': ens_weighted_nll,
       'calibrated log-prob mean': ens_cal_logmean, 'calibrated prob mean': ens_cal_probmean}

# ---------- pooling over views: P[g,v,c] -> Q[g,c]; mask[g,v] = view available ----------
def pool_mean(P, tr, mask): return (P * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
def pool_logmean(P, tr, mask): return softmax((np.log(P + 1e-12) * mask[..., None]).sum(1) / mask.sum(1, keepdims=True))
def pool_conf(P, tr, mask):
    a = P.max(-1) * mask; return (P * a[..., None]).sum(1) / a.sum(1, keepdims=True)
def pool_entropy(P, tr, mask):
    H = -(P * np.log(P + 1e-12)).sum(-1); a = np.exp(-2 * H) * mask; return (P * a[..., None]).sum(1) / a.sum(1, keepdims=True)
VSIMPLEX = [w for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=V) if abs(sum(w) - 1) < 1e-9]
def pool_learned(P, tr, mask):
    best = None
    for w in VSIMPLEX:
        if min(w) == 0: continue
        q = np.einsum('gvc,v->gc', P[tr], np.array(w)); s = metrics(y[tr], q.argmax(1))['ds']
        if best is None or s > best[0]: best = (s, w)
    a = np.array(best[1])[None, :] * mask; return (P * a[..., None]).sum(1) / a.sum(1, keepdims=True)
def pool_stack(P, tr, mask):
    F = np.log(P + 1e-12).reshape(len(P), -1)
    clf = LogisticRegression(C=0.5, max_iter=3000).fit(F[tr], y[tr]); return clf.predict_proba(F)
POOL = {'mean prob (current idea)': pool_mean, 'mean log-prob': pool_logmean, 'confidence-weighted': pool_conf,
        'entropy-weighted': pool_entropy, 'learned per-form weights': pool_learned, 'stacking meta-classifier': pool_stack}

# ---------- decision rules: Q[g,c] -> labels ----------
def dec_argmax(Q, tr): return Q.argmax(1)
def dec_safe_bias(Q, tr):
    L = np.log(Q + 1e-12); best = None
    for b in np.arange(0, 1.51, 0.05):
        Z = L.copy(); Z[:, 0] -= b; s = metrics(y[tr], Z[tr].argmax(1))['ds']
        if best is None or s > best[0] + 1e-12: best = (s, b)
    Z = L.copy(); Z[:, 0] -= best[1]; return Z.argmax(1)
def dec_4bias(Q, tr):
    L = np.log(Q + 1e-12); b = np.zeros(4); best = metrics(y[tr], L[tr].argmax(1))['ds']
    for _ in range(2):
        for c in range(4):
            for v in np.arange(-1.5, 1.51, 0.1):
                t = b.copy(); t[c] = v; s = metrics(y[tr], (L[tr] + t).argmax(1))['ds']
                if s > best + 1e-12: best, b = s, t
    return (L + b).argmax(1)
def dec_threshold(Q, tr):  # Safe only if p(Safe) >= tau, otherwise best unsafe class
    best = None
    for tau in np.arange(0.3, 0.91, 0.025):
        p = np.where(Q[tr, 0] >= tau, 0, Q[tr, 1:].argmax(1) + 1); s = metrics(y[tr], p)['ds']
        if best is None or s > best[0]: best = (s, tau)
    return np.where(Q[:, 0] >= best[1], 0, Q[:, 1:].argmax(1) + 1)
DEC = {'argmax (no bias)': dec_argmax, 'Safe-only bias (CV)': dec_safe_bias,
       '4-class bias (CV)': dec_4bias, 'Safe threshold (CV)': dec_threshold}

def run(ens, pool, dec, mask=None, reps=None, models=None):
    reps = ARGS.reps if reps is None else reps
    lp = LP if models is None else LP[:, models]
    mask = np.ones((G, V)) if mask is None else mask
    out = collections.defaultdict(list)
    for r in range(reps):
        pred = np.zeros(G, int)
        for tr, te in KFold(5, shuffle=True, random_state=r).split(np.arange(G)):
            Pv = ens(lp, tr); Q = pool(Pv, tr, mask); p = dec(Q, tr); pred[te] = p[te]
        for k, v in metrics(y, pred).items(): out[k].append(v)
    return {k: (np.mean(v), np.std(v)) for k, v in out.items()}

def show(name, res, base=None):
    d = f" ({100*(res['ds'][0]-base):+.2f})" if base is not None else ''
    print(f"  {name:32s} DS {res['ds'][0]:.4f}{d:9s} F1 {res['f1'][0]:.4f}  USR {res['usr'][0]:.4f}  SafeFPR {res['safe_fpr'][0]:.3f}  "
          f"recall S/O/H/Sv {res['r0'][0]:.2f}/{res['r1'][0]:.2f}/{res['r2'][0]:.2f}/{res['r3'][0]:.2f}")

if __name__ == '__main__':
    print(f"dev prompts={G}, models={list(MODELS)}; 5 x 5-fold grouped CV; learned parts fit on train folds only\n")
    print("A. Ensemble strategy  (pooling = mean prob, decision = argmax)")
    base = run(ens_logmean, pool_mean, dec_argmax)['ds'][0]
    for n, f in ENS.items(): show(n, run(f, pool_mean, dec_argmax), base)
    print("\nB. View pooling  (ensemble = prob mean, decision = argmax)")
    base = run(ens_probmean, pool_mean, dec_argmax)['ds'][0]
    for n, f in POOL.items(): show(n, run(ens_probmean, f, dec_argmax), base)
    print("\nC. Decision rule  (ensemble = prob mean, pooling = mean prob)")
    base = run(ens_probmean, pool_mean, dec_argmax)['ds'][0]
    for n, f in DEC.items(): show(n, run(ens_probmean, pool_mean, f), base)
    print("\nD. Missing views  (prob mean, mean pooling over the available views, argmax)")
    for avail in [(1, 1, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        mask = np.tile(np.array(avail, float), (G, 1))
        show('views ' + '+'.join(f[:2] for f, a in zip(FORMS, avail) if a), run(ens_probmean, pool_mean, dec_argmax, mask=mask, reps=1))
    print("\nE. Single models for reference (prob mean pooling, argmax)")
    for mi, k in enumerate(MODELS): show(k, run(ens_probmean, pool_mean, dec_argmax, models=[mi], reps=1))

    print("\nF. Paired bootstrap vs current (log-prob mean, mean pooling, argmax): DS diff in points, 95% CI, P(better)")
    def cvpred(ens, pool, dec):
        mask = np.ones((G, V)); pred = np.zeros(G, int)
        for tr, te in KFold(5, shuffle=True, random_state=0).split(np.arange(G)):
            pred[te] = dec(pool(ens(LP, tr), tr, mask), tr)[te]
        return pred
    rng = np.random.RandomState(0); base_pred = cvpred(ens_logmean, pool_mean, dec_argmax)
    cands = {'prob mean': (ens_probmean, pool_mean, dec_argmax),
             'NLL-weighted prob': (ens_weighted_nll, pool_mean, dec_argmax),
             'calibrated log-prob': (ens_cal_logmean, pool_mean, dec_argmax),
             'Safe-only bias': (ens_probmean, pool_mean, dec_safe_bias),
             'calibrated log-prob + Safe bias': (ens_cal_logmean, pool_mean, dec_safe_bias),
             'stacking': (ens_probmean, pool_stack, dec_argmax)}
    for n, c in cands.items():
        p = cvpred(*c); d = []
        for _ in range(ARGS.bootstrap):
            s_ = rng.randint(0, G, G); d.append(metrics(y[s_], p[s_])['ds'] - metrics(y[s_], base_pred[s_])['ds'])
        d = np.array(d)
        print(f"  {n:34s} {100*d.mean():+.2f}  [{100*np.percentile(d,2.5):+.2f}, {100*np.percentile(d,97.5):+.2f}]  P>0 {np.mean(d>0):.2f}")
