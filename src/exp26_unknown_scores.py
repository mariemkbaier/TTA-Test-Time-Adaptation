"""
EXPERIMENT 26 — a better unknown score, under the same Tier-1 constraints.

exp25 rejects on maximum softmax. Its oracle-threshold diagnostic showed that roughly half
the shortfall in unknown detection is a miscalibrated threshold and the other half is the
score itself: even with the threshold chosen on target labels, H reaches only 0.52 and
0.47. That puts the headroom in the score, not in the adaptation method, so this script
replaces the score and leaves everything else alone.

Six candidates, all computable from the frozen source model and the source validation
split, with no target labels and no target statistics:

  msp        maximum softmax probability                       (exp25's score)
  maxlogit   maximum logit, without the softmax normalisation
  entropy    Shannon entropy of the softmax
  energy     -logsumexp(logits), the free-energy OOD score
  maha       Mahalanobis distance to the nearest source class prototype, using the
             per-class mean and diagonal covariance of the source embedding
  cosine     1 - maximum cosine similarity to a source class prototype

Each is oriented so that larger means "more likely unknown". Thresholds are the
(100-q)-th percentile of the score on the SOURCE validation split, fixed before the target
stream is seen, exactly as in exp25.

Detectors are compared two ways. AUROC between known and unknown target samples is
threshold-free, so it measures the score itself rather than the calibration. H at the fixed
q is what a deployment actually obtains. Both are reported, because a score can separate
well and still calibrate badly across buildings.
"""
import os, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import train_mlp, TRANSFORMS
from tta import SourceOnly
from exp25_openset_tta import (load_inv, remap, openset_scores, NAMES, UNKNOWN, KNOWN,
                               Q_GRID, Q_HEAD)

D = os.path.join(os.path.dirname(__file__), "..", "data")
R = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS, BS = (0, 1, 2), 64


def auroc(score, is_unknown):
    """Rank-based AUROC; larger score should mean 'unknown'."""
    o = np.argsort(score); r = np.empty(len(score), float); r[o] = np.arange(len(score))
    npos = int(is_unknown.sum()); nneg = len(score) - npos
    if npos == 0 or nneg == 0: return float("nan")
    return float((r[is_unknown].sum() - npos * (npos - 1) / 2) / (npos * nneg))


@torch.no_grad()
def all_scores(model, Z, proto_mu=None, proto_var=None):
    """Every candidate score for one feature matrix. Larger = more unknown."""
    x = torch.tensor(Z, dtype=torch.float32)
    logit, feat = model(x, return_feat=True)
    p = torch.softmax(logit, 1)
    out = {
        "msp":      (-p.max(1).values).numpy(),
        "maxlogit": (-logit.max(1).values).numpy(),
        "entropy":  (-(p * torch.log(p + 1e-12)).sum(1)).numpy(),
        "energy":   (-torch.logsumexp(logit, 1)).numpy(),
    }
    if proto_mu is not None:
        f = feat.numpy()
        d = ((f[:, None, :] - proto_mu[None]) ** 2 / (proto_var[None] + 1e-6)).mean(2)
        out["maha"] = d.min(1)
        fn = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)
        pn = proto_mu / np.maximum(np.linalg.norm(proto_mu, axis=1, keepdims=True), 1e-9)
        out["cosine"] = 1.0 - (fn @ pn.T).max(1)
    return out


@torch.no_grad()
def prototypes(model, Z, y, Kc):
    _, feat = model(torch.tensor(Z, dtype=torch.float32), return_feat=True)
    f = feat.numpy()
    mu = np.stack([f[y == c].mean(0) for c in range(Kc)])
    var = np.stack([f[y == c].var(0) + 1e-6 for c in range(Kc)])
    return mu, var


if __name__ == "__main__":
    A = {t: load_inv(t) for t in ("S1", "S2")}
    Kc = len(KNOWN); rows = []
    print("=" * 94)
    print("EXPERIMENT 26 — unknown-detection scores, all source-calibrated")
    print(f"  known {[NAMES[k] for k in KNOWN]}   unknown {[NAMES[k] for k in UNKNOWN]}")
    print("=" * 94)

    for srcb, tgt in (("S1", "S2"), ("S2", "S1")):
        Xs, ys, ds = A[srcb]; Xt, yt, _ = A[tgt]
        smask = np.isin(ys, KNOWN); is_unknown = np.isin(yt, UNKNOWN)
        yt_k = remap(yt, KNOWN)
        folds = np.array_split(np.unique(ds), 5)
        va = np.isin(ds, folds[-1]); tr = (~va) & smask; vas = va & smask
        acc = {}
        for sd in SEEDS:
            tf = TRANSFORMS["zscore"](Xs[tr])
            Ztr, Zva, Zt = tf(Xs[tr]), tf(Xs[vas]), tf(Xt)
            ytr_k = remap(ys[tr], KNOWN)
            src, _ = train_mlp(Ztr, ytr_k, Zva, remap(ys[vas], KNOWN),
                               Xs.shape[1], seed=sd, epochs=60, K_out=Kc)
            mu, var = prototypes(src, Ztr, ytr_k, Kc)
            sv = all_scores(src, Zva, mu, var)          # SOURCE validation -> thresholds
            tg = all_scores(src, Zt, mu, var)           # target stream
            ad = SourceOnly(src); pk = np.zeros(len(yt), int)
            for i in range(0, len(Zt), BS):
                pk[i:i + BS] = ad(torch.tensor(Zt[i:i + BS], dtype=torch.float32))[0]
            for name in tg:
                a = auroc(tg[name], is_unknown)
                for q in Q_GRID:
                    t = float(np.percentile(sv[name], 100 - q))
                    s = openset_scores(yt_k, pk, is_unknown, tg[name] > t, Kc)
                    s["auroc"] = a
                    acc.setdefault((name, q), []).append(s)

        print(f"\n{srcb}->{tgt}   unknown share {is_unknown.mean():.0%}")
        print(f"  {'score':<12}{'AUROC':>9}{'OS*':>9}{'UCDR':>9}{'H':>9}{'macroF1(k)':>12}")
        print("  " + "-" * 60)
        best = sorted({n for n, _ in acc},
                      key=lambda n: -np.mean([x["auroc"] for x in acc[(n, Q_HEAD)]]))
        for name in best:
            v = acc[(name, Q_HEAD)]
            m = {k: np.mean([x[k] for x in v]) for k in v[0]}
            star = " *" if name == best[0] else ""
            print(f"  {name:<12}{m['auroc']:>9.4f}{m['os_star']:>9.4f}{m['ucdr']:>9.4f}"
                  f"{m['h']:>9.4f}{m['macroF1']:>12.4f}{star}")
            for q in Q_GRID:
                vq = acc[(name, q)]
                mq = {k: np.mean([x[k] for x in vq]) for k in vq[0]}
                rows.append(dict(direction=f"{srcb}->{tgt}", score=name, q=q, **mq))
        bn = best[0]
        print(f"\n  threshold sensitivity for {bn}:")
        for q in Q_GRID:
            v = acc[(bn, q)]
            m = {k: np.mean([x[k] for x in v]) for k in v[0]}
            print(f"    q={q:<5} OS*={m['os_star']:.4f}  UCDR={m['ucdr']:.4f}  H={m['h']:.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(R, "exp26_unknown_scores.csv"), index=False)
    print("\n-> results/exp26_unknown_scores.csv")
