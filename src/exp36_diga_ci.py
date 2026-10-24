"""
EXPERIMENT 36 — a confidence interval for DIGA's gain, and the right resampling unit.

Table~\ref{tab:harmethods} reports DIGA gaining 2.1 macro-F1 points entering S1, the only
positive entry in the activity evaluation. A single point estimate is not enough to carry
that, so this script attaches a paired bootstrap to it, as Section 3.5 promises for
differences.

The resampling unit matters here more than usual, and the paper's own Section 4 is the
reason. Features are computed over stride-1 windows sharing nineteen of twenty timesteps,
so windows are not independent draws. Resampling them individually treats near-duplicates
as fresh evidence and narrows the interval accordingly. The unit that respects the
dependence is the recording day, which is also the unit the validation split uses. Both are
computed below so the difference is visible rather than asserted.

Protocol matches exp23 exactly: 13-d layout-invariant causal features, last GroupKFold fold
of the source days for the checkpoint, 3 seeds, batch 64, no smoothing, scored over the
whole target apartment.
"""
import os, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import metrics, train_mlp, TRANSFORMS
from tta import SourceOnly
from tta_faithful import DIGAFaithful

D = os.path.join(os.path.dirname(__file__), "..", "data")
R = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS, BS, B = (0, 1, 2), 64, 2000


def load(tag):
    z = np.load(os.path.join(D, f"feat_inv_{tag}.npz"), allow_pickle=True)
    cols = [str(s) for s in z["names"]]
    m = np.array([c.startswith(("g_", "t_")) for c in cols])
    return z["X"].astype(np.float64)[:, m], z["y"], z["day"]


if __name__ == "__main__":
    X1, y1, d1 = load("S1"); X2, y2, d2 = load("S2")
    rows = []
    print("=" * 92)
    print("EXPERIMENT 36 — paired bootstrap on DIGA vs source-only")
    print("=" * 92)
    for dirn, Xs, ys, ds, Xt, yt, dt in (
            ("S2->S1", X2, y2, d2, X1, y1, d1),
            ("S1->S2", X1, y1, d1, X2, y2, d2)):
        folds = np.array_split(np.unique(ds), 5)
        P0, PD, PP = [], [], []
        for sd in SEEDS:
            va = np.isin(ds, folds[-1]); tr = ~va
            tf = TRANSFORMS["zscore"](Xs[tr])
            Ztr, Zva, Zt = tf(Xs[tr]), tf(Xs[va]), tf(Xt)
            src, _ = train_mlp(Ztr, ys[tr], Zva, ys[va], Xs.shape[1],
                               seed=sd, epochs=60)

            def run(mk):
                ad = mk(); pr = np.zeros(len(yt), int)
                for i in range(0, len(Zt), BS):
                    pr[i:i + BS] = ad(torch.tensor(Zt[i:i + BS], dtype=torch.float32))[0]
                return pr
            P0.append(run(lambda: SourceOnly(src)))
            PD.append(run(lambda: DIGAFaithful(src)))
            # lam_BN = 1.0 leaves the normalisation statistics untouched, so this
            # variant isolates the prototype classifier from the normalisation step
            PP.append(run(lambda: DIGAFaithful(src, cfg=dict(lam_bn=1.0))))

        days = np.unique(dt)
        idx_by_day = {d: np.where(dt == d)[0] for d in days}
        print(f"\n{dirn}: target has {len(yt)} windows over {len(days)} recording days")

        for variant, PX in (("DIGA, published", PD), ("DIGA, lam_BN=1.0", PP)):
            pt = np.mean([metrics(yt, d)["macroF1"] - metrics(yt, p)["macroF1"]
                          for p, d in zip(P0, PX)]) * 100
            print(f"  {variant:<20} point estimate {pt:+.2f} macro-F1 points")

            for unit in ("recording day", "window"):
                rng = np.random.default_rng(0); draws = []
                for _ in range(B):
                    if unit == "recording day":
                        pick = rng.choice(days, size=len(days), replace=True)
                        idx = np.concatenate([idx_by_day[d] for d in pick])
                    else:
                        idx = rng.integers(0, len(yt), len(yt))
                    yr = yt[idx]
                    if len(np.unique(yr)) < 2: continue
                    draws.append(np.mean([metrics(yr, d[idx])["macroF1"]
                                          - metrics(yr, p[idx])["macroF1"]
                                          for p, d in zip(P0, PX)]) * 100)
                lo, hi = np.percentile(draws, [2.5, 97.5])
                inc = "INCLUDES zero" if lo <= 0 <= hi else "excludes zero"
                print(f"    resampled by {unit:<14} 95% CI [{lo:+.2f}, {hi:+.2f}]  {inc}")
                rows.append(dict(direction=dirn, variant=variant, unit=unit, delta_pts=pt,
                                 ci_lo=lo, ci_hi=hi, includes_zero=bool(lo <= 0 <= hi),
                                 n_windows=len(yt), n_days=len(days)))
    pd.DataFrame(rows).to_csv(os.path.join(R, "exp36_diga_ci.csv"), index=False)
    print("\n-> results/exp36_diga_ci.csv")
