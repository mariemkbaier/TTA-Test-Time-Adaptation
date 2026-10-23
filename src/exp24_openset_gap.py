"""
EXPERIMENT 24 — the cross-building penalty in all four cells of the open-set grid.

Before any adaptation method is run, each cell needs its own baseline, because withholding
classes from source training changes the source model and therefore changes what transfer
costs. The measurement is the one from Section~\ref{sec:res-gap}: hold the evaluation set
fixed, vary only where the training labels came from, match training-set size.

  closed-set   source trained on all five activities
  open-set     source trained on the three KNOWN activities only
               (meal_prep, watching_tv, music); toileting and cleaning are target-private

In the open-set cells the penalty is measured on known-class samples only. A source model
that never saw the withheld activities cannot be expected to classify them, so including
them would mix the geographic shift with the label-space shift, which is precisely what
this grid is meant to separate. Unknown-class detection is measured separately, by
exp25, under the no-target-label constraint; this script is a diagnostic and legitimately
uses target labels to build the oracle, exactly as exp21 does.

Protocol: 13-d layout-invariant causal features, GroupKFold over year_day, 3 seeds,
training-set sizes matched by subsampling.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import metrics, trivial_baselines, TRANSFORMS, train_mlp, predict
from build_features import load_raw, segments

D = os.path.join(os.path.dirname(__file__), "..", "data")
R = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS = (0, 1, 2)
NAMES = {0: "toileting", 1: "meal_prep", 2: "watching_tv", 3: "music", 4: "cleaning"}
UNKNOWN = (0, 4)                       # toileting, cleaning
KNOWN = tuple(k for k in range(5) if k not in UNKNOWN)
SEG = {t: segments(load_raw(t), "recording") for t in ("S1", "S2")}


def load_inv(tag):
    z = np.load(os.path.join(D, f"feat_inv_{tag}.npz"), allow_pickle=True)
    cols = [str(s) for s in z["names"]]
    m = np.array([c.startswith(("g_", "t_")) for c in cols])
    return z["X"].astype(np.float64)[:, m], z["y"], z["day"]


def causal_median(pred, seg, L=41, K=5):
    out = np.empty_like(pred)
    for i in range(len(pred)):
        lo = i
        while lo > 0 and i - lo < L - 1 and seg[lo - 1] == seg[i]:
            lo -= 1
        out[i] = np.bincount(pred[lo:i + 1], minlength=K).argmax()
    return out


def remap(y, classes):
    """Relabel a subset of classes to 0..k-1 so the head has the right width."""
    lut = {c: i for i, c in enumerate(classes)}
    return np.array([lut[v] for v in y], dtype=np.int64)


if __name__ == "__main__":
    A = {t: load_inv(t) for t in ("S1", "S2")}
    rows = []
    print("=" * 96)
    print("EXPERIMENT 24 — cross-building penalty, closed-set and open-set")
    print(f"  known   {[NAMES[k] for k in KNOWN]}")
    print(f"  unknown {[NAMES[k] for k in UNKNOWN]}  (target-private, withheld from source)")
    print("=" * 96)
    print(f"  {'cell':<22}{'train on':<24}{'n_train':>9}{'macroF1':>10}{'sd':>8}{'penalty':>10}")
    print("  " + "-" * 83)

    for setting in ("closed-set", "open-set"):
        classes = tuple(range(5)) if setting == "closed-set" else KNOWN
        Kc = len(classes)
        for tgt, srcb in (("S2", "S1"), ("S1", "S2")):
            Xt_all, yt_all, dt_all = A[tgt]
            Xs_all, ys_all, _ = A[srcb]
            # training pools contain only the classes the source is allowed to see
            smask = np.isin(ys_all, classes); tmask = np.isin(yt_all, classes)
            folds = np.array_split(np.unique(dt_all), 5)
            res = {}
            for mode in ("own days", "other building"):
                f1, ns = [], []
                for sd in SEEDS:
                    rng = np.random.default_rng(sd)
                    for f in folds:
                        te = np.isin(dt_all, f)
                        # evaluation: held-out target days, KNOWN-class rows only
                        ev = te & tmask
                        avail = (~te) & tmask
                        n_cap = int(avail.sum())
                        if mode == "own days":
                            trX, trY = Xt_all[avail], yt_all[avail]
                        else:
                            idx = np.where(smask)[0]
                            idx = rng.permutation(idx)[:n_cap]
                            trX, trY = Xs_all[idx], ys_all[idx]
                        trY = remap(trY, classes)
                        cut = max(1, int(0.2 * len(trX)))
                        p = rng.permutation(len(trX)); vi, ti = p[:cut], p[cut:]
                        tf = TRANSFORMS["zscore"](trX[ti])
                        m, _ = train_mlp(tf(trX[ti]), trY[ti], tf(trX[vi]), trY[vi],
                                         trX.shape[1], seed=sd, epochs=60, K_out=Kc)
                        yp, _ = predict(m, tf(Xt_all))
                        yh = causal_median(yp, SEG[tgt], K=Kc)[ev]
                        f1.append(metrics(remap(yt_all[ev], classes), yh,
                                          K=Kc)["macroF1"])
                        ns.append(len(ti))
                res[mode] = np.mean(f1)
                pen = "" if mode == "own days" else f"{(res['own days']-np.mean(f1))*100:>9.1f}p"
                print(f"  {setting+' '+srcb+'->'+tgt:<22}{mode:<24}{np.mean(ns):>9.0f}"
                      f"{np.mean(f1):>10.4f}{np.std(f1):>8.3f}{pen:>10}")
                rows.append(dict(setting=setting, direction=f"{srcb}->{tgt}", mode=mode,
                                 n_train=np.mean(ns), macroF1=np.mean(f1), sd=np.std(f1)))
            rows.append(dict(setting=setting, direction=f"{srcb}->{tgt}", mode="penalty",
                             macroF1=res["own days"] - res["other building"]))
        print()
    pd.DataFrame(rows).to_csv(os.path.join(R, "exp24_openset_gap.csv"), index=False)
    print("-> results/exp24_openset_gap.csv")
