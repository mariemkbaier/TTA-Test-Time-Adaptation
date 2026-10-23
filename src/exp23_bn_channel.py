"""
EXPERIMENT 23 — the normalisation channel, and what every method does through it.

Each of the four adaptation methods here touches batch-norm statistics, and the released
code of each one makes that its primary mechanism:

  AdaBN    replaces source statistics with the test batch's outright
  DIGA     blends them, keeping a fraction lam_BN of the source estimate (Eq. 3)
  PETAL    configure_model() sets track_running_stats=False and discards running_mean /
           running_var, so student AND the deepcopy'd EMA teacher run on batch statistics
  PeTTA    installs RobustBN1d, which normalises from the test stream

Our earlier implementation left the source estimates in place and ran the teachers in
eval(), which removes that channel and leaves only the gradient machinery. This script
measures both, in both directions, so the two can be separated:

  as published   the BN channel active, as each method's own code configures it
  no-BN control  identical in every other respect, source statistics frozen

and then sweeps lam_BN from 1.0 (pure source) to 0.0 (pure target) to show the channel is
a continuum rather than a switch.

Protocol: 13-d layout-invariant causal features, last GroupKFold fold for the source
checkpoint, 3 seeds, batch 64, no smoothing.
"""
import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from common import K, MLP, train_mlp, metrics, trivial_baselines, TRANSFORMS
from tta import SourceOnly, AdaBN, TENT, source_feature_stats
from tta_faithful import PETALFaithful, PeTTAFaithful, DIGAFaithful
from petal_fixed import SWAGDiag

D = os.path.join(os.path.dirname(__file__), "..", "data")
R = os.path.join(os.path.dirname(__file__), "..", "results")
BS, SEEDS = 64, (0, 1, 2)


def load(tag):
    z = np.load(os.path.join(D, f"feat_inv_{tag}.npz"), allow_pickle=True)
    return z["X"].astype(np.float64), z["y"], z["day"], [str(s) for s in z["names"]]


def fit_swag(d_in, src, Z, y, seed, epochs=30):
    torch.manual_seed(seed); np.random.seed(seed)
    sw = MLP(d_in); sw.load_state_dict(src.state_dict())
    opt = torch.optim.Adam(sw.parameters(), lr=1e-3); lf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Z, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.long)
    snaps = []
    for _ in range(epochs):
        sw.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 128):
            j = perm[i:i + 128]
            if len(j) < 2: continue
            opt.zero_grad(); lf(sw(Xt[j]), yt[j]).backward(); opt.step()
        snaps.append(torch.cat([p.data.view(-1) for p in sw.parameters()]).clone())
    return SWAGDiag().fit(snaps)


if __name__ == "__main__":
    X1, y1, d1, cols = load("S1"); X2, y2, d2, _ = load("S2")
    mask = np.array([c.startswith(("g_", "t_")) for c in cols]); d_in = int(mask.sum())
    DIRS = {"S1->S2": (X1[:, mask], y1, d1, X2[:, mask], y2),
            "S2->S1": (X2[:, mask], y2, d2, X1[:, mask], y1)}
    rows = []

    for dirn, (Xs, ys, ds, Xt, yt) in DIRS.items():
        folds = np.array_split(np.unique(ds), 5)
        acc = {}
        for sd in SEEDS:
            va = np.isin(ds, folds[-1]); tr = ~va
            tf = TRANSFORMS["zscore"](Xs[tr])
            Ztr, Zva, Zt = tf(Xs[tr]), tf(Xs[va]), tf(Xt)
            src, _ = train_mlp(Ztr, ys[tr], Zva, ys[va], d_in, seed=sd, epochs=60)
            swag = fit_swag(d_in, src, Ztr, ys[tr], sd)
            st = source_feature_stats(src, Ztr, ys[tr])

            def run(mk):
                ad = mk(); pr = np.zeros(len(yt), int)
                for i in range(0, len(Zt), BS):
                    pr[i:i + BS] = ad(torch.tensor(Zt[i:i + BS], dtype=torch.float32))[0]
                return pr

            p0 = run(lambda: SourceOnly(src))
            cases = [
                ("source-only",            lambda: SourceOnly(src)),
                ("AdaBN",                  lambda: AdaBN(src)),
                ("TENT",                   lambda: TENT(src)),
                ("DIGA (published)",       lambda: DIGAFaithful(src)),
                ("PETAL (published)",      lambda: PETALFaithful(src, swag)),
                ("PeTTA (published)",      lambda: PeTTAFaithful(src, src_stats=st)),
                ("PETAL, BN channel off",  lambda: PETALFaithful(src, swag,
                                               cfg=dict(batch_stat_bn=False))),
                ("PeTTA, BN channel off",  lambda: PeTTAFaithful(src, src_stats=st,
                                               cfg=dict(batch_stat_bn=False))),
            ]
            for lb in (1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0):
                cases.append((f"DIGA lam_BN={lb:.1f}",
                              lambda lb=lb: DIGAFaithful(src, cfg=dict(lam_bn=lb))))
            for k, mk in cases:
                pr = run(mk); m = metrics(yt, pr)
                acc.setdefault(k, []).append((m["macroF1"], m["mcc"],
                                              int((pr != p0).sum())))
        base = np.mean([a for a, _, _ in acc["source-only"]])
        print("=" * 92)
        print(f"EXPERIMENT 23 — {dirn}   chance ≈ {trivial_baselines(yt)['strat_macroF1']:.4f}")
        print("=" * 92)
        print(f"  {'method':<26}{'macroF1':>10}{'sd':>8}{'MCC':>9}{'vs src':>10}{'changed':>12}")
        print("  " + "-" * 75)
        for k, v in acc.items():
            f = [a for a, _, _ in v]; mc = [b for _, b, _ in v]; c = [x for _, _, x in v]
            print(f"  {k:<26}{np.mean(f):>10.4f}{np.std(f):>8.3f}{np.mean(mc):>9.4f}"
                  f"{(np.mean(f)-base)*100:>+9.1f}p{np.mean(c):>12.0f}")
            rows.append(dict(direction=dirn, method=k, macroF1=np.mean(f),
                             sd=np.std(f), mcc=np.mean(mc),
                             d_macroF1=np.mean(f) - base, changed=np.mean(c)))
        print()
    pd.DataFrame(rows).to_csv(os.path.join(R, "exp23_bn_channel.csv"), index=False)
    print("-> results/exp23_bn_channel.csv")
