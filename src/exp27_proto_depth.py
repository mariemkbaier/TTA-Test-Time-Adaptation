"""
EXPERIMENT 27 — where should the unknown score be computed, and is its gain real?

Two questions, answered on the same grid as exp26.

(1) Decomposition. Swapping maximum softmax for a Mahalanobis distance raised the H-score
by 10.9 and 20.1 points. H is a harmonic mean, so it rewards lifting whichever component
was the bottleneck, and unknown detection was. This script reports OS* and UCDR separately,
the fraction of the target stream each score rejects, and a matched-rejection-rate
comparison. The last of these is the fair test: if two scores reject the same number of
samples, the one that catches more true unknowns is genuinely the better detector rather
than merely the more trigger-happy one. Matching the rate uses a target quantile, so it is
a DIAGNOSTIC and is labelled as such; it is not a Tier-1 result.

(2) Depth. exp26 built prototypes from the final 64-d embedding, which is trained to
discard everything not needed to separate the three known classes -- including, plausibly,
what distinguishes an unseen activity. Two alternatives cost nothing to test:

    raw        distance on the 13 layout-invariant input features, no learned embedding
    bn1/bn2    distance in the 256-d and 128-d intermediate representations
    final      the 64-d embedding exp26 used

Every variant keeps the Tier-1 contract: prototypes and thresholds come from the source
training and validation splits, fixed before the target stream is read.
"""
import os, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import train_mlp, TRANSFORMS
from tta import SourceOnly
from exp25_openset_tta import (load_inv, remap, openset_scores, NAMES, UNKNOWN, KNOWN,
                               Q_GRID, Q_HEAD)
from exp26_unknown_scores import auroc

R = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS, BS = (0, 1, 2), 64
# self.feat is [Linear,BN,ReLU,Dropout] x 3 -> block k ends at index 4k
TAPS = {"bn1 (256-d)": 4, "bn2 (128-d)": 8, "final (64-d)": 12}


@torch.no_grad()
def rep(model, Z, upto):
    x = torch.tensor(Z, dtype=torch.float32)
    return torch.nn.Sequential(*list(model.feat)[:upto])(x).numpy()


def proto_dist(Ftr, ytr, F, Kc):
    """Mean squared Mahalanobis distance to the nearest source class prototype."""
    mu = np.stack([Ftr[ytr == c].mean(0) for c in range(Kc)])
    var = np.stack([Ftr[ytr == c].var(0) + 1e-6 for c in range(Kc)])
    d = ((F[:, None, :] - mu[None]) ** 2 / (var[None] + 1e-6)).mean(2)
    return d.min(1)


@torch.no_grad()
def msp_score(model, Z):
    p = torch.softmax(model(torch.tensor(Z, dtype=torch.float32)), 1)
    return (-p.max(1).values).numpy()


if __name__ == "__main__":
    A = {t: load_inv(t) for t in ("S1", "S2")}
    Kc = len(KNOWN); rows = []
    print("=" * 104)
    print("EXPERIMENT 27 — decomposition, rejection rates, and prototype depth")
    print(f"  known {[NAMES[k] for k in KNOWN]}   unknown {[NAMES[k] for k in UNKNOWN]}")
    print("=" * 104)

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
            ad = SourceOnly(src); pk = np.zeros(len(yt), int)
            for i in range(0, len(Zt), BS):
                pk[i:i + BS] = ad(torch.tensor(Zt[i:i + BS], dtype=torch.float32))[0]

            cand = {"msp (exp26 baseline)": (msp_score(src, Zva), msp_score(src, Zt)),
                    "raw 13-d input": (proto_dist(Ztr, ytr_k, Zva, Kc),
                                       proto_dist(Ztr, ytr_k, Zt, Kc))}
            for nm, upto in TAPS.items():
                Ftr = rep(src, Ztr, upto)
                cand["proto " + nm] = (proto_dist(Ftr, ytr_k, rep(src, Zva, upto), Kc),
                                       proto_dist(Ftr, ytr_k, rep(src, Zt, upto), Kc))
            for name, (sv, tg) in cand.items():
                a = auroc(tg, is_unknown)
                for q in Q_GRID:
                    t = float(np.percentile(sv, 100 - q))       # SOURCE-calibrated
                    rej = tg > t
                    s = openset_scores(yt_k, pk, is_unknown, rej, Kc)
                    s.update(auroc=a, rej_rate=float(rej.mean()))
                    acc.setdefault((name, q), []).append(s)
                # DIAGNOSTIC: threshold set so every score rejects 20% of the stream
                t2 = float(np.quantile(tg, 0.80))
                s2 = openset_scores(yt_k, pk, is_unknown, tg > t2, Kc)
                acc.setdefault((name, "matched20"), []).append(s2)

        print(f"\n{srcb}->{tgt}   unknown share {is_unknown.mean():.0%}   "
              f"(source-calibrated threshold, q={Q_HEAD}%)")
        print(f"  {'score':<22}{'AUROC':>8}{'OS*':>8}{'UCDR':>8}{'H':>8}"
              f"{'rejects':>9}{'  | UCDR @ 20% rejected':>24}")
        print("  " + "-" * 86)
        names = sorted({n for n, _ in acc},
                       key=lambda n: -np.mean([x["auroc"] for x in acc[(n, Q_HEAD)]]))
        for name in names:
            v = acc[(name, Q_HEAD)]; m = {k: np.mean([x[k] for x in v]) for k in v[0]}
            mm = acc[(name, "matched20")]
            u20 = np.mean([x["ucdr"] for x in mm]); h20 = np.mean([x["h"] for x in mm])
            print(f"  {name:<22}{m['auroc']:>8.4f}{m['os_star']:>8.4f}{m['ucdr']:>8.4f}"
                  f"{m['h']:>8.4f}{m['rej_rate']:>8.1%}   {u20:>10.4f} (H {h20:.4f})")
            for q in Q_GRID:
                vq = acc[(name, q)]
                mq = {k: np.mean([x[k] for x in vq]) for k in vq[0]}
                rows.append(dict(direction=f"{srcb}->{tgt}", score=name, q=q, **mq))
            mq = {k: np.mean([x[k] for x in mm]) for k in mm[0]}
            rows.append(dict(direction=f"{srcb}->{tgt}", score=name,
                             q="matched20 (diagnostic)", **mq))
        print(f"  chance UCDR at 20% rejection = 0.2000")
    pd.DataFrame(rows).to_csv(os.path.join(R, "exp27_proto_depth.csv"), index=False)
    print("\n-> results/exp27_proto_depth.csv")
