"""
EXPERIMENT 30 — can the temporal axis be exploited, and may it be?

exp29 found that t_time_cos alone separates the withheld activities better in S2 (AUROC
0.6972) than the full 13-d Mahalanobis detector does (0.6377). The obvious move is to
build a detector around time of day.

There is a problem with the obvious move, and it is the problem this project keeps
documenting in other people's work. We learned that time of day is the discriminative axis
by looking at the TARGET labels. A detector that upweights that axis because of what we
saw there is selected with target information, through the analyst rather than through the
code, and reporting it as Tier-1 would be the same error as choosing DIGA's
hyperparameters on target macro-F1. Any variant below marked [post-hoc] is a hypothesis
generated on this data and is NOT a deployable result; it needs a held-out building to
become one.

So the script reports two families.

  [post-hoc]  temporal-only, time-of-day-only, and an upweighted 13-d distance. These
              bound what exploiting the axis is worth, and they are honest only as upper
              bounds.

  [tier-1]    leave-one-class-out weighting. For each KNOWN class in turn, hold it out of
              the source prototypes, score the held-out class's SOURCE windows as if they
              were novel, and record which dimensions detect them. Averaging over the
              three known classes gives a per-dimension weight estimated from source data
              alone, with no target information of any kind. If this rule happens to
              select the temporal axis, the gain is deployable; if it does not, the
              post-hoc numbers stay a hypothesis.

Also reported: max-over-dimensions instead of mean, which is the other way to stop a
single informative axis being diluted by twelve uninformative ones.
"""
import os, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import train_mlp, TRANSFORMS
from tta import SourceOnly
from exp25_openset_tta import load_inv, remap, openset_scores, KNOWN, UNKNOWN, NAMES, Q_GRID, Q_HEAD
from exp26_unknown_scores import auroc

D = os.path.join(os.path.dirname(__file__), "..", "data")
R = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS, BS = (0, 1, 2), 64
COLS = ["g_mean_active", "g_max_active", "g_act_entropy", "g_trans_count", "g_trans_rate",
        "g_std_active", "g_min_active", "g_med_active", "g_frac_idle",
        "t_time_sin", "t_time_cos", "t_day_sin", "t_day_cos"]
TEMP = np.array([c.startswith("t_") for c in COLS])
TOD = np.array([c in ("t_time_sin", "t_time_cos") for c in COLS])


def proto_stats(Ftr, ytr, Kc):
    mu = np.stack([Ftr[ytr == c].mean(0) for c in range(Kc)])
    var = np.stack([Ftr[ytr == c].var(0) + 1e-6 for c in range(Kc)])
    return mu, var


def dist(F, mu, var, w=None, reduce="mean"):
    """Per-dimension squared Mahalanobis, reduced to one number per sample."""
    d = (F[:, None, :] - mu[None]) ** 2 / (var[None] + 1e-6)
    if w is not None:
        d = d * w[None, None, :]
        d = d.sum(2) / w.sum()
    else:
        d = d.max(2) if reduce == "max" else d.mean(2)
    return d.min(1)


def loco_weights(Xtr, ytr, Kc):
    """Per-dimension weights from SOURCE ONLY: how well each dimension flags a held-out
    known class as novel, averaged over the three leave-one-class-out folds."""
    w = np.zeros(Xtr.shape[1])
    for c in range(Kc):
        keep = ytr != c
        others = [k for k in range(Kc) if k != c]
        mu = np.stack([Xtr[ytr == k].mean(0) for k in others])
        var = np.stack([Xtr[ytr == k].var(0) + 1e-6 for k in others])
        held = ytr == c
        pos = np.concatenate([np.ones(held.sum(), bool), np.zeros(keep.sum(), bool)])
        for j in range(Xtr.shape[1]):
            dj = np.concatenate([
                ((Xtr[held, j][:, None] - mu[:, j][None]) ** 2 / (var[:, j][None] + 1e-6)).min(1),
                ((Xtr[keep, j][:, None] - mu[:, j][None]) ** 2 / (var[:, j][None] + 1e-6)).min(1)])
            w[j] += max(auroc(dj, pos) - 0.5, 0.0)
    return w / max(w.sum(), 1e-9) * len(w)      # mean weight 1


if __name__ == "__main__":
    A = {t: load_inv(t) for t in ("S1", "S2")}
    Kc = len(KNOWN); rows = []
    print("=" * 100)
    print("EXPERIMENT 30 — temporal-axis detectors")
    print(f"  [post-hoc] variants select the axis using target knowledge; upper bounds only")
    print("=" * 100)

    for srcb, tgt in (("S1", "S2"), ("S2", "S1")):
        Xs, ys, ds = A[srcb]; Xt, yt, _ = A[tgt]
        smask = np.isin(ys, KNOWN); unk = np.isin(yt, UNKNOWN); yt_k = remap(yt, KNOWN)
        folds = np.array_split(np.unique(ds), 5)
        va = np.isin(ds, folds[-1]); tr = (~va) & smask; vas = va & smask
        acc = {}; wlog = []
        for sd in SEEDS:
            tf = TRANSFORMS["zscore"](Xs[tr])
            Ztr, Zva, Zt = tf(Xs[tr]), tf(Xs[vas]), tf(Xt)
            ytr_k = remap(ys[tr], KNOWN)
            src, _ = train_mlp(Ztr, ytr_k, Zva, remap(ys[vas], KNOWN),
                               Xs.shape[1], seed=sd, epochs=60, K_out=Kc)
            ad = SourceOnly(src); pk = np.zeros(len(yt), int)
            for i in range(0, len(Zt), BS):
                pk[i:i + BS] = ad(torch.tensor(Zt[i:i + BS], dtype=torch.float32))[0]
            mu, var = proto_stats(Ztr, ytr_k, Kc)
            wl = loco_weights(Ztr, ytr_k, Kc); wlog.append(wl)
            up = np.where(TEMP, 5.0, 1.0)
            V = {
              "13-d mean [tier-1]":        (dist(Zva, mu, var), dist(Zt, mu, var)),
              "13-d max-dim [tier-1]":     (dist(Zva, mu, var, reduce="max"),
                                            dist(Zt, mu, var, reduce="max")),
              "LOCO-weighted [tier-1]":    (dist(Zva, mu, var, w=wl), dist(Zt, mu, var, w=wl)),
              "temporal 4-d [post-hoc]":   (dist(Zva[:, TEMP], mu[:, TEMP], var[:, TEMP]),
                                            dist(Zt[:, TEMP], mu[:, TEMP], var[:, TEMP])),
              "time-of-day 2-d [post-hoc]": (dist(Zva[:, TOD], mu[:, TOD], var[:, TOD]),
                                             dist(Zt[:, TOD], mu[:, TOD], var[:, TOD])),
              "temporal x5 [post-hoc]":    (dist(Zva, mu, var, w=up), dist(Zt, mu, var, w=up)),
            }
            SCORES = {k: v[1] for k, v in V.items()}
            for name, (sv, tg) in V.items():
                a = auroc(tg, unk)
                for q in Q_GRID:
                    t = float(np.percentile(sv, 100 - q)); rej = tg > t
                    s = openset_scores(yt_k, pk, unk, rej, Kc)
                    s.update(auroc=a, rej_rate=float(rej.mean()))
                    acc.setdefault((name, q), []).append(s)
                t2 = float(np.quantile(tg, 0.80))
                acc.setdefault((name, "m20"), []).append(
                    openset_scores(yt_k, pk, unk, tg > t2, Kc))

        print(f"\n{srcb}->{tgt}   unknown {unk.mean():.0%}    "
              f"(ceiling {0.7354 if tgt=='S2' else 0.6327:.4f}, "
              f"best single feature {0.6972 if tgt=='S2' else 0.3700:.4f})")
        print(f"  {'variant':<26}{'AUROC':>8}{'OS*':>8}{'UCDR':>8}{'H@5%':>8}"
              f"{'rej':>7}{'  UCDR@20%':>11}")
        print("  " + "-" * 76)
        names = sorted({n for n, _ in acc},
                       key=lambda n: -np.mean([x["auroc"] for x in acc[(n, Q_HEAD)]]))
        for name in names:
            v = acc[(name, Q_HEAD)]; m = {k: np.mean([x[k] for x in v]) for k in v[0]}
            u20 = np.mean([x["ucdr"] for x in acc[(name, "m20")]])
            print(f"  {name:<26}{m['auroc']:>8.4f}{m['os_star']:>8.4f}{m['ucdr']:>8.4f}"
                  f"{m['h']:>8.4f}{m['rej_rate']:>7.1%}{u20:>11.4f}")
            for q in Q_GRID:
                vq = acc[(name, q)]
                rows.append(dict(direction=f"{srcb}->{tgt}", variant=name, q=q,
                                 **{k: np.mean([x[k] for x in vq]) for k in vq[0]}))
            rows.append(dict(direction=f"{srcb}->{tgt}", variant=name, q="m20",
                             **{k: np.mean([x[k] for x in acc[(name,'m20')]])
                                for k in acc[(name,'m20')][0]}))
        W = np.mean(wlog, 0)
        for j, c in enumerate(COLS):
            rows.append(dict(direction=f"{srcb}->{tgt}", variant="LOCO weight",
                             q=c, auroc=float(W[j])))
        rows.append(dict(direction=f"{srcb}->{tgt}", variant="LOCO weight",
                         q="temporal mean", auroc=float(W[TEMP].mean())))
        rows.append(dict(direction=f"{srcb}->{tgt}", variant="LOCO weight",
                         q="global mean", auroc=float(W[~TEMP].mean())))
        top = np.argsort(-W)[:4]
        print(f"  LOCO weights (source only), top 4: "
              + ", ".join(f"{COLS[j]}={W[j]:.2f}" for j in top))
        print(f"    temporal dims mean weight {W[TEMP].mean():.2f} vs "
              f"global {W[~TEMP].mean():.2f}")
        # paired bootstrap on the target stream: is each variant separated from the
        # 13-d mean baseline, or within resampling noise?
        rng = np.random.default_rng(0)
        ridx = [rng.integers(0, len(yt), len(yt)) for _ in range(2000)]
        base = SCORES["13-d mean [tier-1]"]
        print(f"  paired bootstrap vs 13-d mean (2000 resamples):")
        for name, tg in SCORES.items():
            d = [auroc(tg[i], unk[i]) - auroc(base[i], unk[i]) for i in ridx]
            lo, hi = np.percentile(d, [2.5, 97.5])
            mark = "separated" if (lo > 0 or hi < 0) else "within noise"
            print(f"    {name:<26}{auroc(tg, unk):>8.4f}  delta [{lo:+.4f},{hi:+.4f}]  {mark}")
            rows.append(dict(direction=f"{srcb}->{tgt}", variant=name, q="boot_vs_13d",
                             auroc=auroc(tg, unk),
                             delta=auroc(tg, unk) - auroc(base, unk),
                             ci_lo=lo, ci_hi=hi, verdict=mark))

    pd.DataFrame(rows).to_csv(os.path.join(R, "exp30_temporal_detector.csv"), index=False)
    print("\n-> results/exp30_temporal_detector.csv")
