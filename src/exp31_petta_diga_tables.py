"""
EXPERIMENT 31 — regenerate two tables that had no stored source.

A number-verification pass over the closed-set manuscript found two tables whose values
could not be traced to any result file:

  Table~\ref{tab:petta}  the PeTTA divergence statistics (summed versus averaged
                         reduction, the gate value, the resulting alpha_t, and the share
                         of Mahalanobis mass carried by near-degenerate dimensions). These
                         were produced by an ad-hoc script and reported from its console
                         output.
  Table~\ref{tab:diga}   DIGA under two readings of Eq. 8 crossed with two hyperparameter
                         settings. Its source comment pointed at /tmp/diga_sim.py, a file
                         that no longer exists.

Both are regenerated here and written to results/, so every value in either table is
reproducible from the repository. This is the fourth instance of a number reaching a draft
from a run log rather than a stored artefact; the README records the standing rule.
"""
import os, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import MLP, train_mlp, metrics, TRANSFORMS
from tta import SourceOnly, source_feature_stats
from tta_faithful import DIGAFaithful

D = os.path.join(os.path.dirname(__file__), "..", "data")
R = os.path.join(os.path.dirname(__file__), "..", "results")
BS = 64


def load(tag):
    z = np.load(os.path.join(D, f"feat_inv_{tag}.npz"), allow_pickle=True)
    cols = [str(s) for s in z["names"]]
    m = np.array([c.startswith(("g_", "t_")) for c in cols])
    return z["X"].astype(np.float64)[:, m], z["y"]


if __name__ == "__main__":
    X1, y1 = load("S1"); X2, y2 = load("S2")
    z = np.load(os.path.join(D, "feat_inv_S1.npz"), allow_pickle=True)
    d1 = z["day"]
    days = np.unique(d1); va = np.isin(d1, np.array_split(days, 5)[-1]); tr = ~va
    tf = TRANSFORMS["zscore"](X1[tr])
    Z1, Z1v, Z2 = tf(X1[tr]), tf(X1[va]), tf(X2)
    src, _ = train_mlp(Z1, y1[tr], Z1v, y1[va], X1.shape[1], seed=0, epochs=60)
    mu0, var0 = source_feature_stats(src, Z1, y1[tr])
    mu0 = np.asarray(mu0 if not torch.is_tensor(mu0) else mu0.numpy())
    var0 = np.asarray(var0 if not torch.is_tensor(var0) else var0.numpy())

    rows = []
    # ---------------------------------------------- PeTTA divergence (Table tab:petta)
    with torch.no_grad():
        _, f2 = src(torch.tensor(Z2, dtype=torch.float32), return_feat=True)
    f2 = f2.numpy()
    per = (f2[:, None, :] - mu0[None]) ** 2 / (var0[None] + 1e-6)
    dead = var0 < 1e-6
    share = 100 * (per * dead[None]).sum() / per.sum()
    print("=" * 84)
    print("EXPERIMENT 31 — regenerating two previously unsourced tables")
    print("=" * 84)
    print(f"\nPeTTA divergence, S1->S2 source model")
    print(f"  Sigma_0 entries < 1e-6: {dead.mean()*100:.1f}%  "
          f"carrying {share:.1f}% of the Mahalanobis mass")
    for name, stat in (("sum over embedding dims", per.sum(2)),
                       ("per-dimension mean (released)", per.mean(2))):
        d = stat.min(1); g = 1 - np.exp(-d)
        a = (1 - g.mean()) * 1e-3
        print(f"  {name:<32} mean d={d.mean():8.2f}  median d={np.median(d):8.2f}  "
              f"mean gamma={g.mean():.6f}  alpha_t={a:.2e}")
        rows.append(dict(table="tab:petta", variant=name,
                         mean_d=float(d.mean()), median_d=float(np.median(d)),
                         mean_gamma=float(g.mean()), median_gamma=float(np.median(g)),
                         alpha_t=float(a), dead_frac=float(dead.mean()),
                         mass_share=float(share)))

    # ---------------------------------------------- DIGA Eq.8 readings (Table tab:diga)
    p0 = np.concatenate([SourceOnly(src)(torch.tensor(Z2[i:i+BS], dtype=torch.float32))[0]
                         for i in range(0, len(Z2), BS)])
    print(f"\nDIGA, two readings of Eq.8 x two hyperparameter settings")
    print(f"  {'reading':<22}{'hyperparameters':<22}{'changed':>12}{'macroF1':>10}{'MCC':>9}")
    for sim in ("neg_inner", "neg_l2", "cosine"):
        for hp, cfg in (("published 0.8/0.9", dict(lam_bn=0.8, P0=0.9)),
                        ("retuned 0.95/0.50", dict(lam_bn=0.95, P0=0.50, lam_F=0.80, topK=32))):
            c = dict(cfg); c["sim"] = sim
            ad = DIGAFaithful(src, cfg=c)
            pr = np.concatenate([ad(torch.tensor(Z2[i:i+BS], dtype=torch.float32))[0]
                                 for i in range(0, len(Z2), BS)])
            m = metrics(y2, pr)
            print(f"  {sim:<22}{hp:<22}{int((pr!=p0).sum()):>8}/{len(y2)}"
                  f"{m['macroF1']:>10.4f}{m['mcc']:>9.4f}")
            rows.append(dict(table="tab:diga", variant=f"{sim} | {hp}",
                             changed=int((pr != p0).sum()),
                             macroF1=m["macroF1"], mcc=m["mcc"]))
    m0 = metrics(y2, p0)
    rows.append(dict(table="tab:diga", variant="source-only",
                     changed=0, macroF1=m0["macroF1"], mcc=m0["mcc"]))
    print(f"  {'source-only':<44}{0:>8}/{len(y2)}{m0['macroF1']:>10.4f}{m0['mcc']:>9.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(R, "exp31_petta_diga_tables.csv"), index=False)
    print("\n-> results/exp31_petta_diga_tables.csv")
