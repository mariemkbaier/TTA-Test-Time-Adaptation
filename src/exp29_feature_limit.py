"""
EXPERIMENT 29 — are the withheld activities separable in the 13-d description at all?

Section~\ref{sec:os-depth} observes that prototype distance on the raw input matches
prototype distance in the learned embedding, and reads that as a limit of the features
rather than of the representation. That reading is an inference, and this script tests it.

Two measurements, both on the target building:

  marginal   for each of the 13 features in turn, the AUROC separating unknown-class
             windows from known-class ones. A feature carrying no information about
             novelty sits at 0.5. Reported with Benjamini-Hochberg control over the 13
             tests, since testing thirteen features and reporting the largest would be
             the same error this project has documented elsewhere.

  ceiling    a supervised binary classifier, known versus unknown, trained ON the target
             building with GroupKFold over recording day. This uses target labels and is
             therefore not a method -- it is an upper bound. It answers the question the
             interpretation turns on: if an oracle with labels cannot separate the two
             groups from these features, the features are the limit; if it can, then the
             information is present and our detector is what fails to reach it.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from scipy.stats import mannwhitneyu
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from common import TRANSFORMS
from exp25_openset_tta import KNOWN, UNKNOWN, NAMES

D = os.path.join(os.path.dirname(__file__), "..", "data")
R = os.path.join(os.path.dirname(__file__), "..", "results")


def load_named(tag):
    z = np.load(os.path.join(D, f"feat_inv_{tag}.npz"), allow_pickle=True)
    cols = [str(s) for s in z["names"]]
    m = np.array([c.startswith(("g_", "t_")) for c in cols])
    return z["X"].astype(np.float64)[:, m], z["y"], z["day"], [c for c, k in zip(cols, m) if k]


def auroc(score, pos):
    o = np.argsort(score); r = np.empty(len(score), float); r[o] = np.arange(len(score))
    np_, nn = int(pos.sum()), int((~pos).sum())
    if np_ == 0 or nn == 0: return float("nan")
    return float((r[pos].sum() - np_ * (np_ - 1) / 2) / (np_ * nn))


def bh(p, q=0.05):
    """Benjamini-Hochberg: returns the boolean reject vector."""
    n = len(p); o = np.argsort(p); keep = np.zeros(n, bool)
    thresh = q * (np.arange(1, n + 1)) / n
    passed = np.where(np.array(p)[o] <= thresh)[0]
    if len(passed): keep[o[:passed.max() + 1]] = True
    return keep


if __name__ == "__main__":
    rows = []
    print("=" * 94)
    print("EXPERIMENT 29 — is novelty present in the 13-d features?")
    print(f"  known {[NAMES[k] for k in KNOWN]}   unknown {[NAMES[k] for k in UNKNOWN]}")
    print("=" * 94)
    for tag in ("S2", "S1"):
        X, y, day, names = load_named(tag)
        unk = np.isin(y, UNKNOWN)
        print(f"\n{tag}   n={len(y)}  unknown={unk.sum()} ({unk.mean():.0%})")
        # ---- marginals
        aus, ps = [], []
        for j in range(X.shape[1]):
            aus.append(auroc(X[:, j], unk))
            ps.append(mannwhitneyu(X[unk, j], X[~unk, j], alternative="two-sided").pvalue)
        sig = bh(ps, 0.05)
        order = np.argsort([-abs(a - 0.5) for a in aus])
        print(f"  {'feature':<26}{'AUROC':>9}{'|dev|':>8}{'p':>12}{'BH 5%':>8}")
        print("  " + "-" * 63)
        for j in order:
            print(f"  {names[j]:<26}{aus[j]:>9.4f}{abs(aus[j]-0.5):>8.4f}"
                  f"{ps[j]:>12.2e}{'  yes' if sig[j] else '   no':>8}")
            rows.append(dict(building=tag, kind="marginal", feature=names[j],
                             auroc=aus[j], p=ps[j], bh_sig=bool(sig[j])))
        print(f"  -> {int(sig.sum())}/{len(names)} features differ after BH correction; "
              f"largest single-feature AUROC {max(aus, key=lambda a: abs(a-0.5)):.4f}")

        # ---- multivariate ceiling, GroupKFold over recording day
        folds = np.array_split(np.unique(day), 5)
        for cname, mk in (("logistic", lambda: LogisticRegression(max_iter=3000)),
                          ("grad-boost", lambda: HistGradientBoostingClassifier(
                              max_iter=300, learning_rate=0.08, random_state=0))):
            sc = np.zeros(len(y))
            for f in folds:
                te = np.isin(day, f); tr = ~te
                tf = TRANSFORMS["zscore"](X[tr])
                m = mk().fit(tf(X[tr]), unk[tr])
                sc[te] = m.predict_proba(tf(X[te]))[:, 1]
            a = auroc(sc, unk)
            print(f"  supervised ceiling ({cname:<10}) AUROC = {a:.4f}")
            rows.append(dict(building=tag, kind="ceiling", feature=cname, auroc=a))
    pd.DataFrame(rows).to_csv(os.path.join(R, "exp29_feature_limit.csv"), index=False)
    print("\n-> results/exp29_feature_limit.csv")
