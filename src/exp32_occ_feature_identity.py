"""
EXPERIMENT 32 — do the occupancy features mean the same thing in both offices?

The occupancy CSVs used until now carry anonymised columns (0-8), which makes this
question unaskable. The raw Grenoble extracts name them, and the names show two problems
that the anonymised form hides.

First, the two offices do not instrument the same things. The source office has `window`,
`power5` and `power6`; the target office has `power4`. Only `power1`, `power2` and
`power3` are nominally shared, and `power3` is zero in 96% of source rows against 5% of
target rows, so the column name is shared but the appliance behind it is not.

Second, and more seriously, a feature can predict occupancy in one office and
anti-predict it in the other. This script measures that per feature, with bootstrap
intervals, and then quantifies what it costs: a model is trained with and without the
offending feature and transferred in both directions.

This is the occupancy counterpart of the per-sensor result in the activity study, where
features tied to a specific sensor index scored below chance across buildings. Here the
tie is to a specific sensor placement rather than an index, and it is invisible until the
columns are named.

NOTE ON DATA VERSION: this runs on the raw extracts (Final_Data_Source.csv,
Final_Data_Target.csv), which do not reconcile row-for-row with the anonymised CSVs used
for the main occupancy tables: 1210 vs 1345 source rows, 1041 vs 972 target rows, and no
binarisation threshold reproduces both label balances. The feature-identity finding does
not depend on that reconciliation, since it is a property of the sensors rather than of
any particular extract.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import TRANSFORMS, train_mlp, predict

RAW = "/Users/mariemkbaier/Downloads"
R = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS = (0, 1, 2, 3, 4)
B = 2000


def load():
    S = pd.read_csv(f"{RAW}/Final_Data_Source.csv").rename(columns={"Door_contact": "door"})
    T = pd.read_csv(f"{RAW}/Final_Data_Target.csv")
    for d in (S, T):
        d["occ"] = (d["labels"] > 0).astype(int)
        d["day"] = (d["time"] // 86400).astype(int)
    return S, T


def mcc(y, p):
    tp = ((p == 1) & (y == 1)).sum(); tn = ((p == 0) & (y == 0)).sum()
    fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
    d = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return 0.0 if d == 0 else (tp * tn - fp * fn) / d


if __name__ == "__main__":
    S, T = load()
    shared = [c for c in S.columns if c in set(T.columns)
              and c not in ("time", "labels", "occ", "day")]
    rows = []
    print("=" * 96)
    print("EXPERIMENT 32 — feature identity across the two offices")
    print("=" * 96)
    print(f"\nSensor sets differ:")
    print(f"  only in source : {sorted(set(S.columns) - set(T.columns) - {'occ','day'})}")
    print(f"  only in target : {sorted(set(T.columns) - set(S.columns) - {'occ','day'})}")
    print(f"  nominally shared: {shared}")
    for c in sorted(set(S.columns) - set(T.columns) - {"occ", "day"}):
        rows.append(dict(kind="sensor_set", feature=c, present_in="source only"))
    for c in sorted(set(T.columns) - set(S.columns) - {"occ", "day"}):
        rows.append(dict(kind="sensor_set", feature=c, present_in="target only"))
    # per-channel distribution, so "same name, different appliance" is checkable
    print(f"\n  {'channel':<16}{'src median':>12}{'src %zero':>11}{'tgt median':>12}{'tgt %zero':>11}")
    for c in shared:
        sm, sz = float(S[c].median()), float((S[c] == 0).mean())
        tm, tz = float(T[c].median()), float((T[c] == 0).mean())
        print(f"  {c:<16}{sm:>12.3f}{sz:>10.0%}{tm:>12.3f}{tz:>10.0%}")
        rows.append(dict(kind="channel_dist", feature=c, src_median=sm, src_frac_zero=sz,
                         tgt_median=tm, tgt_frac_zero=tz))

    print(f"\n{'feature':<16}{'source r':>12}{'95% CI':>20}{'target r':>12}{'95% CI':>20}  verdict")
    print("-" * 96)
    rng = np.random.default_rng(0)
    iS = [rng.integers(0, len(S), len(S)) for _ in range(B)]
    iT = [rng.integers(0, len(T), len(T)) for _ in range(B)]
    for c in shared:
        rs = np.corrcoef(S[c], S.occ)[0, 1]; rt = np.corrcoef(T[c], T.occ)[0, 1]
        bs = [np.corrcoef(S[c].values[i], S.occ.values[i])[0, 1] for i in iS]
        bt = [np.corrcoef(T[c].values[i], T.occ.values[i])[0, 1] for i in iT]
        sl, sh = np.percentile(bs, [2.5, 97.5]); tl, th = np.percentile(bt, [2.5, 97.5])
        flip = (sl > 0 and th < 0) or (sh < 0 and tl > 0)
        v = "SIGN REVERSES" if flip else ("same sign" if rs * rt > 0 else "sign differs (CIs overlap 0)")
        print(f"{c:<16}{rs:>+12.3f}  [{sl:+.3f},{sh:+.3f}]{rt:>+12.3f}  [{tl:+.3f},{th:+.3f}]  {v}")
        rows.append(dict(kind="correlation", feature=c, r_source=rs, src_lo=sl, src_hi=sh,
                         r_target=rt, tgt_lo=tl, tgt_hi=th, reverses=bool(flip)))

    # ---- what does keeping a reversing feature cost in transfer?
    rev = [r["feature"] for r in rows if r.get("reverses")]
    print(f"\nFeatures whose sign reverses with non-overlapping intervals: {rev or 'none'}")
    if rev:
        print(f"\nTransfer cost of keeping them (MCC, day-grouped source selection, 5 seeds):")
        print(f"  {'direction':<14}{'feature set':<22}{'target MCC':>12}{'sd':>8}")
        print("  " + "-" * 56)
        for dname, A, Bo in (("source->target", S, T), ("target->source", T, S)):
            for tag, cols in (("all shared", shared),
                              (f"minus {'+'.join(rev)}", [c for c in shared if c not in rev])):
                sc = []
                for sd in SEEDS:
                    days = np.unique(A.day); va = np.isin(A.day, days[-max(1, len(days)//5):])
                    tf = TRANSFORMS["zscore"](A.loc[~va, cols].values.astype(float))
                    m, _ = train_mlp(tf(A.loc[~va, cols].values.astype(float)),
                                     A.loc[~va, "occ"].values,
                                     tf(A.loc[va, cols].values.astype(float)),
                                     A.loc[va, "occ"].values,
                                     len(cols), seed=sd, epochs=60, K_out=2)
                    yp, _ = predict(m, tf(Bo[cols].values.astype(float)))
                    sc.append(mcc(Bo.occ.values, yp))
                print(f"  {dname:<14}{tag:<22}{np.mean(sc):>12.4f}{np.std(sc):>8.3f}")
                rows.append(dict(kind="transfer", direction=dname, feature_set=tag,
                                 mcc=np.mean(sc), sd=np.std(sc), n_features=len(cols)))
    pd.DataFrame(rows).to_csv(os.path.join(R, "exp32_occ_feature_identity.csv"), index=False)
    print("\n-> results/exp32_occ_feature_identity.csv")
