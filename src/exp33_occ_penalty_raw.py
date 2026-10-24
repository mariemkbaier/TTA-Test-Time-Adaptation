"""
EXPERIMENT 33 — the cross-office penalty on occupancy, measured properly.

The anonymised occupancy CSVs used for the earlier occupancy tables carry no timestamp,
so a non-leaky held-out split could not be constructed from them and the penalty could not
be measured the way Section~\ref{sec:res-gap} measures it for activity recognition. The
raw Grenoble extracts carry `time`, which gives a recording-day index: 10 days in the
source office, 8 in the target. That makes the same protocol available here.

Protocol, identical in structure to the activity measurement:

  oracle    train on the TARGET office's own days, evaluate on its held-out days
  transfer  train on the SOURCE office, evaluate on the SAME held-out days
  matched   training-set sizes equalised by subsampling, so the comparison is not
            confounded by one office simply having more data

Splits are grouped by recording day throughout. Two feature sets are reported, because
exp32 found that two of the nine shared columns reverse sign against occupancy between
the offices:

  all shared         the nine columns common to both offices
  transferable       the seven that remain after dropping rms and derivative_co2

DATA VERSION. This runs on the raw extracts. They do not reconcile with the anonymised
CSVs behind the earlier occupancy tables: 1210 against 1345 source rows, 1041 against 972
target rows, and no binarisation threshold reproduces both label balances. No data-
preparation script exists in either of the published repositories, and the cited papers
describe a three-level occupancy protocol rather than a binary one, so the discrepancy
could not be resolved. The raw extracts are therefore taken as the reference and the
mismatch is reported as an open limitation rather than assumed away.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import TRANSFORMS, train_mlp, predict

RAW = "/Users/mariemkbaier/Downloads"
R = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS = (0, 1, 2, 3, 4)
SHARED = ["co2", "motion", "rms", "door", "power1", "power2", "power3",
          "sum_power", "derivative_co2"]
REVERSING = ["rms", "derivative_co2"]


def load():
    S = pd.read_csv(f"{RAW}/Final_Data_Source.csv").rename(columns={"Door_contact": "door"})
    T = pd.read_csv(f"{RAW}/Final_Data_Target.csv")
    for d in (S, T):
        d["occ"] = (d["labels"] > 0).astype(int)
        d["day"] = (d["time"] // 86400).astype(int)
    return S, T


def scores(y, p):
    tp = ((p == 1) & (y == 1)).sum(); tn = ((p == 0) & (y == 0)).sum()
    fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = 0.0 if den == 0 else (tp * tn - fp * fn) / den
    acc = (tp + tn) / max(len(y), 1)
    f1 = []
    for c, (t, f_p, f_n) in ((0, (tn, fn, fp)), (1, (tp, fp, fn))):
        pr = t / (t + f_p) if t + f_p else 0.0
        rc = t / (t + f_n) if t + f_n else 0.0
        f1.append(0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))
    rec1 = tp / (tp + fn) if tp + fn else 0.0
    rec0 = tn / (tn + fp) if tn + fp else 0.0
    return dict(mcc=mcc, acc=acc, macroF1=float(np.mean(f1)),
                balacc=(rec0 + rec1) / 2, tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))


if __name__ == "__main__":
    S, T = load()
    rows = []
    print("=" * 100)
    print("EXPERIMENT 33 — cross-office occupancy penalty, day-grouped, matched n")
    print(f"  source {len(S)} rows over {S.day.nunique()} days, {S.occ.mean():.1%} occupied")
    print(f"  target {len(T)} rows over {T.day.nunique()} days, {T.occ.mean():.1%} occupied")
    print("=" * 100)

    for fname, cols in (("all shared (9)", SHARED),
                        ("transferable (7)", [c for c in SHARED if c not in REVERSING])):
        print(f"\n--- feature set: {fname} ---")
        print(f"  {'evaluated on':<16}{'trained on':<22}{'n_tr':>7}{'MCC':>9}{'macroF1':>10}"
              f"{'bal.acc':>9}{'acc':>8}{'penalty':>10}")
        print("  " + "-" * 82)
        for tgt_name, A, B in (("target office", T, S), ("source office", S, T)):
            days = np.unique(A.day); folds = np.array_split(days, 4)
            res = {}
            for mode in ("own days", "other office"):
                acc = {k: [] for k in ("mcc", "macroF1", "balacc", "acc")}; ns = []
                for sd in SEEDS:
                    rng = np.random.default_rng(sd)
                    for f in folds:
                        te = np.isin(A.day, f)
                        cap = int((~te).sum())
                        if mode == "own days":
                            trX = A.loc[~te, cols].values.astype(float); trY = A.loc[~te, "occ"].values
                        else:
                            idx = rng.permutation(len(B))[:cap]
                            trX = B.iloc[idx][cols].values.astype(float); trY = B.iloc[idx]["occ"].values
                        if len(np.unique(trY)) < 2: continue
                        cut = max(1, int(0.2 * len(trX)))
                        p = rng.permutation(len(trX)); vi, ti = p[:cut], p[cut:]
                        if len(np.unique(trY[ti])) < 2: continue
                        tf = TRANSFORMS["zscore"](trX[ti])
                        m, _ = train_mlp(tf(trX[ti]), trY[ti], tf(trX[vi]), trY[vi],
                                         len(cols), seed=sd, epochs=60, K_out=2)
                        yp, _ = predict(m, tf(A.loc[te, cols].values.astype(float)))
                        s = scores(A.loc[te, "occ"].values, yp)
                        for k in acc: acc[k].append(s[k])
                        ns.append(len(ti))
                res[mode] = {k: float(np.mean(v)) for k, v in acc.items()}
                pen = "" if mode == "own days" else f"{(res['own days']['mcc']-res[mode]['mcc'])*100:>9.1f}p"
                print(f"  {tgt_name:<16}{mode:<22}{np.mean(ns):>7.0f}{res[mode]['mcc']:>9.4f}"
                      f"{res[mode]['macroF1']:>10.4f}{res[mode]['balacc']:>9.4f}"
                      f"{res[mode]['acc']:>8.4f}{pen:>10}")
                rows.append(dict(feature_set=fname, evaluated_on=tgt_name, trained_on=mode,
                                 n_train=float(np.mean(ns)), **res[mode]))
            rows.append(dict(feature_set=fname, evaluated_on=tgt_name, trained_on="penalty",
                             mcc=res["own days"]["mcc"] - res["other office"]["mcc"]))
    pd.DataFrame(rows).to_csv(os.path.join(R, "exp33_occ_penalty_raw.csv"), index=False)
    print("\n-> results/exp33_occ_penalty_raw.csv")
