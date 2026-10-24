"""
EXPERIMENT 34 — does the occupancy asymmetry survive the change of data version?

The occupancy tables report a strong asymmetry: transferring one way gives MCC 0.66,
transferring the other gives -0.08, worse than chance. That contrast is the clearest
evidence in the study for reading an adaptation gain as a measure of source-model
deficiency, because the two directions differ in nothing but which office supplies the
source model.

Those numbers come from the anonymised extract. This script asks whether they are a
property of the two offices or of that extract, by running the identical ungrouped
protocol -- same architecture, same split procedure, no day grouping -- on both the
anonymised CSVs and the raw Grenoble files. Holding the protocol fixed isolates the data
version as the only thing that differs.

Ten seeds, because one of the cells turns out to straddle zero and a mean alone would
misrepresent it.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import TRANSFORMS, train_mlp, predict
from exp33_occ_penalty_raw import load, scores, SHARED, REVERSING

OPDA = "/Users/mariemkbaier/Downloads/OPDA_FINAL/data"
R = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS = 10


def run(Xtr, ytr, Xte, yte):
    v = []
    for sd in range(SEEDS):
        rng = np.random.default_rng(sd); p = rng.permutation(len(Xtr))
        cut = max(1, int(0.2 * len(Xtr))); vi, ti = p[:cut], p[cut:]
        tf = TRANSFORMS["zscore"](Xtr[ti])
        m, _ = train_mlp(tf(Xtr[ti]), ytr[ti], tf(Xtr[vi]), ytr[vi], Xtr.shape[1],
                         seed=sd, epochs=60, K_out=2)
        yp, _ = predict(m, tf(Xte))
        v.append(scores(yte, yp)["mcc"])
    return np.array(v)


if __name__ == "__main__":
    rows = []
    print("=" * 94)
    print("EXPERIMENT 34 — occupancy asymmetry: extract or offices?")
    print("  identical ungrouped protocol throughout; only the data version changes")
    print("=" * 94)
    print(f"\n  {'version':<22}{'direction':<18}{'MCC':>9}{'sd':>8}{'neg seeds':>11}{'range':>18}")
    print("  " + "-" * 86)

    o = {k: pd.read_csv(f"{OPDA}/{k}.csv") for k in
         ("Source_train", "Source_test", "Target_train", "Target_test")}
    g = lambda k: (o[k].iloc[:, :9].values.astype(float),
                   o[k]["labels"].astype(float).values.astype(int))
    for tag, a, b in (("source->target", "Source_train", "Target_test"),
                      ("target->source", "Target_train", "Source_test")):
        Xtr, ytr = g(a); Xte, yte = g(b)
        v = run(Xtr, ytr, Xte, yte)
        print(f"  {'anonymised (9-d)':<22}{tag:<18}{v.mean():>+9.4f}{v.std():>8.3f}"
              f"{sum(v<0):>8}/10{f'[{v.min():+.2f},{v.max():+.2f}]':>18}")
        rows.append(dict(version="anonymised", features="9-d", direction=tag,
                         mcc=v.mean(), sd=v.std(), neg_seeds=int(sum(v < 0)),
                         lo=v.min(), hi=v.max()))

    S, T = load()
    for fs, cols in (("raw, 9 shared", SHARED),
                     ("raw, 7 transferable", [c for c in SHARED if c not in REVERSING])):
        for tag, A, B in (("source->target", S, T), ("target->source", T, S)):
            v = run(A[cols].values.astype(float), A.occ.values,
                    B[cols].values.astype(float), B.occ.values)
            print(f"  {fs:<22}{tag:<18}{v.mean():>+9.4f}{v.std():>8.3f}"
                  f"{sum(v<0):>8}/10{f'[{v.min():+.2f},{v.max():+.2f}]':>18}")
            rows.append(dict(version="raw", features=fs.split(", ")[1], direction=tag,
                             mcc=v.mean(), sd=v.std(), neg_seeds=int(sum(v < 0)),
                             lo=v.min(), hi=v.max()))
    pd.DataFrame(rows).to_csv(os.path.join(R, "exp34_occ_version_asymmetry.csv"), index=False)
    print("\n-> results/exp34_occ_version_asymmetry.csv")
