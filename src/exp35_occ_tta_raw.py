"""
EXPERIMENT 35 — the four adaptation methods on occupancy, raw extracts, Tier-1.

Rebuilds the occupancy method comparison on the version of the data that can be fully
inspected. Three things the earlier occupancy tables could not do are done here.

Tier-1 compliance, verified rather than assumed. The source model is trained once on the
source office's days and frozen. The target office is presented as a single pass in
recording order -- the raw files carry `time`, so "recording order" is meaningful here for
the first time. Every batch is predicted before the model is updated on it. No target
labels are used for anything, including normalisation, which is fitted on the source
split alone.

Full metrics rather than MCC alone. Accuracy, macro-F1, balanced accuracy and the
confusion matrix are reported together, because a binary task at roughly 30% positive
rewards collapsing to the majority class with an accuracy that looks respectable. The
per-class recalls make that collapse visible: a method that predicts "not occupied"
everywhere scores recall 1.00 / 0.00 and accuracy near 0.70.

Both feature sets. The nine shared columns, and the seven that remain after dropping the
two whose correlation with occupancy reverses between offices (exp32).
"""
import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from common import MLP, train_mlp, predict, TRANSFORMS
from tta import SourceOnly, AdaBN, TENT, source_feature_stats
from tta_faithful import DIGAFaithful, PETALFaithful, PeTTAFaithful
from petal_fixed import SWAGDiag
from exp33_occ_penalty_raw import load, scores, SHARED, REVERSING

R = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS, BS = (0, 1, 2, 3, 4), 64


def fit_swag(d_in, src, Z, y, seed, epochs=30):
    torch.manual_seed(seed); np.random.seed(seed)
    sw = MLP(d_in, K=2); sw.load_state_dict(src.state_dict())
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
    S, T = load()
    S = S.sort_values("time").reset_index(drop=True)      # recording order
    T = T.sort_values("time").reset_index(drop=True)
    rows = []
    print("=" * 104)
    print("EXPERIMENT 35 — occupancy TTA on the raw extracts, Tier-1, full metrics")
    print("  stream in recording order, predict-before-adapt, source-fitted scaler, no target labels")
    print("=" * 104)

    for fs, cols in (("9 shared", SHARED),
                     ("7 transferable", [c for c in SHARED if c not in REVERSING])):
        for dirn, A, B in (("source->target", S, T), ("target->source", T, S)):
            d_in = len(cols)
            Xtr_all = A[cols].values.astype(float); ytr_all = A.occ.values
            Xte = B[cols].values.astype(float); yte = B.occ.values
            days = np.unique(A.day); va = np.isin(A.day, days[-max(1, len(days) // 5):])
            acc = {}
            for sd in SEEDS:
                tf = TRANSFORMS["zscore"](Xtr_all[~va])
                Ztr, Zva, Zt = tf(Xtr_all[~va]), tf(Xtr_all[va]), tf(Xte)
                src, _ = train_mlp(Ztr, ytr_all[~va], Zva, ytr_all[va], d_in,
                                   seed=sd, epochs=60, K_out=2)
                swag = fit_swag(d_in, src, Ztr, ytr_all[~va], sd)
                st = source_feature_stats(src, Ztr, ytr_all[~va])

                def stream(mk):
                    ad = mk(); pr = np.zeros(len(yte), int)
                    for i in range(0, len(Zt), BS):
                        pr[i:i + BS] = ad(torch.tensor(Zt[i:i + BS], dtype=torch.float32))[0]
                    return pr
                for name, mk in [("Source-only", lambda: SourceOnly(src)),
                                 ("AdaBN", lambda: AdaBN(src)),
                                 ("TENT", lambda: TENT(src)),
                                 ("DIGA", lambda: DIGAFaithful(src, K=2)),
                                 ("PETAL", lambda: PETALFaithful(src, swag)),
                                 ("PeTTA", lambda: PeTTAFaithful(src, src_stats=st))]:
                    p = stream(mk); s = scores(yte, p)
                    s["rec_occ"] = s["tp"] / max(s["tp"] + s["fn"], 1)
                    s["rec_emp"] = s["tn"] / max(s["tn"] + s["fp"], 1)
                    s["pred_occ_rate"] = float((p == 1).mean())
                    acc.setdefault(name, []).append(s)
            maj = 1 - yte.mean()
            print(f"\n--- {fs}, {dirn}   (target {yte.mean():.1%} occupied; "
                  f"always-'empty' gives acc {maj:.3f}, MCC 0.000) ---")
            print(f"  {'method':<13}{'MCC':>9}{'macroF1':>10}{'bal.acc':>9}{'acc':>8}"
                  f"{'rec occ':>9}{'rec empty':>11}{'pred occ%':>11}")
            print("  " + "-" * 80)
            for name, v in acc.items():
                m = {k: float(np.mean([x[k] for x in v])) for k in v[0]}
                flag = "  <- collapsed" if m["pred_occ_rate"] < 0.02 or m["rec_occ"] < 0.05 else ""
                print(f"  {name:<13}{m['mcc']:>9.4f}{m['macroF1']:>10.4f}{m['balacc']:>9.4f}"
                      f"{m['acc']:>8.4f}{m['rec_occ']:>9.4f}{m['rec_emp']:>11.4f}"
                      f"{m['pred_occ_rate']:>10.1%}{flag}")
                rows.append(dict(feature_set=fs, direction=dirn, method=name, **m))
    pd.DataFrame(rows).to_csv(os.path.join(R, "exp35_occ_tta_raw.csv"), index=False)
    print("\n-> results/exp35_occ_tta_raw.csv")
