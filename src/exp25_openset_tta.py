"""
EXPERIMENT 25 — open-set TTA under Tier-1 constraints.

The source model is trained once on the source building's KNOWN activities, then frozen.
Adaptation happens online, in one pass over the unlabelled target stream, in recording
order, with no source data, no target labels, and no target-driven calibration of any
quantity -- including the unknown-rejection threshold.

Rejection. A sample is called unknown when the adapted model's maximum softmax falls below
tau. tau is the q-th percentile of the maximum softmax over the SOURCE validation split,
computed before the target stream is touched. q is fixed a priori; the sensitivity curve
over q is reported so that no headline number depends on a threshold that could have been
chosen with hindsight.

What would break TTA, and is avoided here:

  * multi-epoch offline training on the target set. LEAD (ispc-lab/LEAD) runs
    args.epochs = 50 (config/model_config.py:15, train_target.py:375) over the target
    data. That is source-free domain adaptation, not test-time adaptation: it needs the
    whole target set in advance and revisits it fifty times.
  * reusing the evaluation set as the adaptation set. In train_target.py:360-362 both
    target_train_dataloader and target_test_dataloader wrap the SAME target_dataset
    object, so every sample scored at line 383 was trained on at line 377.
  * seeing the evaluation set before adapting. obtain_LEAD_pseudo_labels (line 209) makes
    a full pass over that test loader at the start of every epoch to build pseudo-labels.
  * destroying stream order. shuffle=True on the adaptation loader (line 360).

Here each batch is predicted BEFORE the model is updated on it, the pass is single, and
order is the recording order of the target building.

Metrics, all on the held-out target stream:
  OS*     mean per-class accuracy over known classes
  UCDR    fraction of truly-unknown samples rejected
  H       harmonic mean of OS* and UCDR (standard OSDA score)
  macroF1 over known classes, rejection counted as an error for the true class
with reject-all and reject-none as the two degenerate baselines.
"""
import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from common import MLP, train_mlp, predict, TRANSFORMS, confusion
from tta import SourceOnly, AdaBN, TENT, source_feature_stats
from tta_faithful import DIGAFaithful, PETALFaithful, PeTTAFaithful
from petal_fixed import SWAGDiag
from build_features import load_raw, segments

D = os.path.join(os.path.dirname(__file__), "..", "data")
R = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS, BS = (0, 1, 2), 64
NAMES = {0: "toileting", 1: "meal_prep", 2: "watching_tv", 3: "music", 4: "cleaning"}
UNKNOWN = (0, 4)
KNOWN = tuple(k for k in range(5) if k not in UNKNOWN)
Q_GRID = (1.0, 2.5, 5.0, 10.0, 20.0)
Q_HEAD = 5.0


def load_inv(tag):
    z = np.load(os.path.join(D, f"feat_inv_{tag}.npz"), allow_pickle=True)
    cols = [str(s) for s in z["names"]]
    m = np.array([c.startswith(("g_", "t_")) for c in cols])
    return z["X"].astype(np.float64)[:, m], z["y"], z["day"]


def remap(y, classes):
    lut = {c: i for i, c in enumerate(classes)}
    return np.array([lut.get(v, -1) for v in y], dtype=np.int64)


def openset_scores(y_true_k, y_pred_k, is_unknown, rejected, Kc):
    """y_*_k are known-class indices (-1 for truly-unknown rows)."""
    known = ~is_unknown
    # OS*: per-class accuracy over known classes, a rejected known sample is wrong
    accs = []
    for c in range(Kc):
        sel = known & (y_true_k == c)
        if sel.sum() == 0: continue
        accs.append(float(((y_pred_k[sel] == c) & ~rejected[sel]).mean()))
    os_star = float(np.mean(accs)) if accs else 0.0
    ucdr = float(rejected[is_unknown].mean()) if is_unknown.sum() else float("nan")
    h = 0.0 if (os_star + ucdr) == 0 else 2 * os_star * ucdr / (os_star + ucdr)
    # macro-F1 over known classes, rejection = error for the true class
    yp = np.where(rejected, Kc, y_pred_k)          # Kc is the "unknown" bin
    M = confusion(np.where(known, y_true_k, Kc), yp, Kc + 1)
    tp = np.diag(M).astype(float); fp = M.sum(0) - tp; fn = M.sum(1) - tp
    f1 = []
    for c in range(Kc):
        p = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        r = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        f1.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return dict(os_star=os_star, ucdr=ucdr, h=h, macroF1=float(np.mean(f1)))


if __name__ == "__main__":
    A = {t: load_inv(t) for t in ("S1", "S2")}
    Kc = len(KNOWN)
    rows = []
    print("=" * 100)
    print("EXPERIMENT 25 — open-set TTA, Tier-1 constraints")
    print(f"  known {[NAMES[k] for k in KNOWN]}   unknown {[NAMES[k] for k in UNKNOWN]}")
    print(f"  tau = {Q_HEAD}th percentile of SOURCE-validation max-softmax; "
          f"sensitivity over q in {Q_GRID}")
    print("=" * 100)

    for srcb, tgt in (("S1", "S2"), ("S2", "S1")):
        Xs, ys, ds = A[srcb]; Xt, yt, dt = A[tgt]
        smask = np.isin(ys, KNOWN)
        is_unknown = np.isin(yt, UNKNOWN)
        yt_k = remap(yt, KNOWN)
        folds = np.array_split(np.unique(ds), 5)
        va = np.isin(ds, folds[-1]); tr = (~va) & smask; vas = va & smask
        acc = {}
        for sd in SEEDS:
            tf = TRANSFORMS["zscore"](Xs[tr])
            Ztr, Zva, Zt = tf(Xs[tr]), tf(Xs[vas]), tf(Xt)
            src, _ = train_mlp(Ztr, remap(ys[tr], KNOWN), Zva, remap(ys[vas], KNOWN),
                               Xs.shape[1], seed=sd, epochs=60, K_out=Kc)
            # ---- tau from SOURCE validation only, before the target stream is touched
            with torch.no_grad():
                sv = torch.softmax(src(torch.tensor(Zva, dtype=torch.float32)), 1)
                sv = sv.max(1).values.numpy()
            taus = {q: float(np.percentile(sv, q)) for q in Q_GRID}

            torch.manual_seed(sd); np.random.seed(sd)
            sw = MLP(Xs.shape[1], K=Kc); sw.load_state_dict(src.state_dict())
            o = torch.optim.Adam(sw.parameters(), lr=1e-3); lf = nn.CrossEntropyLoss()
            Xtt = torch.tensor(Ztr, dtype=torch.float32)
            ytt = torch.tensor(remap(ys[tr], KNOWN), dtype=torch.long); snaps = []
            for _ in range(30):
                sw.train(); perm = torch.randperm(len(Xtt))
                for i in range(0, len(perm), 128):
                    j = perm[i:i + 128]
                    if len(j) < 2: continue
                    o.zero_grad(); lf(sw(Xtt[j]), ytt[j]).backward(); o.step()
                snaps.append(torch.cat([p.data.view(-1) for p in sw.parameters()]).clone())
            swag = SWAGDiag().fit(snaps)
            st = source_feature_stats(src, Ztr, remap(ys[tr], KNOWN))

            def stream(mk):
                """Single pass, recording order, predict BEFORE adapting on the batch."""
                ad = mk(); pk = np.zeros(len(yt), int); conf = np.zeros(len(yt))
                for i in range(0, len(Zt), BS):
                    p, pr = ad(torch.tensor(Zt[i:i + BS], dtype=torch.float32))
                    pk[i:i + BS] = p
                    q = np.asarray(pr, dtype=np.float64)
                    q = q / np.maximum(q.sum(1, keepdims=True), 1e-12)
                    conf[i:i + BS] = q.max(1)
                return pk, conf

            for name, mk in [("source-only", lambda: SourceOnly(src)),
                             ("AdaBN", lambda: AdaBN(src)),
                             ("TENT", lambda: TENT(src)),
                             ("DIGA", lambda: DIGAFaithful(src, K=Kc)),
                             ("PETAL", lambda: PETALFaithful(src, swag)),
                             ("PeTTA", lambda: PeTTAFaithful(src, src_stats=st))]:
                pk, conf = stream(mk)
                for q in Q_GRID:
                    rej = conf < taus[q]
                    s = openset_scores(yt_k, pk, is_unknown, rej, Kc)
                    acc.setdefault((name, q), []).append(s)
                # DIAGNOSTIC ONLY, not a Tier-1 result: the best threshold obtainable
                # if target labels were available. Bounds how much of the UCDR shortfall
                # is a miscalibrated tau rather than a weak detector.
                best = max(((openset_scores(yt_k, pk, is_unknown, conf < t, Kc), t)
                            for t in np.quantile(conf, np.linspace(0.01, 0.9, 40))),
                           key=lambda z: z[0]["h"])
                acc.setdefault((name + " [oracle tau]", Q_HEAD), []).append(best[0])
            # degenerate baselines
            pk0, conf0 = stream(lambda: SourceOnly(src))
            for nm, rej in (("reject-none", np.zeros(len(yt), bool)),
                            ("reject-all", np.ones(len(yt), bool))):
                acc.setdefault((nm, Q_HEAD), []).append(
                    openset_scores(yt_k, pk0, is_unknown, rej, Kc))

        print(f"\n{srcb}->{tgt}   unknown share {is_unknown.mean():.0%} "
              f"({is_unknown.sum()} of {len(yt)})")
        print(f"  {'method':<14}{'q':>6}{'OS*':>9}{'UCDR':>9}{'H':>9}{'macroF1(k)':>12}")
        print("  " + "-" * 59)
        for (name, q), v in acc.items():
            if q != Q_HEAD and name not in ("reject-none", "reject-all"): continue
            m = {k: np.mean([x[k] for x in v]) for k in v[0]}
            print(f"  {name:<14}{q:>6.1f}{m['os_star']:>9.4f}{m['ucdr']:>9.4f}"
                  f"{m['h']:>9.4f}{m['macroF1']:>12.4f}")
        for (name, q), v in acc.items():
            m = {k: np.mean([x[k] for x in v]) for k in v[0]}
            rows.append(dict(direction=f"{srcb}->{tgt}", method=name, q=q, **m))
        print(f"\n  tau sensitivity (DIGA):")
        for q in Q_GRID:
            v = acc.get(("DIGA", q))
            if v:
                m = {k: np.mean([x[k] for x in v]) for k in v[0]}
                print(f"    q={q:<5} OS*={m['os_star']:.4f}  UCDR={m['ucdr']:.4f}  H={m['h']:.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(R, "exp25_openset_tta.csv"), index=False)
    print("\n-> results/exp25_openset_tta.csv")
