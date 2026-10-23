"""
EXPERIMENT 28 — is "below chance" a claim or a point estimate?

exp26 reports every confidence-based unknown score at an AUROC under 0.5. That is the
headline of the detector section, and a point estimate is not enough to support it: a
reviewer is entitled to ask whether 0.4952 is distinguishable from 0.5. This script
bootstraps the target stream, 2000 resamples, and reports an interval for each score
averaged over the three source seeds.
"""
import os, sys, numpy as np, pandas as pd, torch
sys.path.insert(0,'src')
from common import train_mlp, TRANSFORMS
from exp25_openset_tta import load_inv, remap, KNOWN, UNKNOWN
from exp26_unknown_scores import auroc, all_scores, prototypes
A={t:load_inv(t) for t in ("S1","S2")}; Kc=len(KNOWN); B=2000
ROWS=[]
for srcb,tgt in (("S1","S2"),("S2","S1")):
    Xs,ys,ds=A[srcb]; Xt,yt,_=A[tgt]
    smask=np.isin(ys,KNOWN); unk=np.isin(yt,UNKNOWN)
    folds=np.array_split(np.unique(ds),5); va=np.isin(ds,folds[-1])
    tr=(~va)&smask; vas=va&smask
    per={}
    for sd in (0,1,2):
        tf=TRANSFORMS["zscore"](Xs[tr]); Ztr,Zva,Zt=tf(Xs[tr]),tf(Xs[vas]),tf(Xt)
        ytr=remap(ys[tr],KNOWN)
        src,_=train_mlp(Ztr,ytr,Zva,remap(ys[vas],KNOWN),Xs.shape[1],seed=sd,epochs=60,K_out=Kc)
        mu,var=prototypes(src,Ztr,ytr,Kc)
        for k,v in all_scores(src,Zt,mu,var).items(): per.setdefault(k,[]).append(v)
    print(f"\n{srcb}->{tgt}   n={len(yt)}  unknown={unk.sum()}")
    print(f"  {'score':<10}{'AUROC':>9}{'95% CI':>22}{'':>6}")
    rng=np.random.default_rng(0)
    idx=[rng.integers(0,len(yt),len(yt)) for _ in range(B)]
    for k,v in sorted(per.items(), key=lambda z:-np.mean([auroc(x,unk) for x in z[1]])):
        pt=np.mean([auroc(x,unk) for x in v])
        bs=[]
        for i in idx:
            u=unk[i]
            if u.sum()==0 or u.sum()==len(u): continue
            bs.append(np.mean([auroc(x[i],u) for x in v]))
        lo,hi=np.percentile(bs,2.5),np.percentile(bs,97.5)
        flag = "below chance" if hi<0.5 else ("above chance" if lo>0.5 else "spans 0.5")
        print(f"  {k:<10}{pt:>9.4f}   [{lo:.4f}, {hi:.4f}]   {flag}")
        ROWS.append(dict(direction=f"{srcb}->{tgt}", score=k, auroc=pt,
                         ci_lo=lo, ci_hi=hi, verdict=flag))

pd.DataFrame(ROWS).to_csv(os.path.join(os.path.dirname(__file__), "..", "results",
                                       "exp28_auroc_ci.csv"), index=False)
print("\n-> results/exp28_auroc_ci.csv")
