"""EXPERIMENT 19 - how much of DIGA's gain survives implementation choices?

Does DIGA's gain depend on which implementation of its two modules is used?

Paper Eq.3  blends BN *statistics*            : mu = l*mu_S + (1-l)*mu_T, then normalise once
Repo code   blends normalised *outputs*       : out = l*src_norm + (1-l)*ins_norm
Paper Eq.10 fuses *probabilities*             : p = lF*p_tilde + (1-lF)*p_hat
Repo code   fuses cosine similarity with logits: logits = lF*sim + (1-lF)*logits
"""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from copy import deepcopy
sys.path.insert(0,'src')
from common import *
from tta import SourceOnly, _bn_layers
from exp12_lambda_bar import load, stream

class DIGAv(nn.Module):
    def __init__(self, model, dam="stats", fuse="probs", lam_bn=.95, P0=.5, rho=.1, lamF=.8, K_=5):
        super().__init__(); self.c=dict(dam=dam,fuse=fuse,l=lam_bn,P0=P0,rho=rho,lF=lamF); self.K=K_
        self.m=deepcopy(model).eval(); self.bns=_bn_layers(self.m)
        self.src=[(b.running_mean.clone(),b.running_var.clone()) for b in self.bns]
        self.hist=None; self.hooks=[]
    @torch.no_grad()
    def __call__(self,x):
        c=self.c
        if c["dam"]=="stats":
            st={}; hs=[]
            def mk(i):
                def h(mod,inp,out): z=inp[0].detach(); st[i]=(z.mean(0),z.var(0,unbiased=False))
                return h
            for i,b in enumerate(self.bns): hs.append(b.register_forward_hook(mk(i)))
            self.m(x)
            for h in hs: h.remove()
            for i,b in enumerate(self.bns):
                ms,vs=self.src[i]; mt,vt=st[i]
                b.running_mean=c["l"]*ms+(1-c["l"])*mt; b.running_var=c["l"]*vs+(1-c["l"])*vt
            logit,z=self.m(x,return_feat=True)
        else:                                            # blend normalised outputs
            hs=[]
            def mk(i,b):
                ms,vs=self.src[i]
                def h(mod,inp,out):
                    v=inp[0].detach()
                    sn=(v-ms)/torch.sqrt(vs+mod.eps)
                    if v.size(0)>1: im,iv=v.mean(0),v.var(0,unbiased=False)
                    else: im,iv=ms,vs
                    inn=(v-im)/torch.sqrt(iv+mod.eps)
                    return mod.weight*(c["l"]*sn+(1-c["l"])*inn)+mod.bias
                return h
            for i,b in enumerate(self.bns): hs.append(b.register_forward_hook(mk(i,b)))
            logit,z=self.m(x,return_feat=True)
            for h in hs: h.remove()
        p_hat=F.softmax(logit,1); conf,cls=p_hat.max(1)
        inst=torch.zeros(self.K,z.shape[1]); hit=torch.zeros(self.K,dtype=torch.bool)
        for k in range(self.K):
            s=(cls==k)&(conf>=c["P0"])
            if s.sum(): inst[k]=z[s].mean(0); hit[k]=True
        if self.hist is None: self.hist=inst.clone()
        else: self.hist[hit]=c["rho"]*self.hist[hit]+(1-c["rho"])*inst[hit]
        sim=F.normalize(z,dim=1)@F.normalize(self.hist,dim=1).T
        if c["fuse"]=="probs":
            p=c["lF"]*F.softmax(sim/0.1,1)+(1-c["lF"])*p_hat
        else:                                            # repo: similarity + raw logits
            p=c["lF"]*sim+(1-c["lF"])*logit
        return p.argmax(1).numpy(), p.numpy()

X1,y1,d1,cols=load("S1"); X2,y2,_,_=load("S2")
mask=np.array([c.startswith(("g_","t_")) for c in cols])
days=np.unique(d1); va=np.isin(d1,np.array_split(days,5)[-1]); tr=~va
tf=TRANSFORMS["zscore"](X1[tr][:,mask])
Z1,Z1v,Z2=tf(X1[tr][:,mask]),tf(X1[va][:,mask]),tf(X2[:,mask])
VARIANTS=[("paper Eq.3 + Eq.10 (ours)",dict(dam="stats",fuse="probs")),
          ("repo: output-blend + Eq.10",dict(dam="outs", fuse="probs")),
          ("paper Eq.3 + logit-fuse",  dict(dam="stats",fuse="logits")),
          ("repo: output-blend + logit-fuse", dict(dam="outs",fuse="logits"))]
f1s={}; preds={}
for sd in range(3):
    src,_=train_mlp(Z1,y1[tr],Z1v,y1[va],int(mask.sum()),seed=sd,epochs=60)
    p0,_=stream(lambda:SourceOnly(src),Z2,y2)
    f1s.setdefault("source-only",[]).append(metrics(y2,p0)["macroF1"])
    preds.setdefault("source-only",[]).append(p0)
    for tag,kw in VARIANTS:
        pr,_=stream(lambda: DIGAv(src,**kw), Z2,y2)
        f1s.setdefault(tag,[]).append(metrics(y2,pr)["macroF1"])
        preds.setdefault(tag,[]).append(pr)

def paired_ci(tag, B=2000, seed=0):
    """Paired bootstrap over target samples: resample indices once per draw and score
    both the variant and source-only on the same resample, so the interval is on the
    difference and the shared source checkpoint cancels."""
    rng=np.random.default_rng(seed); n=len(y2); d=[]
    for _ in range(B):
        idx=rng.integers(0,n,n)
        yr=y2[idx]
        if len(np.unique(yr))<2: continue
        d.append(np.mean([metrics(yr,preds[tag][k][idx])["macroF1"]
                          -metrics(yr,preds["source-only"][k][idx])["macroF1"]
                          for k in range(len(preds[tag]))]))
    return float(np.percentile(d,2.5)), float(np.percentile(d,97.5))
print(f"  {'DIGA variant':<36}{'TGT macroF1':>13}{'sd':>8}{'vs source':>12}  95% CI on delta")
base=np.mean(f1s["source-only"])
rows=[]
for k,v in f1s.items():
    lo,hi=(np.nan,np.nan) if k=="source-only" else paired_ci(k)
    ci="" if k=="source-only" else f"  [{lo:+.4f}, {hi:+.4f}]"
    print(f"  {k:<36}{np.mean(v):>13.4f}{np.std(v):>8.3f}{(np.mean(v)-base)*100:>+11.1f}pp{ci}")
    rows.append(dict(variant=k, macroF1=np.mean(v), sd=np.std(v),
                     d_macroF1=np.mean(v)-base, ci_lo=lo, ci_hi=hi, n_seeds=len(v)))

import pandas as pd, os
out=os.path.join(os.path.dirname(__file__),"..","results","exp19_diga_impl.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print("\n-> results/exp19_diga_impl.csv")
