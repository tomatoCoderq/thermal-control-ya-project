"""Итер 4: ImageNet-претрейн энкодер (resnet34, in_chans=14) + U-Net-декодер,
аугментации из datasets/aug.py, кэш 14ch. Цель test IoU 0.75-0.8."""
import os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm
ROOT="/Users/tomatocoder/Documents/thermal-control-ya-project"
import sys; sys.path.insert(0, f"{ROOT}/datasets")
from aug import HorizontalFlip, VerticalFlip, Transpose, RandomRotate90, Compose  # аугментации из datasets/
SEED=67; dev="mps" if torch.backends.mps.is_available() else "cpu"; rng=np.random.default_rng(SEED)
d=np.load("/tmp/seg14_cache.npz"); FEAT,MASK,perm=d["FEAT"],d["MASK"],d["perm"]
INCH=FEAT.shape[1]; Sz=FEAT.shape[-1]; Ft=torch.from_numpy(FEAT)
teI=perm[:10]; vaI=perm[10:14]; trI=perm[14:]
print(f"кэш {FEAT.shape} | train={len(trI)} val={len(vaI)} test={len(teI)} INCH={INCH}",flush=True)
AUG=Compose([HorizontalFlip(0.5),VerticalFlip(0.5),Transpose(0.5),RandomRotate90(0.5)])

class PretrainedUNet(nn.Module):
    def __init__(s,inch,backbone="resnet34"):
        super().__init__()
        s.enc=timm.create_model(backbone,pretrained=True,features_only=True,in_chans=inch)
        ch=s.enc.feature_info.channels()          # [64,64,128,256,512] strides 2,4,8,16,32
        def up(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.GroupNorm(8,o),nn.ReLU(),
                                          nn.Conv2d(o,o,3,padding=1),nn.GroupNorm(8,o),nn.ReLU())
        s.u4=up(ch[4]+ch[3],256); s.u3=up(256+ch[2],128); s.u2=up(128+ch[1],64); s.u1=up(64+ch[0],32)
        s.u0=up(32,16); s.outc=nn.Conv2d(16,1,1); s.upx=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False)
    def forward(s,x):
        f0,f1,f2,f3,f4=s.enc(x)                    # 128,64,32,16,8
        x=s.u4(torch.cat([s.upx(f4),f3],1)); x=s.u3(torch.cat([s.upx(x),f2],1))
        x=s.u2(torch.cat([s.upx(x),f1],1)); x=s.u1(torch.cat([s.upx(x),f0],1))
        x=s.u0(s.upx(x)); return s.outc(x)

bce=nn.BCEWithLogitsLoss()
def focal_tversky(l,t,a=0.7,b=0.3,g=0.75,eps=1.):
    p=torch.sigmoid(l).flatten(1);t=t.flatten(1); tp=(p*t).sum(1);fn=((1-p)*t).sum(1);fp=(p*(1-t)).sum(1)
    return ((1-(tp+eps)/(tp+a*fn+b*fp+eps))**g).mean()
def augment(feat,mask):
    hwc=np.ascontiguousarray(feat.transpose(1,2,0))       # (H,W,C) для datasets-аугментаций
    hwc,m=AUG(hwc,mask); return np.ascontiguousarray(hwc.transpose(2,0,1)), m
TTA=[(0,0,0),(0,1,0),(0,0,1),(2,0,0)]
def tta_prob(net,x):
    ps=[]
    for k,hf,vf in TTA:
        xi=x
        if hf: xi=torch.flip(xi,[-1])
        if vf: xi=torch.flip(xi,[-2])
        if k: xi=torch.rot90(xi,k,[-2,-1])
        with torch.no_grad(): p=torch.sigmoid(net(xi))
        if k: p=torch.rot90(p,-k,[-2,-1])
        if vf: p=torch.flip(p,[-2])
        if hf: p=torch.flip(p,[-1])
        ps.append(p)
    return torch.stack(ps).mean(0)[0,0].cpu().numpy()
def evalset(net,idx,thr,tta=True):
    io=[];di=[]
    for i in idx:
        x=Ft[i:i+1].to(dev); prob=tta_prob(net,x) if tta else torch.sigmoid(net(x))[0,0].detach().cpu().numpy()
        pm=prob>thr; gm=MASK[i]>0.5; inter=(pm&gm).sum()
        io.append(inter/max((pm|gm).sum(),1)); di.append(2*inter/max(pm.sum()+gm.sum(),1))
    return np.mean(io),np.mean(di)
def best_thr(net):
    b=(0.5,-1)
    for t in np.linspace(0.25,0.7,10):
        _,dd=evalset(net,vaI,t);
        if dd>b[1]: b=(t,dd)
    return b[0]
def train(seed=SEED,maxep=70,pat=15):
    torch.manual_seed(seed); net=PretrainedUNet(INCH).to(dev)
    opt=torch.optim.Adam([{"params":net.enc.parameters(),"lr":1e-4},
                          {"params":[p for n,p in net.named_parameters() if not n.startswith("enc.")],"lr":1e-3}])
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,maxep)
    best=-1;bs=None;wait=0
    for ep in range(maxep):
        net.train(); order=rng.permutation(trI)
        for j in range(0,len(order),4):
            xs=[];ms=[]
            for i in order[j:j+4]:
                a,m=augment(FEAT[i],MASK[i]); xs.append(a);ms.append(m)
            x=torch.from_numpy(np.stack(xs)).float().to(dev); m=torch.from_numpy(np.stack(ms))[:,None].float().to(dev)
            out=net(x); L=0.5*bce(out,m)+focal_tversky(out,m)
            opt.zero_grad(); L.backward(); opt.step()
        sch.step(); net.eval(); _,vd=evalset(net,vaI,0.5)
        if vd>best: best=vd;bs={k:v.cpu().clone() for k,v in net.state_dict().items()};wait=0
        else: wait+=1
        if ep%10==9: print(f"  ep{ep+1} valDice={vd:.3f} best={best:.3f}",flush=True)
        if wait>=pat: print(f"  early stop ep{ep+1}",flush=True); break
    net.load_state_dict(bs); net.eval(); return net

print("\n===== ИТЕР 4: ImageNet-претрейн resnet34 + U-Net =====",flush=True)
t0=time.time(); net=train(); thr=best_thr(net); io,di=evalset(net,teI,thr)
print(f"[iter4] TEST IoU={io:.3f} Dice={di:.3f} (thr={thr:.2f}, TTA) | {time.time()-t0:.0f}s",flush=True)
# 3-fold робастность
ios=[]
for sh in range(0,30,10):
    p=np.roll(perm,sh); globals()['teI']=p[:10]; globals()['vaI']=p[10:14]; globals()['trI']=p[14:]
    n=train(); io2,_=evalset(n,teI,best_thr(n)); ios.append(io2); print(f"  fold{sh}: IoU={io2:.3f}",flush=True)
print(f"\n[iter4] 3-fold IoU={np.mean(ios):.3f} ± {np.std(ios):.3f}",flush=True)
