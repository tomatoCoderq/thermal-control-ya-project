"""Итер 5: boundary-loss (п.1) и EfficientNet-бэкбон (п.3) — по отдельности и вместе.
Оценка 3-fold. База: resnet34+focal-tversky (3-fold 0.660)."""
import os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm
from scipy.ndimage import distance_transform_edt
ROOT="/Users/tomatocoder/Documents/thermal-control-ya-project"
import sys; sys.path.insert(0, f"{ROOT}/datasets")
from aug import HorizontalFlip, VerticalFlip, Transpose, RandomRotate90, Compose
SEED=67; dev="mps" if torch.backends.mps.is_available() else "cpu"; rng=np.random.default_rng(SEED)
d=np.load("/tmp/seg14_cache.npz"); FEAT,MASK,perm0=d["FEAT"],d["MASK"],d["perm"]
INCH=FEAT.shape[1]; Sz=FEAT.shape[-1]; Ft=torch.from_numpy(FEAT)
AUG=Compose([HorizontalFlip(0.5),VerticalFlip(0.5),Transpose(0.5),RandomRotate90(0.5)])

class PretrainedUNet(nn.Module):
    def __init__(s,inch,backbone):
        super().__init__(); s.enc=timm.create_model(backbone,pretrained=True,features_only=True,in_chans=inch)
        ch=s.enc.feature_info.channels()
        def up(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.GroupNorm(8,o),nn.ReLU(),
                                          nn.Conv2d(o,o,3,padding=1),nn.GroupNorm(8,o),nn.ReLU())
        s.u4=up(ch[4]+ch[3],256);s.u3=up(256+ch[2],128);s.u2=up(128+ch[1],64);s.u1=up(64+ch[0],32)
        s.u0=up(32,16);s.outc=nn.Conv2d(16,1,1);s.upx=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False)
    def forward(s,x):
        f0,f1,f2,f3,f4=s.enc(x)
        x=s.u4(torch.cat([s.upx(f4),f3],1));x=s.u3(torch.cat([s.upx(x),f2],1))
        x=s.u2(torch.cat([s.upx(x),f1],1));x=s.u1(torch.cat([s.upx(x),f0],1))
        return s.outc(s.u0(s.upx(x)))

bce=nn.BCEWithLogitsLoss()
def focal_tversky(l,t,a=0.7,b=0.3,g=0.75,eps=1.):
    p=torch.sigmoid(l).flatten(1);t=t.flatten(1);tp=(p*t).sum(1);fn=((1-p)*t).sum(1);fp=(p*(1-t)).sum(1)
    return ((1-(tp+eps)/(tp+a*fn+b*fp+eps))**g).mean()
def sdf(m):                      # знаковая дистанция: >0 снаружи, <0 внутри, норм [-1,1]
    mb=m>0.5
    if mb.any() and (~mb).any():
        s=distance_transform_edt(~mb)-distance_transform_edt(mb); return (s/(np.abs(s).max()+1e-6)).astype(np.float32)
    return np.zeros_like(m,np.float32)
def augment(feat,mask):
    hwc=np.ascontiguousarray(feat.transpose(1,2,0)); hwc,m=AUG(hwc,mask)
    return np.ascontiguousarray(hwc.transpose(2,0,1)), m
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
def evalset(net,idx,thr):
    io=[]
    for i in idx:
        prob=tta_prob(net,Ft[i:i+1].to(dev)); pm=prob>thr; gm=MASK[i]>0.5; inter=(pm&gm).sum()
        io.append(inter/max((pm|gm).sum(),1))
    return np.mean(io)
def best_thr(net,vaI):
    b=(0.5,-1)
    for t in np.linspace(0.25,0.7,10):
        v=evalset(net,vaI,t);
        if v>b[1]: b=(t,v)
    return b[0]
def train(trI,vaI,backbone,use_bd,seed=SEED,maxep=70,pat=15):
    torch.manual_seed(seed); net=PretrainedUNet(INCH,backbone).to(dev)
    opt=torch.optim.Adam([{"params":net.enc.parameters(),"lr":1e-4},
                          {"params":[p for n,p in net.named_parameters() if not n.startswith("enc.")],"lr":1e-3}])
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,maxep); best=-1;bs=None;wait=0
    for ep in range(maxep):
        net.train(); order=rng.permutation(trI)
        for j in range(0,len(order),4):
            xs=[];ms=[];sd=[]
            for i in order[j:j+4]:
                a,m=augment(FEAT[i],MASK[i]); xs.append(a);ms.append(m); sd.append(sdf(m))
            x=torch.from_numpy(np.stack(xs)).float().to(dev); m=torch.from_numpy(np.stack(ms))[:,None].float().to(dev)
            out=net(x); L=0.5*bce(out,m)+focal_tversky(out,m)
            if use_bd:
                sdt=torch.from_numpy(np.stack(sd))[:,None].float().to(dev)
                L=L+0.5*(torch.sigmoid(out)*sdt).mean()      # boundary loss (Kervadec)
            opt.zero_grad(); L.backward(); opt.step()
        sch.step(); net.eval(); v=evalset(net,vaI,0.5)
        if v>best: best=v;bs={k:vv.cpu().clone() for k,vv in net.state_dict().items()};wait=0
        else: wait+=1
        if wait>=pat: break
    net.load_state_dict(bs); net.eval(); return net
def run_cfg(tag,backbone,use_bd):
    ios=[]
    for sh in range(0,30,10):
        p=np.roll(perm0,sh); teI=p[:10];vaI=p[10:14];trI=p[14:]
        net=train(trI,vaI,backbone,use_bd); ios.append(evalset(net,teI,best_thr(net,vaI)))
    m,s=np.mean(ios),np.std(ios); print(f"[{tag}] 3-fold IoU={m:.3f} ± {s:.3f}  folds={[round(x,3) for x in ios]}",flush=True); return m

t0=time.time()
print("=== A: resnet34 (база) ===",flush=True); A=run_cfg("A resnet34",       "resnet34",       False)
print("=== B (п.1): resnet34 + boundary ===",flush=True); B=run_cfg("B res+bd","resnet34",       True)
print("=== C (п.3): efficientnet_b0 ===",flush=True); C=run_cfg("C effnet",    "efficientnet_b0",False)
print(f"\nA={A:.3f} B(bd)={B:.3f} C(effnet)={C:.3f}",flush=True)
if B>A:
    print("=== D (п.1+3): efficientnet + boundary (т.к. B>A) ===",flush=True)
    D=run_cfg("D eff+bd","efficientnet_b0",True); print(f"D={D:.3f}",flush=True)
else:
    print("boundary (B) не улучшил базу — совместный тест D пропущен.",flush=True)
print(f"\nвремя {time.time()-t0:.0f}s",flush=True)
