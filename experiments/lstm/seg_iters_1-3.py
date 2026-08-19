"""3 итерации улучшения сегментации → цель test IoU 0.75-0.8+. Кэш фич для скорости."""
import glob, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import scipy.io as sio
from scipy import ndimage, fft
from sklearn.decomposition import PCA
from PIL import Image
import cv2
ROOT="/Users/tomatocoder/Documents/thermal-control-ya-project"
DATA=f"{ROOT}/datasets/datasets_list/dataset_kaggle/data"; MASKD=f"{ROOT}/datasets/datasets_list/dataset_kaggle/masks"
GRAY2CLS={0:0,51:1,102:2,153:3,204:4,255:5}
Sz,SEED=256,67; dev="mps" if torch.backends.mps.is_available() else "cpu"; rng=np.random.default_rng(SEED)
CACHE="/tmp/seg14_cache.npz"

def build_cache():
    def load_video(p):
        m=sio.loadmat(p); k=next(k for k in ("imageArray","data","IMAGES") if k in m and np.asarray(m[k]).ndim==3)
        return np.transpose(np.asarray(m[k]).astype(np.float32),(2,0,1))
    def mask_cls(vid):
        im=np.array(Image.open(f"{MASKD}/{vid}.png")); c=np.zeros_like(im,np.uint8)
        for g,cl in GRAY2CLS.items(): c[im==g]=cl
        return c
    def rs(a): return cv2.resize(a,(Sz,Sz),interpolation=cv2.INTER_LINEAR)
    def rsn(a): return cv2.resize(a.astype(np.uint8),(Sz,Sz),interpolation=cv2.INTER_NEAREST)
    def zc(a): return (a-a.mean((1,2),keepdims=True))/(a.std((1,2),keepdims=True)+1e-6)
    FEAT=[];MASK=[];VID=[]
    for p in sorted(glob.glob(f"{DATA}/*.mat")):
        vid=os.path.splitext(os.path.basename(p))[0]; fp=f"{ROOT}/features_p5/{vid}.npy"
        if not os.path.exists(f"{MASKD}/{vid}.png") or not os.path.exists(fp): continue
        X=load_video(p);T,H,W=X.shape;peak=int(X.reshape(T,-1).mean(1).argmax());base=X[:max(1,peak//4)].mean(0)
        dT=np.clip(X[peak:]-base[None],1e-3,None)
        tsr=np.stack([rs(c) for c in np.load(fp)]).astype(np.float32)
        eof=PCA(3,random_state=0).fit(dT.reshape(dT.shape[0],-1)).components_.reshape(3,H,W)
        pca=np.stack([rs(e) for e in eof]).astype(np.float32)
        ph=np.angle(fft.rfft(dT,axis=0)[1:4]); ppt=np.stack([rs(x) for x in ph.astype(np.float32)])
        c=np.load(fp); d2=lambda u:20*c[0]*u**3+12*c[1]*u**2+6*c[2]*u+2*c[3]
        u1,u2=np.log(max(2,(T-peak)//8)),np.log(max(3,(T-peak)//2))
        dd=np.stack([rs(d2(u1)),rs(d2(u2))]).astype(np.float32)
        FEAT.append(np.concatenate([zc(tsr),zc(pca),zc(ppt),zc(dd)],0)); MASK.append((rsn(mask_cls(vid))>0).astype(np.float32)); VID.append(vid)
    FEAT=np.stack(FEAT);MASK=np.stack(MASK);V=len(VID)
    perm=rng.permutation(V)
    np.savez(CACHE,FEAT=FEAT,MASK=MASK,perm=perm)
    return FEAT,MASK,perm

if os.path.exists(CACHE):
    d=np.load(CACHE); FEAT,MASK,perm=d["FEAT"],d["MASK"],d["perm"]; print("кэш загружен",FEAT.shape,flush=True)
else:
    print("строю кэш…",flush=True); FEAT,MASK,perm=build_cache(); print("кэш готов",FEAT.shape,flush=True)
INCH=FEAT.shape[1]; teI=perm[:10]; vaI=perm[10:14]; trI=perm[14:]
Ft=torch.from_numpy(FEAT); print(f"train={len(trI)} val={len(vaI)} test={len(teI)} INCH={INCH}",flush=True)

# ── блоки ────────────────────────────────────────────────────────────────────
def cbr(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.GroupNorm(8,o),nn.ReLU(),
                                   nn.Conv2d(o,o,3,padding=1),nn.GroupNorm(8,o),nn.ReLU())
class AttGate(nn.Module):
    def __init__(s,g,x,n):
        super().__init__(); s.wg=nn.Conv2d(g,n,1); s.wx=nn.Conv2d(x,n,1); s.psi=nn.Conv2d(n,1,1)
    def forward(s,g,x): return x*torch.sigmoid(s.psi(F.relu(s.wg(g)+s.wx(x))))
class AttUNet(nn.Module):
    def __init__(s,inch,deep=False):
        super().__init__(); s.deep=deep
        s.d1=cbr(inch,32);s.d2=cbr(32,64);s.d3=cbr(64,128);s.d4=cbr(128,256);s.bott=cbr(256,512)
        s.pool=nn.MaxPool2d(2);s.up=nn.Upsample(scale_factor=2)
        s.a4=AttGate(512,256,128);s.u4=cbr(512+256,256)
        s.a3=AttGate(256,128,64);s.u3=cbr(256+128,128)
        s.a2=AttGate(128,64,32);s.u2=cbr(128+64,64)
        s.a1=AttGate(64,32,16);s.u1=cbr(64+32,32)
        s.outc=nn.Conv2d(32,1,1)
        if deep: s.ds3=nn.Conv2d(128,1,1); s.ds2=nn.Conv2d(64,1,1)
    def forward(s,x):
        e1=s.d1(x);e2=s.d2(s.pool(e1));e3=s.d3(s.pool(e2));e4=s.d4(s.pool(e3));b=s.bott(s.pool(e4))
        g=s.up(b);x4=s.u4(torch.cat([g,s.a4(g,e4)],1))
        g=s.up(x4);x3=s.u3(torch.cat([g,s.a3(g,e3)],1))
        g=s.up(x3);x2=s.u2(torch.cat([g,s.a2(g,e2)],1))
        g=s.up(x2);x1=s.u1(torch.cat([g,s.a1(g,e1)],1))
        out=s.outc(x1)
        if s.deep and s.training:
            H=out.shape[-1]
            return out,[F.interpolate(s.ds3(x3),size=H,mode='bilinear',align_corners=False),
                        F.interpolate(s.ds2(x2),size=H,mode='bilinear',align_corners=False)]
        return out,[]

# ── лоссы ────────────────────────────────────────────────────────────────────
bce=nn.BCEWithLogitsLoss()
def dice_l(l,t,eps=1.):
    p=torch.sigmoid(l).flatten(1);t=t.flatten(1); return 1-((2*(p*t).sum(1)+eps)/(p.sum(1)+t.sum(1)+eps)).mean()
def focal_tversky(l,t,a=0.7,b=0.3,g=0.75,eps=1.):
    p=torch.sigmoid(l).flatten(1);t=t.flatten(1)
    tp=(p*t).sum(1);fn=((1-p)*t).sum(1);fp=(p*(1-t)).sum(1)
    tv=(tp+eps)/(tp+a*fn+b*fp+eps); return ((1-tv)**g).mean()

# ── аугментации ──────────────────────────────────────────────────────────────
def aug(x,m,strong=False):
    if rng.random()<0.5: x=x[...,::-1].copy();m=m[...,::-1].copy()
    if rng.random()<0.5: x=x[...,::-1,:].copy();m=m[...,::-1,:].copy()
    k=rng.integers(0,4)
    if k: x=np.rot90(x,k,(-2,-1)).copy();m=np.rot90(m,k,(-2,-1)).copy()
    if strong:
        if rng.random()<0.5: x=x+rng.normal(0,0.1,x.shape).astype(np.float32)      # шум
        if rng.random()<0.5: x=x*rng.uniform(0.8,1.2)                               # контраст
        if rng.random()<0.4:                                                        # небольшой зум
            z=rng.uniform(0.85,1.15); Snew=int(Sz*z)
            xi=np.stack([cv2.resize(c,(Snew,Snew)) for c in x]); mi=cv2.resize(m,(Snew,Snew),interpolation=cv2.INTER_NEAREST)
            if Snew>=Sz: o=(Snew-Sz)//2; x=xi[:,o:o+Sz,o:o+Sz].copy(); m=(mi[o:o+Sz,o:o+Sz]>0.5).astype(np.float32)
            else: p=(Sz-Snew)//2; x=np.pad(xi,((0,0),(p,Sz-Snew-p),(p,Sz-Snew-p))).astype(np.float32); m=np.pad(mi,((p,Sz-Snew-p),(p,Sz-Snew-p)))
    return x.astype(np.float32),m.astype(np.float32)

TTA=[(0,False,False),(0,True,False),(0,False,True),(2,False,False)]  # rot90k, hflip, vflip
def tta_prob(net,x):                    # x (1,C,S,S) tensor
    ps=[]
    for k,hf,vf in TTA:
        xi=x
        if hf: xi=torch.flip(xi,[-1])
        if vf: xi=torch.flip(xi,[-2])
        if k: xi=torch.rot90(xi,k,[-2,-1])
        with torch.no_grad(): p=torch.sigmoid(net(xi)[0])
        if k: p=torch.rot90(p,-k,[-2,-1])
        if vf: p=torch.flip(p,[-2])
        if hf: p=torch.flip(p,[-1])
        ps.append(p)
    return torch.stack(ps).mean(0)[0,0].cpu().numpy()

def iou_dice(pm,gm):
    inter=(pm&gm).sum(); return inter/max((pm|gm).sum(),1), 2*inter/max(pm.sum()+gm.sum(),1)
def eval_set(nets,idx,thr=0.5,use_tta=False):
    ious=[];dices=[]
    for i in idx:
        x=Ft[i:i+1].to(dev)
        if use_tta: prob=np.mean([tta_prob(n,x) for n in nets],0)
        else:
            with torch.no_grad(): prob=np.mean([torch.sigmoid(n(x)[0])[0,0].cpu().numpy() for n in nets],0)
        pm=prob>thr; gm=MASK[i]>0.5; io,di=iou_dice(pm,gm); ious.append(io);dices.append(di)
    return np.mean(ious),np.mean(dices)
def best_thr(nets,use_tta=False):
    best=(0.5,-1)
    for t in np.linspace(0.25,0.7,10):
        _,d=eval_set(nets,vaI,t,use_tta)
        if d>best[1]: best=(t,d)
    return best[0]

def train_one(seed,loss_kind,deep,strong,maxep=80,pat=15):
    torch.manual_seed(seed); net=AttUNet(INCH,deep=deep).to(dev)
    opt=torch.optim.Adam(net.parameters(),1e-3); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,maxep)
    best=-1;bs=None;wait=0
    for ep in range(maxep):
        net.train();order=rng.permutation(trI)
        for j in range(0,len(order),4):
            xs=[];ms=[]
            for i in order[j:j+4]:
                a,b=aug(FEAT[i],MASK[i],strong); xs.append(a);ms.append(b)
            x=torch.from_numpy(np.stack(xs)).to(dev); m=torch.from_numpy(np.stack(ms))[:,None].to(dev)
            out,ds=net(x)
            if loss_kind=="bcedice": L=bce(out,m)+dice_l(out,m)
            else: L=0.5*bce(out,m)+focal_tversky(out,m)
            for k,d in enumerate(ds): L=L+(0.4 if k==0 else 0.2)*focal_tversky(d,m)
            opt.zero_grad();L.backward();opt.step()
        sch.step()
        net.eval(); _,vd=eval_set([net],vaI)
        if vd>best: best=vd;bs={k:v.cpu().clone() for k,v in net.state_dict().items()};wait=0
        else: wait+=1
        if wait>=pat: break
    net.load_state_dict(bs); net.eval(); return net,best

def report(tag,nets,use_tta):
    thr=best_thr(nets,use_tta); io,di=eval_set(nets,teI,thr,use_tta)
    print(f"[{tag}] TEST IoU={io:.3f} Dice={di:.3f} (thr={thr:.2f}, tta={use_tta}, nets={len(nets)})",flush=True)
    return io,di

print("\n===== ИТЕР 1: Attention U-Net + Focal-Tversky =====",flush=True)
t0=time.time(); n1,_=train_one(SEED,"ft",deep=False,strong=False); io1,_=report("iter1",[n1],False); print(f"  {time.time()-t0:.0f}s",flush=True)
print("\n===== ИТЕР 2: + deep supervision + strong aug + TTA + thr-tuning =====",flush=True)
t0=time.time(); n2,_=train_one(SEED,"ft",deep=True,strong=True); io2,_=report("iter2",[n2],True); print(f"  {time.time()-t0:.0f}s",flush=True)
print("\n===== ИТЕР 3: + ансамбль (3 сида) =====",flush=True)
t0=time.time(); ens=[n2]+[train_one(s,"ft",deep=True,strong=True)[0] for s in (11,23)]; io3,_=report("iter3-ens",ens,True); print(f"  {time.time()-t0:.0f}s",flush=True)
print(f"\nИТОГ: iter1={io1:.3f} iter2={io2:.3f} iter3={io3:.3f} | цель 0.75-0.8",flush=True)
