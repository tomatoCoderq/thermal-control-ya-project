"""Итер 6: добавить tpu-домен (через новый data-слой) → kaggle+tpu, лучший конфиг
(efficientnet_b0 + ImageNet + datasets-аугментации). Замер по доменам, 3-fold."""
import os, time, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm
import scipy.io as sio
from scipy import fft
from scipy.ndimage import distance_transform_edt  # noqa
from sklearn.decomposition import PCA
from PIL import Image
import cv2
ROOT="/Users/tomatocoder/Documents/thermal-control-ya-project"
import sys; sys.path.insert(0, f"{ROOT}/datasets")
from aug import HorizontalFlip, VerticalFlip, Transpose, RandomRotate90, Compose
SEED=67; Sz=256; dev="mps" if torch.backends.mps.is_available() else "cpu"; rng=np.random.default_rng(SEED)
TPU_D=f"{ROOT}/datasets/datasets_list/dataset_tpu"
TPU_CACHE="/tmp/seg14_tpu.npz"
def rs(a): return cv2.resize(a,(Sz,Sz),interpolation=cv2.INTER_LINEAR)
def rsn(a): return cv2.resize(a.astype(np.uint8),(Sz,Sz),interpolation=cv2.INTER_NEAREST)
def zc(a): return (a-a.mean((1,2),keepdims=True))/(a.std((1,2),keepdims=True)+1e-6)

def build_tpu():
    FEAT=[];MASK=[];VID=[]
    for p in sorted(glob.glob(f"{TPU_D}/data/*.mat")):
        vid=os.path.splitext(os.path.basename(p))[0]; mp=f"{TPU_D}/masks/{vid}.png"
        if not os.path.exists(mp): continue
        X=np.transpose(sio.loadmat(p)["data"].astype(np.float32),(2,0,1))  # (T,240,320)
        T,H,W=X.shape; peak=int(X.reshape(T,-1).mean(1).argmax()); base=X[:max(1,peak//4)].mean(0)
        dT=np.clip(X[peak:]-base[None],1e-3,None); n=dT.shape[0]
        logt=np.log(np.arange(1,n+1)); coef=np.polyfit(logt,np.log(dT).reshape(n,-1),5).reshape(6,H,W).astype(np.float32)
        tsr=np.stack([rs(c) for c in coef])
        eof=PCA(3,random_state=0).fit(dT.reshape(n,-1)).components_.reshape(3,H,W)
        pca=np.stack([rs(e) for e in eof]).astype(np.float32)
        ph=np.angle(fft.rfft(dT,axis=0)[1:4]); ppt=np.stack([rs(x) for x in ph.astype(np.float32)])
        d2=lambda u:20*coef[0]*u**3+12*coef[1]*u**2+6*coef[2]*u+2*coef[3]
        u1,u2=np.log(max(2,n//8)),np.log(max(3,n//2)); dd=np.stack([rs(d2(u1)),rs(d2(u2))]).astype(np.float32)
        feat=np.concatenate([zc(tsr),zc(pca),zc(ppt),zc(dd)],0)
        msk=(rsn(np.array(Image.open(mp)))>0).astype(np.float32)
        FEAT.append(feat);MASK.append(msk);VID.append(vid)
    F_=np.stack(FEAT);M_=np.stack(MASK); np.savez(TPU_CACHE,FEAT=F_,MASK=M_); return F_,M_
if os.path.exists(TPU_CACHE):
    z=np.load(TPU_CACHE); TF,TM=z["FEAT"],z["MASK"]; print("tpu кэш",TF.shape,flush=True)
else:
    print("строю tpu-кэш…",flush=True); TF,TM=build_tpu(); print("tpu кэш",TF.shape,flush=True)

kz=np.load("/tmp/seg14_cache.npz"); KF,KM=kz["FEAT"],kz["MASK"]
FEAT=np.concatenate([KF,TF],0); MASK=np.concatenate([KM,TM],0)
DOM=np.array(["kaggle"]*len(KF)+["tpu"]*len(TF)); INCH=FEAT.shape[1]; Ft=torch.from_numpy(FEAT)
print(f"combined: kaggle={len(KF)} tpu={len(TF)} всего={len(FEAT)} INCH={INCH}",flush=True)
AUG=Compose([HorizontalFlip(0.5),VerticalFlip(0.5),Transpose(0.5),RandomRotate90(0.5)])

class PUNet(nn.Module):
    def __init__(s,inch,bb="efficientnet_b0"):
        super().__init__(); s.enc=timm.create_model(bb,pretrained=True,features_only=True,in_chans=inch)
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
def ftl(l,t,a=0.7,b=0.3,g=0.75,e=1.):
    p=torch.sigmoid(l).flatten(1);t=t.flatten(1);tp=(p*t).sum(1);fn=((1-p)*t).sum(1);fp=(p*(1-t)).sum(1)
    return ((1-(tp+e)/(tp+a*fn+b*fp+e))**g).mean()
def augment(feat,mask):
    hwc=np.ascontiguousarray(feat.transpose(1,2,0)); hwc,m=AUG(hwc,mask); return np.ascontiguousarray(hwc.transpose(2,0,1)),m
TTA=[(0,0,0),(0,1,0),(0,0,1),(2,0,0)]
def tta(net,x):
    ps=[]
    for k,hf,vf in TTA:
        xi=x
        if hf: xi=torch.flip(xi,[-1])
        if vf: xi=torch.flip(xi,[-2])
        if k: xi=torch.rot90(xi,k,[-2,-1])
        with torch.no_grad(): p=torch.sigmoid(net(xi))
        if k:p=torch.rot90(p,-k,[-2,-1])
        if vf:p=torch.flip(p,[-2])
        if hf:p=torch.flip(p,[-1])
        ps.append(p)
    return torch.stack(ps).mean(0)[0,0].cpu().numpy()
def iou_at(net,idx,thr):
    io=[]
    for i in idx:
        pm=tta(net,Ft[i:i+1].to(dev))>thr; gm=MASK[i]>0.5; inter=(pm&gm).sum(); io.append(inter/max((pm|gm).sum(),1))
    return np.array(io)
def bthr(net,idx):
    b=(0.5,-1)
    for t in np.linspace(0.25,0.7,10):
        v=iou_at(net,idx,t).mean()
        if v>b[1]:b=(t,v)
    return b[0]
def train(trI,vaI,seed=SEED,maxep=70,pat=15):
    torch.manual_seed(seed); net=PUNet(INCH).to(dev)
    opt=torch.optim.Adam([{"params":net.enc.parameters(),"lr":1e-4},
                          {"params":[p for n,p in net.named_parameters() if not n.startswith("enc.")],"lr":1e-3}])
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,maxep); best=-1;bs=None;wait=0
    for ep in range(maxep):
        net.train(); order=rng.permutation(trI)
        for j in range(0,len(order),4):
            xs=[];ms=[]
            for i in order[j:j+4]:
                a,m=augment(FEAT[i],MASK[i]); xs.append(a);ms.append(m)
            x=torch.from_numpy(np.stack(xs)).float().to(dev); m=torch.from_numpy(np.stack(ms))[:,None].float().to(dev)
            out=net(x); L=0.5*bce(out,m)+ftl(out,m); opt.zero_grad(); L.backward(); opt.step()
        sch.step(); net.eval(); v=iou_at(net,vaI,0.5).mean()
        if v>best:best=v;bs={k:vv.cpu().clone() for k,vv in net.state_dict().items()};wait=0
        else:wait+=1
        if wait>=pat: break
    net.load_state_dict(bs); net.eval(); return net

# 3-fold: тест по 10 kaggle + 4 tpu (оба домена в train/test)
kI=np.where(DOM=="kaggle")[0]; tI=np.where(DOM=="tpu")[0]
allc=[]; allk=[]; allt=[]
for sh in range(3):
    kk=np.roll(kI,sh*10); tt=np.roll(tI,sh*4)
    teI=np.concatenate([kk[:10],tt[:4]]); vaI=np.concatenate([kk[10:12],tt[4:5]])
    trI=np.array([i for i in range(len(FEAT)) if i not in set(teI.tolist())|set(vaI.tolist())])
    net=train(trI,vaI); thr=bthr(net,vaI)
    io=iou_at(net,teI,thr); dk=DOM[teI]=="kaggle"; dt=DOM[teI]=="tpu"
    allc.append(io.mean()); allk.append(io[dk].mean()); allt.append(io[dt].mean())
    print(f"  fold{sh}: all={io.mean():.3f} kaggle={io[dk].mean():.3f} tpu={io[dt].mean():.3f} (thr={thr:.2f})",flush=True)
print(f"\n[iter6 kaggle+tpu] 3-fold IoU: all={np.mean(allc):.3f}±{np.std(allc):.3f} | "
      f"kaggle={np.mean(allk):.3f} | tpu={np.mean(allt):.3f}",flush=True)
print(f"(база efficientnet только kaggle: 0.671) время …",flush=True)
