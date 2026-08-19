"""Голова глубины (U-Net-энкодер + ROI-пул + GRU, K=16 crop-level): предсказанные
значения по классам глубины + метрики (MAE/RMSE/R²). kaggle из нового data-слоя."""
import glob, os, numpy as np, torch, torch.nn as nn
import scipy.io as sio
from scipy import ndimage
from PIL import Image
ROOT="/Users/tomatocoder/Documents/thermal-control-ya-project"
DATA=f"{ROOT}/datasets/datasets_list/dataset_kaggle/data"; MASKD=f"{ROOT}/datasets/datasets_list/dataset_kaggle/masks"
GRAY2CLS={0:0,51:1,102:2,153:3,204:4,255:5}; CLS_DEPTH_MM={1:5.,2:10.,3:15.,4:20.,5:25.}
N,K,SEED=64,16,67; dev="mps" if torch.backends.mps.is_available() else "cpu"; rng=np.random.default_rng(SEED)
def load(p): m=sio.loadmat(p)["imageArray"].astype(np.float32); return np.transpose(m,(2,0,1))
def mcls(v):
    im=np.array(Image.open(f"{MASKD}/{v}.png")); c=np.zeros_like(im,np.uint8)
    for g,cl in GRAY2CLS.items(): c[im==g]=cl
    return c
def logidx(peak,T):
    hi=min(peak+800,T-1); li=np.unique(np.clip((peak+np.geomspace(1,max(hi-peak,2),K)).astype(int),0,T-1))
    while len(li)<K: li=np.append(li,li[-1])
    return li[:K]
print("извлекаю…",flush=True)
Xl,M,D,VID=[],[],[],[]
for p in sorted(glob.glob(f"{DATA}/*.mat")):
    v=os.path.splitext(os.path.basename(p))[0]
    if not os.path.exists(f"{MASKD}/{v}.png"): continue
    X=load(p);T,H,W=X.shape;peak=int(X.reshape(T,-1).mean(1).argmax());base=X[:max(1,peak//4)].mean(0)
    cls=mcls(v);li=logidx(peak,T)
    for c in range(1,6):
        lbl,n=ndimage.label(cls==c)
        for k in range(1,n+1):
            rr,cc=ndimage.center_of_mass(lbl==k);r0=int(np.clip(rr-N//2,0,H-N));c0=int(np.clip(cc-N//2,0,W-N))
            s=(X[li]-base[None])[:,r0:r0+N,c0:c0+N];s=((s-s.mean())/(s.std()+1e-6)).astype(np.float32)
            Xl.append(s);M.append((cls[r0:r0+N,c0:c0+N]>0).astype(np.float32));D.append(CLS_DEPTH_MM[c]);VID.append(v)
Xl=np.stack(Xl);M=np.stack(M);D=np.array(D,np.float32);VID=np.array(VID)
vids=np.array(sorted(set(VID.tolist())));te=set(rng.permutation(vids)[:10].tolist())
tr=~np.isin(VID,list(te));teM=np.isin(VID,list(te));dmean,dstd=D[tr].mean(),D[tr].std()+1e-6
print(f"дефектов={len(D)} train={tr.sum()} test={teM.sum()}",flush=True)
class Enc(nn.Module):
    def __init__(s):
        super().__init__()
        def blk(i,o):return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.GroupNorm(8,o),nn.ReLU(),nn.MaxPool2d(2))
        s.net=nn.Sequential(blk(1,32),blk(32,64),blk(64,128),blk(128,256))
    def forward(s,x):return s.net(x)
class Net(nn.Module):
    def __init__(s):super().__init__();s.enc=Enc();s.gru=nn.GRU(256,128,batch_first=True);s.head=nn.Linear(128,1)
    def forward(s,x,roi):
        B,Kk,_,_=x.shape;b=s.enc(x.reshape(B*Kk,1,N,N));hb,wb=b.shape[-2:];b=b.reshape(B,Kk,256,hb,wb)
        import torch.nn.functional as F
        w=F.adaptive_avg_pool2d(roi[:,None],(hb,wb))+0.1
        pooled=(b*w[:,None]).sum((-2,-1))/w.sum((-2,-1),keepdim=True).squeeze(-1)
        _,hn=s.gru(pooled);return s.head(hn[-1]).squeeze(-1)
huber=nn.SmoothL1Loss();torch.manual_seed(SEED)
net=Net().to(dev);opt=torch.optim.Adam(net.parameters(),1e-3)
Xt=torch.from_numpy(Xl);Mt=torch.from_numpy(M);Dt=torch.from_numpy((D-dmean)/dstd)
ii=np.where(tr)[0]
for ep in range(30):
    rng.shuffle(ii)
    for j in range(0,len(ii),32):
        bi=ii[j:j+32];x=Xt[bi].to(dev);roi=Mt[bi].to(dev);d=Dt[bi].to(dev)
        dep=net(x,roi);loss=huber(dep,d);opt.zero_grad();loss.backward();opt.step()
net.eval();pr=[]
with torch.no_grad():
    for j in range(0,teM.sum(),64):
        bi=np.where(teM)[0][j:j+64];pr+=list(net(Xt[bi].to(dev),Mt[bi].to(dev)).cpu().numpy()*dstd+dmean)
pr=np.array(pr);gt=D[teM]
mae=np.abs(pr-gt).mean();rmse=np.sqrt(((pr-gt)**2).mean())
ss=1-((pr-gt)**2).sum()/(((gt-gt.mean())**2).sum()+1e-9)
print(f"\n=== ГОЛОВА ГЛУБИНЫ (GRU, crop-level, тест) ===")
print(f"MAE={mae:.2f}мм  RMSE={rmse:.2f}мм  R²={ss:.3f}  (test дефектов={len(gt)})")
print(f"\nпредсказанные значения по истинному классу глубины:")
print(f"{'истина, мм':>10} {'n':>3} {'pred среднее':>12} {'pred min..max':>16} {'MAE':>6}")
for c in [5.,10.,15.,20.,25.]:
    m=gt==c
    if m.sum(): print(f"{c:>10.0f} {int(m.sum()):>3} {pr[m].mean():>12.2f} {pr[m].min():>7.1f}..{pr[m].max():<7.1f} {np.abs(pr[m]-c).mean():>6.2f}")
print(f"\nдиапазон всех предсказаний: {pr.min():.1f}..{pr.max():.1f} мм (истина {gt.min():.0f}..{gt.max():.0f})")
