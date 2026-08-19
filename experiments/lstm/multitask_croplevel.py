"""v2: U-Net + ROI-пулинг латента по предсказанной маске + K=16 (logspaced)."""
import glob, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import scipy.io as sio
from scipy import ndimage
from PIL import Image

ROOT="/Users/tomatocoder/Documents/thermal-control-ya-project"
DATA=f"{ROOT}/datasets/dataset_kaggle/data"; MASKD=f"{ROOT}/datasets/dataset_kaggle/labels/automated_mask"
GRAY2CLS={0:0,51:1,102:2,153:3,204:4,255:5}; CLS_DEPTH_MM={1:5.,2:10.,3:15.,4:20.,5:25.}
N,K,SEED=64,16,67
dev="mps" if torch.backends.mps.is_available() else "cpu"; rng=np.random.default_rng(SEED)

def load_video(p):
    m=sio.loadmat(p); k=next(k for k in ("imageArray","data","IMAGES") if k in m and np.asarray(m[k]).ndim==3)
    return np.transpose(np.asarray(m[k]).astype(np.float32),(2,0,1))
def mask_cls(vid):
    im=np.array(Image.open(f"{MASKD}/{vid}.png")); c=np.zeros_like(im,np.uint8)
    for g,cl in GRAY2CLS.items(): c[im==g]=cl
    return c
def log_idx(peak,T):
    hi=min(peak+800,T-1); li=np.unique(np.clip((peak+np.geomspace(1,max(hi-peak,2),K)).astype(int),0,T-1))
    while len(li)<K: li=np.append(li,li[-1])
    return li[:K]

print("извлекаю…",flush=True)
Xl,M,D,VID=[],[],[],[]
for p in sorted(glob.glob(f"{DATA}/*.mat")):
    vid=os.path.splitext(os.path.basename(p))[0]
    if not os.path.exists(f"{MASKD}/{vid}.png"): continue
    X=load_video(p); T,H,W=X.shape; peak=int(X.reshape(T,-1).mean(1).argmax()); base=X[:max(1,peak//4)].mean(0)
    cls=mask_cls(vid); li=log_idx(peak,T)
    for c in range(1,6):
        lbl,n=ndimage.label(cls==c)
        for k in range(1,n+1):
            rr,cc=ndimage.center_of_mass(lbl==k); r0=int(np.clip(rr-N//2,0,H-N)); c0=int(np.clip(cc-N//2,0,W-N))
            s=(X[li]-base[None])[:,r0:r0+N,c0:c0+N]; s=((s-s.mean())/(s.std()+1e-6)).astype(np.float32)
            Xl.append(s); M.append((cls[r0:r0+N,c0:c0+N]>0).astype(np.float32)); D.append(CLS_DEPTH_MM[c]); VID.append(vid)
Xl=np.stack(Xl); M=np.stack(M); D=np.array(D,np.float32); VID=np.array(VID)
vids=np.array(sorted(set(VID.tolist()))); test_vids=set(rng.permutation(vids)[:10].tolist())
tr=~np.isin(VID,list(test_vids)); te=np.isin(VID,list(test_vids)); dmean,dstd=D[tr].mean(),D[tr].std()+1e-6
print(f"дефектов={len(D)} X={Xl.shape} train={tr.sum()} test={te.sum()}",flush=True)

# ── U-Net ────────────────────────────────────────────────────────────────────
def cbr(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.GroupNorm(8,o),nn.ReLU(),
                                   nn.Conv2d(o,o,3,padding=1),nn.GroupNorm(8,o),nn.ReLU())
class UNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.d1=cbr(1,32); s.d2=cbr(32,64); s.d3=cbr(64,128); s.bott=cbr(128,256)
        s.pool=nn.MaxPool2d(2); s.up=nn.Upsample(scale_factor=2)
        s.u3=cbr(256+128,128); s.u2=cbr(128+64,64); s.u1=cbr(64+32,32); s.outc=nn.Conv2d(32,1,1)
    def encode(s,x):
        e1=s.d1(x); e2=s.d2(s.pool(e1)); e3=s.d3(s.pool(e2)); b=s.bott(s.pool(e3))  # b: (B,256,8,8)
        return b,(e1,e2,e3)
    def decode(s,b,sk):
        e1,e2,e3=sk
        x=s.u3(torch.cat([s.up(b),e3],1)); x=s.u2(torch.cat([s.up(x),e2],1)); x=s.u1(torch.cat([s.up(x),e1],1))
        return s.outc(x)                                                            # (B,1,N,N)

class MultiTaskUNet(nn.Module):
    def __init__(s,d=256):
        super().__init__(); s.unet=UNet(); s.gru=nn.GRU(d,128,batch_first=True); s.head=nn.Linear(128,1); s.d=d
    def forward(s,x):                          # x (B,K,N,N)
        B,Kk,_,_=x.shape
        b,sk=s.unet.encode(x.reshape(B*Kk,1,N,N))         # b (B*K,256,8,8)
        hb,wb=b.shape[-2:]
        b0=b.reshape(B,Kk,s.d,hb,wb)[:,0]; sk0=[e.reshape(B,Kk,*e.shape[1:])[:,0] for e in sk]
        seg=s.unet.decode(b0,sk0)                          # (B,1,N,N) сегментация пик-кадра
        # ROI-пул: латент по РЕГИОНУ сегментации (soft-mask attention), floor 0.1
        w=torch.sigmoid(seg); w=F.adaptive_avg_pool2d(w,(hb,wb))+0.1               # (B,1,8,8)
        bk=b.reshape(B,Kk,s.d,hb,wb)
        pooled=(bk*w.unsqueeze(1)).sum((-2,-1))/w.sum((-2,-1),keepdim=True).squeeze(-1)  # (B,K,256)
        _,hn=s.gru(pooled); depth=s.head(hn[-1]).squeeze(-1)
        return seg,depth

def soft_dice(l,t,eps=1.):
    p=torch.sigmoid(l).flatten(1); t=t.flatten(1); return 1-((2*(p*t).sum(1)+eps)/(p.sum(1)+t.sum(1)+eps)).mean()
bce=nn.BCEWithLogitsLoss(); huber=nn.SmoothL1Loss()
def batches(idx,bs=16,shuf=True):
    ii=np.where(idx)[0]
    if shuf: rng.shuffle(ii)
    for j in range(0,len(ii),bs): yield ii[j:j+bs]

def run(epochs=25):
    torch.manual_seed(SEED); net=MultiTaskUNet().to(dev); opt=torch.optim.Adam(net.parameters(),1e-3)
    Xt=torch.from_numpy(Xl); Mt=torch.from_numpy(M); Dt=torch.from_numpy((D-dmean)/dstd)
    for ep in range(epochs):
        net.train()
        for bi in batches(tr):
            x=Xt[bi].to(dev); m=Mt[bi].unsqueeze(1).to(dev); d=Dt[bi].to(dev)
            seg,dep=net(x); loss=bce(seg,m)+soft_dice(seg,m)+huber(dep,d)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep%5==4: print(f"  ep{ep+1} loss={loss.item():.3f}",flush=True)
    net.eval(); ious=[];dices=[];pr=[];gt=[]
    with torch.no_grad():
        for bi in batches(te,32,False):
            x=Xt[bi].to(dev); seg,dep=net(x)
            pm=(torch.sigmoid(seg).cpu().numpy()[:,0]>0.5); gm=M[bi]>0.5
            inter=(pm&gm).sum((1,2)); uni=(pm|gm).sum((1,2))
            ious+=(inter/np.maximum(uni,1)).tolist(); dices+=(2*inter/np.maximum(pm.sum((1,2))+gm.sum((1,2)),1)).tolist()
            pr+=(dep.cpu().numpy()*dstd+dmean).tolist(); gt+=D[bi].tolist()
    pr=np.array(pr);gt=np.array(gt)
    print(f"\n[U-Net+ROI, K=16] segIoU={np.mean(ious):.3f} segDice={np.mean(dices):.3f} | "
          f"depthMAE={np.abs(pr-gt).mean():.2f}мм RMSE={np.sqrt(((pr-gt)**2).mean()):.2f}",flush=True)

t0=time.time(); run(); print(f"время {time.time()-t0:.0f}s")
