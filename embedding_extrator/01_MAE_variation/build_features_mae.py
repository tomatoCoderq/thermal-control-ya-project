"""Адаптер: MAE-эмбеддинги → features_mae/ (C,256,320) под кадр регрессии.

Пайплайн (маршрут B): для каждого kaggle-видео усредняем extract_features по
клипам и по временным токенам → карта (256,11,10) на ROI; PCA 256→C по всем
видео; апсемпл (C,11,10)→(C,176,160) и укладка в (C,256,320) на ROI
[35:211, 81:241] (x+1 из-за drop_first_col). mode=p5 в регрессии, in_ch=C.
"""
import sys, os
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

ROOT = Path("/Users/tomatocoder/Documents/thermal-control-ya-project")
MAEDIR = ROOT/"embedding_extrator/01_MAE_variation"
sys.path.insert(0, str(MAEDIR))
import mae_lib as MAE

C_PCA = 32
CLIPS_PER_VIDEO = 15         # подвыборка клипов на видео (скорость)
ROI_Y0, ROI_Y1 = 35, 211     # строки в исходном 256x320
ROI_X0, ROI_X1 = 81, 241     # +1 к roi x из-за drop_first_col
OUT = ROOT/"features_cache"/"mae_kaggle"
dev = "mps" if torch.backends.mps.is_available() else "cpu"

cfg = MAE.Config()
cfg.dataset_root = str(ROOT/"datasets/dataset_kaggle"); cfg.data_subdir = "data"
cfg.labels_subdirs = ("automated_mask",)
cfg.roi_crop = (35,211,80,240); cfg.img_h=176; cfg.img_w=160
GT, GH, GW = cfg.clip_len//cfg.tubelet_size, cfg.img_h//cfg.patch_size, cfg.img_w//cfg.patch_size

man = MAE.build_manifest(cfg)
model = MAE.VideoMAE(cfg)
ck = torch.load(MAEDIR/"checkpoints/mae_full.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"], strict=False); model.eval().to(dev)
ds = MAE.ThermoClipDataset(man, cfg, augment=False)

# clips по видео
by_vid = defaultdict(list)
for i,(mat,mask,start) in enumerate(ds.index):
    by_vid[Path(mat).stem].append(i)

@torch.no_grad()
def video_tokenmap(idxs):
    if len(idxs) > CLIPS_PER_VIDEO:
        idxs = list(np.array(idxs)[np.linspace(0,len(idxs)-1,CLIPS_PER_VIDEO).astype(int)])
    acc = np.zeros((GH, GW, cfg.embed_dim), np.float32); n=0
    for i in idxs:
        clip,_ = ds[i]
        t = model.extract_features(clip.unsqueeze(0).to(dev)).squeeze(0).cpu().numpy()  # (880,256)
        t = t.reshape(GT, GH, GW, cfg.embed_dim).mean(0)   # усреднить по времени → (11,10,256)
        acc += t; n+=1
    return acc/max(n,1)

print(f"строю token-карты по {len(by_vid)} видео (dev={dev})…")
maps = {v: video_tokenmap(idxs) for v,idxs in sorted(by_vid.items())}

# PCA 256→C по всем пространственным точкам всех видео
stack = np.concatenate([m.reshape(-1, cfg.embed_dim) for m in maps.values()], 0)
pca = PCA(n_components=C_PCA, random_state=0).fit(stack)
print(f"PCA {cfg.embed_dim}→{C_PCA}; объяснённая дисперсия={pca.explained_variance_ratio_.sum():.2f}")

OUT.mkdir(exist_ok=True)
for v, m in maps.items():
    red = pca.transform(m.reshape(-1, cfg.embed_dim)).reshape(GH, GW, C_PCA)  # (11,10,C)
    ten = torch.from_numpy(red.transpose(2,0,1)[None]).float()                # (1,C,11,10)
    up = F.interpolate(ten, size=(ROI_Y1-ROI_Y0, ROI_X1-ROI_X0), mode="bilinear",
                       align_corners=False).squeeze(0).numpy()                # (C,176,160)
    full = np.zeros((C_PCA, 256, 320), np.float32)
    full[:, ROI_Y0:ROI_Y1, ROI_X0:ROI_X1] = up
    np.save(OUT/f"{v}.npy", full)
print(f"features_mae: {len(list(OUT.glob('*.npy')))} файлов, форма {full.shape}")
np.save(ROOT/"features_cache"/"_mae_pca_components.npy", pca.components_)
