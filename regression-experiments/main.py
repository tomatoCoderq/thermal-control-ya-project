"""Регрессия глубины дефектов (вариант B) — точка входа и переиспользуемые функции.

Запуск:
    cd regression-experiments && python main.py

Отличия от 5p (классификация):
  * таргет — глубина залегания в МИЛЛИМЕТРАХ (kaggle: см ×10; tpu: из table_mask);
  * индекс кропов несёт domain ('kaggle'|'tpu') и depth_mm (float);
  * только ДЕФЕКТНЫЕ кропы (фон — забота сегментации, гейт-головы нет);
  * нормировка признаков — по-домену; таргет — z-score по train;
  * модель — 1-выходная (RegressionModel), лосс — SmoothL1/L1/MSE;
  * метрики — RegressionMetrics (mae_mm/rmse_mm/r2 + дискретизированные acc/qwk).

Функции ниже импортируются ноутбуком notebooks/regression_experiments.ipynb.
"""
import glob
import os
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import ndimage
from PIL import Image

import torch
from torch.utils.data import DataLoader

from config import CFG
from datasets import RegressionDataset, Modes, feature_dir_for
from losses import Losses
from model import build_model
from metrics import RegressionMetrics
from optimizers import build_optimizer
from engine import fit

# ── Настройки запуска ────────────────────────────────────────────────────────
MODE = Modes.p5_d1          # набор каналов: p5 | p5_d1 | p5_d1_d2
TPU_CROP_TB = 60            # обрезка фона сверху/снизу у tpu (как при сборке фич)


# ── 1. TSR-признаки (полином 5°) ─────────────────────────────────────────────
def list_videos():
    ids = [os.path.splitext(os.path.basename(p))[0]
           for p in glob.glob(os.path.join(CFG.paths.data_dir, "*.mat"))]
    ids = sorted(ids)
    return ids if CFG.train.max_videos is None else ids[:CFG.train.max_videos]


def tsr_coeffs(path, deg=CFG.tsr.poly_degree):
    m = sio.loadmat(path)
    key = next(k for k in ("imageArray", "data", "IMAGES")
               if k in m and np.asarray(m[k]).ndim == 3)
    X = np.transpose(np.asarray(m[key]).astype(np.float32), (2, 0, 1))  # (n,H,W)
    n, H, W = X.shape
    peak = int(np.argmax(X.reshape(n, -1).mean(1)))
    base = X[:max(1, peak // 4)].mean(0)
    dT = np.clip(X[peak:] - base[None], 1e-3, None)
    tc = (np.arange(dT.shape[0]) + 1).astype(np.float32)
    logt = np.log(tc)
    coef = np.polyfit(logt, np.log(dT).reshape(len(tc), -1), deg)
    return coef.reshape(deg + 1, H, W).astype(np.float32)


def precompute_features():
    os.makedirs(CFG.paths.feature_dir, exist_ok=True)
    for i, vid in enumerate(list_videos(), 1):
        out = CFG.paths.feature_dir / f"{vid}.npy"
        if out.exists():
            continue
        np.save(out, tsr_coeffs(os.path.join(CFG.paths.data_dir, f"{vid}.mat")))
        print(f"[{i}] features {vid}")


def _kaggle_hw():
    files = sorted(glob.glob(str(CFG.paths.feature_dir / "*.npy")))
    return tuple(np.load(files[0]).shape[1:]) if files else (256, 320)


def _tpu_crop_resize(a, Hk, Wk):
    import cv2
    a = a[..., TPU_CROP_TB:a.shape[-2] - TPU_CROP_TB, :]
    if a.ndim == 2:
        return cv2.resize(a, (Wk, Hk), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return np.stack([cv2.resize(a[i], (Wk, Hk), interpolation=cv2.INTER_LINEAR)
                     for i in range(a.shape[0])]).astype(np.float32)


def _tpu_id(path):
    return os.path.splitext(os.path.basename(path))[0].replace(" ", "_")


def list_tpu_videos():
    return sorted(glob.glob(os.path.join(CFG.paths.tpu_dir, "*.mat")))


def precompute_tpu_features():
    os.makedirs(CFG.paths.tpu_feature_dir, exist_ok=True)
    Hk, Wk = _kaggle_hw()
    existing = sorted(CFG.paths.tpu_feature_dir.glob("*.npy"))
    if existing and tuple(np.load(existing[0]).shape[1:]) != (Hk, Wk):
        for f in existing:
            f.unlink()
    for p in list_tpu_videos():
        out = CFG.paths.tpu_feature_dir / f"{_tpu_id(p)}.npy"
        if out.exists():
            continue
        coef = _tpu_crop_resize(tsr_coeffs(p), Hk, Wk)
        np.save(out, coef)
        print("TPU features:", _tpu_id(p))


# ── 2. Индекс кропов (два домена, только дефекты, таргет — мм) ────────────────
def load_mask_cls(vid):
    im = np.array(Image.open(CFG.paths.mask_dir / f"{vid}.png"))
    cls = np.zeros_like(im, dtype=np.uint8)
    for g, c in CFG.classes.gray2cls.items():
        cls[im == g] = c
    return cls


def build_index_kaggle():
    """Дефектные кропы kaggle. Метка = глубина класса в мм. domain='kaggle'."""
    s = CFG.crop.crop
    idx = []
    for vid in list_videos():
        if not (CFG.paths.feature_dir / f"{vid}.npy").exists():
            continue
        cls = load_mask_cls(vid)
        H, W = cls.shape
        for c in range(1, CFG.classes.n_classes):
            depth_mm = CFG.classes.cls_depth_mm[c]
            lbl, n = ndimage.label(cls == c)
            for k in range(1, n + 1):
                rr, cc = ndimage.center_of_mass(lbl == k)
                r0 = int(np.clip(rr - s // 2, 0, H - s))
                c0 = int(np.clip(cc - s // 2, 0, W - s))
                idx.append((vid, r0, c0, float(depth_mm), "kaggle"))
    return idx


def build_index_tpu():
    """Дефектные кропы tpu по table_mask (мм, фон=NaN). domain='tpu'.

    Каждая связная область конечных значений маски = отдельный дефект; глубина =
    медиана его значений (в образце она константна).
    """
    s = CFG.crop.crop
    idx = []
    for mp in sorted(CFG.paths.tpu_mask_dir.glob("*.npy")):
        vid = mp.stem
        if not (CFG.paths.tpu_feature_dir / f"{vid}.npy").exists():
            continue
        dm = np.load(mp).astype(np.float32)          # (H,W), мм, фон=NaN
        H, W = dm.shape
        finite = np.isfinite(dm)
        lbl, n = ndimage.label(finite)
        for k in range(1, n + 1):
            region = lbl == k
            depth_mm = float(np.median(dm[region]))
            rr, cc = ndimage.center_of_mass(region)
            r0 = int(np.clip(rr - s // 2, 0, H - s))
            c0 = int(np.clip(cc - s // 2, 0, W - s))
            idx.append((vid, r0, c0, depth_mm, "tpu"))
    return idx


# ── 3. Сплит по видео + нормировки ───────────────────────────────────────────
def split_by_video(index, n_test=None, seed=None, domains=None):
    """Сплит по видео (без утечки). Если domains задан — тест берём из этих
    доменов; тестовых видео на домен — n_test (или CFG.train.n_test_videos)."""
    rng = np.random.default_rng(CFG.train.seed if seed is None else seed)
    n_test = CFG.train.n_test_videos if n_test is None else n_test
    doms = domains or sorted({r[4] for r in index})
    test_vids = set()
    for d in doms:
        vids = np.array(sorted({r[0] for r in index if r[4] == d}))
        if len(vids) == 0:
            continue
        perm = rng.permutation(vids)
        test_vids |= set(perm[:min(n_test, len(vids))])
    train = [r for r in index if r[0] not in test_vids]
    test = [r for r in index if r[0] in test_vids]
    return train, test


def compute_norm_by_domain(index):
    """Поканальная μ/σ признаков ОТДЕЛЬНО по каждому домену (по train-кропам)."""
    s = CFG.crop.crop
    out = {}
    for d in sorted({r[4] for r in index}):
        rows = [r for r in index if r[4] == d]
        cache = {v: np.load(feature_dir_for(d) / f"{v}.npy")
                 for v in sorted({r[0] for r in rows})}
        stk = np.stack([cache[v][:, r0:r0 + s, c0:c0 + s]
                        for (v, r0, c0, _, _) in rows])
        out[d] = (stk.mean(axis=(0, 2, 3)), stk.std(axis=(0, 2, 3)) + 1e-6)
    return out


def compute_target_stats(index):
    """z-score статистики таргета (мм) по train."""
    ys = np.array([r[3] for r in index], dtype=np.float64)
    return float(ys.mean()), float(ys.std() + 1e-8)


def n_channels(mode: Modes) -> int:
    b = CFG.tsr.n_channels
    return {Modes.p5: b, Modes.p5_d1: b + (b - 1),
            Modes.p5_d1_d2: b + (b - 1) + (b - 2)}[mode]


def make_regression_loss():
    return Losses.regression(CFG.regression.loss_name, CFG.regression.huber_beta)


def make_loaders(train_idx, test_idx, mode=MODE):
    """Собрать train/test DataLoader'ы + вернуть нормировки для денормализации."""
    norm = compute_norm_by_domain(train_idx)
    t_mean, t_std = compute_target_stats(train_idx)
    # если в тесте есть домен, которого не было в train — берём его же статистику
    for d in {r[4] for r in test_idx} - set(norm):
        norm[d] = compute_norm_by_domain([r for r in test_idx if r[4] == d])[d]
    train_ds = RegressionDataset(train_idx, norm, t_mean, t_std, mode, train=True)
    test_ds = RegressionDataset(test_idx, norm, t_mean, t_std, mode, train=False)
    train_ld = DataLoader(train_ds, batch_size=CFG.train.batch_size, shuffle=True)
    test_ld = DataLoader(test_ds, batch_size=CFG.train.batch_size, shuffle=False)
    return train_ld, test_ld, norm, t_mean, t_std


def get_device():
    return ("cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu")


def run(train_idx, test_idx, mode=MODE, tag="reg", device=None):
    """Полный прогон: loaders → модель → fit → RegressionMetrics по доменам."""
    device = device or get_device()
    torch.manual_seed(CFG.train.seed)
    np.random.seed(CFG.train.seed)

    train_ld, test_ld, norm, t_mean, t_std = make_loaders(train_idx, test_idx, mode)
    model = build_model(in_channels=n_channels(mode)).to(device)
    print("Current model:", model.__class__.__name__, f"(device={device})")
    loss_fn = make_regression_loss().to(device)
    optimizer = build_optimizer(model)          # adam | adamw | muon (CFG.optim)

    run_dir, history, result = fit(
        train_ld, model, loss_fn, optimizer, device,
        t_mean, t_std, val_loader=test_ld, tag=tag)

    # раздельные метрики по доменам (порядок кропов = порядок test_idx, shuffle=False)
    dom = np.array([r[4] for r in test_idx])
    rm = RegressionMetrics.from_loader(model, test_ld, device, t_mean, t_std, domain=dom)
    print("\n=== RegressionMetrics (по доменам) ===")
    rm.summary()
    return dict(model=model, run_dir=run_dir, history=history,
                metrics=rm, norm=norm, target_mean=t_mean, target_std=t_std)


# ── 4. Эксперимент 1 по умолчанию: train=kaggle, test=tpu ────────────────────
def main():
    print("device:", get_device(), "| mode:", MODE.value,
          "| loss:", CFG.regression.loss_name)
    precompute_features()
    precompute_tpu_features()

    idx_kaggle = build_index_kaggle()
    idx_tpu = build_index_tpu()
    print(f"кропов: kaggle={len(idx_kaggle)} | tpu={len(idx_tpu)}")

    # train — весь kaggle, test — весь tpu (held-out домен, перенос)
    out = run(idx_kaggle, idx_tpu, tag="reg_kaggle2tpu")
    print("готово:", out["run_dir"])


if __name__ == "__main__":
    main()
