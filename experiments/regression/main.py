"""Регрессия глубины дефектов (вариант B) — точка входа и функции для ноутбука.

Отличие этой (main-)версии от ветки regression_experiments: сырые данные и маски
берутся НЕ через ad-hoc loadmat + фиксированные пути, а через общий загрузчик
`datasets.TermoDataset` над `datasets/datasets_list/` (манифесты поддатасетов).
Остальной пайплайн прежний: из кадров считаем TSR-полином (6 каналов), по маске
режем дефектные кропы 48×48 с таргетом-глубиной (мм), обучаем 1-выходной регрессор.

Соответствие доменов: подпапка `dataset_kaggle` → domain='kaggle',
`dataset_tpu` → 'tpu'. Оба домена TermoDataset приводит к 256×256.

  * kaggle-таргет: серый уровень маски → класс → глубина (см ×10 = мм);
  * tpu-таргет:   png-маска кодирует глубину линейно, depth_mm = gray*depth_max/255
                  (см. scripts/make_tpu_masks.py) — «png как в kaggle».

Функции ниже импортируются ноутбуком notebooks/regression_experiments.ipynb.
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader

from config import CFG, ROOT
from datasets import RegressionDataset, Modes, feature_dir_for
from losses import Losses
from model import build_model
from metrics import RegressionMetrics
from optimizers import build_optimizer
from engine import fit

# ── Настройки запуска ────────────────────────────────────────────────────────
MODE = Modes.p5_d1          # набор каналов: p5 | p5_d1 | p5_d1_d2
STD_SIZE = (256, 256)       # общий размер кадра/маски (= TermoDataset.standard_size)


# ── TermoDataset: изолированная загрузка (обход коллизии имён config/datasets) ─
def _load_termo_cls():
    """Импортировать TermoDataset из пакета `datasets/`, не ломая одноимённые
    модули `config`/`datasets` этого пакета (оба используют bare-import)."""
    ds_dir = ROOT / "datasets"
    saved = {k: sys.modules.get(k) for k in ("config", "datasets")}
    try:
        spec_c = importlib.util.spec_from_file_location("config", ds_dir / "config.py")
        cfgmod = importlib.util.module_from_spec(spec_c)
        sys.modules["config"] = cfgmod
        spec_c.loader.exec_module(cfgmod)

        spec_d = importlib.util.spec_from_file_location("_termo_datasets", ds_dir / "datasets.py")
        dsmod = importlib.util.module_from_spec(spec_d)
        spec_d.loader.exec_module(dsmod)      # внутри: from config import DatasetConfig
        return dsmod.TermoDataset
    finally:
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v
            else:
                sys.modules.pop(k, None)


def build_termo(include=None):
    """TermoDataset над datasets_root с приведением к STD_SIZE."""
    TermoDataset = _load_termo_cls()
    return TermoDataset(root_dir=str(CFG.paths.datasets_root),
                        include=include, standard_size=STD_SIZE)


def _domain_of(mat_path: str) -> str:
    """Домен по подпапке поддатасета: .../<dataset_dir>/data/<file>.mat."""
    sub = Path(mat_path).parent.parent.name          # dataset_kaggle | dataset_tpu
    return sub.replace("dataset_", "")


def _load_mask_resized(mask_path: str) -> np.ndarray:
    """Маска, приведённая к STD_SIZE тем же nearest-resize, что и в TermoDataset."""
    m = np.array(Image.open(mask_path)).astype(np.float32)
    t = torch.from_numpy(m)[None, None]
    t = F.interpolate(t, size=STD_SIZE, mode="nearest")
    return t[0, 0].numpy()


# ── 1. TSR-признаки (полином 5°) из кадров TermoDataset ──────────────────────
def tsr_from_frames(frames: np.ndarray, deg: int = CFG.tsr.poly_degree) -> np.ndarray:
    """TSR-коэффициенты (deg+1, H, W) из стека кадров (n, H, W).

    Глобальная нормировка TermoDataset — аффинная по всему тензору, поэтому пик
    (argmax средней яркости кадра) и наклоны лог-полинома сохраняются; сдвигается
    лишь свободный член. Для регрессии это несущественно (признаки затем ещё раз
    нормируются поканально по домену).
    """
    X = np.asarray(frames, np.float32)               # (n, H, W)
    n, H, W = X.shape
    peak = int(np.argmax(X.reshape(n, -1).mean(1)))
    base = X[:max(1, peak // 4)].mean(0)
    dT = np.clip(X[peak:] - base[None], 1e-3, None)
    tc = (np.arange(dT.shape[0]) + 1).astype(np.float32)
    logt = np.log(tc)
    coef = np.polyfit(logt, np.log(dT).reshape(len(tc), -1), deg)
    return coef.reshape(deg + 1, H, W).astype(np.float32)


def precompute_features(include=None):
    """Посчитать и закэшировать TSR-признаки по всем сэмплам TermoDataset.

    Кэш раскладывается по доменам: features_p5/<vid>.npy (kaggle),
    features_tpu/<vid>.npy (tpu). Повторные запуски пропускают готовое.
    """
    ds = build_termo(include=include)
    for d in ("kaggle", "tpu"):
        os.makedirs(feature_dir_for(d), exist_ok=True)
    for i in range(len(ds)):
        mat_path = ds.items[i][0]
        vid = os.path.splitext(os.path.basename(mat_path))[0]
        domain = _domain_of(mat_path)
        out = feature_dir_for(domain) / f"{vid}.npy"
        if out.exists():
            continue
        data, _ = ds[i]                              # data: (C, 256, 256)
        np.save(out, tsr_from_frames(data.numpy()))
        print(f"[{i + 1}/{len(ds)}] TSR {domain}:{vid}")


# ── 2. Индекс дефектных кропов (таргет — глубина, мм) ────────────────────────
def _crops_from_regions(mask_bin, s, H, W):
    """Центроиды связных областей → координаты кропов (r0, c0)."""
    lbl, n = ndimage.label(mask_bin)
    out = []
    for k in range(1, n + 1):
        rr, cc = ndimage.center_of_mass(lbl == k)
        r0 = int(np.clip(rr - s // 2, 0, H - s))
        c0 = int(np.clip(cc - s // 2, 0, W - s))
        out.append((r0, c0))
    return out


def build_index_kaggle():
    """Дефектные кропы kaggle. Метка = глубина класса (мм). domain='kaggle'."""
    s = CFG.crop.crop
    idx = []
    ds = build_termo(include=["dataset_kaggle"])
    for i in range(len(ds)):
        mat_path, mask_path, _ = ds.items[i]
        vid = os.path.splitext(os.path.basename(mat_path))[0]
        if not (feature_dir_for("kaggle") / f"{vid}.npy").exists():
            continue
        gray = _load_mask_resized(mask_path)
        cls = np.zeros_like(gray, dtype=np.uint8)
        for g, c in CFG.classes.gray2cls.items():
            cls[np.isclose(gray, g)] = c
        H, W = cls.shape
        for c in range(1, CFG.classes.n_classes):
            depth_mm = CFG.classes.cls_depth_mm[c]
            for r0, c0 in _crops_from_regions(cls == c, s, H, W):
                idx.append((vid, r0, c0, float(depth_mm), "kaggle"))
    return idx


def build_index_tpu():
    """Дефектные кропы tpu из png-маски (как kaggle). domain='tpu'.

    png-маска tpu кодирует глубину линейно: depth_mm = gray*depth_max/255. Все
    блоки образца — на одной глубине; каждый блок = отдельный кроп.
    """
    s = CFG.crop.crop
    scale = CFG.tpu.depth_max_mm / 255.0
    idx = []
    ds = build_termo(include=["dataset_tpu"])
    for i in range(len(ds)):
        mat_path, mask_path, _ = ds.items[i]
        vid = os.path.splitext(os.path.basename(mat_path))[0]
        if not (feature_dir_for("tpu") / f"{vid}.npy").exists():
            continue
        gray = _load_mask_resized(mask_path)
        fg = gray > 0
        if not fg.any():
            continue
        depth_mm = float(np.median(gray[fg]) * scale)
        H, W = gray.shape
        for r0, c0 in _crops_from_regions(fg, s, H, W):
            idx.append((vid, r0, c0, depth_mm, "tpu"))
    return idx


# ── 3. Сплит по видео + нормировки ───────────────────────────────────────────
def split_by_video(index, n_test=None, seed=None, domains=None):
    """Сплит по видео (без утечки). Если domains задан — тест из этих доменов;
    тестовых видео на домен — n_test (или CFG.train.n_test_videos)."""
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
    loss_fn = make_regression_loss().to(device)
    optimizer = build_optimizer(model)          # adam | adamw | muon (CFG.optim)

    run_dir, history, result = fit(
        train_ld, model, loss_fn, optimizer, device,
        t_mean, t_std, val_loader=test_ld, tag=tag)

    dom = np.array([r[4] for r in test_idx])
    rm = RegressionMetrics.from_loader(model, test_ld, device, t_mean, t_std, domain=dom)
    print("\n=== RegressionMetrics (по доменам) ===")
    rm.summary()
    return dict(model=model, run_dir=run_dir, history=history,
                metrics=rm, norm=norm, target_mean=t_mean, target_std=t_std)


# ── 4. Эксперимент по умолчанию: train=kaggle, test=tpu ──────────────────────
def main():
    print("device:", get_device(), "| mode:", MODE.value,
          "| loss:", CFG.regression.loss_name)
    precompute_features()

    idx_kaggle = build_index_kaggle()
    idx_tpu = build_index_tpu()
    print(f"кропов: kaggle={len(idx_kaggle)} | tpu={len(idx_tpu)}")

    out = run(idx_kaggle, idx_tpu, tag="reg_kaggle2tpu")
    print("готово:", out["run_dir"])


if __name__ == "__main__":
    main()
