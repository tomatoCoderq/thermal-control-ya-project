# Auto-extracted MAE code from MAE-featyres-extraction.ipynb (defs only)

# ===== cell 3 =====
import os
import random
import struct
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===== cell 5 =====
class Config:
    dataset_root: str = "/kaggle/input/datasets/ziangwei/irt-pvc-depth"
    data_subdir: str = "data"

    labels_subdirs: tuple = ("manual_mask", "automated_mask")

    # --- предобработка кадра ---
    drop_first_col: bool = True
    informative_frames: int = 500
    baseline_subtract: bool = True
    baseline_frame_idx: int = 1

    per_file_norm: bool = True
    norm_sample_frames: int = 40

    # --- TSR денойзинг кадра ---
    use_tsr_reconstruction: bool = True
    tsr_poly_degree: int = 5
    tsr_sample_frames: int = 40
    tsr_spatial_smooth: bool = True
    tsr_spatial_smooth_sigma: float = 0.8

    # --- Частичное чтение .mat файла, чтобы не гонять CPU
    fast_mat_reader: bool = True
    mat_cache_size: int = 2
    use_file_grouped_sampler: bool = False

    roi_crop: tuple | None = None

    clip_len: int = 16
    clip_stride: int = 4
    max_clips_per_file: int = 60
    peak_window_frames: int = 60

    # --- Параметры архитектуры ---
    img_h: int = 256
    img_w: int = 304
    patch_size: int = 16
    tubelet_size: int = 2
    embed_dim: int = 256
    encoder_depth: int = 6
    decoder_depth: int = 2
    num_heads: int = 8

    # --- Маскирование чатей видеопотока ---    
    mask_ratio: float = 0.75 # Доля замаскированных блоков
    bias_masking_to_defects: bool = True
    defect_mask_bias: float = 0.06
    spacetime_mask_bias: bool = True
    temporal_bias_sigma: float = 3.0
    temporal_bias_floor: float = 0.15

    # -- Аугментация видео ---
    augment_shift_max: int = 8
    augment_block_swap_n: int = 3
    use_informative_sampling: bool = True

    # --- Штучка для валидации --
    informative_window_boost: float = 3.0

    # --- обучение ---
    batch_size: int = 8
    lr: float = 1.5e-4
    weight_decay: float = 0.05
    epochs: int = 50
    num_workers: int = 4

    # --- train/val split ---
    val_fraction: float = 0.15
    seed: int = 0
# ===== cell 6 =====
def build_manifest(cfg: Config) -> pd.DataFrame:
    root = Path(cfg.dataset_root)
    data_dir = root / cfg.data_subdir
    mat_files = sorted(data_dir.glob("*.mat"))

    rows = []
    for mat_path in mat_files:
        stem = mat_path.stem  # "R_002" / "Z_013"
        shape_code = stem.split("_")[0]  # "R" (прямоугольный) или "Z" (круглый) дефект
        mask_path, mask_type = None, None
        for sub in cfg.labels_subdirs:
            candidate = root / "labels" / sub / f"{stem}.png"
            if candidate.exists():
                mask_path, mask_type = candidate, sub
                break
        rows.append({
            "stem": stem,
            "shape": shape_code,
            "mat_path": str(mat_path),
            "mask_path": str(mask_path) if mask_path else None,
            "mask_type": mask_type,
        })

    df = pd.DataFrame(rows)
    n_missing = df["mask_path"].isna().sum()
    print(f"Найдено .mat файлов: {len(df)}, из них без маски: {n_missing}")
    print("По форме дефекта:\n", df["shape"].value_counts())
    print("По типу разметки:\n", df["mask_type"].value_counts(dropna=False))
    return df
# ===== cell 7 =====
def crop_to_patch_grid(arr2d, cfg: Config):
    if cfg.drop_first_col:
        arr2d = arr2d[:, 1:]
    if cfg.roi_crop is not None:
        y0, y1, x0, x1 = cfg.roi_crop
        arr2d = arr2d[y0:y1, x0:x1]
    h, w = arr2d.shape[0], arr2d.shape[1]
    h_crop = (h // cfg.patch_size) * cfg.patch_size
    w_crop = (w // cfg.patch_size) * cfg.patch_size
    return arr2d[:h_crop, :w_crop]
# ===== cell 18 =====
def _file_peak_frames(mat_path: str, mask_path: str | None, cfg: Config,
                       ring_pad_factor: float = 2.0, skip_frames: int = 5,
                       header_cache: dict | None = None):
    if mask_path is None:
        return []

    t_end = cfg.informative_frames
    frames = _fast_or_none(mat_path, 0, t_end, cfg, header_cache)
    if frames is None:
        arr = loadmat(mat_path)["imageArray"]
        t_end = min(t_end, arr.shape[2])
        frames = arr[:, :, :t_end].astype(np.float32)
    t_end = frames.shape[2]
    baseline = crop_to_patch_grid(frames[:, :, cfg.baseline_frame_idx].astype(np.float64), cfg)

    mask = np.array(Image.open(mask_path))
    mask = crop_to_patch_grid(mask, cfg)
    classes = sorted(v for v in np.unique(mask) if v > 0)
    if not classes:
        return []

    diff_stack = np.stack(
        [crop_to_patch_grid(frames[:, :, t].astype(np.float64), cfg) - baseline for t in range(t_end)],
        axis=0,
    )

    peaks = []
    for value in classes:
        ys, xs = np.where(mask == value)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        h, w = y1 - y0, x1 - x0
        pad_y, pad_x = max(1, int(h * ring_pad_factor)), max(1, int(w * ring_pad_factor))
        ry0, ry1 = max(0, y0 - pad_y), min(mask.shape[0], y1 + pad_y)
        rx0, rx1 = max(0, x0 - pad_x), min(mask.shape[1], x1 + pad_x)

        local_mask = mask[ry0:ry1, rx0:rx1]
        defect_px = (local_mask == value)
        bg_px = (local_mask != value)
        if not defect_px.any() or not bg_px.any():
            continue

        region = diff_stack[:, ry0:ry1, rx0:rx1]
        defect_ts = region[:, defect_px].mean(axis=1)
        bg_ts = region[:, bg_px].mean(axis=1)
        contrast = np.abs(defect_ts - bg_ts)

        start = min(skip_frames, t_end - 1)
        peak_frame = start + int(np.argmax(contrast[start:]))
        peaks.append(peak_frame)

    return peaks
def compute_temporal_weights(ds, cfg: Config, boost: float = 3.0):
    weights = []
    for mat_path, mask_path, start in ds.index:
        if mat_path not in ds._peak_cache:
            ds._peak_cache[mat_path] = _file_peak_frames(
                mat_path, mask_path, cfg, header_cache=ds._mat_header_cache
            )
        peaks = ds._peak_cache[mat_path]
        covers_peak = any(start <= p < start + cfg.clip_len for p in peaks)
        weights.append(boost if covers_peak else 1.0)
    return torch.tensor(weights, dtype=torch.double)
# ===== cell 20 =====
def shift_2d(arr2d, dy: int, dx: int, mode: str = "edge"):
    if dy == 0 and dx == 0:
        return arr2d
    h, w = arr2d.shape
    ady, adx = abs(dy), abs(dx)
    kwargs = {} if mode == "edge" else {"constant_values": 0}
    padded = np.pad(arr2d, ((ady, ady), (adx, adx)), mode=mode, **kwargs)
    y0 = ady - dy
    x0 = adx - dx
    return padded[y0:y0 + h, x0:x0 + w]
def sample_block_swaps(grid_h: int, grid_w: int, n_swaps: int):
    swaps = []
    if grid_h * grid_w < 2:
        return swaps
    for _ in range(n_swaps):
        by = random.randint(0, grid_h - 1)
        bx = random.randint(0, grid_w - 1)
        candidates = [(by - 1, bx), (by + 1, bx), (by, bx - 1), (by, bx + 1)]
        candidates = [(y, x) for y, x in candidates if 0 <= y < grid_h and 0 <= x < grid_w]
        ny, nx = candidates[random.randint(0, len(candidates) - 1)]
        swaps.append((by, bx, ny, nx))
    return swaps
def swap_neighbor_blocks(arr2d, patch_size: int, swaps):
    if not swaps:
        return arr2d
    out = arr2d.copy()
    p = patch_size
    for by, bx, ny, nx in swaps:
        y0, x0 = by * p, bx * p
        y1, x1 = ny * p, nx * p
        block_a = out[y0:y0 + p, x0:x0 + p].copy()
        block_b = out[y1:y1 + p, x1:x1 + p].copy()
        out[y0:y0 + p, x0:x0 + p] = block_b
        out[y1:y1 + p, x1:x1 + p] = block_a
    return out
def _mat_header_info(mat_path: str):
    try:
        with open(mat_path, "rb") as f:
            header = f.read(128)
            if len(header) < 128:
                return None
            endian_bytes = header[126:128]
            if endian_bytes not in (b"IM", b"MI"):
                return None
            endian = "<" if endian_bytes == b"IM" else ">"

            tag = f.read(8)
            if len(tag) < 8:
                return None
            mtype, _nbytes = struct.unpack(endian + "II", tag)
            if mtype != 14:
                return None

            def read_subelement():
                t = f.read(8)
                if len(t) < 8:
                    return None, None
                st, sn = struct.unpack(endian + "II", t)
                data = f.read(sn)
                if len(data) < sn:
                    return None, None
                pad = (8 - sn % 8) % 8
                f.read(pad)
                return st, data

            _flags_type, flags_data = read_subelement()
            if not flags_data:
                return None
            mclass = flags_data[0] & 0x0F

            _dims_type, dims_data = read_subelement()
            if not dims_data or len(dims_data) % 4 != 0 or len(dims_data) // 4 != 3:
                return None
            H, W, T = struct.unpack(endian + "iii", dims_data)

            _name_type, name_data = read_subelement()
            if name_data != b"imageArray":
                return None

            data_tag = f.read(8)
            if len(data_tag) < 8:
                return None
            data_type, data_nbytes = struct.unpack(endian + "II", data_tag)
            data_offset = f.tell()

        if mclass != 11 or data_type != 4:
            return None
        if data_nbytes != H * W * T * 2:
            return None

        return {"endian": endian, "H": H, "W": W, "T": T, "data_offset": data_offset}
    except Exception:
        return None
def _fast_read_mat_frames(mat_path: str, header_info: dict, start: int, end: int):
    H, W = header_info["H"], header_info["W"]
    itemsize = 2
    dtype = "<u2" if header_info["endian"] == "<" else ">u2"
    count = (end - start) * H * W
    raw = np.fromfile(mat_path, dtype=dtype, count=count,
                       offset=header_info["data_offset"] + start * H * W * itemsize)
    if raw.size != count:
        raise ValueError(f"_fast_read_mat_frames: короткое чтение {mat_path} [{start}:{end}]")
    return raw.reshape((end - start, W, H)).transpose(0, 2, 1)
def _fast_or_none(mat_path: str, start: int, end: int, cfg: Config, header_cache: dict | None = None):
    if not cfg.fast_mat_reader:
        return None
    if header_cache is not None and mat_path in header_cache:
        header_info = header_cache[mat_path]
    else:
        header_info = _mat_header_info(mat_path)
        if header_cache is not None:
            header_cache[mat_path] = header_info
    if header_info is None:
        return None
    clamped_end = min(end, header_info["T"])
    try:
        frames = _fast_read_mat_frames(mat_path, header_info, start, clamped_end)  # (n, H, W)
        return np.transpose(frames, (1, 2, 0)).astype(np.float32)  # (H, W, n)
    except Exception:
        if header_cache is not None:
            header_cache[mat_path] = None
        return None
class ThermoClipDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, cfg: Config, stems: list[str] | None = None,
                 augment: bool = False):
        self.cfg = cfg
        self.augment = augment and (cfg.augment_shift_max > 0 or cfg.augment_block_swap_n > 0)
        df = manifest if stems is None else manifest[manifest["stem"].isin(stems)]

        self._mask_cache = {}
        self._peak_cache = {}
        self._mat_cache = OrderedDict()
        self._mat_header_cache = {}
        self._norm_cache = {}
        self._tsr_cache = {}

        self.index = []
        for _, row in df.iterrows():
            mat_path, mask_path = row["mat_path"], row.get("mask_path")
            for s in self._build_clip_starts(mat_path, mask_path):
                self.index.append((mat_path, mask_path, s))

    def _build_clip_starts(self, mat_path: str, mask_path: str | None):
        cfg = self.cfg
        n_frames = cfg.informative_frames
        last_start = max(0, n_frames - cfg.clip_len)
        dense_starts = list(range(0, last_start + 1, cfg.clip_stride)) or [0]

        if mask_path is None:
            return self._thin_to_budget(dense_starts, cfg.max_clips_per_file)

        if mat_path not in self._peak_cache:
            try:
                self._peak_cache[mat_path] = _file_peak_frames(
                    mat_path, mask_path, cfg, header_cache=self._mat_header_cache
                )
            except Exception:
                self._peak_cache[mat_path] = []
        peaks = self._peak_cache[mat_path]
        if not peaks:
            return self._thin_to_budget(dense_starts, cfg.max_clips_per_file)

        peak_budget = max(1, cfg.max_clips_per_file // 2)
        bg_budget = max(0, cfg.max_clips_per_file - peak_budget)
        per_peak = max(1, peak_budget // len(peaks))
        half_w = cfg.peak_window_frames // 2

        peak_starts = set()
        for p in peaks:
            w0, w1 = max(0, p - half_w), min(last_start, p + half_w)
            local = list(range(w0, w1 + 1, cfg.clip_stride)) or [max(0, min(w0, last_start))]
            peak_starts.update(self._thin_to_budget(local, per_peak))

        remaining = [s for s in dense_starts if s not in peak_starts]
        bg_starts = self._thin_to_budget(remaining, bg_budget) if remaining and bg_budget > 0 else []

        return sorted(peak_starts | set(bg_starts))

    @staticmethod
    def _thin_to_budget(starts, budget: int):
        if budget <= 0 or not starts:
            return []
        if len(starts) <= budget:
            return list(starts)
        idx = np.linspace(0, len(starts) - 1, budget).astype(int)
        return sorted({starts[i] for i in idx})

    def __len__(self):
        return len(self.index)

    def _load_mat_array_fallback(self, mat_path: str):
        cache = self._mat_cache
        if mat_path in cache:
            cache.move_to_end(mat_path)
            return cache[mat_path]
        arr = loadmat(mat_path)["imageArray"]
        cache[mat_path] = arr
        cache.move_to_end(mat_path)
        if len(cache) > self.cfg.mat_cache_size:
            cache.popitem(last=False)
        return arr

    def _load_frames(self, mat_path: str, start: int, end: int):
        fast = _fast_or_none(mat_path, start, end, self.cfg, self._mat_header_cache)
        if fast is not None:
            return fast
        arr = self._load_mat_array_fallback(mat_path)
        return arr[:, :, start:end].astype(np.float32)

    def _load_clip(self, mat_path: str, start: int, dy: int = 0, dx: int = 0, swaps=None):
        cfg = self.cfg
        end = start + cfg.clip_len
        if cfg.use_tsr_reconstruction:
            clip = self._tsr_reconstruct_frames(mat_path, start, end)
            baseline_frame = np.zeros(clip.shape[:2], dtype=np.float32)
        else:
            clip = self._load_frames(mat_path, start, end)
            baseline_frame = self._load_frames(
                mat_path, cfg.baseline_frame_idx, cfg.baseline_frame_idx + 1
            )[:, :, 0]

        def _augment_frame(frame2d):
            frame2d = shift_2d(frame2d, dy, dx)
            frame2d = swap_neighbor_blocks(frame2d, cfg.patch_size, swaps)
            return frame2d

        clip = np.stack(
            [_augment_frame(crop_to_patch_grid(clip[:, :, i], cfg)) for i in range(clip.shape[2])],
            axis=-1,
        )
        baseline_frame = _augment_frame(crop_to_patch_grid(baseline_frame, cfg))
        clip = np.transpose(clip, (2, 0, 1))  # (t, H, W)
        if clip.shape[0] < cfg.clip_len:
            pad = cfg.clip_len - clip.shape[0]
            clip = np.pad(clip, ((0, pad), (0, 0), (0, 0)), mode="edge")
        return clip, baseline_frame

    def _clip_value_frame(self, mat_path: str, idx: int):
        cfg = self.cfg
        if cfg.use_tsr_reconstruction:
            return self._tsr_reconstruct_frames(mat_path, idx, idx + 1)[:, :, 0]
        frame = self._load_frames(mat_path, idx, idx + 1)[:, :, 0]
        if cfg.baseline_subtract:
            baseline = self._load_frames(
                mat_path, cfg.baseline_frame_idx, cfg.baseline_frame_idx + 1
            )[:, :, 0]
            frame = frame - baseline
        return frame

    def _file_norm_stats(self, mat_path: str, mask_path: str | None = None):
        cache_key = mat_path
        if cache_key in self._norm_cache:
            return self._norm_cache[cache_key]
        cfg = self.cfg
        n_frames = cfg.informative_frames
        n_sample = max(2, min(cfg.norm_sample_frames, n_frames))
        bg_idxs = sorted(set(np.linspace(0, n_frames - 1, n_sample).astype(int).tolist()))

        peak_idxs = []
        if mask_path is not None:
            if mat_path not in self._peak_cache:
                try:
                    self._peak_cache[mat_path] = _file_peak_frames(
                        mat_path, mask_path, cfg, header_cache=self._mat_header_cache
                    )
                except Exception:
                    self._peak_cache[mat_path] = []
            for p in self._peak_cache[mat_path]:
                peak_idxs.extend(fi for fi in range(p - 2, p + 3) if 0 <= fi < n_frames)
        peak_idxs = sorted(set(peak_idxs))
        def _read_stack(idxs):
            frames = [crop_to_patch_grid(self._clip_value_frame(mat_path, int(i)), cfg) for i in idxs]
            return np.stack(frames, axis=0)

        bg_stack = _read_stack(bg_idxs)
        lo = float(np.percentile(bg_stack, 1.0))

        if peak_idxs:
            peak_stack = _read_stack(peak_idxs)
            hi = float(np.percentile(peak_stack, 99.0))
        else:
            hi = float(np.percentile(bg_stack, 99.0))

        if hi - lo < 1e-6:
            lo, hi = float(bg_stack.min()), float(max(bg_stack.max(), hi))
        stats = (lo, hi)
        self._norm_cache[cache_key] = stats
        return stats

    def _file_tsr_coeffs(self, mat_path: str):
        if mat_path in self._tsr_cache:
            return self._tsr_cache[mat_path]
        cfg = self.cfg
        n_frames = cfg.informative_frames
        start_frame = cfg.baseline_frame_idx + 1
        end_frame = max(start_frame + 1, n_frames - 1)
        n_sample = max(cfg.tsr_poly_degree + 2, min(cfg.tsr_sample_frames, end_frame - start_frame))
        idxs = np.unique(np.round(np.geomspace(start_frame, end_frame, n_sample)).astype(int))
        idxs = np.clip(idxs, start_frame, n_frames - 1)

        baseline_frame = self._load_frames(
            mat_path, cfg.baseline_frame_idx, cfg.baseline_frame_idx + 1
        )[:, :, 0]
        H, W = baseline_frame.shape

        dT_frames = []
        t_phys = []
        for i in idxs:
            i = int(i)
            frame = self._load_frames(mat_path, i, i + 1)[:, :, 0]
            dT_frames.append(frame - baseline_frame)
            t_phys.append(i - cfg.baseline_frame_idx)
        dT_stack = np.stack(dT_frames, axis=0).reshape(len(idxs), -1).astype(np.float64)

    
        eps = max(1e-3, float(np.percentile(np.abs(dT_stack), 5.0)))
        log_dT = np.log(np.clip(np.abs(dT_stack), eps, None))
        log_t = np.log(np.asarray(t_phys, dtype=np.float64))
        coeffs = np.polynomial.polynomial.polyfit(log_t, log_dT, cfg.tsr_poly_degree)

        result = (coeffs.astype(np.float32), H, W)
        self._tsr_cache[mat_path] = result
        return result

    def _tsr_reconstruct_frames(self, mat_path: str, start: int, end: int):
        cfg = self.cfg
        coeffs, H, W = self._file_tsr_coeffs(mat_path)
        idxs = np.arange(start, end)
        t_phys = np.clip((idxs - cfg.baseline_frame_idx).astype(np.float64), 1.0, None)
        log_t = np.log(t_phys)  # (n,)
        log_dT = np.polynomial.polynomial.polyval(log_t, coeffs)
        dT = np.exp(log_dT).reshape(H, W, len(idxs))
        if cfg.tsr_spatial_smooth and cfg.tsr_spatial_smooth_sigma > 0:
            for i in range(dT.shape[2]):
                dT[:, :, i] = gaussian_filter(dT[:, :, i], sigma=cfg.tsr_spatial_smooth_sigma)
        return dT.astype(np.float32)

    def _load_mask_cropped(self, mask_path):
        cfg = self.cfg
        cache_key = (mask_path, cfg.roi_crop, cfg.drop_first_col, cfg.patch_size)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]
        mask = np.array(Image.open(mask_path))
        mask = crop_to_patch_grid(mask, cfg)
        self._mask_cache[cache_key] = mask
        return mask

    def _token_weight(self, mask_path, dy: int = 0, dx: int = 0, swaps=None):
        cfg = self.cfg
        grid_h, grid_w = cfg.img_h // cfg.patch_size, cfg.img_w // cfg.patch_size
        if mask_path is None or not cfg.bias_masking_to_defects:
            return np.zeros(grid_h * grid_w, dtype=np.float32)

        mask = self._load_mask_cropped(mask_path)
        mask = shift_2d(mask, dy, dx, mode="constant") if (dy or dx) else mask
        mask = swap_neighbor_blocks(mask, cfg.patch_size, swaps)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        pooled = F.max_pool2d(mask_t, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        return (pooled.squeeze().numpy() > 0).astype(np.float32).reshape(-1)

    def _temporal_bias_profile(self, mat_path, mask_path, start: int):
        cfg = self.cfg
        grid_t = cfg.clip_len // cfg.tubelet_size
        if not cfg.bias_masking_to_defects or not cfg.spacetime_mask_bias or mask_path is None:
            return np.ones(grid_t, dtype=np.float32)

        if mat_path not in self._peak_cache:
            try:
                self._peak_cache[mat_path] = _file_peak_frames(
                        mat_path, mask_path, cfg, header_cache=self._mat_header_cache
                    )
            except Exception:
                self._peak_cache[mat_path] = []
        peaks = self._peak_cache[mat_path]
        if not peaks:
            return np.ones(grid_t, dtype=np.float32)

        t_idx = np.arange(grid_t, dtype=np.float32)
        frame_centers = start + t_idx * cfg.tubelet_size + cfg.tubelet_size / 2.0
        peaks_arr = np.asarray(peaks, dtype=np.float32)
        min_dist = np.abs(frame_centers[:, None] - peaks_arr[None, :]).min(axis=1)
        profile = np.exp(-0.5 * (min_dist / cfg.temporal_bias_sigma) ** 2)
        return np.maximum(profile, cfg.temporal_bias_floor).astype(np.float32)

    def __getitem__(self, idx):
        cfg = self.cfg
        mat_path, mask_path, start = self.index[idx]

        dy, dx = 0, 0
        swaps = None
        if self.augment:
            if cfg.augment_shift_max > 0:
                m = cfg.augment_shift_max
                dy = random.randint(-m, m)
                dx = random.randint(-m, m)
            if cfg.augment_block_swap_n > 0:
                grid_h, grid_w = cfg.img_h // cfg.patch_size, cfg.img_w // cfg.patch_size
                swaps = sample_block_swaps(grid_h, grid_w, cfg.augment_block_swap_n)

        clip, baseline_frame = self._load_clip(mat_path, start, dy, dx, swaps)

        if cfg.baseline_subtract:
            clip = clip - baseline_frame[None, :, :]
        if cfg.per_file_norm:
            lo, hi = self._file_norm_stats(mat_path, mask_path)
            denom = max(hi - lo, 1e-6)
            clip = np.clip((clip - lo) / denom, 0.0, 1.0)
        else:
            c_min, c_max = clip.min(), clip.max()
            denom = max(c_max - c_min, 1e-6)
            clip = (clip - c_min) / denom
        clip = torch.from_numpy(clip).unsqueeze(0)


        spatial_weight = self._token_weight(mask_path, dy, dx, swaps)
        grid_t = cfg.clip_len // cfg.tubelet_size
        temporal_profile = self._temporal_bias_profile(mat_path, mask_path, start)
        token_weight = (temporal_profile[:, None] * spatial_weight[None, :]).reshape(-1)
        token_weight = torch.from_numpy(token_weight.astype(np.float32))

        return clip, token_weight
class FileGroupedSampler(Sampler):
    def __init__(self, ds: "ThermoClipDataset", seed: int = 0):
        self.ds = ds
        self.seed = seed
        self.epoch = 0
        groups = {}
        for i, (mat_path, _mask_path, _start) in enumerate(ds.index):
            groups.setdefault(mat_path, []).append(i)
        self.groups = groups

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        file_order = list(self.groups.keys())
        rng.shuffle(file_order)
        order = []
        for mat_path in file_order:
            idxs = self.groups[mat_path][:]
            rng.shuffle(idxs)
            order.extend(idxs)
        return iter(order)

    def __len__(self):
        return len(self.ds.index)
def make_splits(manifest: pd.DataFrame, cfg: Config):
    stems = manifest["stem"].tolist()
    rng = random.Random(cfg.seed)
    rng.shuffle(stems)
    n_val = max(1, int(len(stems) * cfg.val_fraction))
    val_stems = set(stems[:n_val])
    train_stems = set(stems[n_val:])
    return sorted(train_stems), sorted(val_stems)
# ===== cell 22 =====
class TubeletEmbed(nn.Module):
    def __init__(self, cfg: Config, in_chans: int = 1):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Conv3d(
            in_chans, cfg.embed_dim,
            kernel_size=(cfg.tubelet_size, cfg.patch_size, cfg.patch_size),
            stride=(cfg.tubelet_size, cfg.patch_size, cfg.patch_size),
        )
        self.grid_t = cfg.clip_len // cfg.tubelet_size
        self.grid_h = cfg.img_h // cfg.patch_size
        self.grid_w = cfg.img_w // cfg.patch_size
        self.num_tokens = self.grid_t * self.grid_h * self.grid_w

    def forward(self, x):
        x = self.proj(x)
        B, D = x.shape[:2]
        x = x.flatten(2).transpose(1, 2)
        return x
def sincos_pos_embed_3d(grid_t, grid_h, grid_w, dim):
    n = grid_t * grid_h * grid_w
    pe = torch.zeros(n, dim)
    nn.init.trunc_normal_(pe, std=0.02)
    return pe
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x
def random_masking(x, mask_ratio, token_weights=None, bias_strength=0.0):
    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))
    noise = torch.rand(B, N, device=x.device)
    if token_weights is not None and bias_strength > 0:
        noise = noise + bias_strength * token_weights.to(x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

    mask = torch.ones(B, N, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    return x_visible, mask, ids_restore, ids_keep
class VideoMAE(nn.Module):
    def __init__(self, cfg: Config, in_chans: int = 1):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = TubeletEmbed(cfg, in_chans)
        N = self.patch_embed.num_tokens
        self.pos_embed = nn.Parameter(sincos_pos_embed_3d(
            self.patch_embed.grid_t, self.patch_embed.grid_h, self.patch_embed.grid_w, cfg.embed_dim
        ).unsqueeze(0), requires_grad=True)

        self.encoder = nn.ModuleList([
            TransformerBlock(cfg.embed_dim, cfg.num_heads) for _ in range(cfg.encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(cfg.embed_dim)

        self.decoder_embed = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.decoder_pos_embed = self.pos_embed
        self.decoder = nn.ModuleList([
            TransformerBlock(cfg.embed_dim, cfg.num_heads) for _ in range(cfg.decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(cfg.embed_dim)
        patch_dim = in_chans * cfg.tubelet_size * cfg.patch_size * cfg.patch_size
        self.decoder_pred = nn.Linear(cfg.embed_dim, patch_dim)

    def patchify(self, x):
        cfg = self.cfg
        B, C, T, H, W = x.shape
        pt, p = cfg.tubelet_size, cfg.patch_size
        x = x.reshape(B, C, T // pt, pt, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
        x = x.reshape(B, (T // pt) * (H // p) * (W // p), C * pt * p * p)
        return x

    def forward_encoder(self, x, token_weights=None):
        tokens = self.patch_embed(x) + self.pos_embed
        visible, mask, ids_restore, ids_keep = random_masking(
            tokens, self.cfg.mask_ratio,
            token_weights=token_weights, bias_strength=self.cfg.defect_mask_bias,
        )
        for blk in self.encoder:
            visible = blk(visible)
        visible = self.encoder_norm(visible)
        return visible, mask, ids_restore

    def forward_decoder(self, visible, ids_restore):
        B, N_vis, D = visible.shape
        N = ids_restore.shape[1]
        x = self.decoder_embed(visible)
        mask_tokens = self.mask_token.expand(B, N - N_vis, -1)
        x_full = torch.cat([x, mask_tokens], dim=1)
        x_full = torch.gather(x_full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x_full = x_full + self.decoder_pos_embed
        for blk in self.decoder:
            x_full = blk(x_full)
        x_full = self.decoder_norm(x_full)
        pred = self.decoder_pred(x_full)
        return pred

    def forward(self, x, token_weights=None):
        target = self.patchify(x)
        visible, mask, ids_restore = self.forward_encoder(x, token_weights=token_weights)
        pred = self.forward_decoder(visible, ids_restore)

        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target_norm = (target - mean) / (var + 1e-6) ** 0.5

        loss = (pred - target_norm) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss, pred, mask

    @torch.no_grad()
    def extract_features(self, x):
        tokens = self.patch_embed(x) + self.pos_embed
        for blk in self.encoder:
            tokens = blk(tokens)
        tokens = self.encoder_norm(tokens)
        return tokens
# ===== cell 28 =====
def save_checkpoint(model, cfg: Config, path: str = "checkpoints/mae_full.pt"):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg_dict": asdict(cfg)}, path)
    print(f"Сохранено {len(model.state_dict())} тензоров -> {path}")
def load_checkpoint(path: str, map_location=None):
    ckpt = torch.load(path, map_location=map_location or device)
    cfg = Config(**ckpt["cfg_dict"])
    model = VideoMAE(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model, cfg
