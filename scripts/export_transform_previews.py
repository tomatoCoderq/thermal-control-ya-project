#!/usr/bin/env python3
"""
Выгрузка превью преобразований для презентации.

Строит для одного термографического видео:
  A. Кривые температуры в пикселе (дефект vs фон) и как их меняет
     каждое преобразование: raw T(t), ΔT, log-log (logt/logdt),
     полином 5°, его 1-я/2-я производная, FFT-спектр (амплитуда+фаза).
  B. Карты кадра под каждым преобразованием: raw(peak), Min/Max, Max-First,
     Std, PCA-1, коэффициенты TSR (6 каналов), фазограммы FFT (8 бинов).

Все преобразования повторяют реализацию из репозитория
(datasets/transforms/extract.py, thermo/features/{tsr,fourier}.py).
Оформление — в цветах шаблона Яндекс-студкемпа.

Запуск:
    .venv/bin/python scripts/export_transform_previews.py
Параметры (пути/видео) — в блоке CONFIG ниже.
"""
from __future__ import annotations
import os, glob
import numpy as np
import scipy.io as sio
from scipy import fft, signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ----------------------------- CONFIG -----------------------------
DATA_DIR = "datasets/datasets_list/dataset_kaggle/data"
MASK_DIR = "datasets/datasets_list/dataset_kaggle/masks"
VIDEO_ID = "Z_015"                 # какое видео визуализировать
OUT_DIR  = "transform_previews"    # куда складывать PNG
PATCH    = 3                       # полу-размер патча усреднения кривой (пикс.)
DEG      = 5                       # степень полинома TSR
N_FREQ   = 8                       # число фазовых бинов FFT
DPI      = 160

# --------------------- Яндекс-стиль (matplotlib) ------------------
INK, MUTE, GRID = "#21201F", "#7A7A7A", "#E2E2E2"
PINK, OLIVE, BLUE, PEACH, PURPLE = "#FF37B9", "#DAD06E", "#5A9BB9", "#FDB492", "#9586E5"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "font.size": 12,
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
})
os.makedirs(OUT_DIR, exist_ok=True)

def save(fig, name):
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    print("saved", p)

# ----------------------------- LOAD -------------------------------
def load_video(vid):
    m = sio.loadmat(os.path.join(DATA_DIR, f"{vid}.mat"))
    key = next(k for k in ("imageArray", "data", "IMAGES")
               if k in m and np.asarray(m[k]).ndim == 3)
    arr = np.asarray(m[key]).astype(np.float32)          # (H,W,T)
    fs = 1.0
    if "Fs" in m:
        try: fs = float(np.asarray(m["Fs"]).squeeze())
        except Exception: pass
    return np.transpose(arr, (2, 0, 1)), fs               # (T,H,W), Fs

def load_mask(vid, HW):
    p = os.path.join(MASK_DIR, f"{vid}.png")
    if not os.path.exists(p):
        return np.zeros(HW, np.uint8)
    return np.array(Image.open(p).convert("L"))

def pick_pixels(mask):
    """Дефектный пиксель (центроид самого глубокого класса) и фоновый."""
    vals = sorted(v for v in np.unique(mask) if v > 0)
    if vals:
        ys, xs = np.where(mask == vals[-1])
        dy, dx = int(ys.mean()), int(xs.mean())
    else:
        dy, dx = mask.shape[0] // 2, mask.shape[1] // 2
    # фон: пиксель с нулевой маской, максимально далёкий от дефектов
    H, W = mask.shape
    by, bx = 8, 8
    if (mask == 0).any():
        ys0, xs0 = np.where(mask == 0)
        d2 = (ys0 - dy) ** 2 + (xs0 - dx) ** 2
        k = int(np.argmax(d2))
        by, bx = int(ys0[k]), int(xs0[k])
    return (dy, dx), (by, bx)

def curve(video, yx):
    y, x = yx
    y0, y1 = max(0, y - PATCH), y + PATCH + 1
    x0, x1 = max(0, x - PATCH), x + PATCH + 1
    return video[:, y0:y1, x0:x1].mean(axis=(1, 2))       # (T,)

# ------------------- преобразования (как в репо) ------------------
def peak_index(video):
    return int(np.argmax(video.reshape(video.shape[0], -1).mean(1)))

def maxmin(video):   return video.max(0) - video.min(0)
def maxfirst(video): return video.max(0) - video[0]
def tstd(video):     return video.std(0)

def pca1(video, iters=8, eps=1e-6):
    T, H, W = video.shape
    x = video.reshape(T, -1).astype(np.float32)
    x = x - x.mean(0, keepdims=True)
    ref = (video.max(0) - video.min(0)).reshape(-1)
    v = ref - ref.mean(); v /= max(np.linalg.norm(v), eps)
    for _ in range(iters):
        u = x @ v; u /= max(np.linalg.norm(u), eps)
        v = x.T @ u; v /= max(np.linalg.norm(v), eps)
    score = (x.T @ (x @ v)).reshape(H, W)
    c = score.reshape(-1) - score.mean()
    if np.dot(c, ref - ref.mean()) < 0: score = -score
    return score

def tsr_coeffs(video, deg=DEG):
    T, H, W = video.shape
    peak = peak_index(video)
    base = video[:max(1, peak // 4)].mean(0)
    dT = np.clip(video[peak:] - base[None], 1e-3, None)
    logt = np.log(np.arange(1, dT.shape[0] + 1, dtype=np.float32))
    coef = np.polyfit(logt, np.log(dT).reshape(dT.shape[0], -1), deg)   # (deg+1,H*W) high->low
    return coef[::-1].reshape(deg + 1, H, W), peak, base                # -> low->high (c0..cN)

def fft_phase(video, first_bin=1, n=N_FREQ):
    sp = fft.rfft(video, axis=0)                          # (F,H,W)
    bins = np.arange(first_bin, first_bin + n)
    return np.angle(sp[bins]), bins                       # (n,H,W)

# ------------------------------ RUN -------------------------------
def main():
    video, fs = load_video(VIDEO_ID)
    T, H, W = video.shape
    mask = load_mask(VIDEO_ID, (H, W))
    dfx, bgx = pick_pixels(mask)
    print(f"{VIDEO_ID}: (T,H,W)=({T},{H},{W}) Fs={fs}  defect={dfx} bg={bgx}")

    cd = curve(video, dfx)      # температура в дефекте
    cb = curve(video, bgx)      # температура в фоне
    peak = peak_index(video)
    t = np.arange(T)

    # ---------- A. КРИВЫЕ ----------
    # 1) raw T(t)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, cd, color=PINK, lw=2, label="дефект")
    ax.plot(t, cb, color=INK, lw=1.6, label="фон")
    ax.axvline(peak, color=OLIVE, lw=2, ls="--", label="пик")
    ax.set_xlabel("кадр t"); ax.set_ylabel("сигнал")
    ax.set_title("Базовый сигнал T(t)", fontweight="bold"); ax.legend(frameon=False)
    ax.tick_params(labelleft=False)
    save(fig, "A1_raw_temperature.png")

    # 1b) сравнение трёх сырых кадров: пик / середина / конец
    #     авто-шкала на каждый кадр отдельно — чтобы дефекты были видны на всех стадиях
    mid, end = T // 2, T - 1
    idxs = [(peak, "пик"), (mid, "середина"), (end, "конец")]
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.8))
    for ax, (fi, name) in zip(axs, idxs):
        lo, hi = np.percentile(video[fi], [2, 98])        # своя шкала на кадр
        im = ax.imshow(video[fi], cmap="magma", vmin=lo, vmax=hi)
        ax.set_title(f"{name}: кадр {fi}  (t≈{fi / fs:.1f} с)", fontweight="bold")
        ax.axis("off")
        ax.scatter([dfx[1]], [dfx[0]], s=40, edgecolor=PINK, facecolor="none", lw=1.8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cbar.set_ticks([])
    fig.suptitle("Сырое видео: пик / середина / конец (авто-шкала на кадр)",
                 fontweight="bold", fontsize=14)
    save(fig, "A1b_three_frames.png")

    # 2) ΔT на остывании
    base = video[:max(1, peak // 4)].mean(0)
    bpx = base[dfx[0], dfx[1]]; bbx = base[bgx[0], bgx[1]]
    dtd = np.clip(cd[peak:] - bpx, 1e-3, None)
    dtb = np.clip(cb[peak:] - bbx, 1e-3, None)
    tt = np.arange(1, len(dtd) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tt, dtd, color=PINK, lw=2, label="дефект")
    ax.plot(tt, dtb, color=INK, lw=1.6, label="фон")
    ax.set_xlabel("t от пика (кадры)"); ax.set_ylabel("ΔT над базой")
    ax.set_title("ΔT на фазе остывания", fontweight="bold"); ax.legend(frameon=False)
    ax.tick_params(labelleft=False)
    save(fig, "A2_deltaT.png")

    # 3) log-log (logt / logdt)
    lx, ld, lb = np.log(tt), np.log(dtd), np.log(dtb)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lx, ld, color=PINK, lw=2, label="дефект")
    ax.plot(lx, lb, color=INK, lw=1.6, label="фон")
    ax.set_xlabel("log t"); ax.set_ylabel("log ΔT")
    ax.set_title("Лог-лог домен  (logt / logΔt)", fontweight="bold"); ax.legend(frameon=False)
    ax.tick_params(labelleft=False)
    save(fig, "A3_loglog.png")

    # 4) полином 5° поверх лог-лог (дефект)
    cf = np.polyfit(lx, ld, DEG)                          # high->low
    fitd = np.polyval(cf, lx)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(lx, ld, s=8, color=PEACH, label="данные (дефект)", zorder=2)
    ax.plot(lx, fitd, color=PINK, lw=2.4, label=f"полином {DEG}°", zorder=3)
    ax.set_xlabel("log t"); ax.set_ylabel("log ΔT")
    ax.set_title("Полином 5° (TSR-аппроксимация)", fontweight="bold"); ax.legend(frameon=False)
    ax.tick_params(labelleft=False)
    save(fig, "A4_polynom5_fit.png")

    # 5) 1-я и 2-я производные полинома по log t
    d1 = np.polyder(cf, 1); d2 = np.polyder(cf, 2)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lx, np.polyval(d1, lx), color=PINK, lw=2.2, label="1-я производная")
    ax.plot(lx, np.polyval(d2, lx), color=BLUE, lw=2.2, label="2-я производная")
    ax.axhline(0, color=MUTE, lw=1)
    ax.set_xlabel("log t"); ax.set_ylabel("d(log ΔT)/d(log t)")
    ax.set_title("Производные TSR (1-я, 2-я)", fontweight="bold"); ax.legend(frameon=False)
    save(fig, "A5_tsr_derivatives.png")

    # 6) FFT: амплитуда + фаза первых бинов
    sd = fft.rfft(cd); sb = fft.rfft(cb)
    fr = fft.rfftfreq(T, d=1.0 / fs)
    kmax = min(40, len(fr))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.semilogy(fr[1:kmax], np.abs(sd)[1:kmax], color=PINK, lw=2, label="дефект")
    a1.semilogy(fr[1:kmax], np.abs(sb)[1:kmax], color=INK, lw=1.6, label="фон")
    a1.axvspan(fr[1], fr[N_FREQ], color=OLIVE, alpha=0.25)
    a1.set_xlabel("частота, Гц"); a1.set_ylabel("|FFT|"); a1.set_title("Амплитуда", fontweight="bold"); a1.tick_params(labelleft=False)
    a1.legend(frameon=False)
    a2.plot(fr[1:N_FREQ + 1], np.angle(sd)[1:N_FREQ + 1], "o-", color=PINK, lw=2, label="дефект")
    a2.plot(fr[1:N_FREQ + 1], np.angle(sb)[1:N_FREQ + 1], "o-", color=INK, lw=1.6, label="фон")
    a2.set_xlabel("частота, Гц"); a2.set_ylabel("фаза, рад"); a2.set_title(f"Фаза (бины 1..{N_FREQ})", fontweight="bold")
    a2.legend(frameon=False)
    fig.suptitle("FFT сигнала пикселя", fontweight="bold")
    save(fig, "A6_fft_spectrum.png")

    # ---------- B. КАРТЫ КАДРА ----------
    def frame_map(arr, name, title, cmap="magma", diverging=False):
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        if diverging:
            v = np.nanpercentile(np.abs(arr), 99)
            im = ax.imshow(arr, cmap="RdBu_r", vmin=-v, vmax=v)
        else:
            lo, hi = np.nanpercentile(arr, [1, 99])
            im = ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title, fontweight="bold"); ax.axis("off")
        ax.scatter([dfx[1]], [dfx[0]], s=40, edgecolor=PINK, facecolor="none", lw=1.8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cbar.set_ticks([])
        save(fig, name)

    frame_map(video[peak], "B0_raw_peak.png", "Кадр на пике (raw)")
    frame_map(maxmin(video),   "B1_minmax.png",   "Min/Max  (max − min)")
    frame_map(maxfirst(video), "B2_maxfirst.png", "Max − First")
    frame_map(tstd(video),     "B3_std.png",      "Std по времени")
    frame_map(pca1(video),     "B4_pca1.png",     "PCA-1 (первая мода)", diverging=True)

    coef, _, _ = tsr_coeffs(video)
    fig, axs = plt.subplots(2, 3, figsize=(12, 7.2))
    for i, ax in enumerate(axs.ravel()):
        a = coef[i]; v = np.nanpercentile(np.abs(a - np.median(a)), 99)
        im = ax.imshow(a, cmap="RdBu_r", vmin=np.median(a) - v, vmax=np.median(a) + v)
        ax.set_title(f"TSR c{i}", fontweight="bold"); ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cbar.set_ticks([])
    fig.suptitle("Коэффициенты полинома 5° (TSR, 6 каналов)", fontweight="bold", fontsize=15)
    save(fig, "B5_tsr_coeffs.png")

    ph, bins = fft_phase(video)
    fig, axs = plt.subplots(2, 4, figsize=(15, 7))
    for i, ax in enumerate(axs.ravel()):
        im = ax.imshow(ph[i], cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax.set_title(f"FFT фаза · бин {bins[i]}", fontweight="bold"); ax.axis("off")
    fig.suptitle(f"Фазограммы FFT (первые {N_FREQ} бинов)", fontweight="bold", fontsize=15)
    fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.02, pad=0.02)
    save(fig, "B6_fft_phase_bins.png")

    print(f"\nГотово. Все PNG в: {os.path.abspath(OUT_DIR)}")

if __name__ == "__main__":
    main()
