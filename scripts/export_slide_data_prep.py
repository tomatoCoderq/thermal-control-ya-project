#!/usr/bin/env python3
"""Slide-ready SVG for data preparation (sample17). English labels, no % / n× markers."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from PIL import Image
from scipy import ndimage
from scipy.fft import rfft, rfftfreq
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "datasets"))

from transforms.channel import PercentileNorm
from transforms.extract import MaxFirst, MaxMin, PCA1, Std, TSRDeriv, _poly_deriv
from transforms.video import SelectFrames

MAT = ROOT / "datasets/datasets_list/dataset_tpu/data/sample17.mat"
MASK = ROOT / "datasets/datasets_list/dataset_tpu/masks/sample17.png"
OUT = ROOT / "transform_previews/slide_data_prep"
PARTS = OUT / "parts"

PATCH = 3
FS = 10.0
N_FFT_BINS = 8
INK, MUTE, GRID = "#21201F", "#7A7A7A", "#E8E8E8"
PINK, BLUE, OLIVE, PEACH = "#FF37B9", "#5A9BB9", "#C4B84A", "#FDB492"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "font.size": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlepad": 8,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
    "savefig.facecolor": "white",
})


def save_svg(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg")
    plt.close(fig)
    print("saved", path.relative_to(ROOT))
    return path


def finish(ax, title, xlabel, ylabel, loc="best", fs=11):
    ax.set_title(title, loc="left", fontsize=fs)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    h, _ = ax.get_legend_handles_labels()
    if h:
        ax.legend(loc=loc, fontsize=8)


def patch_curve(video, yx):
    y, x = yx
    y0, y1 = max(0, y - PATCH), y + PATCH + 1
    x0, x1 = max(0, x - PATCH), x + PATCH + 1
    return video[:, y0:y1, x0:x1].mean(axis=(1, 2))


def pick_pixels(mask):
    binary = mask > 0
    lab, n = ndimage.label(binary)
    sizes = ndimage.sum(binary, lab, index=np.arange(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    cy, cx = ndimage.center_of_mass(binary, lab, k)
    dy, dx = int(round(cy)), int(round(cx))
    ys0, xs0 = np.where(~binary)
    j = int(np.argmax((ys0 - dy) ** 2 + (xs0 - dx) ** 2))
    return (dy, dx), (int(ys0[j]), int(xs0[j]))


def map_panel(ax, arr, title, mask, *, phase=False):
    if phase:
        im = ax.imshow(arr, cmap="twilight", vmin=-np.pi, vmax=np.pi, interpolation="nearest")
        ticks, labels = [-np.pi, 0, np.pi], [r"$-\pi$", "0", r"$\pi$"]
    else:
        lo, hi = np.nanpercentile(arr, [1, 99])
        if hi <= lo:
            lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr) + 1e-6)
        im = ax.imshow(arr, cmap="magma", vmin=lo, vmax=hi, interpolation="nearest")
        ticks, labels = [lo, hi], ["0", "1"]
    if (mask > 0).any():
        ax.contour(mask > 0, levels=[0.5], colors=["#ffffff"], linewidths=0.55, alpha=0.8)
    ax.set_title(title, fontsize=10, pad=6)
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.025, ticks=ticks)
    cbar.set_ticklabels(labels)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)


def plot_T(ax, t_s, cd, cb, peak):
    ax.plot(t_s, cd, color=PINK, lw=2.0, label="Defect")
    ax.plot(t_s, cb, color=INK, lw=1.4, label="Background")
    ax.axvline(peak / FS, color=OLIVE, lw=1.3, ls="--", label="Heat peak")
    finish(ax, "Raw signal  T(t)", "Time (s)", "Temperature (°C)", loc="upper right")


def plot_dT(ax, tt_s, dtd, dtb):
    ax.plot(tt_s, dtd, color=PINK, lw=2.0, label="Defect")
    ax.plot(tt_s, dtb, color=INK, lw=1.4, label="Background")
    finish(ax, "Cooling contrast  ΔT", "Time after peak (s)", "ΔT (°C)", loc="upper right")


def plot_loglog(ax, lx, ld, lb):
    ax.plot(lx, ld, color=PINK, lw=2.0, label="Defect")
    ax.plot(lx, lb, color=INK, lw=1.4, label="Background")
    finish(ax, "Log–log domain", r"$\log t$", r"$\log\,\Delta T$", loc="lower left")


def plot_tsr(ax, lx, ld, fit, useful):
    ax.plot(lx[useful], ld[useful], color=PEACH, lw=1.3, label="Defect data")
    ax.plot(lx[useful], fit[useful], color=PINK, lw=2.1, label="5th-degree TSR")
    ax.set_xlim(lx[useful][0], lx[useful][-1])
    finish(ax, "Thermographic signal reconstruction", r"$\log t$", r"$\log\,\Delta T$", loc="lower left")


def plot_derivs(ax, lx, p1, p2, useful):
    sl = useful & (lx >= 2.0)
    ax.plot(lx[sl], p1[sl], color=PINK, lw=2.0, label="1st derivative  p'")
    ax.plot(lx[sl], p2[sl], color=BLUE, lw=2.0, label="2nd derivative  p''")
    ax.axhline(0, color=MUTE, lw=0.7)
    ax.set_xlim(lx[sl][0], lx[sl][-1])
    finish(ax, "TSR derivatives", r"$\log t$", r"$d^{m}\log\Delta T\ /\ d(\log t)^{m}$")


def plot_fft(a1, a2, freq, amp_d, amp_b, ph_d, ph_b):
    show = slice(1, min(len(freq), 25))
    feat = slice(1, N_FFT_BINS + 1)
    a1.plot(freq[show], amp_d[show], color=PINK, lw=2.0, label="Defect")
    a1.plot(freq[show], amp_b[show], color=INK, lw=1.4, label="Background")
    a1.axvspan(freq[feat][0], freq[feat][-1], color=OLIVE, alpha=0.22, zorder=0)
    finish(a1, "FFT amplitude  ·  cooling ΔT", "Frequency (Hz)", "Amplitude", loc="upper right")
    ticks = freq[feat]
    a2.plot(ticks, ph_d[feat], "o-", color=PINK, lw=1.8, ms=4.5, label="Defect")
    a2.plot(ticks, ph_b[feat], "o-", color=INK, lw=1.4, ms=4, label="Background")
    a2.set_xticks(ticks[::2])
    finish(a2, "FFT phase  ·  first bins", "Frequency (Hz)", "Phase (rad)")


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    PARTS.mkdir()

    mat = loadmat(MAT)
    video = np.transpose(mat["data"].astype(np.float32), (2, 0, 1))
    video, _ = SelectFrames(num_frames=10**6)(video, None)
    mask = np.array(Image.open(MASK).convert("L"), dtype=np.float32)
    T = video.shape[0]
    dfx, bgx = pick_pixels(mask)
    cd, cb = patch_curve(video, dfx), patch_curve(video, bgx)
    peak = int(video.reshape(T, -1).mean(1).argmax())
    t_s = np.arange(T) / FS
    print(f"sample17  T={T}  peak={peak}  defect={dfx}  bg={bgx}")

    base_d = float(cd[: max(1, peak // 4)].mean())
    base_b = float(cb[: max(1, peak // 4)].mean())
    dtd = np.maximum(cd[peak:] - base_d, 1e-3)
    dtb = np.maximum(cb[peak:] - base_b, 1e-3)
    n = len(dtd)
    tt = np.arange(1, n + 1)
    tt_s = tt / FS
    useful = dtd > 0.05
    if useful.sum() < 20:
        useful[:] = True
    lx = np.log(tt.astype(np.float64))
    ld, lb = np.log(dtd), np.log(dtb)
    coef_d = np.linalg.lstsq(np.stack([lx[useful] ** j for j in range(6)], axis=1), ld[useful], rcond=None)[0]
    fit = np.stack([lx ** j for j in range(6)], axis=1) @ coef_d
    coef_hw = coef_d.reshape(6, 1, 1)
    p1 = np.array([float(_poly_deriv(coef_hw, float(u), 1)[0, 0]) for u in lx])
    p2 = np.array([float(_poly_deriv(coef_hw, float(u), 2)[0, 0]) for u in lx])

    win = np.hanning(n)
    spec_d = rfft((dtd - dtd.mean()) * win)
    spec_b = rfft((dtb - dtb.mean()) * win)
    freq = rfftfreq(n, d=1.0 / FS)
    amp_d, amp_b = np.abs(spec_d), np.abs(spec_b)
    ph_d, ph_b = np.unwrap(np.angle(spec_d)), np.unwrap(np.angle(spec_b))

    extractors = [
        ("maxmin", "Max − min", MaxMin()),
        ("maxfirst", "Max − T(0)", MaxFirst()),
        ("std", "Std  σ(t)", Std()),
        ("pca1", "PCA-1", PCA1()),
        ("p1a", "TSR  p'", TSRDeriv(order=1, at=(0.10,))),
        ("p1b", "TSR  p'", TSRDeriv(order=1, at=(0.25,))),
        ("p2a", "TSR  p''", TSRDeriv(order=2, at=(0.05,))),
        ("p2b", "TSR  p''", TSRDeriv(order=2, at=(0.15,))),
    ]
    norm = PercentileNorm()
    maps = []
    for key, title, ext in extractors:
        ch, _ = ext(video, mask)
        ch, _ = norm(ch, mask)
        maps.append((key, title, ch[0]))

    base = video[: max(1, peak // 4)].mean(0)
    cube = np.maximum(video[peak:] - base, 1e-3)
    cube = cube - cube.mean(axis=0, keepdims=True)
    w = np.hanning(cube.shape[0]).astype(np.float32)[:, None, None]
    phase = np.angle(rfft(cube * w, axis=0)[1:5])

    # ---- individual SVGs ----
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    plot_T(ax, t_s, cd, cb, peak)
    save_svg(fig, PARTS / "A1_T.svg")

    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    plot_dT(ax, tt_s, dtd, dtb)
    save_svg(fig, PARTS / "A2_dT.svg")

    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    plot_loglog(ax, lx, ld, lb)
    save_svg(fig, PARTS / "A3_loglog.svg")

    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    plot_tsr(ax, lx, ld, fit, useful)
    save_svg(fig, PARTS / "A4_tsr.svg")

    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    plot_derivs(ax, lx, p1, p2, useful)
    save_svg(fig, PARTS / "A5_derivs.svg")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.1), layout="constrained")
    plot_fft(a1, a2, freq, amp_d, amp_b, ph_d, ph_b)
    save_svg(fig, PARTS / "A6_fft.svg")

    for key, title, arr in maps:
        fig, ax = plt.subplots(figsize=(3.8, 3.2))
        map_panel(ax, arr, title, mask)
        save_svg(fig, PARTS / f"B_{key}.svg")

    fig, axs = plt.subplots(1, 4, figsize=(12.4, 3.1))
    for i, ax in enumerate(axs):
        map_panel(ax, phase[i], f"FFT phase  ·  bin {i + 1}", mask, phase=True)
    fig.suptitle("Pulsed phase thermography", fontweight="bold", fontsize=12)
    save_svg(fig, PARTS / "B_fft_phase.svg")

    # ---- one drop-in 16:9 slide ----
    fig = plt.figure(figsize=(19.2, 10.8), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.06, wspace=0.04, hspace=0.06)
    gs = GridSpec(4, 6, figure=fig, height_ratios=[1.05, 1.05, 1.05, 1.15])
    fig.suptitle("Data preparation", fontsize=20, fontweight="bold", color=INK, ha="left", x=0.01)

    plot_T(fig.add_subplot(gs[0, 0:2]), t_s, cd, cb, peak)
    plot_dT(fig.add_subplot(gs[0, 2:4]), tt_s, dtd, dtb)
    plot_loglog(fig.add_subplot(gs[1, 0:2]), lx, ld, lb)
    plot_tsr(fig.add_subplot(gs[1, 2:4]), lx, ld, fit, useful)
    plot_derivs(fig.add_subplot(gs[2, 0:2]), lx, p1, p2, useful)
    fft_gs = gs[2, 2:4].subgridspec(1, 2, wspace=0.18)
    plot_fft(fig.add_subplot(fft_gs[0, 0]), fig.add_subplot(fft_gs[0, 1]),
             freq, amp_d, amp_b, ph_d, ph_b)

    map_pos = [(0, 4), (0, 5), (1, 4), (1, 5), (2, 4), (2, 5)]
    # 8 maps don't fit 6 cells; put 4 contrast+pca on the right of rows 0-1
    # and TSR + FFT phase on row 3
    right = [(0, 4), (0, 5), (1, 4), (1, 5)]
    for (r, c), (_, title, arr) in zip(right, maps[:4]):
        map_panel(fig.add_subplot(gs[r, c]), arr, title, mask)

    for c, (_, title, arr) in enumerate(maps[4:]):
        map_panel(fig.add_subplot(gs[3, c]), arr, title, mask)
    for i in range(2):
        map_panel(fig.add_subplot(gs[3, 4 + i]), phase[i], f"FFT phase  ·  bin {i + 1}",
                  mask, phase=True)

    save_svg(fig, OUT / "slide.svg")
    print(f"\nslide → {OUT / 'slide.svg'}")


if __name__ == "__main__":
    main()
