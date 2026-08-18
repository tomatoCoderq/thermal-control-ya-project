#!/usr/bin/env python3
"""Convert a thermal .mat capture into a colormapped video.

The .mat file is expected to hold:
  - `data` : float array of shape (H, W, N) — N thermal frames
  - `FPS`  : scalar frame rate (may be 0/absent; override with --fps)

Each frame is normalized to a temperature range and mapped through a
matplotlib colormap, then written to an .mp4 (or .gif).

Example:
    python thermal_to_video.py sample_11.mat -o sample_11.mp4 --fps 30
"""
import argparse
import sys

import numpy as np
import scipy.io as sio
import matplotlib as mpl
import matplotlib.colors as mcolors
import imageio.v2 as imageio
import os


DATA_KEYS = ("data", "imageArray", "IMAGES", "frames")
FPS_KEYS = ("FPS", "Fs", "fps", "FrameRate")


def _scalar(v):
    """Coerce a mat value (numeric or char array) to float, or 0.0."""
    a = np.ravel(v)
    if a.size == 0:
        return 0.0
    x = a[0]
    if isinstance(x, (bytes, str, np.str_)):
        try:
            return float("".join(np.ravel(v).astype(str)))
        except ValueError:
            return 0.0
    return float(x)


def load_thermal(path):
    
    mat_path = path
    
    m = sio.loadmat(mat_path)
    key = next((k for k in DATA_KEYS if k in m), None)
    if key is None:
        real = [k for k in m if not k.startswith("__")]
        raise KeyError(f"no thermal array found in {mat_path}; variables: {real}")
    data = np.asarray(m[key])
    fkey = next((k for k in FPS_KEYS if k in m), None)
    fps = _scalar(m[fkey]) if fkey else 0.0
    return data, fps


def to_rgb_frames(data, cmap_name, vmin, vmax, add_colorbar_text=False):
    """Yield uint8 RGB frames from an (H, W, N) thermal array."""
    cmap = mpl.colormaps[cmap_name]
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    n = data.shape[2]
    for i in range(n):
        frame = data[:, :, i]
        rgba = cmap(norm(frame))          # H x W x 4 float in [0,1]
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        yield rgb


def convert_one(mat_path, out, args):
    """Convert a single .mat file to a video. Returns the output path."""
    data, file_fps = load_thermal(mat_path)
    if data.ndim != 3:
        raise ValueError(f"expected 3D array (H,W,N); got shape {data.shape}")

    if args.stride > 1:
        data = data[:, :, ::args.stride]

    fps = args.fps or (file_fps if file_fps > 0 else 30.0)

    # Determine color range across the whole clip for a stable colormap.
    if args.percentile is not None:
        vmin = np.percentile(data, args.percentile)
        vmax = np.percentile(data, 100 - args.percentile)
    else:
        vmin = args.vmin if args.vmin is not None else float(data.min())
        vmax = args.vmax if args.vmax is not None else float(data.max())

    print(f"input   : {mat_path}  shape={data.shape}")
    print(f"fps     : {fps} (file reported {file_fps})")
    print(f"range   : vmin={vmin:.2f} vmax={vmax:.2f}  cmap={args.cmap}")
    print(f"output  : {out}")

    n = data.shape[2]
    is_gif = out.lower().endswith(".gif")
    writer_kwargs = {"fps": fps}
    if not is_gif:
        writer_kwargs.update(codec="libx264", quality=8,
                             macro_block_size=None)  # allow non-16-multiple sizes
    with imageio.get_writer(out, **writer_kwargs) as w:
        for i, rgb in enumerate(to_rgb_frames(data, args.cmap, vmin, vmax)):
            w.append_data(rgb)
            if (i + 1) % 200 == 0 or i + 1 == n:
                print(f"  wrote {i + 1}/{n} frames", end="\r", flush=True)
    print(f"\ndone: {out}")
    return out


def collect_inputs(path, pattern):
    """Return a list of .mat files from a file or a directory."""
    import glob
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, pattern)))
        if not files:
            sys.exit(f"no files matching {pattern!r} in {path}")
        return files
    return [path]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mat", help="input .mat file OR a directory (batch mode)")
    p.add_argument("-o", "--out", help="output video (.mp4 or .gif); "
                                       "default: <input>.mp4. Ignored in batch mode.")
    p.add_argument("--glob", default="*.mat",
                   help="in batch mode, filename pattern to match (default: *.mat)")
    p.add_argument("--outdir", default=None,
                   help="in batch mode, directory for outputs (default: alongside "
                        "each input)")
    p.add_argument("--ext", default="mp4", choices=["mp4", "gif"],
                   help="in batch mode, output container/extension (default: mp4)")
    p.add_argument("--fps", type=float, default=None,
                   help="frames/sec; overrides FPS in the file (default 30 if unset)")
    p.add_argument("--cmap", default="inferno", help="matplotlib colormap")
    p.add_argument("--vmin", type=float, default=None,
                   help="temperature mapped to colormap min (default: data min)")
    p.add_argument("--vmax", type=float, default=None,
                   help="temperature mapped to colormap max (default: data max)")
    p.add_argument("--percentile", type=float, default=None, metavar="P",
                   help="clip color range to [P, 100-P] percentiles, e.g. 1")
    p.add_argument("--stride", type=int, default=1,
                   help="keep every Nth frame (default 1 = all frames)")
    args = p.parse_args(argv)

    inputs = collect_inputs(args.mat, args.glob)
    batch = len(inputs) > 1 or os.path.isdir(args.mat)

    if batch and args.out:
        print("note: --out is ignored in batch mode; using per-file names")
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    ok, failed = 0, []
    for i, mat_path in enumerate(inputs, 1):
        if batch:
            print(f"\n[{i}/{len(inputs)}] {mat_path}")
        if batch:
            stem = os.path.splitext(os.path.basename(mat_path))[0]
            outdir = args.outdir or os.path.dirname(mat_path) or "."
            out = os.path.join(outdir, f"{stem}.{args.ext}")
        else:
            out = args.out or (args.mat.rsplit(".", 1)[0] + ".mp4")
        try:
            convert_one(mat_path, out, args)
            ok += 1
        except Exception as e:  # keep going through the rest of the batch
            print(f"  FAILED: {mat_path}: {e}")
            failed.append(mat_path)

    if batch:
        print(f"\nbatch done: {ok} ok, {len(failed)} failed")
        for f in failed:
            print(f"  x {f}")


if __name__ == "__main__":
    main()
