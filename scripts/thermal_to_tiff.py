#!/usr/bin/env python3
"""Export thermal .mat frames as 32-bit float TIFFs preserving actual °C.

Each pixel keeps its real temperature value (float32), so the TIFF is
analysis-grade rather than display-ready: in Photoshop it opens as a
32-bit image and looks very dark until you adjust Exposure/Levels
(Image > Adjustments > Exposure, or set the 32-bit preview white point).

Examples:
    # one frame
    python thermal_to_tiff.py sample_11.mat --frame 1000 -o frame1000.tif

    # several frames -> frame_0000.tif, frame_1000.tif, ...
    python thermal_to_tiff.py sample_11.mat --frames 0,1000,1999 --outdir tiffs

    # every 100th frame as a single multi-page TIFF (note: Photoshop
    # only shows the first page of a multi-page float TIFF)
    python thermal_to_tiff.py sample_11.mat --stack --stride 100 -o stack.tif
"""
import argparse
import os
import sys

import numpy as np
import scipy.io as sio
import tifffile


def load_thermal(path):
    m = sio.loadmat(path)
    if "data" not in m:
        raise KeyError(f"'data' variable not found in {path}; got {list(m)}")
    data = np.asarray(m["data"], dtype=np.float32)
    fps = float(np.ravel(m["FPS"])[0]) if "FPS" in m else 0.0
    return data, fps


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mat", help="input .mat file")
    p.add_argument("--frame", type=int,
                   help="single frame index to export")
    p.add_argument("--frames", type=str,
                   help="comma-separated frame indices, e.g. 0,1000,1999")
    p.add_argument("--stack", action="store_true",
                   help="write all (or --stride) frames as one multi-page TIFF")
    p.add_argument("--stride", type=int, default=1,
                   help="with --stack: keep every Nth frame (default 1)")
    p.add_argument("-o", "--out", help="output file (single frame or --stack)")
    p.add_argument("--outdir", default=".",
                   help="directory for multi-frame export (default: .)")
    args = p.parse_args(argv)

    data, fps = load_thermal(args.mat)
    if data.ndim != 3:
        sys.exit(f"expected 3D 'data' (H,W,N); got shape {data.shape}")
    n = data.shape[2]
    print(f"input : {args.mat}  shape={data.shape}  fps(file)={fps}")
    print(f"range : {data.min():.2f}..{data.max():.2f} °C (whole clip)")

    # tifffile keeps float32 sample format; tag it so viewers know it's linear.
    meta = {"unit": "degC"}

    if args.stack:
        sub = data[:, :, ::args.stride]                 # H,W,K
        vol = np.moveaxis(sub, 2, 0)                     # K,H,W (pages)
        out = args.out or (args.mat.rsplit(".", 1)[0] + "_stack.tif")
        tifffile.imwrite(out, vol, dtype=np.float32,
                         photometric="minisblack", metadata=meta)
        print(f"wrote : {out}  ({vol.shape[0]} pages, 32-bit float)")
        return

    if args.frame is not None:
        idx = [args.frame]
    elif args.frames:
        idx = [int(x) for x in args.frames.split(",") if x.strip() != ""]
    else:
        idx = [n // 2]                                   # default: middle frame
        print(f"(no frame specified; exporting middle frame {idx[0]})")

    for i in idx:
        if not 0 <= i < n:
            sys.exit(f"frame {i} out of range 0..{n - 1}")

    if len(idx) == 1 and args.out:
        frame = np.ascontiguousarray(data[:, :, idx[0]])
        tifffile.imwrite(args.out, frame, dtype=np.float32,
                         photometric="minisblack", metadata=meta)
        print(f"wrote : {args.out}  frame {idx[0]}  "
              f"[{frame.min():.2f}..{frame.max():.2f} °C]  32-bit float")
        return

    os.makedirs(args.outdir, exist_ok=True)
    for i in idx:
        frame = np.ascontiguousarray(data[:, :, i])
        out = os.path.join(args.outdir, f"frame_{i:04d}.tif")
        tifffile.imwrite(out, frame, dtype=np.float32,
                         photometric="minisblack", metadata=meta)
        print(f"wrote : {out}  [{frame.min():.2f}..{frame.max():.2f} °C]")


if __name__ == "__main__":
    main()
