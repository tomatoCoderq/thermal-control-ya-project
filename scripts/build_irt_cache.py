#!/usr/bin/env python3
"""Build float16 .npy frame cache for all sources in a dataset yaml.

Reads ``cache_dir``, ``sources[].root/pattern/time_axis`` from the yaml and
merges into one ``index.json`` (kaggle R_/Z_ + sample* with time_axis=2).

Example (from repo root)::

    python scripts/build_irt_cache.py --yaml segmentation/U-Net/dataset_tsr.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from irt_data.cache import build_cache
from irt_data.config import DatasetConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("build_irt_cache")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--yaml",
        type=Path,
        default=ROOT / "segmentation" / "U-Net" / "dataset_tsr.yaml",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Override cache_dir from yaml (default: yaml cache_dir)",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    yaml_path = args.yaml if args.yaml.is_absolute() else (ROOT / args.yaml)
    cfg = DatasetConfig.from_yaml(yaml_path)
    out = Path(args.out) if args.out else Path(cfg.cache_dir)
    if not out.is_absolute():
        out = ROOT / out

    logger.info("yaml=%s", yaml_path)
    logger.info("out=%s", out)

    for src in cfg.sources:
        root = Path(src.root)
        if not root.is_absolute():
            root = ROOT / root
        pattern = src.pattern or "*.mat"
        time_axis = src.time_axis
        logger.info(
            "caching root=%s pattern=%s time_axis=%s",
            root,
            pattern,
            time_axis,
        )
        build_cache(
            [root],
            out_dir=out,
            pattern=pattern,
            time_axis=time_axis,
            overwrite=args.overwrite,
        )

    index = out / "index.json"
    logger.info("done → %s", index)
    if index.exists():
        import json

        data = json.loads(index.read_text(encoding="utf-8"))
        logger.info("videos in index: %d", len(data.get("videos", {})))


if __name__ == "__main__":
    main()
