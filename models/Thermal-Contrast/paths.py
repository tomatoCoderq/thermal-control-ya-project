"""Repository locations, and sys.path setup so these scripts can import `datasets` and `common`.

This directory is not an importable package (`Thermal-Contrast` is not a valid
identifier), so every entry point here runs as a script with its own directory on
`sys.path`. Importing this module first is what makes `from datasets import ...`
and `from common.split import ...` resolve.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS_ROOT = HERE.parent
REPO_ROOT = MODELS_ROOT.parent

DATASETS_ROOT = REPO_ROOT / "datasets" / "datasets_list"
CACHE_DIR = HERE / "cache"
RUNS_DIR = HERE / "runs"

for _path in (HERE, MODELS_ROOT, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
