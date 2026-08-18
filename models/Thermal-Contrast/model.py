"""Static U-Net on collapsed thermal contrast maps (reuses classic U-Net)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_UNET_MAIN = Path(__file__).resolve().parents[1] / "U-Net" / "main.py"
_spec = importlib.util.spec_from_file_location("_thermal_contrast_unet_main", _UNET_MAIN)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

UNetModel = _mod.UNetModel
SoftDiceLoss = _mod.SoftDiceLoss

__all__ = ["UNetModel", "SoftDiceLoss"]
