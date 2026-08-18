"""MPS / high-RAM training helpers (Apple Silicon + ~36GB unified memory)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


def setup_mps_env() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # slightly less allocator fragmentation on long runs
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    # albumentations phone-home → SSL noise on some macOS Python installs
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


def optimize_model_mps(
    model: nn.Module,
    device: torch.device,
    *,
    channels_last: bool = True,
    compile_model: bool = True,
) -> nn.Module:
    """channels_last (CUDA) + optional torch.compile."""
    use_cl = channels_last and device.type == "cuda"
    if use_cl:
        model = model.to(device=device, memory_format=torch.channels_last)
    else:
        model = model.to(device)
    if not compile_model or device.type not in ("mps", "cuda"):
        return model
    try:
        model = torch.compile(model, mode="default", fullgraph=False)
        print(f"torch.compile enabled on {device.type}")
    except Exception as exc:  # noqa: BLE001
        print(f"torch.compile skipped: {exc}")
    return model


def suggest_batch_size(
    device: torch.device,
    *,
    base: int = 8,
    kind: str = "attn",
) -> int:
    """Larger batches on MPS with 36GB unified memory → fewer Python dispatches."""
    # Sized for INPUT_SIZE=256 (~4× pixels vs 128); drop further if OOM.
    if device.type == "mps":
        return 8 if kind == "attn" else 4
    if device.type == "cuda":
        return 16 if kind == "attn" else 8
    return max(base, 4)


def suggest_num_workers(device: torch.device) -> int:
    # MPS + DataLoader workers often hang / stall on first batch (spawn + Metal).
    if device.type == "mps":
        return 0
    if device.type == "cuda":
        return 8
    return 2


def loader_kwargs(device: torch.device, num_workers: int | None = None) -> dict[str, Any]:
    nw = suggest_num_workers(device) if num_workers is None else num_workers
    kw: dict[str, Any] = {
        "num_workers": nw,
        "pin_memory": False,  # no benefit on MPS / unified memory
        "persistent_workers": nw > 0,
    }
    if nw > 0:
        kw["prefetch_factor"] = 3
    return kw


def strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(k.startswith("_orig_mod.") for k in state_dict):
        return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    return state_dict


def load_model_weights(
    model: nn.Module,
    path: str | Path,
    *,
    strict: bool = True,
) -> Path:
    """Load a ``state_dict`` checkpoint (strips ``_orig_mod.`` from torch.compile)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(sd, dict):
        raise TypeError(f"expected state_dict dict in {path}, got {type(sd)}")
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    sd = strip_compile_prefix(sd)
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing or unexpected:
        print(f"load {path.name}: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"loaded weights <- {path}")
    return path


def mps_empty_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
