"""Training state saved as a single pickle, so a run can be resumed exactly.

A bare `state_dict` is not enough to continue training: the optimizer's momentum,
the LR schedule position and the epoch counter all have to come back too, or a
resumed run silently restarts with a fresh optimizer at the wrong learning rate.
Everything needed lives in one `.pkl`.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from channels import CHANNEL_NAMES, ChannelParams

FORMAT_VERSION = 1


@dataclass
class Checkpoint:
    epoch: int
    best_iou: float
    best_epoch: int
    params: ChannelParams
    channel_names: tuple[str, ...] = CHANNEL_NAMES
    model_state: dict[str, Any] = field(default_factory=dict, repr=False)
    optimizer_state: dict[str, Any] = field(default_factory=dict, repr=False)
    scheduler_state: dict[str, Any] = field(default_factory=dict, repr=False)
    history: list[dict[str, Any]] = field(default_factory=list, repr=False)
    version: int = FORMAT_VERSION


def _strip_compile_prefix(state: dict[str, Any]) -> dict[str, Any]:
    """`torch.compile` wraps the module, prefixing every key with `_orig_mod.`."""
    return {key.removeprefix("_orig_mod."): value for key, value in state.items()}


def save_checkpoint(
    path: Path | str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_iou: float,
    best_epoch: int,
    params: ChannelParams,
    history: list[dict[str, Any]],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = Checkpoint(
        epoch=epoch,
        best_iou=best_iou,
        best_epoch=best_epoch,
        params=params,
        model_state={k: v.detach().cpu() for k, v in _strip_compile_prefix(model.state_dict()).items()},
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
        history=list(history),
    )
    with path.open("wb") as handle:
        pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_checkpoint(path: Path | str) -> Checkpoint:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    with path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if not isinstance(checkpoint, Checkpoint):
        raise TypeError(f"{path} holds {type(checkpoint).__name__}, expected Checkpoint")
    if checkpoint.version != FORMAT_VERSION:
        raise ValueError(f"{path} is format v{checkpoint.version}, this code reads v{FORMAT_VERSION}")
    if tuple(checkpoint.channel_names) != CHANNEL_NAMES:
        raise ValueError(
            f"{path} was trained on channels {checkpoint.channel_names}, code defines {CHANNEL_NAMES}"
        )
    return checkpoint


def load_model(path: Path | str, model: nn.Module) -> Checkpoint:
    """Restore weights into `model` and hand back the rest of the state."""
    checkpoint = load_checkpoint(path)
    model.load_state_dict(checkpoint.model_state, strict=True)
    return checkpoint
