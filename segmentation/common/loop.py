"""One epoch over a loader: train or eval."""
from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import SegMetrics, combined_bce_dice

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    desc: str = "",
    channels_last: bool = False,
    loss_fn: LossFn = combined_bce_dice,
    grad_clip: float | None = None,
) -> tuple[float, dict[str, float]]:
    train = optimizer is not None
    model.train(train)
    metrics = SegMetrics()
    total, seen = 0.0, 0
    use_cl = channels_last and device.type == "cuda"

    bar = tqdm(loader, leave=False, desc=desc or ("train" if train else "test"))
    with torch.set_grad_enabled(train):
        for batch in bar:
            if isinstance(batch, (tuple, list)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch["image"], batch["mask"]
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if use_cl and x.dim() == 4:
                x = x.contiguous(memory_format=torch.channels_last)

            logits = model(x)
            # MPS + channels_last can break backward (.view in BN/Upsample); keep NCHW.
            if not logits.is_contiguous():
                logits = logits.contiguous()
            loss = loss_fn(logits, y)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            bs = x.shape[0]
            total += float(loss.detach()) * bs
            seen += bs
            metrics.update(logits.detach(), y)
            bar.set_postfix(loss=total / max(seen, 1))

    return total / max(seen, 1), metrics.compute()
