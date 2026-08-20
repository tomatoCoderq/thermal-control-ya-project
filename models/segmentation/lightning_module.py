import torch
import torch.nn as nn
import pytorch_lightning as pl

from .metrics import compute_iou, compute_dice


class SegmentationLightningModule(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            criterion: nn.Module,
            lr: float = 3e-4,
            weight_decay: float = 1e-4,
            ):

        super().__init__()
        self.model = model
        self.criterion = criterion

        self.lr = lr
        self.weight_decay = weight_decay

        self.save_hyperparameters(ignore=["model", "criterion"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data, mask = batch
        mask = (mask > 0).float()
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        logits = self(data)
        loss = self.criterion(logits, mask)
        iou = compute_iou(logits, mask)
        dice = compute_dice(logits, mask)

        return loss, iou, dice

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        data, _ = batch
        loss, iou, dice = self._shared_step(batch)
        log = dict(on_step=False, on_epoch=True, batch_size=data.size(0))
        self.log("train_loss", loss, prog_bar=True, **log)
        self.log("train_iou", iou, **log)
        self.log("train_dice", dice, **log)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        data, _ = batch
        loss, iou, dice = self._shared_step(batch)
        log = dict(on_step=False, on_epoch=True, batch_size=data.size(0), prog_bar=True)
        self.log("val_loss", loss, **log)
        self.log("val_iou", iou, **log)
        self.log("val_dice", dice, **log)
        return loss

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        data, _ = batch
        logits = self(data)
        return torch.sigmoid(logits)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=self.lr * 0.05
            )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}