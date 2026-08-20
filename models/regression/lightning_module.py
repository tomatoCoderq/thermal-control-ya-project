import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any, Callable, Optional


class RegressionLightningModule(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            criterion: nn.Module,
            optimizer: Callable[[nn.Module], torch.optim.Optimizer] = lambda model: torch.optim.AdamW(
                model.parameters(), lr=3e-4, weight_decay=1e-4
            ),
            scheduler: Optional[Callable[[torch.optim.Optimizer], Any]] = None,
            ):
        super().__init__()
        self.model = model
        self.criterion = criterion

        self.optimizer_fn = optimizer
        self.scheduler_fn = scheduler

        self.save_hyperparameters(ignore=["model", "criterion", "optimizer", "scheduler"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)

        error = y_pred - y
        mae = error.abs().mean()
        rmse = (error ** 2).mean().sqrt()

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_mae", mae, prog_bar=True, on_epoch=True)
        self.log("val_rmse", rmse, prog_bar=True, on_epoch=True)

        return loss

    def predict_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, _ = batch
        return self(x)

    def configure_optimizers(self) -> Any:
        optimizer = self.optimizer_fn(self.model)

        if self.scheduler_fn is None:
            return optimizer

        scheduler = self.scheduler_fn(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}