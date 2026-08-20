import torch
import torch.nn as nn
import pytorch_lightning as pl


class RegressionLightningModule(pl.LightningModule):
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=self.lr * 0.05
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}