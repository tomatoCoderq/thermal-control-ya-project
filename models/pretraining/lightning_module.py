import torch
import pytorch_lightning as pl

from model import MAE, patchify
from loss import MAEReconstructionLoss
from metrics import compute_psnr, compute_visible_mse


class MAELightningModule(pl.LightningModule):
    def __init__(
            self,
            img_size: int = 256,
            patch_size: int = 16,
            in_channels: int = 1,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            decoder_dim: int = 512,
            decoder_depth: int = 4,
            decoder_heads: int = 8,
            mask_ratio: float = 0.4,
            lr: float = 1.5e-4,
            weight_decay: float = 0.05,
            ) -> None:
        super().__init__()
        self.model = MAE(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            decoder_dim=decoder_dim, decoder_depth=decoder_depth, decoder_heads=decoder_heads,
            mask_ratio=mask_ratio,
        )
        self.criterion = MAEReconstructionLoss(patch_size=patch_size)
        self.patch_size = patch_size

        self.lr = lr
        self.weight_decay = weight_decay

        self.save_hyperparameters()

    def forward(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(imgs)

    def _shared_step(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        pred, mask = self(imgs)
        loss = self.criterion(pred, imgs, mask)

        target = patchify(imgs, self.patch_size)
        psnr = compute_psnr(pred, target)
        visible_mse = compute_visible_mse(pred, target, mask)

        return loss, psnr, visible_mse

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, psnr, visible_mse = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_psnr", psnr, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, psnr, visible_mse = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_psnr", psnr, prog_bar=True, on_epoch=True)
        self.log("val_visible_mse", visible_mse, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=self.lr * 0.05
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}