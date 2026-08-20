import torch
import torch.nn as nn


def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    B, C, H, W = imgs.shape
    h, w = H // patch_size, W // patch_size

    x = imgs.reshape(B, C, h, patch_size, w, patch_size)
    x = x.permute(0, 2, 4, 3, 5, 1)
    x = x.reshape(B, h * w, patch_size * patch_size * C)

    return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 256, patch_size: int = 16, in_channels: int = 1, embed_dim: int = 768) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MAEEncoder(nn.Module):
    def __init__(self, embed_dim: int = 768, depth: int = 12, num_heads: int = 12) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class MAEDecoder(nn.Module):
    def __init__(
            self, embed_dim: int = 768, decoder_dim: int = 512,
            depth: int = 4, num_heads: int = 8, patch_size: int = 16,
            out_channels: int = 1
            ) -> None:
        super().__init__()
        self.embed_to_decoder = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.blocks = nn.ModuleList([TransformerBlock(decoder_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, patch_size * patch_size * out_channels)

    def forward(self, x: torch.Tensor, restore_idx: torch.Tensor) -> torch.Tensor:
        x = self.embed_to_decoder(x)

        B, N_visible, D = x.shape
        N_total = restore_idx.shape[1]
        num_mask = N_total - N_visible

        mask_tokens = self.mask_token.repeat(B, num_mask, 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=restore_idx.unsqueeze(-1).repeat(1, 1, D))

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        return self.pred(x)


class MAE(nn.Module):
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
            ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.encoder = MAEEncoder(embed_dim, depth, num_heads)
        self.decoder = MAEDecoder(embed_dim, decoder_dim, decoder_depth, decoder_heads, patch_size, in_channels)

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.mask_ratio = mask_ratio

    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        num_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :num_keep]
        x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_visible, mask, ids_restore

    def forward(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.patch_embed(imgs)
        x_visible, mask, restore_idx = self.random_masking(x, self.mask_ratio)

        latent = self.encoder(x_visible)
        pred = self.decoder(latent, restore_idx)

        return pred, mask   