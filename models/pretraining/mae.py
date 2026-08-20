# import torch
# import torch.nn as nn

# class PatchEmbed(nn.Module):
#     def __init__(self, img_size: int = 256, patch_size: int = 16, in_channels: int = 1, embed_dim: int = 768):
#         super().__init__()
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = (img_size // patch_size) ** 2
#         self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.proj(x)
#         x = x.flatten(2).transpose(1, 2)

#         return x

# class TransformerBlock(nn.Module):
#     def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
#             nn.GELU(),
#             nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
#         x = x + attn_out
#         x = x + self.mlp(self.norm2(x))

#         return x

# class MAEEncoder(nn.Module):
#     def __init__(self, embed_dim: int = 768, depth: int = 12, num_heads: int = 12):
#         super().__init__()
#         self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
#         self.norm = nn.LayerNorm(embed_dim)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         for block in self.blocks:
#             x = block(x)
#         return self.norm(x)


# class MAEDecoder(nn.Module):
#     def __init__(
#             self, embed_dim: int = 768, decoder_dim: int = 512,
#             depth: int = 4, num_heads: int = 8, patch_size: int = 16,
#             out_channels: int = 1
#             ):
#         super().__init__()
#         self.embed_to_decoder = nn.Linear(embed_dim, decoder_dim)
#         self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
#         self.blocks = nn.ModuleList([TransformerBlock(decoder_dim, num_heads) for _ in range(depth)])
#         self.norm = nn.LayerNorm(decoder_dim)
#         self.pred = nn.Linear(decoder_dim, patch_size * patch_size * out_channels)

#     def forward(self, x: torch.Tensor, restore_idx: torch.Tensor) -> torch.Tensor:
