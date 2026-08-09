"""Image encoders. ResNet-18 comes from torchvision (random init — the point
is training dynamics, not re-deriving conv arithmetic). ViT-Tiny is built
from scratch, reusing the text encoder's TransformerBlock.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18

from .text_encoder import TransformerBlock


class ResNet18Encoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        m = resnet18(weights=None)  # from scratch — no pretrained weights
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # -> (B, 512, 1, 1)
        self.proj = nn.Linear(512, embed_dim, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(images).flatten(1)  # (B, 512)
        return self.proj(feats)                   # (B, embed_dim), NOT normalized


class ViTTinyEncoder(nn.Module):
    """ViT-Tiny: 12 layers, d_model=192, 3 heads, patch 16. Reuses the text
    encoder's pre-norm TransformerBlock with an all-ones pad mask."""

    def __init__(self, embed_dim: int, image_size: int = 224, patch: int = 16,
                 d_model: int = 192, layers: int = 12, heads: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        assert image_size % patch == 0
        num_patches = (image_size // patch) ** 2
        self.patchify = nn.Conv2d(3, d_model, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, heads, ffn_mult=4, dropout=dropout)
            for _ in range(layers))
        self.ln_final = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, embed_dim, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B = images.size(0)
        x = self.patchify(images).flatten(2).transpose(1, 2)  # (B, N, d_model)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_emb
        ones = torch.ones(B, x.size(1), dtype=torch.long, device=x.device)
        for block in self.blocks:
            x = block(x, ones)
        x = self.ln_final(x)
        return self.proj(x[:, 0])  # CLS pool -> (B, embed_dim), NOT normalized


def build_image_encoder(name: str, embed_dim: int) -> nn.Module:
    if name == "resnet18":
        return ResNet18Encoder(embed_dim)
    if name == "vit_tiny":
        return ViTTinyEncoder(embed_dim)
    raise ValueError(f"Unknown image encoder: {name}")
