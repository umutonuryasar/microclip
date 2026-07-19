"""Image encoders. ResNet-18 comes from torchvision (random init — the point
is training dynamics, not re-deriving conv arithmetic). ViT-Tiny is a from-
scratch stub for the week-3 ablation; it can reuse your TransformerBlock.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18


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
    """Week-3 ablation. TODO(week 3): patchify (16x16 -> Conv2d stride 16),
    prepend CLS token, add pos emb, reuse TransformerBlock (all-ones pad mask),
    pool CLS, project. ~12 layers, d_model=192, 3 heads = ViT-Tiny."""

    def __init__(self, embed_dim: int):
        super().__init__()
        raise NotImplementedError("Week-3 ablation — implement after main runs.")


def build_image_encoder(name: str, embed_dim: int) -> nn.Module:
    if name == "resnet18":
        return ResNet18Encoder(embed_dim)
    if name == "vit_tiny":
        return ViTTinyEncoder(embed_dim)
    raise ValueError(f"Unknown image encoder: {name}")
