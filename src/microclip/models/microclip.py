"""Dual-encoder MicroCLIP model."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_encoder import build_image_encoder
from .text_encoder import TextEncoder


class MicroCLIP(nn.Module):
    def __init__(self, cfg: dict, vocab_size: int, max_len: int):
        super().__init__()
        m = cfg["model"]
        self.image_encoder = build_image_encoder(m["image_encoder"], m["embed_dim"])
        t = m["text_encoder"]
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size, embed_dim=m["embed_dim"], layers=t["layers"],
            d_model=t["d_model"], heads=t["heads"], ffn_mult=t["ffn_mult"],
            dropout=t["dropout"], max_len=max_len,
        )
        # Learnable temperature (and bias, used only by the sigmoid loss).
        self.logit_scale = nn.Parameter(torch.tensor(float(cfg["loss"]["logit_scale_init"])))
        self.logit_bias = nn.Parameter(torch.tensor(float(cfg["loss"]["logit_bias_init"])))
        self._init_weights(m.get("init", "default"))

    def _init_weights(self, scheme: str) -> None:
        # Covers Linear AND Conv2d so the init ablation touches the whole
        # model (ResNet backbone included), not just projection heads.
        if scheme == "default":
            return
        for mod in self.modules():
            if isinstance(mod, (nn.Linear, nn.Conv2d)):
                if scheme == "xavier":
                    nn.init.xavier_uniform_(mod.weight)
                elif scheme == "he":
                    nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    def forward(self, images, token_ids, pad_mask):
        """Returns L2-normalized (img_feats, txt_feats), each (B, embed_dim)."""
        img = F.normalize(self.image_encoder(images), dim=-1)
        txt = F.normalize(self.text_encoder(token_ids, pad_mask), dim=-1)
        return img, txt

    def logits(self, img_feats, txt_feats):
        """Scaled pairwise similarity matrix (B_img, B_txt). Bias is added by
        the sigmoid loss, not here."""
        return self.logit_scale.exp() * img_feats @ txt_feats.t()
