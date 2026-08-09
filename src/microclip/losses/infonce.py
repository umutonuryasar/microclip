"""Softmax InfoNCE contrastive loss (CLIP, Radford et al., 2021, Sec. 2.3).

Symmetric cross-entropy over the similarity matrix: each image must pick its
text out of the batch (row-wise CE) and each text its image (column-wise CE).
The batch IS the negative pool — this is exactly why CLIP wants huge batches,
and why we expect this loss to degrade at B=128 relative to sigmoid.
"""
import torch
import torch.nn.functional as F


def infonce_loss(img_feats: torch.Tensor, txt_feats: torch.Tensor,
                 logit_scale: torch.Tensor) -> torch.Tensor:
    """img_feats, txt_feats: (B, D), L2-normalized. Returns scalar.
    No logit_bias here — bias is a sigmoid-loss concept."""
    logits = logit_scale.exp() * img_feats @ txt_feats.t()  # (B, B)
    targets = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) +
                  F.cross_entropy(logits.t(), targets))


def build_loss(cfg: dict):
    """Returns callable(model_outputs...) -> scalar, dispatched on cfg."""
    from .siglip import siglip_loss
    kind = cfg["loss"]["type"]
    if kind == "sigmoid":
        return lambda img, txt, model: siglip_loss(img, txt, model.logit_scale, model.logit_bias)
    if kind == "softmax":
        return lambda img, txt, model: infonce_loss(img, txt, model.logit_scale)
    raise ValueError(f"Unknown loss type: {kind}")
