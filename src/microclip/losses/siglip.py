"""SigLIP sigmoid loss (Zhai et al., 2023, arXiv:2303.15343, Eq. 1).

Every (image_i, text_j) pair in the batch is an independent binary
classification: label +1 on the diagonal (matched pair), -1 elsewhere.

    L = -(1/B) * sum_{i,j} log sigmoid( z_ij * (s * <img_i, txt_j> + b) )

where z_ij = +1 if i == j else -1, s = exp(logit_scale), b = logit_bias.

Why it matters here: no softmax over the batch → each pair's gradient does not
depend on batch composition → far weaker batch-size dependence than InfoNCE.
This is the property the whole project interrogates at small scale.

Numerical note: -log(sigmoid(x)) == softplus(-x); softplus is used instead of
log(sigmoid(...)), which underflows for large -x.
"""
import torch
import torch.nn.functional as F


def siglip_loss(img_feats: torch.Tensor, txt_feats: torch.Tensor,
                logit_scale: torch.Tensor, logit_bias: torch.Tensor) -> torch.Tensor:
    """img_feats, txt_feats: (B, D), already L2-normalized. Returns scalar."""
    B = img_feats.size(0)
    logits = logit_scale.exp() * img_feats @ txt_feats.t() + logit_bias  # (B, B)
    z = 2.0 * torch.eye(B, device=logits.device, dtype=logits.dtype) - 1.0
    # SigLIP normalizes by B, not B^2 — matches the paper.
    return F.softplus(-z * logits).sum() / B
