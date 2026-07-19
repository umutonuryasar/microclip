"""SigLIP sigmoid loss (Zhai et al., 2023, arXiv:2303.15343, Eq. 1).

Every (image_i, text_j) pair in the batch is an independent binary
classification: label +1 on the diagonal (matched pair), -1 elsewhere.

    L = -(1/B) * sum_{i,j} log sigmoid( z_ij * (s * <img_i, txt_j> + b) )

where z_ij = +1 if i == j else -1, s = exp(logit_scale), b = logit_bias.

Why it matters here: no softmax over the batch → each pair's gradient does not
depend on batch composition → far weaker batch-size dependence than InfoNCE.
This is the property the whole project interrogates at small scale.

>>> IMPLEMENT-ME. Numerical hint: -log(sigmoid(x)) == softplus(-x) — use
>>> F.softplus, never log(sigmoid(...)) directly (underflows for large -x).
>>> tests/test_losses.py checks value + symmetry properties.
"""
import torch
import torch.nn.functional as F


def siglip_loss(img_feats: torch.Tensor, txt_feats: torch.Tensor,
                logit_scale: torch.Tensor, logit_bias: torch.Tensor) -> torch.Tensor:
    """img_feats, txt_feats: (B, D), already L2-normalized. Returns scalar.

    TODO:
    1. logits = logit_scale.exp() * img_feats @ txt_feats.t() + logit_bias
    2. labels z: 2*eye(B) - 1   (+1 diagonal, -1 off-diagonal)
    3. loss = softplus(-z * logits).sum() / B     # sum over pairs, / batch
       (SigLIP normalizes by B, not B^2 — keep it, it matches the paper.)
    """
    raise NotImplementedError
