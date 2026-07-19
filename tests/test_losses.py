"""Contract tests for the loss implementations.

These are written BEFORE the implementations (they currently fail with
NotImplementedError) — implement until green. Run: pytest tests/ -v
"""
import math

import pytest
import torch
import torch.nn.functional as F

from microclip.losses.infonce import infonce_loss
from microclip.losses.siglip import siglip_loss


def _feats(b=8, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    img = F.normalize(torch.randn(b, d, generator=g), dim=-1)
    txt = F.normalize(torch.randn(b, d, generator=g), dim=-1)
    return img, txt


class TestInfoNCE:
    def test_scalar_and_finite(self):
        img, txt = _feats()
        loss = infonce_loss(img, txt, torch.tensor(math.log(10.0)))
        assert loss.dim() == 0 and torch.isfinite(loss)

    def test_perfect_alignment_low_loss(self):
        # identical feats + high temperature scale -> near-zero loss
        img, _ = _feats()
        loss = infonce_loss(img, img.clone(), torch.tensor(math.log(100.0)))
        assert loss.item() < 0.05

    def test_random_baseline(self):
        # scale=0 -> uniform logits -> loss == log(B) exactly
        img, txt = _feats(b=8)
        loss = infonce_loss(img, txt, torch.tensor(float("-inf")).clamp(min=-30))
        assert abs(loss.item() - math.log(8)) < 1e-3

    def test_symmetric(self):
        img, txt = _feats()
        s = torch.tensor(math.log(10.0))
        assert torch.allclose(infonce_loss(img, txt, s), infonce_loss(txt, img, s), atol=1e-5)


class TestSigLIP:
    def test_scalar_and_finite(self):
        img, txt = _feats()
        loss = siglip_loss(img, txt, torch.tensor(math.log(10.0)), torch.tensor(-10.0))
        assert loss.dim() == 0 and torch.isfinite(loss)

    def test_hand_computed_value(self):
        # B=1, single positive pair: loss = softplus(-(s*sim + b)) / 1
        img = F.normalize(torch.ones(1, 4), dim=-1)
        txt = img.clone()  # sim = 1
        s, b = torch.tensor(0.0), torch.tensor(0.0)  # scale=1, bias=0
        expected = F.softplus(torch.tensor(-1.0)).item()
        loss = siglip_loss(img, txt, s, b)
        assert abs(loss.item() - expected) < 1e-5

    def test_numerically_stable_extreme_logits(self):
        img, txt = _feats()
        # huge scale would underflow a naive log(sigmoid(...))
        loss = siglip_loss(img, txt, torch.tensor(10.0), torch.tensor(-10.0))
        assert torch.isfinite(loss)

    def test_gradients_flow(self):
        img, txt = _feats()
        img = img.clone().requires_grad_(True)
        loss = siglip_loss(img, txt, torch.tensor(math.log(10.0)), torch.tensor(-10.0))
        loss.backward()
        assert img.grad is not None and torch.isfinite(img.grad).all()
