"""Image-text retrieval: R@1/5/10 both directions (COCO 5K, Flickr30k 1K).

Careful with the standard protocol detail: COCO/Flickr tests use ALL ~5
captions per image. I->T: a hit if any of the image's captions is in top-K.
T->I: each caption queries independently, its image must be in top-K.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def encode_corpus(model, dataset, cfg, device="cuda"):
    """TODO(week 4): return (img_feats (N,D), txt_feats (M,D),
    txt2img (M,) mapping caption index -> image index). Batch it; this must
    also run on the RTX 3050 for the HF Space demo."""
    raise NotImplementedError


def recall_at_k(img_feats, txt_feats, txt2img, ks=(1, 5, 10)) -> dict:
    """TODO(week 4): similarity matrix (chunked if needed), compute R@K for
    I->T and T->I per the protocol above. Return e.g.
    {'i2t/R@1': ..., 't2i/R@1': ..., ...}"""
    raise NotImplementedError
