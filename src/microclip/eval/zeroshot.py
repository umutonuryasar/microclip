"""Zero-shot classification (CIFAR-10/100) via prompt embedding.

Protocol (CLIP paper, Sec. 3.1.2): embed "a photo of a {class}" for every
class, L2-normalize, classify each image by max cosine similarity.
"""
from __future__ import annotations

import torch


PROMPT = "a photo of a {}"


@torch.no_grad()
def build_class_embeddings(model, tokenizer, class_names: list[str],
                           max_len: int, device: str) -> torch.Tensor:
    """TODO(week 4): tokenize PROMPT.format(name) for each class, run text
    encoder, F.normalize, stack -> (num_classes, D)."""
    raise NotImplementedError


@torch.no_grad()
def zeroshot_accuracy(model, tokenizer, dataset, class_names, cfg, device="cuda"):
    """TODO(week 4): DataLoader over dataset (use eval transforms!), encode
    images, logits = img @ class_emb.T, top-1 accuracy. Return float."""
    raise NotImplementedError
