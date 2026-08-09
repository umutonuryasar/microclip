"""Zero-shot classification (CIFAR-10/100) via prompt embedding.

Protocol (CLIP paper, Sec. 3.1.2): embed "a photo of a {class}" for every
class, L2-normalize, classify each image by max cosine similarity.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


PROMPT = "a photo of a {}"


@torch.no_grad()
def build_class_embeddings(model, tokenizer, class_names: list[str],
                           max_len: int, device: str) -> torch.Tensor:
    """-> (num_classes, D), L2-normalized."""
    all_ids, all_masks = [], []
    for name in class_names:
        ids, mask = tokenizer.encode(PROMPT.format(name), max_len)
        all_ids.append(ids)
        all_masks.append(mask)
    ids = torch.tensor(all_ids, device=device)
    mask = torch.tensor(all_masks, device=device)
    return F.normalize(model.text_encoder(ids, mask), dim=-1)


@torch.no_grad()
def zeroshot_accuracy(model, tokenizer, dataset, class_names, cfg,
                      device="cuda", batch_size=256) -> float:
    """dataset must yield (image, label) with EVAL transforms. Returns top-1."""
    model.eval()
    class_emb = build_class_embeddings(
        model, tokenizer, class_names, cfg["data"]["max_text_len"], device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=cfg["data"]["num_workers"])
    correct, total = 0, 0
    for images, labels in loader:
        img = F.normalize(model.image_encoder(images.to(device)), dim=-1)
        pred = (img @ class_emb.t()).argmax(dim=1).cpu()
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return correct / total
