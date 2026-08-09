"""Image-text retrieval: R@1/5/10 both directions (COCO 5K, Flickr30k 1K).

Careful with the standard protocol detail: COCO/Flickr tests use ALL ~5
captions per image. I->T: a hit if any of the image's captions is in top-K.
T->I: each caption queries independently, its image must be in top-K.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image


@torch.no_grad()
def encode_corpus(model, dataset, cfg, device="cuda", batch_size=64):
    """Encodes every image and EVERY caption of a CocoCaptions-style dataset.
    Returns (img_feats (N,D), txt_feats (M,D), txt2img (M,) mapping caption
    index -> image index). Batched so it also fits the RTX 3050."""
    model.eval()
    img_feats, txt_feats, txt2img = [], [], []
    img_batch, tok_batch, mask_batch = [], [], []

    def flush_images():
        if not img_batch:
            return
        x = torch.stack(img_batch).to(device)
        img_feats.append(F.normalize(model.image_encoder(x), dim=-1).cpu())
        img_batch.clear()

    def flush_texts():
        if not tok_batch:
            return
        ids = torch.tensor(tok_batch, device=device)
        mask = torch.tensor(mask_batch, device=device)
        txt_feats.append(F.normalize(model.text_encoder(ids, mask), dim=-1).cpu())
        tok_batch.clear()
        mask_batch.clear()

    for img_idx, (file_name, captions) in enumerate(dataset.items):
        image = Image.open(dataset.images_dir / file_name).convert("RGB")
        img_batch.append(dataset.transform(image))
        if len(img_batch) == batch_size:
            flush_images()
        for cap in captions:
            ids, mask = dataset.tokenizer.encode(cap, dataset.max_text_len)
            tok_batch.append(ids)
            mask_batch.append(mask)
            txt2img.append(img_idx)
            if len(tok_batch) == batch_size:
                flush_texts()
    flush_images()
    flush_texts()
    return torch.cat(img_feats), torch.cat(txt_feats), torch.tensor(txt2img)


def recall_at_k(img_feats, txt_feats, txt2img, ks=(1, 5, 10)) -> dict:
    """R@K for I->T and T->I per the protocol above."""
    sim = txt_feats @ img_feats.t()  # (M captions, N images)
    out = {}
    # T->I: each caption queries independently; its image must be in top-K.
    for k in ks:
        topk = sim.topk(k, dim=1).indices  # (M, k) image indices
        out[f"t2i/R@{k}"] = (topk == txt2img[:, None]).any(dim=1).float().mean().item()
    # I->T: hit if ANY of the image's captions is in top-K retrieved captions.
    sim_it = sim.t()  # (N, M)
    n_images = sim_it.size(0)
    for k in ks:
        topk = sim_it.topk(k, dim=1).indices          # (N, k) caption indices
        retrieved_imgs = txt2img[topk]                # (N, k) their image ids
        hits = retrieved_imgs == torch.arange(n_images)[:, None]
        out[f"i2t/R@{k}"] = hits.any(dim=1).float().mean().item()
    return out
