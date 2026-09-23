#!/usr/bin/env python
"""Precompute everything the Hugging Face Space needs.

The Space runs on a free CPU box, so it must never touch the COCO images at
request time: encoding 5K images takes minutes. This script does that work
once, offline, and writes a self-contained asset bundle:

  demo/assets/
    index.json                 image file names + one ground-truth caption each
    thumbs/<file>.jpg          small JPEGs the gallery displays
    image_emb_<loss>.npy       (N, D) float16, L2-normalized image embeddings
    <loss>_b512_s42.safetensors  model weights (converted from the .pt)
    tokenizer/bpe16k.json      copied so the Space needs no repo checkout
    configs/                   copied for the same reason

Usage:
  PYTHONPATH=src python scripts/precompute_demo.py
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file
from tqdm import tqdm

from microclip.config import load_config
from microclip.data.tokenizer import CaptionTokenizer
from microclip.data.transforms import build_transforms
from microclip.models.microclip import MicroCLIP

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "demo" / "assets"

# (label, config, checkpoint) — both models are the seed-42, batch-512 runs.
MODELS = [
    ("softmax", "configs/softmax_b512.yml", "weights/softmax_b512_s42.pt"),
    ("sigmoid", "configs/sigmoid_b512.yml", "weights/sigmoid_b512_s42.pt"),
]

THUMB_MAX_SIDE = 320
THUMB_QUALITY = 78


def load_model(cfg: dict, ckpt: Path, tokenizer: CaptionTokenizer) -> MicroCLIP:
    model = MicroCLIP(cfg, vocab_size=tokenizer.vocab_size,
                      max_len=cfg["data"]["max_text_len"])
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"] if "model" in state else state)
    return model.eval()


def build_index(cfg: dict) -> list[dict]:
    """One entry per val2017 image, with its first caption for display."""
    d = cfg["data"]
    ann_path = REPO / d["root"] / d["val_ann"]
    with open(ann_path) as f:
        ann = json.load(f)
    id2file = {img["id"]: img["file_name"] for img in ann["images"]}
    first_caption: dict[int, str] = {}
    for a in ann["annotations"]:
        first_caption.setdefault(a["image_id"], a["caption"].strip())
    # Sorted by image id so the index order is stable across re-runs.
    return [{"file": id2file[i], "caption": first_caption.get(i, "")}
            for i in sorted(id2file) if i in first_caption]


def write_thumbnails(index: list[dict], images_dir: Path) -> None:
    out = ASSETS / "thumbs"
    out.mkdir(parents=True, exist_ok=True)
    for entry in tqdm(index, desc="thumbnails"):
        dst = out / entry["file"]
        if dst.exists():
            continue
        img = Image.open(images_dir / entry["file"]).convert("RGB")
        img.thumbnail((THUMB_MAX_SIDE, THUMB_MAX_SIDE), Image.LANCZOS)
        img.save(dst, "JPEG", quality=THUMB_QUALITY, optimize=True)


@torch.no_grad()
def encode_images(model: MicroCLIP, index: list[dict], images_dir: Path,
                  cfg: dict, batch_size: int = 64) -> np.ndarray:
    """(N, D) float16, L2-normalized. Uses the eval transforms, exactly as
    scripts/evaluate.py does, so the demo matches the reported numbers."""
    transform = build_transforms(cfg["data"]["image_size"], train=False)
    feats, batch = [], []

    def flush():
        if batch:
            x = torch.stack(batch)
            feats.append(F.normalize(model.image_encoder(x), dim=-1))
            batch.clear()

    for entry in tqdm(index, desc="encoding"):
        batch.append(transform(Image.open(images_dir / entry["file"]).convert("RGB")))
        if len(batch) == batch_size:
            flush()
    flush()
    return torch.cat(feats).numpy().astype(np.float16)


def export_weights(ckpt: Path, dst: Path) -> None:
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = state["model"] if "model" in state else state
    save_file({k: v.contiguous() for k, v in sd.items()}, dst)


def copy_static(cfg_paths: list[str], tokenizer_path: str) -> None:
    """Copy configs + tokenizer so the Space is standalone."""
    cfg_out = ASSETS / "configs"
    cfg_out.mkdir(parents=True, exist_ok=True)
    for name in set(cfg_paths) | {"configs/base.yml"}:
        shutil.copy2(REPO / name, cfg_out / Path(name).name)
    tok_out = ASSETS / "tokenizer"
    tok_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / tokenizer_path, tok_out / Path(tokenizer_path).name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="only process the first N images (smoke test)")
    p.add_argument("--skip-thumbs", action="store_true")
    args = p.parse_args()

    ASSETS.mkdir(parents=True, exist_ok=True)
    base_cfg = load_config(REPO / MODELS[0][1])
    tokenizer = CaptionTokenizer(str(REPO / base_cfg["tokenizer"]["path"]))
    images_dir = REPO / base_cfg["data"]["root"] / base_cfg["data"]["val_images"]

    index = build_index(base_cfg)
    if args.limit:
        index = index[: args.limit]
    print(f"{len(index)} images from {images_dir}")

    if not args.skip_thumbs:
        write_thumbnails(index, images_dir)

    for label, cfg_path, ckpt_path in MODELS:
        cfg = load_config(REPO / cfg_path)
        model = load_model(cfg, REPO / ckpt_path, tokenizer)
        emb = encode_images(model, index, images_dir, cfg)
        np.save(ASSETS / f"image_emb_{label}.npy", emb)
        export_weights(REPO / ckpt_path, ASSETS / f"{label}_b512_s42.safetensors")
        print(f"{label}: embeddings {emb.shape}, "
              f"temp {model.logit_scale.exp().item():.2f}")

    copy_static([m[1] for m in MODELS], base_cfg["tokenizer"]["path"])
    with open(ASSETS / "index.json", "w") as f:
        json.dump(index, f)
    print(f"wrote {ASSETS}")


if __name__ == "__main__":
    main()
