"""COCO Captions dataset for contrastive training.

One (image, caption) pair per __getitem__; the caption is sampled uniformly
from the image's ~5 captions each epoch (cheap augmentation, matches CLIP).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .tokenizer import CaptionTokenizer
from .transforms import build_transforms


class CocoCaptions(Dataset):
    def __init__(self, root: str, images_dir: str, ann_file: str,
                 tokenizer: CaptionTokenizer, image_size: int = 224,
                 max_text_len: int = 64, train: bool = True,
                 limit: int | None = None):
        self.images_dir = Path(root) / images_dir
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.train = train
        self.transform = build_transforms(image_size, train=train)

        with open(Path(root) / ann_file) as f:
            ann = json.load(f)
        id2file = {img["id"]: img["file_name"] for img in ann["images"]}
        caps: dict[int, list[str]] = {}
        for a in ann["annotations"]:
            caps.setdefault(a["image_id"], []).append(a["caption"])
        self.items = [(id2file[i], c) for i, c in caps.items() if i in id2file]
        if limit is not None:
            self.items = self.items[:limit]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        file_name, captions = self.items[idx]
        image = Image.open(self.images_dir / file_name).convert("RGB")
        image = self.transform(image)
        caption = random.choice(captions) if self.train else captions[0]
        token_ids, attn_mask = self.tokenizer.encode(caption, self.max_text_len)
        return image, torch.tensor(token_ids), torch.tensor(attn_mask)
