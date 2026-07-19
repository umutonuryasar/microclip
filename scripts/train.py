#!/usr/bin/env python
"""Main training entry point. Auto-resumes from runs/<run_name>/last.pt."""
import argparse
import random

import numpy as np
import torch

from microclip.config import apply_overrides, load_config
from microclip.data.coco_captions import CocoCaptions
from microclip.data.tokenizer import CaptionTokenizer
from microclip.losses.infonce import build_loss
from microclip.models.microclip import MicroCLIP
from microclip.training.trainer import Trainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.lr=1e-4")
    p.add_argument("--limit", type=int, default=None, help="cap dataset size (smoke tests)")
    args = p.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"]); random.seed(cfg["seed"])

    tokenizer = CaptionTokenizer(cfg["tokenizer"]["path"])
    d = cfg["data"]
    train_ds = CocoCaptions(d["root"], d["train_images"], d["train_ann"], tokenizer,
                            d["image_size"], d["max_text_len"], train=True, limit=args.limit)
    val_ds = CocoCaptions(d["root"], d["val_images"], d["val_ann"], tokenizer,
                          d["image_size"], d["max_text_len"], train=False, limit=args.limit)

    model = MicroCLIP(cfg, vocab_size=tokenizer.vocab_size, max_len=d["max_text_len"])
    loss_fn = build_loss(cfg)

    if cfg["wandb"]["enabled"]:
        import wandb
        wandb.init(project=cfg["wandb"]["project"], name=cfg["run_name"], config=cfg)

    Trainer(model, loss_fn, train_ds, val_ds, cfg).fit()


if __name__ == "__main__":
    main()
