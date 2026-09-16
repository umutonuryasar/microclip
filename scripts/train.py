#!/usr/bin/env python
"""Main training entry point. Auto-resumes from runs/<run_name>/last.pt."""
import argparse
import random

import numpy as np
import torch

from microclip.config import apply_overrides, load_config
from microclip.data.coco_captions import CocoCaptions
from microclip.data.tokenizer import CaptionTokenizer
from microclip.losses import build_loss
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
        # id=run_name + resume="allow": a preempted run that restarts continues the
        # SAME W&B run instead of opening a new one and leaving the old "crashed".
        wandb.init(project=cfg["wandb"]["project"], name=cfg["run_name"],
                   id=cfg["run_name"], resume="allow", config=cfg)

    try:
        Trainer(model, loss_fn, train_ds, val_ds, cfg).fit()
    finally:
        # Mark the run finished even if fit() raised — otherwise W&B shows "crashed".
        if cfg["wandb"]["enabled"]:
            import wandb
            if wandb.run is not None:
                wandb.finish()


if __name__ == "__main__":
    main()
