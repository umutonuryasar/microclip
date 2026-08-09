#!/usr/bin/env python
"""1K-sample smoke test — MUST pass before burning A100 hours.

Checks: data loads, shapes are right, loss decreases over ~50 steps on both
loss types, checkpoint save/resume round-trips.

Run on the RTX 3050 — if it fits there at batch 64, A100 runs are safe.
Usage: python scripts/smoke_test.py --config configs/base.yml
"""
import argparse
import copy
import math
import shutil
from pathlib import Path

import torch

from microclip.config import apply_overrides, load_config
from microclip.data.coco_captions import CocoCaptions
from microclip.data.tokenizer import CaptionTokenizer
from microclip.losses.infonce import build_loss
from microclip.models.microclip import MicroCLIP
from microclip.training.trainer import Trainer

TARGET_STEPS = 50
LIMIT = 1000
BATCH = 64


def run_one(cfg: dict, loss_type: str) -> None:
    cfg = copy.deepcopy(cfg)  # overrides must not leak into the next run
    run_name = f"smoke_{loss_type}"
    out_dir = Path(cfg["train"]["out_dir"]) / run_name
    shutil.rmtree(out_dir, ignore_errors=True)  # always start fresh

    apply_overrides(cfg, [
        f"run_name={run_name}",
        f"loss.type={loss_type}",
        f"train.batch_size={BATCH}",
        "train.warmup_steps=10",
        "train.ckpt_every_steps=25",
        "wandb.enabled=false",
        "data.num_workers=2",
    ])

    tokenizer = CaptionTokenizer(cfg["tokenizer"]["path"])
    d = cfg["data"]
    train_ds = CocoCaptions(d["root"], d["train_images"], d["train_ann"], tokenizer,
                            d["image_size"], d["max_text_len"], train=True, limit=LIMIT)
    val_ds = CocoCaptions(d["root"], d["val_images"], d["val_ann"], tokenizer,
                          d["image_size"], d["max_text_len"], train=False, limit=LIMIT)

    steps_per_epoch = LIMIT // BATCH
    epochs = max(2, math.ceil(TARGET_STEPS / steps_per_epoch))
    apply_overrides(cfg, [f"train.epochs={epochs}"])

    def build_trainer():
        model = MicroCLIP(cfg, vocab_size=tokenizer.vocab_size, max_len=d["max_text_len"])
        return Trainer(model, build_loss(cfg), train_ds, val_ds, cfg)

    print(f"\n=== smoke [{loss_type}]: {epochs} epochs x {steps_per_epoch} steps ===")
    trainer = build_trainer()
    first_loss = None
    losses = []
    for _ in range(epochs):
        epoch_loss = trainer.train_one_epoch()
        if first_loss is None:
            first_loss = epoch_loss
        losses.append(epoch_loss)
        trainer.epoch += 1
        trainer._save(trainer.out_dir / "last.pt")
    val_loss = trainer.validate()
    assert torch.isfinite(torch.tensor(losses)).all(), f"non-finite loss: {losses}"
    assert losses[-1] < first_loss, \
        f"[{loss_type}] loss did not decrease: {first_loss:.4f} -> {losses[-1]:.4f}"
    print(f"[{loss_type}] loss {first_loss:.4f} -> {losses[-1]:.4f}, val {val_loss:.4f} OK")

    # Kill and re-create — resume must pick up where we left off.
    saved_step = trainer.global_step
    del trainer
    resumed = build_trainer()
    assert resumed.global_step == saved_step, \
        f"resume broken: expected step {saved_step}, got {resumed.global_step}"
    print(f"[{loss_type}] resume OK at step {saved_step}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yml")
    args = p.parse_args()
    base_cfg = load_config(args.config)
    for loss_type in ("sigmoid", "softmax"):
        run_one(base_cfg, loss_type)
    print("\nSMOKE TEST PASSED — safe to launch A100 runs.")


if __name__ == "__main__":
    main()
