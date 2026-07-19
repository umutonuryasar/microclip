"""Training loop with Colab-proof checkpoint/resume.

Resume contract (learned the hard way on rt-detr-kd):
- `last.pt` is written every `ckpt_every_steps` AND at every epoch end, and
  contains model + optimizer + scheduler + scaler + epoch + global_step +
  RNG states. Restarting `scripts/train.py` with the same config resumes
  bit-for-bit from it automatically.
- `best.pt` tracks best val loss (weights only).
- Writes go to a temp file then os.replace() — a preempted Colab session
  can't leave a half-written checkpoint.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(self, model, loss_fn, train_ds, val_ds, cfg: dict, device: str = "cuda"):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.device = device
        t = cfg["train"]
        self.out_dir = Path(t["out_dir"]) / cfg["run_name"]
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.train_loader = DataLoader(
            train_ds, batch_size=t["batch_size"], shuffle=True, drop_last=True,
            num_workers=cfg["data"]["num_workers"], pin_memory=True)
        self.val_loader = DataLoader(
            val_ds, batch_size=t["batch_size"], shuffle=False,
            num_workers=cfg["data"]["num_workers"], pin_memory=True)

        self.optimizer = self._build_optimizer()
        total_steps = len(self.train_loader) * t["epochs"]
        from .schedulers import build_scheduler
        self.scheduler = build_scheduler(self.optimizer, cfg, total_steps)

        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[t["amp_dtype"]]
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.amp_dtype == torch.float16))

        self.epoch = 0
        self.global_step = 0
        self.best_val = float("inf")
        self._maybe_resume()

    def _build_optimizer(self):
        t = self.cfg["train"]
        params = self.model.parameters()
        if t["optimizer"] == "adamw":
            return torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"],
                                     betas=tuple(t["betas"]))
        if t["optimizer"] == "sgd":
            return torch.optim.SGD(params, lr=t["lr"], momentum=t["momentum"],
                                   weight_decay=t["weight_decay"])
        raise ValueError(t["optimizer"])

    # ---------------- checkpointing ----------------

    def _save(self, path: Path, weights_only: bool = False) -> None:
        state = {"model": self.model.state_dict()}
        if not weights_only:
            state.update({
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "epoch": self.epoch,
                "global_step": self.global_step,
                "best_val": self.best_val,
                "rng": {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all(),
                    "numpy": np.random.get_state(),
                    "python": random.getstate(),
                },
            })
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)  # atomic — no torn checkpoints on preemption

    def _maybe_resume(self) -> None:
        last = self.out_dir / "last.pt"
        if not last.exists():
            return
        state = torch.load(last, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.epoch = state["epoch"]
        self.global_step = state["global_step"]
        self.best_val = state["best_val"]
        torch.set_rng_state(state["rng"]["torch"])
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        np.random.set_state(state["rng"]["numpy"])
        random.setstate(state["rng"]["python"])
        print(f"[resume] epoch={self.epoch} step={self.global_step} best_val={self.best_val:.4f}")

    # ---------------- loops ----------------

    def train_one_epoch(self) -> float:
        """TODO(week 1): the actual step. Per batch:
        1. move (images, token_ids, pad_mask) to device
        2. with torch.autocast(self.device, dtype=self.amp_dtype):
               img, txt = self.model(images, token_ids, pad_mask)
               loss = self.loss_fn(img, txt, self.model)
        3. scaler.scale(loss).backward(); unscale; clip to cfg train.grad_clip;
           scaler.step(optimizer); scaler.update(); optimizer.zero_grad();
           scheduler.step(); self.global_step += 1
        4. every cfg train.ckpt_every_steps: self._save(self.out_dir/'last.pt')
        5. log loss + lr + logit_scale.exp() to wandb (watch logit_scale —
           if it explodes, temperature learning is broken)
        Return mean epoch loss.
        """
        raise NotImplementedError

    @torch.no_grad()
    def validate(self) -> float:
        """TODO(week 1): mean val loss over self.val_loader (same forward,
        no grad). Keep it to loss only — retrieval metrics run separately via
        scripts/evaluate.py, they're too slow for every epoch."""
        raise NotImplementedError

    def fit(self) -> None:
        epochs = self.cfg["train"]["epochs"]
        while self.epoch < epochs:
            train_loss = self.train_one_epoch()
            val_loss = self.validate()
            self.epoch += 1
            self._save(self.out_dir / "last.pt")
            if val_loss < self.best_val:
                self.best_val = val_loss
                self._save(self.out_dir / "best.pt", weights_only=True)
            print(f"epoch {self.epoch}/{epochs} train={train_loss:.4f} "
                  f"val={val_loss:.4f} best={self.best_val:.4f}")
