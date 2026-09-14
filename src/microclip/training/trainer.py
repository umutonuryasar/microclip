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

import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm


class EpochBatchSampler(Sampler[list[int]]):
    """Stable per-epoch shuffle, with a cursor for checkpoint recovery."""

    def __init__(self, size: int, batch_size: int, seed: int):
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.epoch = 0
        self.start_batch = 0

    def __len__(self):
        return self.size // self.batch_size - self.start_batch

    def __iter__(self):
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(
            self.seed + self.epoch)).tolist()
        for batch in range(self.start_batch, self.size // self.batch_size):
            start = batch * self.batch_size
            yield order[start:start + self.batch_size]


class Trainer:
    def __init__(self, model, loss_fn, train_ds, val_ds, cfg: dict,
                 device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.device = device
        t = cfg["train"]
        self.out_dir = Path(t["out_dir"]) / cfg["run_name"]
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.train_ds = train_ds
        self.train_batches = EpochBatchSampler(len(train_ds), t["batch_size"], cfg["seed"])
        self.train_loader = DataLoader(
            train_ds, batch_sampler=self.train_batches,
            num_workers=cfg["data"]["num_workers"], pin_memory=True,
            generator=torch.Generator().manual_seed(cfg["seed"]))
        self.val_loader = DataLoader(
            val_ds, batch_size=t["batch_size"], shuffle=False,
            num_workers=cfg["data"]["num_workers"], pin_memory=True)

        self.optimizer = self._build_optimizer()
        total_steps = len(self.train_batches) * t["epochs"]
        from .schedulers import build_scheduler
        self.scheduler = build_scheduler(self.optimizer, cfg, total_steps)

        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[t["amp_dtype"]]
        self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        # Loss scaling only matters for fp16 on CUDA; bf16/CPU run unscaled.
        scaler_enabled = self.amp_dtype == torch.float16 and self.device_type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device_type, enabled=scaler_enabled)

        self.epoch = 0
        self.batch_idx = 0
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
                "batch_idx": self.batch_idx,
                "global_step": self.global_step,
                "best_val": self.best_val,
                "rng": {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
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
        state = torch.load(last, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.epoch = state["epoch"]
        self.batch_idx = state.get("batch_idx", 0)
        self.global_step = state["global_step"]
        self.best_val = state["best_val"]
        torch.set_rng_state(state["rng"]["torch"].cpu())
        if torch.cuda.is_available() and len(state["rng"]["cuda"]) > 0:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["rng"]["cuda"]])
        np.random.set_state(state["rng"]["numpy"])
        random.setstate(state["rng"]["python"])
        print(f"[resume] epoch={self.epoch} step={self.global_step} best_val={self.best_val:.4f}")

    # ---------------- loops ----------------

    def _wandb(self):
        if not self.cfg["wandb"]["enabled"]:
            return None
        import wandb
        return wandb if wandb.run is not None else None

    def train_one_epoch(self) -> float:
        self.model.train()
        t = self.cfg["train"]
        wb = self._wandb()
        total, n = 0.0, 0
        self.train_batches.epoch = self.epoch
        self.train_batches.start_batch = self.batch_idx
        if hasattr(self.train_ds, "epoch"):
            self.train_ds.epoch = self.epoch
            self.train_ds.seed = self.cfg["seed"]
        pbar = tqdm(self.train_loader, desc=f"epoch {self.epoch}", leave=False)
        for images, token_ids, pad_mask in pbar:
            images = images.to(self.device, non_blocking=True)
            token_ids = token_ids.to(self.device, non_blocking=True)
            pad_mask = pad_mask.to(self.device, non_blocking=True)

            with torch.autocast(self.device_type, dtype=self.amp_dtype):
                img, txt = self.model(images, token_ids, pad_mask)
                loss = self.loss_fn(img, txt, self.model)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), t["grad_clip"])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # Clamp temperature like CLIP: an unbounded logit_scale can drift up,
            # saturate the softmax (InfoNCE arm) and silently vanish gradients.
            with torch.no_grad():
                self.model.logit_scale.clamp_(max=math.log(100.0))
            self.scheduler.step()
            self.global_step += 1
            self.batch_idx += 1

            item = loss.item()
            total += item
            n += 1
            pbar.set_postfix(loss=f"{item:.4f}")
            if wb is not None:
                # Watch logit_scale — if it explodes, temperature learning is broken.
                wb.log({
                    "train/loss": item,
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "train/logit_scale": self.model.logit_scale.exp().item(),
                }, step=self.global_step)
            if self.global_step % t["ckpt_every_steps"] == 0:
                self._save(self.out_dir / "last.pt")
        return total / max(n, 1)

    @torch.no_grad()
    def validate(self) -> float:
        """Mean val loss only — retrieval metrics run separately via
        scripts/evaluate.py, they're too slow for every epoch."""
        self.model.eval()
        total, n = 0.0, 0
        for images, token_ids, pad_mask in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            token_ids = token_ids.to(self.device, non_blocking=True)
            pad_mask = pad_mask.to(self.device, non_blocking=True)
            with torch.autocast(self.device_type, dtype=self.amp_dtype):
                img, txt = self.model(images, token_ids, pad_mask)
                loss = self.loss_fn(img, txt, self.model)
            total += loss.item()
            n += 1
        return total / max(n, 1)

    def fit(self) -> None:
        epochs = self.cfg["train"]["epochs"]
        wb = self._wandb()
        while self.epoch < epochs:
            train_loss = self.train_one_epoch()
            val_loss = self.validate()
            self.epoch += 1
            self.batch_idx = 0
            if val_loss < self.best_val:
                self.best_val = val_loss
                self._save(self.out_dir / "best.pt", weights_only=True)
            self._save(self.out_dir / "last.pt")
            if wb is not None:
                wb.log({"val/loss": val_loss, "epoch": self.epoch}, step=self.global_step)
            print(f"epoch {self.epoch}/{epochs} train={train_loss:.4f} "
                  f"val={val_loss:.4f} best={self.best_val:.4f}")
