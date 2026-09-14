"""Checkpoint recovery regressions."""
import torch
from torch import nn
from torch.utils.data import Dataset

from microclip.training.trainer import EpochBatchSampler, Trainer


class TinyDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, idx):
        x = torch.tensor([float(idx)])
        return x, x, x


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, images, token_ids, pad_mask):
        return images * self.weight, token_ids


def config(tmp_path):
    return {
        "seed": 7, "run_name": "test", "data": {"num_workers": 0},
        "wandb": {"enabled": False},
        "train": {
            "out_dir": str(tmp_path), "batch_size": 2, "epochs": 2,
            "optimizer": "sgd", "lr": 0.01, "momentum": 0.0,
            "weight_decay": 0.0, "schedule": "constant",
            "amp_dtype": "bf16", "grad_clip": 1.0, "ckpt_every_steps": 2,
        },
    }


def test_mid_epoch_resume_uses_remaining_batches(tmp_path):
    cfg = config(tmp_path)
    seen = []

    def loss(img, txt, model):
        if len(seen) == 4:
            raise RuntimeError("interrupted")
        seen.extend(img[:, 0].detach().tolist())
        return ((img - txt) ** 2).mean()

    trainer = Trainer(TinyModel(), loss, TinyDataset(), TinyDataset(), cfg, device="cpu")
    try:
        trainer.train_one_epoch()
    except RuntimeError as exc:
        assert str(exc) == "interrupted"
    else:
        raise AssertionError("expected interruption")
    assert trainer.global_step == 2
    resumed_seen = []

    def resumed_loss(img, txt, model):
        resumed_seen.extend(txt[:, 0].tolist())
        return ((img - txt) ** 2).mean()

    resumed = Trainer(TinyModel(), resumed_loss, TinyDataset(), TinyDataset(), cfg, device="cpu")
    assert resumed.batch_idx == 2
    resumed.train_one_epoch()
    assert resumed.global_step == 4
    expected = list(EpochBatchSampler(8, 2, cfg["seed"]).__iter__())
    assert resumed_seen == [float(i) for batch in expected[2:] for i in batch]


def test_last_checkpoint_contains_updated_best_val(tmp_path):
    cfg = config(tmp_path)
    trainer = Trainer(TinyModel(), lambda img, txt, model: ((img - txt) ** 2).mean(),
                      TinyDataset(), TinyDataset(), cfg, device="cpu")
    trainer.train_one_epoch = lambda: 1.0
    trainer.validate = lambda: 0.25
    trainer.fit()
    state = torch.load(trainer.out_dir / "last.pt", weights_only=False)
    assert state["best_val"] == 0.25
    assert state["batch_idx"] == 0
    assert (trainer.out_dir / "best.pt").exists()
