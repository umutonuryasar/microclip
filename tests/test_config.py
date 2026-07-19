"""Config system tests — these pass already (infra is implemented)."""
from pathlib import Path

from microclip.config import apply_overrides, load_config

ROOT = Path(__file__).parent.parent


def test_base_inheritance():
    cfg = load_config(ROOT / "configs/sigmoid_b512.yml")
    assert cfg["loss"]["type"] == "sigmoid"
    assert cfg["model"]["embed_dim"] == 256  # inherited from base
    assert "_base_" not in cfg


def test_nested_ablation_inheritance():
    cfg = load_config(ROOT / "configs/ablations/optimizer_sgd.yml")
    assert cfg["train"]["optimizer"] == "sgd"
    assert cfg["train"]["epochs"] == 10
    assert cfg["train"]["batch_size"] == 512  # untouched base field survives merge


def test_cli_overrides():
    cfg = load_config(ROOT / "configs/base.yml")
    apply_overrides(cfg, ["train.lr=1e-4", "loss.type=softmax", "wandb.enabled=false"])
    assert cfg["train"]["lr"] == 1e-4
    assert cfg["loss"]["type"] == "softmax"
    assert cfg["wandb"]["enabled"] is False
