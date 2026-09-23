#!/usr/bin/env python
"""Export each model's text encoder to ONNX for the in-browser demo.

Only the text tower is needed at request time: image embeddings are
precomputed. The exported graph takes a padded token sequence and returns the
L2-normalized embedding, so the browser does no post-processing.

Shapes are fixed at (1, max_text_len) because the tokenizer always pads to
max_text_len — fixed shapes let onnxruntime-web skip reallocation.

Parity against PyTorch is asserted here, not assumed: an export that silently
changes the masking would be invisible in the demo but wrong.

Usage:
  PYTHONPATH=src python scripts/export_onnx.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from onnxruntime.quantization import QuantType, quantize_dynamic

from microclip.config import load_config
from microclip.data.tokenizer import CaptionTokenizer
from microclip.models.microclip import MicroCLIP

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "demo" / "static" / "assets"

MODELS = [
    ("softmax", "configs/softmax_b512.yml", "weights/softmax_b512_s42.pt"),
    ("sigmoid", "configs/sigmoid_b512.yml", "weights/sigmoid_b512_s42.pt"),
]

PARITY_QUERIES = [
    "a man riding a surfboard on a wave",
    "two dogs playing in the snow",
    "an empty street",
    "x",  # near-degenerate input: almost all padding
]


class TextTower(nn.Module):
    """Text encoder plus the L2 normalization the demo needs."""

    def __init__(self, model: MicroCLIP):
        super().__init__()
        self.text_encoder = model.text_encoder

    def forward(self, token_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_encoder(token_ids, pad_mask), dim=-1)


def export_one(label: str, cfg_path: str, ckpt_path: str,
               tokenizer: CaptionTokenizer) -> None:
    cfg = load_config(REPO / cfg_path)
    max_len = cfg["data"]["max_text_len"]
    model = MicroCLIP(cfg, vocab_size=tokenizer.vocab_size, max_len=max_len)
    state = torch.load(REPO / ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"] if "model" in state else state)
    tower = TextTower(model.eval()).eval()

    ids, mask = tokenizer.encode(PARITY_QUERIES[0], max_len)
    sample = (torch.tensor([ids]), torch.tensor([mask]))
    dst = OUT / f"text_encoder_{label}.onnx"
    dst.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        tower, sample, str(dst),
        input_names=["token_ids", "pad_mask"], output_names=["embedding"],
        opset_version=17, dynamo=False,
    )

    session = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    worst = 0.0
    for query in PARITY_QUERIES:
        ids, mask = tokenizer.encode(query, max_len)
        ids_np = np.array([ids], dtype=np.int64)
        mask_np = np.array([mask], dtype=np.int64)
        with torch.no_grad():
            expected = tower(torch.tensor(ids_np), torch.tensor(mask_np)).numpy()
        got = session.run(None, {"token_ids": ids_np, "pad_mask": mask_np})[0]
        assert np.isfinite(got).all(), f"{label}: non-finite output for {query!r}"
        worst = max(worst, float(np.abs(expected - got).max()))
    assert worst < 1e-4, f"{label}: ONNX output differs from PyTorch by {worst}"
    print(f"{label}: {dst.name}, {dst.stat().st_size / 1e6:.1f} MB, "
          f"max parity error {worst:.2e}")
    quantize(label, dst, tokenizer, max_len)


def quantize(label: str, fp32: Path, tokenizer: CaptionTokenizer, max_len: int) -> None:
    """int8 weights cut the download 4x. Verify that ranking barely moves:
    a demo that silently returns different images is not worth the bytes."""
    int8 = fp32.with_suffix(".int8.onnx")
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)

    images = np.load(REPO / "demo" / "assets" / f"image_emb_{label}.npy").astype(np.float32)
    a = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])

    with open(REPO / "data" / "coco" / "annotations" / "captions_val2017.json") as f:
        captions = [x["caption"] for x in json.load(f)["annotations"]]
    rng = random.Random(0)
    queries = rng.sample(captions, 200)

    agree_top1, overlap = 0, 0.0
    for query in queries:
        ids, mask = tokenizer.encode(query, max_len)
        feeds = {"token_ids": np.array([ids], dtype=np.int64),
                 "pad_mask": np.array([mask], dtype=np.int64)}
        ra = (a.run(None, feeds)[0][0] @ images.T).argsort()[::-1][:8]
        rb = (b.run(None, feeds)[0][0] @ images.T).argsort()[::-1][:8]
        agree_top1 += int(ra[0] == rb[0])
        overlap += len(set(ra) & set(rb)) / 8

    top1 = 100 * agree_top1 / len(queries)
    assert top1 >= 95, f"{label}: int8 changes top-1 on {100 - top1:.0f}% of queries"
    print(f"{label}: {int8.name}, {int8.stat().st_size / 1e6:.1f} MB, "
          f"top-1 unchanged on {top1:.1f}% of 200 captions, "
          f"top-8 overlap {100 * overlap / len(queries):.1f}%")


def main() -> None:
    cfg = load_config(REPO / MODELS[0][1])
    tokenizer = CaptionTokenizer(str(REPO / cfg["tokenizer"]["path"]))
    for label, cfg_path, ckpt_path in MODELS:
        export_one(label, cfg_path, ckpt_path, tokenizer)


if __name__ == "__main__":
    main()
