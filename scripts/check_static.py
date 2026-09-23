#!/usr/bin/env python
"""Generate PyTorch reference rankings, then check the browser code against them."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from microclip.config import load_config  # noqa: E402
from microclip.data.tokenizer import CaptionTokenizer  # noqa: E402
from microclip.models.microclip import MicroCLIP  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
# onnxruntime-web for Node, installed on first use into a gitignored folder.
NODE_DIR = REPO / "demo" / ".node"
ORT_VERSION = "1.26.0"  # keep in step with scripts/fetch_ort.sh
QUERIES = [
    "a man riding a surfboard on a wave",
    "two dogs playing in the snow",
    "a plate of pizza on a wooden table",
    "a giraffe standing next to a tree",
    "a kitchen with white cabinets and a window",
    "émoji café naïve",
]


def reference() -> dict:
    out = {}
    for loss in ("softmax", "sigmoid"):
        cfg = load_config(REPO / "configs" / f"{loss}_b512.yml")
        tokenizer = CaptionTokenizer(str(REPO / cfg["tokenizer"]["path"]))
        model = MicroCLIP(cfg, tokenizer.vocab_size, cfg["data"]["max_text_len"])
        state = torch.load(REPO / "weights" / f"{loss}_b512_s42.pt",
                           map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        model.eval()
        images = torch.from_numpy(
            np.load(REPO / "demo" / "assets" / f"image_emb_{loss}.npy").astype(np.float32))
        cases = []
        for query in QUERIES:
            ids, mask = tokenizer.encode(query, cfg["data"]["max_text_len"])
            with torch.no_grad():
                feat = F.normalize(
                    model.text_encoder(torch.tensor([ids]), torch.tensor([mask])), dim=-1)
            top = (feat @ images.t()).squeeze(0).topk(8).indices.tolist()
            cases.append({"query": query, "expected": top})
        out[loss] = cases
    return out


def ensure_ort() -> Path:
    if not (NODE_DIR / "node_modules" / "onnxruntime-web").exists():
        NODE_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["npm", "install", "--silent", "--prefix", str(NODE_DIR),
                        f"onnxruntime-web@{ORT_VERSION}"], check=True)
    return NODE_DIR / "index.js"  # createRequire() resolves from this path


def main() -> None:
    require_from = ensure_ort()
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(reference(), f)
        path = f.name
    result = subprocess.run(
        ["node", str(REPO / "scripts" / "check_static_js.mjs"), path, str(require_from)])
    Path(path).unlink()
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
