#!/usr/bin/env python
"""Assemble a standalone Hugging Face Space folder in demo/_build.

The Space cannot `pip install -e .` from this repo, so the model code is copied
in next to the app. Only the modules the demo imports are needed: the config
loader, the tokenizer and the encoders. Training and eval code stays out.

Usage:
  python scripts/build_space.py            # assemble demo/_build
  python scripts/build_space.py --check    # also verify nothing is missing
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / "demo" / "_build"

# Modules the app imports, directly or transitively.
PACKAGE_FILES = [
    "__init__.py",
    "config.py",
    "data/__init__.py",
    "data/tokenizer.py",
    "data/transforms.py",
    "models/__init__.py",
    "models/microclip.py",
    "models/image_encoder.py",
    "models/text_encoder.py",
]

TOP_LEVEL = ["app.py", "requirements.txt", "README.md"]

REQUIRED_ASSETS = [
    "index.json",
    "image_emb_softmax.npy",
    "image_emb_sigmoid.npy",
    "softmax_b512_s42.safetensors",
    "sigmoid_b512_s42.safetensors",
    "tokenizer/bpe16k.json",
    "configs/base.yml",
    "configs/softmax_b512.yml",
    "configs/sigmoid_b512.yml",
]


def build() -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)

    for name in TOP_LEVEL:
        shutil.copy2(REPO / "demo" / name, BUILD / name)

    for rel in PACKAGE_FILES:
        dst = BUILD / "microclip" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / "src" / "microclip" / rel, dst)

    shutil.copytree(REPO / "demo" / "assets", BUILD / "assets",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # Running the app in-place leaves bytecode caches behind; they must not
    # ship to the Space.
    for junk in list(BUILD.rglob("__pycache__")):
        shutil.rmtree(junk, ignore_errors=True)


def check() -> None:
    missing = [a for a in REQUIRED_ASSETS if not (BUILD / "assets" / a).exists()]
    thumbs = list((BUILD / "assets" / "thumbs").glob("*.jpg"))
    import json
    with open(BUILD / "assets" / "index.json") as f:
        index = json.load(f)
    if len(thumbs) != len(index):
        missing.append(f"thumbnails: {len(thumbs)} files for {len(index)} index entries")
    if missing:
        raise SystemExit("MISSING:\n  " + "\n  ".join(missing))
    size = sum(p.stat().st_size for p in BUILD.rglob("*") if p.is_file())
    print(f"ok — {len(index)} images, {len(thumbs)} thumbnails, {size / 1e6:.0f} MB total")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    build()
    print(f"built {BUILD}")
    if args.check:
        check()
