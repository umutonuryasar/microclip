#!/usr/bin/env python
"""Assemble the static (browser-only) Space in demo/_static.

Free Hugging Face Spaces host static sites but not Gradio, so the demo runs
client-side: int8 ONNX text encoders plus precomputed image embeddings. This
script converts the precomputed assets into browser-friendly files and copies
the page next to them.

Prerequisites:
  PYTHONPATH=src python scripts/precompute_demo.py   # embeddings + thumbnails
  PYTHONPATH=src python scripts/export_onnx.py       # ONNX text encoders

Usage:
  python scripts/build_static.py [--check]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "demo" / "static"
PRECOMPUTED = REPO / "demo" / "assets"
BUILD = REPO / "demo" / "_static"
LOSSES = ["softmax", "sigmoid"]

PAGE_FILES = ["index.html", "app.js", "tokenizer.js", "README.md"]

# onnxruntime-web, vendored by scripts/fetch_ort.sh. Shipping it with the page
# removes any dependency on a third-party CDN at load time.
# When wasmPaths is set, the runtime loads its glue module (.mjs) from that
# folder as well as the binary, so both must ship.
VENDOR_FILES = ["ort.wasm.bundle.min.mjs", "ort-wasm-simd-threaded.mjs",
                "ort-wasm-simd-threaded.wasm"]


def build() -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    (BUILD / "assets" / "tokenizer").mkdir(parents=True)

    for name in PAGE_FILES:
        shutil.copy2(SRC / name, BUILD / name)

    for loss in LOSSES:
        # float16 on disk saves bandwidth but every browser can read float32
        # directly into a Float32Array, so the 2x size is worth the simplicity.
        emb = np.load(PRECOMPUTED / f"image_emb_{loss}.npy").astype(np.float32)
        emb.tofile(BUILD / "assets" / f"image_emb_{loss}.bin")
        shutil.copy2(SRC / "assets" / f"text_encoder_{loss}.int8.onnx",
                     BUILD / "assets" / f"text_encoder_{loss}.int8.onnx")

    shutil.copy2(PRECOMPUTED / "index.json", BUILD / "assets" / "index.json")
    shutil.copy2(PRECOMPUTED / "tokenizer" / "bpe16k.json",
                 BUILD / "assets" / "tokenizer" / "bpe16k.json")
    shutil.copytree(PRECOMPUTED / "thumbs", BUILD / "assets" / "thumbs",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    (BUILD / "vendor" / "ort").mkdir(parents=True)
    for name in VENDOR_FILES:
        src = SRC / "vendor" / "ort" / name
        if not src.exists():
            raise SystemExit(f"{src} missing — run scripts/fetch_ort.sh first")
        shutil.copy2(src, BUILD / "vendor" / "ort" / name)


def check() -> None:
    with open(BUILD / "assets" / "index.json") as f:
        index = json.load(f)
    problems = []

    for loss in LOSSES:
        bin_path = BUILD / "assets" / f"image_emb_{loss}.bin"
        expected = len(index) * 256 * 4  # float32
        if bin_path.stat().st_size != expected:
            problems.append(f"{bin_path.name}: {bin_path.stat().st_size} bytes, expected {expected}")
        if not (BUILD / "assets" / f"text_encoder_{loss}.int8.onnx").exists():
            problems.append(f"missing text_encoder_{loss}.int8.onnx")

    thumbs = {p.name for p in (BUILD / "assets" / "thumbs").glob("*.jpg")}
    missing = [e["file"] for e in index if e["file"] not in thumbs]
    if missing:
        problems.append(f"{len(missing)} thumbnails missing, e.g. {missing[:3]}")

    for name in PAGE_FILES:
        if not (BUILD / name).exists():
            problems.append(f"missing {name}")
    for name in VENDOR_FILES:
        if not (BUILD / "vendor" / "ort" / name).exists():
            problems.append(f"missing vendor/ort/{name}")

    if problems:
        raise SystemExit("PROBLEMS:\n  " + "\n  ".join(problems))

    files = [p for p in BUILD.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    upfront = sum(p.stat().st_size for p in files if p.suffix != ".jpg")
    print(f"ok — {len(index)} images, {len(files)} files, {total / 1e6:.0f} MB total; "
          f"{upfront / 1e6:.0f} MB loaded before the first search")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    build()
    print(f"built {BUILD}")
    if args.check:
        check()
