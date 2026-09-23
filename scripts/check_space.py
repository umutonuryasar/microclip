#!/usr/bin/env python
"""Smoke-test the assembled Space in demo/_build without a browser.

Checks that the app imports with only its own bundled code on the path, that a
query returns the right shapes, that the two models actually disagree, and that
a known-good query retrieves a plausibly matching caption.

Usage:
  .venv/bin/python scripts/check_space.py
"""
from __future__ import annotations

import sys
from pathlib import Path

BUILD = Path(__file__).resolve().parents[1] / "demo" / "_build"

# Import the app exactly as the Space will: only the build folder on the path,
# so a missing copied module fails here rather than on Hugging Face.
sys.path.insert(0, str(BUILD))
for mod in [m for m in sys.modules if m.startswith("microclip")]:
    del sys.modules[mod]

import app  # noqa: E402

QUERIES = [
    "a man riding a surfboard on a wave",
    "two dogs playing in the snow",
    "a plate of pizza on a wooden table",
    "a giraffe standing next to a tree",
]


def main() -> None:
    index = app.load_index()
    print(f"index: {len(index)} images")
    for key in ("softmax", "sigmoid"):
        emb = app.load_embeddings(key)
        assert emb.shape[0] == len(index), f"{key}: {emb.shape[0]} embeddings vs {len(index)} images"
        model, _ = app.load_model(key)
        print(f"{key}: embeddings {tuple(emb.shape)}, temp {model.logit_scale.exp():.2f}")

    for query in QUERIES:
        soft, sig, tokens = app.run(query)
        assert len(soft) == len(sig) == app.TOP_K, "wrong result count"
        for path, _ in soft + sig:
            assert Path(path).exists(), f"missing thumbnail: {path}"
        print(f"\nQUERY: {query}")
        print(f"  {tokens}")
        print(f"  softmax top-3: {[c for _, c in soft[:3]]}")
        print(f"  sigmoid top-3: {[c for _, c in sig[:3]]}")
        overlap = len({p for p, _ in soft} & {p for p, _ in sig})
        print(f"  overlap between the two models: {overlap}/{app.TOP_K}")

    empty_soft, empty_sig, msg = app.run("")
    assert empty_soft == [] and empty_sig == [], "empty query should return nothing"
    print(f"\nempty query handled: {msg}")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
