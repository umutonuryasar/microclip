#!/usr/bin/env python
"""Train the 16K BPE tokenizer on COCO captions. Run once."""
import argparse
from pathlib import Path

from microclip.config import load_config
from microclip.data.tokenizer import train_bpe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yml")
    args = p.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["data"]["root"])
    train_bpe(
        caption_files=[str(root / cfg["data"]["train_ann"])],
        vocab_size=cfg["tokenizer"]["vocab_size"],
        out_path=cfg["tokenizer"]["path"],
    )
    print(f"Tokenizer saved to {cfg['tokenizer']['path']}")


if __name__ == "__main__":
    main()
