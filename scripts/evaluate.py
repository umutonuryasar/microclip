#!/usr/bin/env python
"""Evaluate a checkpoint: zero-shot classification or retrieval.

Usage:
  python scripts/evaluate.py --checkpoint runs/sigmoid_b512/best.pt \
      --config configs/sigmoid_b512.yml --task zeroshot
  python scripts/evaluate.py --checkpoint runs/sigmoid_b512/best.pt \
      --config configs/sigmoid_b512.yml --task retrieval
"""
import argparse
import json

import torch

from microclip.config import load_config
from microclip.data.coco_captions import CocoCaptions
from microclip.data.tokenizer import CaptionTokenizer
from microclip.data.transforms import build_transforms
from microclip.eval.retrieval import encode_corpus, recall_at_k
from microclip.eval.zeroshot import zeroshot_accuracy
from microclip.models.microclip import MicroCLIP


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True, help="config the checkpoint was trained with")
    p.add_argument("--task", required=True, choices=["zeroshot", "retrieval"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None, help="cap retrieval corpus size")
    args = p.parse_args()

    cfg = load_config(args.config)
    tokenizer = CaptionTokenizer(cfg["tokenizer"]["path"])
    model = MicroCLIP(cfg, vocab_size=tokenizer.vocab_size,
                      max_len=cfg["data"]["max_text_len"])
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(args.device).eval()

    if args.task == "zeroshot":
        from torchvision.datasets import CIFAR10, CIFAR100
        tf = build_transforms(cfg["data"]["image_size"], train=False)
        results = {}
        for name, ds_cls in [("cifar10", CIFAR10), ("cifar100", CIFAR100)]:
            ds = ds_cls("data/cifar", train=False, download=True, transform=tf)
            acc = zeroshot_accuracy(model, tokenizer, ds, ds.classes, cfg,
                                    device=args.device)
            results[f"{name}/top1"] = acc
    else:
        d = cfg["data"]
        ds = CocoCaptions(d["root"], d["val_images"], d["val_ann"], tokenizer,
                          d["image_size"], d["max_text_len"], train=False,
                          limit=args.limit)
        img_feats, txt_feats, txt2img = encode_corpus(model, ds, cfg,
                                                      device=args.device)
        results = recall_at_k(img_feats, txt_feats, txt2img)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
