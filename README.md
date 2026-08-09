# MicroCLIP

Training a CLIP-style vision-language model **from scratch** on a single-GPU budget.

> **Status:** implementation complete — model, losses, trainer, eval. Runs pending.
> Self-paced project in CS230 & CS231n final-project format · [Proposal](docs/PROPOSAL.md)

## What this is

A small dual-encoder VLM (ResNet-18 image encoder + a 4-layer Transformer text
encoder written from scratch) trained on COCO Captions (~118K images) with a
**SigLIP-style sigmoid contrastive loss**, evaluated via zero-shot classification
(CIFAR-10/100) and image–text retrieval (COCO 5K, Flickr30k 1K).

The central question: **under small-batch, single-GPU constraints, how much does
the choice of contrastive loss (sigmoid vs. softmax InfoNCE) matter relative to
batch size?**

This is a compute-normalized study. Absolute zero-shot numbers will be modest by
design (~100K training pairs vs. CLIP's 400M); the value is in the controlled
ablations, not leaderboard scores.

## Setup

```bash
pip install -e .
# or: pip install -r requirements.txt
```

Data: place COCO `train2017/`, `val2017/` and `annotations/captions_*.json`
under `data/coco/` (or set `data.root` in the config).

## Usage

```bash
# 1. Train the BPE tokenizer on COCO captions (one-off, ~1 min)
python scripts/train_tokenizer.py --config configs/base.yml

# 2. Smoke test on 1K samples (must pass before any real run)
python scripts/smoke_test.py --config configs/base.yml

# 3. Full training run
python scripts/train.py --config configs/sigmoid_b512.yml

# 4. Evaluation (config must match the checkpoint's training config)
python scripts/evaluate.py --checkpoint runs/<run>/best.pt --config configs/<run>.yml --task zeroshot
python scripts/evaluate.py --checkpoint runs/<run>/best.pt --config configs/<run>.yml --task retrieval
```

Training resumes automatically from `runs/<run>/last.pt` if present (Colab-proof).

## Ablation matrix

| Axis | Variants | Config |
|---|---|---|
| Loss | sigmoid (SigLIP) vs softmax (InfoNCE) | `sigmoid_b512.yml` / `softmax_b512.yml` |
| Batch size | 128 / 256 / 512 (both losses) | `ablations/bs*.yml` |
| Optimizer | AdamW vs SGD+momentum | `ablations/optimizer_sgd.yml` |
| LR schedule | warmup+cosine vs constant | `ablations/lr_constant.yml` |
| Init | default vs Xavier vs He | `ablations/init_*.yml` |
| Image encoder | ResNet-18 vs ViT-Tiny | `ablations/vit_tiny.yml` |

## Results

*(populated at milestone — all runs logged to W&B)*

## License

MIT
