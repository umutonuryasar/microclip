# MicroCLIP — Project Proposal

*Self-paced study following the CS230 & CS231n final project format:
Proposal → Milestone → Final Report. Not officially enrolled in either course.*
*Author: Umut Onur Yaşar · Date: 2026-07*

## 1. Problem statement

Contrastive vision-language pretraining (CLIP, ALIGN) canonically relies on very
large batches (32K in CLIP) because the softmax InfoNCE objective's negative pool
is the batch itself. Practitioners on single-GPU budgets cannot access this
regime. SigLIP (Zhai et al., 2023) replaces the softmax with a pairwise sigmoid
loss and reports much weaker batch-size dependence — but at web scale.

**Research question:** at *small* scale (~118K image–text pairs, batch ≤ 512,
single A100), does the sigmoid loss retain its advantage over softmax InfoNCE,
and how does the loss choice interact with batch size, relative to standard
optimization choices (optimizer, LR schedule, init)?

## 2. Motivation / relation to prior work

- CLIP (Radford et al., 2021, arXiv:2103.00020) — softmax contrastive dual encoder.
- SigLIP (Zhai et al., 2023, arXiv:2303.15343) — sigmoid pairwise loss, batch-robust.
- Personal thread: in my RT-DETR-KD work I found that applying softmax-KL
  distillation to sigmoid-trained detection logits mismatches the distribution
  family. This project extends the same "distribution family matters" theme to
  contrastive pretraining.

## 3. Method

**Architecture (dual encoder):**
- Image encoder: ResNet-18 (random init, no pretraining), projection to d=256.
- Text encoder: 4-layer pre-norm Transformer written from scratch
  (d_model=256, 4 heads, FFN 4×, max_len=64, bidirectional, EOS-token pooling).
- Tokenizer: 16K BPE trained on COCO captions (HF `tokenizers`).
- Shared embedding dim 256, L2-normalized, learnable temperature
  (+ learnable bias for the sigmoid loss, init t'=log 10, b=−10 per SigLIP).

**Training recipe (baseline):** AdamW (lr 5e-4, wd 0.2, betas 0.9/0.98),
2K-step linear warmup + cosine decay, bf16 AMP, batch 512, 30 epochs,
grad clip 1.0. Augmentation: RandomResizedCrop(224, scale 0.8–1.0) + hflip.

**Data:** COCO Captions train2017 (~118K images × 5 captions; one caption
sampled per image per epoch). Val: COCO val2017 captions.

## 4. Experiments

1. **Main:** sigmoid vs softmax at batch {128, 256, 512} → 6 runs, full schedule.
2. **Optimization ablations** (short 10-epoch runs, sigmoid b512 base):
   AdamW vs SGD+momentum; warmup+cosine vs constant LR; default vs Xavier vs He init.
3. **Architecture ablation:** ViT-Tiny image encoder vs ResNet-18 (1 run).

## 5. Evaluation

- Zero-shot classification: CIFAR-10 / CIFAR-100, prompt "a photo of a {class}".
- Retrieval: COCO 5K test and Flickr30k 1K test, R@1 / R@5 / R@10 (I→T and T→I).
- Success criterion: clean, reproducible ablation trends; CIFAR-10 zero-shot in
  the 55–70% band would match compute-normalized expectations.

## 6. Compute & logistics

- Training: Colab A100 (checkpoint-resume mandatory; sessions are preemptible).
- Inference / demo: RTX 3050 4GB (HF Space retrieval demo).
- Logging: Weights & Biases.

## 7. Timeline

| Week | Deliverable |
|---|---|
| 1 | Repo, tokenizer, encoders, losses, smoke test on 1K samples, this proposal |
| 2 | Main runs (6), MILESTONE.md with first curves |
| 3 | Optimization + ViT ablations |
| 4 | Final report (README results), HF Space demo, blog post |

## 8. Risks

- **Colab preemption** → checkpoint-resume from day one.
- **Modest absolute scores** → framed up front as compute-normalized study.
- **Softmax runs degenerating at small batch** → that *is* a result; report it.

---

## Postscript (2026-09) — what changed between proposal and result

Kept for the record; the [README](../README.md) is the final report.

- **Research question answered in the negative.** At this scale softmax InfoNCE
  matched or beat sigmoid at every batch size, and the gap widened as batch
  shrank. Reported as a non-replication of SigLIP's small-batch advantage, not
  a refutation.
- **Flickr30k retrieval was dropped.** Only COCO 5K val retrieval was run.
- **CIFAR-10 zero-shot never reached the 55–70% band.** Results are near chance
  and seed-noise dominated; they are reported but not used for any claim.
- **HF Space demo and blog post were not built.**
- **Seeds:** the six main runs became 3-seed for b128/b512 (42/43/44); b256 and
  all ablations stayed single-seed.
