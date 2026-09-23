---
title: MicroCLIP
emoji: 🔍
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
license: mit
---

# MicroCLIP

A CLIP-style image–text model built from scratch on a single-GPU budget. The
tokenizer, both encoders, the attention, the two contrastive losses, the
training loop and the evaluation are hand-written, with no pretrained weights.

This Space searches 5,000 COCO val2017 images by description, running every
query through two models that differ only in their loss function: softmax
(InfoNCE, as in CLIP) and sigmoid (SigLIP).

## The experiment

SigLIP reports that its sigmoid loss beats softmax at small batch sizes. Tested
here at small scale with identical architecture and data, 3 seeds and batch
sizes 128 to 512, softmax won at every batch size, and its lead grew as the
batch shrank. That is a non-replication at small scale, not a refutation:
SigLIP's advantage is reported at far larger batch and data scales.

| | Softmax | Sigmoid |
|---|---|---|
| Image→text recall@10 | 48.2% | 45.6% |
| Text→image recall@1 | 10.5% | 10.0% |

Both are batch-512, seed-42, 30 epochs on COCO train2017, 18.6M parameters.

## Limits

Absolute quality is modest by design. The model saw only COCO, so COCO-like
scenes work best. Roughly half of image→text queries land a correct match in
the top 10.

Full code, results and ablations:
[github.com/umutonuryasar/microclip](https://github.com/umutonuryasar/microclip).
Images from [COCO](https://cocodataset.org) val2017; captions CC BY 4.0.
