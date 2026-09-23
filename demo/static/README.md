---
title: MicroCLIP
emoji: 🔍
colorFrom: indigo
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
---

# MicroCLIP

A CLIP-style image–text model built from scratch on a single-GPU budget. The
tokenizer, both encoders, the attention, the two contrastive losses, the
training loop and the evaluation are hand-written, with no pretrained weights.

Search 5,000 COCO val2017 images by description. Every query runs through two
models that differ only in their loss function: softmax (InfoNCE, as in CLIP)
and sigmoid (SigLIP).

Everything runs in the browser. The int8 ONNX text encoders and the precomputed
image embeddings are served as static files, so there is no server, no cold
start and no upload of anything you type.

## The experiment

SigLIP reports that its sigmoid loss beats softmax at small batch sizes. Tested
here at small scale with identical architecture and data, 3 seeds and batch
sizes 128 to 512, softmax won at every batch size, and its lead grew as the
batch shrank. That is a non-replication at small scale, not a refutation:
SigLIP's advantage is reported at far larger batch and data scales.

| COCO 5K retrieval | Softmax | Sigmoid |
|---|---|---|
| Image→text recall@10 | 48.2% | 45.6% |
| Text→image recall@1 | 10.5% | 10.0% |

Both are batch-512, seed-42, 30 epochs on COCO train2017, 18.6M parameters.

## Limits

Absolute quality is modest by design. The model has only ever seen COCO, so
COCO-like scenes work best. Roughly half of image→text queries land a correct
match in the top 10.

Full code, results and ablations:
[github.com/umutonuryasar/microclip](https://github.com/umutonuryasar/microclip).
Images from [COCO](https://cocodataset.org) val2017; captions CC BY 4.0.
