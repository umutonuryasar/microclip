"""MicroCLIP demo — text-to-image search over COCO val2017, two losses side by side.

Everything heavy is precomputed by scripts/precompute_demo.py: image embeddings
and thumbnails ship with the Space. A query costs one text-encoder pass plus a
matrix multiply, which is milliseconds on a free CPU box.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from microclip.config import load_config
from microclip.data.tokenizer import CaptionTokenizer
from microclip.models.microclip import MicroCLIP

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
TOP_K = 8

MODELS = {
    "softmax": {
        "title": "Softmax — InfoNCE (CLIP)",
        "config": ASSETS / "configs" / "softmax_b512.yml",
        "weights": ASSETS / "softmax_b512_s42.safetensors",
        "embeddings": ASSETS / "image_emb_softmax.npy",
    },
    "sigmoid": {
        "title": "Sigmoid — SigLIP",
        "config": ASSETS / "configs" / "sigmoid_b512.yml",
        "weights": ASSETS / "sigmoid_b512_s42.safetensors",
        "embeddings": ASSETS / "image_emb_sigmoid.npy",
    },
}

EXAMPLES = [
    "a man riding a surfboard on a wave",
    "two dogs playing in the snow",
    "a plate of pizza on a wooden table",
    "a red double decker bus on a city street",
    "a giraffe standing next to a tree",
    "a child holding an umbrella in the rain",
    "a kitchen with white cabinets and a window",
    "a tennis player about to hit the ball",
]


@lru_cache(maxsize=1)
def load_index() -> list[dict]:
    with open(ASSETS / "index.json") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_tokenizer() -> CaptionTokenizer:
    return CaptionTokenizer(str(ASSETS / "tokenizer" / "bpe16k.json"))


@lru_cache(maxsize=2)
def load_model(key: str) -> tuple[MicroCLIP, dict]:
    spec = MODELS[key]
    cfg = load_config(spec["config"])
    tokenizer = load_tokenizer()
    model = MicroCLIP(cfg, vocab_size=tokenizer.vocab_size,
                      max_len=cfg["data"]["max_text_len"])
    model.load_state_dict(load_file(spec["weights"]))
    return model.eval(), cfg


@lru_cache(maxsize=2)
def load_embeddings(key: str) -> torch.Tensor:
    arr = np.load(MODELS[key]["embeddings"])
    return torch.from_numpy(arr).float()  # stored fp16, compared in fp32


@torch.no_grad()
def search(key: str, query: str) -> list[tuple[str, str]]:
    """Returns (thumbnail path, caption) pairs for the gallery."""
    model, cfg = load_model(key)
    tokenizer = load_tokenizer()
    ids, mask = tokenizer.encode(query, cfg["data"]["max_text_len"])
    feat = F.normalize(
        model.text_encoder(torch.tensor([ids]), torch.tensor([mask])), dim=-1)
    sims = (feat @ load_embeddings(key).t()).squeeze(0)
    scores, idx = sims.topk(TOP_K)
    index = load_index()
    return [
        (str(ASSETS / "thumbs" / index[i]["file"]),
         f"{s:.3f} — {index[i]['caption']}")
        for s, i in zip(scores.tolist(), idx.tolist())
    ]


def describe_tokens(query: str) -> str:
    """Show how the from-scratch BPE tokenizer splits the query."""
    tokenizer = load_tokenizer()
    encoding = tokenizer.tok.encode(query)
    pieces = [t.replace("Ġ", "_") for t in encoding.tokens]
    return (f"**{len(pieces)} tokens** (plus [BOS] and [EOS]): "
            + " ".join(f"`{p}`" for p in pieces))


def run(query: str):
    query = (query or "").strip()
    if not query:
        return [], [], "Type a query to see how it is tokenized."
    return search("softmax", query), search("sigmoid", query), describe_tokens(query)


INTRO = """
# MicroCLIP

A CLIP-style image–text model **built from scratch** on a single-GPU budget:
the tokenizer, both encoders, the attention, the two contrastive losses, the
training loop and the evaluation are all hand-written. No pretrained weights.

Search 5,000 COCO val2017 images by description. The same query runs through
**two models trained identically except for the loss function**, which is the
experiment this project was built to run.
"""

HOW = """
### The pipeline

Your text is split by a 16K BPE tokenizer trained on COCO captions, then read by
a 4-layer Transformer encoder written by hand, with no framework attention
module. The hidden state at the end-of-sequence position becomes the sentence
vector. Images went through a ResNet-18 trained from random init. Both sides
project into the same 256-dimensional space and are L2-normalized, so a cosine
similarity ranks them. Image embeddings are precomputed, so a search is one
text pass plus a dot product.

### The experiment

SigLIP reports that a sigmoid loss beats softmax at small batch sizes. This
project tested that at small scale: identical architecture and data, 3 seeds,
batch sizes 128 to 512. **Softmax won at every batch size**, and its lead grew
as the batch shrank, the opposite of the hypothesis. That is a non-replication
at small scale, not a refutation. SigLIP's advantage is reported at far larger
batch and data scales than anything here.

| | Softmax | Sigmoid |
|---|---|---|
| Image→text recall@10 | 48.2% | 45.6% |
| Text→image recall@1 | 10.5% | 10.0% |
| Learned temperature | 18.7 | 13.3 |

Both are batch-512, seed-42, 30 epochs on COCO train2017.

### What to expect

Absolute quality is modest by design. This is an 18.6M parameter model trained
from scratch on 118K images, not a production retrieval system. Around half of
image→text queries land a correct match in the top 10. It knows only COCO, so
COCO-like scenes work best and unusual queries will return loose matches.
"""


with gr.Blocks(title="MicroCLIP") as demo:
    gr.Markdown(INTRO)

    with gr.Row():
        query = gr.Textbox(label="Search COCO val2017", scale=4,
                           placeholder="a man riding a surfboard on a wave")
        button = gr.Button("Search", variant="primary", scale=1)

    gr.Examples(examples=EXAMPLES, inputs=query)
    tokens = gr.Markdown()

    with gr.Row():
        with gr.Column():
            gr.Markdown(f"### {MODELS['softmax']['title']}")
            softmax_gallery = gr.Gallery(columns=4, height=420,
                                         show_label=False, object_fit="cover")
        with gr.Column():
            gr.Markdown(f"### {MODELS['sigmoid']['title']}")
            sigmoid_gallery = gr.Gallery(columns=4, height=420,
                                         show_label=False, object_fit="cover")

    gr.Markdown("Each caption shows the cosine similarity and the image's "
                "ground-truth COCO caption.")

    with gr.Accordion("How it works, and how well", open=False):
        gr.Markdown(HOW)

    gr.Markdown(
        "Code and full results: "
        "[github.com/umutonuryasar/microclip](https://github.com/umutonuryasar/microclip) · "
        "Images: [COCO](https://cocodataset.org) val2017, captions CC BY 4.0."
    )

    outputs = [softmax_gallery, sigmoid_gallery, tokens]
    button.click(run, inputs=query, outputs=outputs)
    query.submit(run, inputs=query, outputs=outputs)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
