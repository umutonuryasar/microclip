"""16K BPE tokenizer trained on COCO captions (HF `tokenizers`).

Special tokens: [PAD]=0, [BOS]=1, [EOS]=2, [UNK]=3.
Text encoding: [BOS] tokens [EOS], padded/truncated to max_len.
The EOS position is what the text encoder pools (CLIP-style).
"""
from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]


def train_bpe(caption_files: list[str], vocab_size: int, out_path: str) -> None:
    """caption_files: COCO captions_*.json annotation files."""
    def caption_iter():
        for cf in caption_files:
            with open(cf) as f:
                ann = json.load(f)
            for a in ann["annotations"]:
                yield a["caption"]

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS)
    tok.train_from_iterator(caption_iter(), trainer)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(out_path)


class CaptionTokenizer:
    def __init__(self, path: str):
        self.tok = Tokenizer.from_file(path)

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str, max_len: int) -> tuple[list[int], list[int]]:
        ids = self.tok.encode(text).ids[: max_len - 2]
        ids = [BOS] + ids + [EOS]
        attn = [1] * len(ids)
        pad = max_len - len(ids)
        return ids + [PAD] * pad, attn + [0] * pad
