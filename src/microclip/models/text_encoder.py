"""From-scratch Transformer text encoder (the CS230/C5 showcase piece).

Design decisions (see PROPOSAL.md §3):
- Pre-norm blocks (LayerNorm before attention/FFN) — more stable for
  from-scratch training than post-norm.
- Bidirectional attention (NO causal mask — this is an encoder, not an LM).
  Padding positions must still be masked out.
- Learned positional embeddings (max_len is small and fixed).
- Pooling: take the hidden state at the EOS token position (CLIP-style),
  then project to embed_dim.

>>> IMPLEMENT-ME: the classes below are intentionally skeletal. Writing
>>> MultiHeadAttention and the block wiring yourself is the point of the
>>> project. Shape comments give you the contract; tests/test_text_encoder.py
>>> checks it. No nn.MultiheadAttention, no nn.TransformerEncoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data.tokenizer import EOS


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        assert d_model % heads == 0
        self.heads = heads
        self.d_head = d_model // heads
        # TODO: q/k/v projections (one fused nn.Linear(d_model, 3*d_model) is
        # fine), output projection, dropout.
        raise NotImplementedError

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model); pad_mask: (B, L) with 1=real token, 0=pad.
        Returns (B, L, d_model).

        TODO:
        1. Project to q, k, v; reshape to (B, heads, L, d_head).
        2. scores = q @ k^T / sqrt(d_head)            -> (B, heads, L, L)
        3. Mask pad *keys*: scores where pad_mask[:, None, None, :] == 0
           get -inf BEFORE softmax. (Common bug: masking queries instead of
           keys, or masking after softmax.)
        4. softmax over the last dim, dropout, @ v, merge heads, out proj.
        """
        raise NotImplementedError


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        # TODO: pre-norm wiring —
        #   x = x + attn(ln1(x), pad_mask)
        #   x = x + ffn(ln2(x))        ffn: Linear -> GELU -> Linear (+dropout)
        raise NotImplementedError

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TextEncoder(nn.Module):
    """(B, L) token ids + (B, L) pad mask -> (B, embed_dim) text features."""

    def __init__(self, vocab_size: int, embed_dim: int, layers: int,
                 d_model: int, heads: int, ffn_mult: int, dropout: float,
                 max_len: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        # TODO: nn.ModuleList of TransformerBlock, final LayerNorm,
        # projection Linear(d_model, embed_dim, bias=False).
        raise NotImplementedError

    def forward(self, token_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """TODO:
        1. x = token_emb(token_ids) + pos_emb[:, :L]
        2. run blocks with pad_mask
        3. final LayerNorm
        4. pool: for each sequence, take the hidden state at the position of
           the EOS token (hint: token_ids == EOS gives you the index; there is
           exactly one EOS per sequence by construction of the tokenizer).
        5. project to embed_dim. Do NOT L2-normalize here — MicroCLIP does it.
        """
        raise NotImplementedError
