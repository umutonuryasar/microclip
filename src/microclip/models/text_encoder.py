"""From-scratch Transformer text encoder (the CS230/C5 showcase piece).

Design decisions (see PROPOSAL.md §3):
- Pre-norm blocks (LayerNorm before attention/FFN) — more stable for
  from-scratch training than post-norm.
- Bidirectional attention (NO causal mask — this is an encoder, not an LM).
  Padding positions must still be masked out.
- Learned positional embeddings (max_len is small and fixed).
- Pooling: take the hidden state at the EOS token position (CLIP-style),
  then project to embed_dim.

No nn.MultiheadAttention, no nn.TransformerEncoder — attention and block
wiring are written by hand. Contract checked by tests/test_text_encoder.py.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..data.tokenizer import EOS


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        assert d_model % heads == 0
        self.heads = heads
        self.d_head = d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model); pad_mask: (B, L) with 1=real token, 0=pad.
        Returns (B, L, d_model)."""
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, heads, L, d_head)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, heads, L, L)
        # Mask pad *keys* before softmax: no query may attend to a pad position.
        scores = scores.masked_fill(pad_mask[:, None, None, :] == 0, float("-inf"))
        attn = self.attn_drop(scores.softmax(dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)  # merge heads
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), pad_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class TextEncoder(nn.Module):
    """(B, L) token ids + (B, L) pad mask -> (B, embed_dim) text features."""

    def __init__(self, vocab_size: int, embed_dim: int, layers: int,
                 d_model: int, heads: int, ffn_mult: int, dropout: float,
                 max_len: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, heads, ffn_mult, dropout) for _ in range(layers))
        self.ln_final = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, embed_dim, bias=False)

    def forward(self, token_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        B, L = token_ids.shape
        x = self.token_emb(token_ids) + self.pos_emb[:, :L]
        for block in self.blocks:
            x = block(x, pad_mask)
        x = self.ln_final(x)
        # Pool the hidden state at the EOS position (exactly one per sequence
        # by construction of the tokenizer).
        eos_pos = (token_ids == EOS).int().argmax(dim=1)  # (B,)
        pooled = x[torch.arange(B, device=x.device), eos_pos]  # (B, d_model)
        return self.proj(pooled)  # NOT normalized — MicroCLIP does it
