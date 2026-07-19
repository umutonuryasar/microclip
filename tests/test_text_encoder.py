"""Contract tests for the from-scratch text encoder. Fail until implemented."""
import torch

from microclip.models.text_encoder import TextEncoder


def _encoder():
    return TextEncoder(vocab_size=100, embed_dim=32, layers=2, d_model=64,
                       heads=4, ffn_mult=4, dropout=0.0, max_len=16)


def _batch(b=4, L=16):
    # ids: [BOS]=1, tokens, [EOS]=2, [PAD]=0
    ids = torch.zeros(b, L, dtype=torch.long)
    mask = torch.zeros(b, L, dtype=torch.long)
    for i in range(b):
        n = 5 + i  # varying lengths
        ids[i, 0] = 1
        ids[i, 1:n] = torch.randint(4, 100, (n - 1,))
        ids[i, n] = 2
        mask[i, : n + 1] = 1
    return ids, mask


def test_output_shape():
    enc = _encoder()
    ids, mask = _batch()
    out = enc(ids, mask)
    assert out.shape == (4, 32)


def test_padding_invariance():
    """The classic masking bug detector: changing PAD token ids must not
    change the output. If this fails, pad keys are leaking into attention."""
    enc = _encoder().eval()
    ids, mask = _batch()
    out1 = enc(ids, mask)
    ids2 = ids.clone()
    ids2[mask == 0] = 99  # garbage in pad positions
    out2 = enc(ids2, mask)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_gradients_flow_to_embeddings():
    enc = _encoder()
    ids, mask = _batch()
    enc(ids, mask).sum().backward()
    assert enc.token_emb.weight.grad is not None
