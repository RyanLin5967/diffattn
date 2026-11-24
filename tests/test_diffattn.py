import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import GPT, GPTConfig, DiffAttention, diff_core, lambda_init_fn  # noqa: E402


def naive_softmax_attn(q, k):
    T = q.shape[-2]
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    return torch.softmax(s.masked_fill(~mask, float("-inf")), dim=-1)


def naive_diff(q, k, v, lam):
    """Literal transcription of the paper / official code, independent of diff_core."""
    B, H2, T, hd = q.shape
    A = naive_softmax_attn(q, k).view(B, H2 // 2, 2, T, T)
    return (A[:, :, 0] - lam * A[:, :, 1]) @ v


def rand_qkv(B=2, H=3, T=17, hd=8, dtype=torch.float64):
    g = torch.Generator().manual_seed(0)
    q = torch.randn(B, 2 * H, T, hd, generator=g, dtype=dtype)
    k = torch.randn(B, 2 * H, T, hd, generator=g, dtype=dtype)
    v = torch.randn(B, H, T, 2 * hd, generator=g, dtype=dtype)
    return q, k, v


def test_lambda_zero_is_standard_attention():
    q, k, v = rand_qkv()
    out = diff_core(q, k, v, torch.tensor(0.0, dtype=q.dtype))
    q1 = q.view(2, 3, 2, 17, 8)[:, :, 0]
    k1 = k.view(2, 3, 2, 17, 8)[:, :, 0]
    ref = naive_softmax_attn(q1, k1) @ v
    torch.testing.assert_close(out, ref, rtol=1e-10, atol=1e-10)


def test_matches_naive_for_nonzero_lambda():
    q, k, v = rand_qkv()
    for lam in (0.2, 0.8, -0.3):
        out = diff_core(q, k, v, torch.tensor(lam, dtype=q.dtype))
        torch.testing.assert_close(out, naive_diff(q, k, v, lam), rtol=1e-10, atol=1e-10)


def test_lambda_one_identical_maps_cancel():
    q, k, v = rand_qkv()
    q = q.view(2, 3, 2, 17, 8)
    q[:, :, 1] = q[:, :, 0]
    k = k.view(2, 3, 2, 17, 8)
    k[:, :, 1] = k[:, :, 0]
    out = diff_core(q.reshape(2, 6, 17, 8), k.reshape(2, 6, 17, 8), v, torch.tensor(1.0, dtype=v.dtype))
    assert out.abs().max().item() < 1e-12


def test_shapes_and_param_match():
    cfg = dict(n_layer=2, n_head=4, d_model=64, mlp_hidden=176, vocab_size=97, seq_len=32)
    m = {a: GPT(GPTConfig(arch=a, **cfg)) for a in ("std", "qknorm", "diff")}
    x = torch.randint(0, 97, (3, 32))
    for a, mod in m.items():
        assert mod(x).shape == (3, 32, 97), a
    da = m["diff"].blocks[0].attn
    assert da.h == 2 and da.hd == 16  # d / (n_head/2) / 2
    y = da(torch.randn(3, 32, 64), m["diff"].cos, m["diff"].sin)
    assert y.shape == (3, 32, 64)
    # matmul params identical; extra params are only norms / lambdas
    mm = {a: sum(p.numel() for p in mod.parameters() if p.dim() == 2) for a, mod in m.items()}
    assert mm["std"] == mm["qknorm"] == mm["diff"]


def test_full_size_param_counts_within_1pct():
    with torch.device("meta"):
        n = {a: GPT(GPTConfig(arch=a)).n_params() for a in ("std", "qknorm", "diff")}
    assert n["std"] >= 350e6
    for a in n:
        assert abs(n[a] - n["std"]) / n["std"] < 0.01, n


def test_lambda_init_schedule():
    assert abs(lambda_init_fn(0) - 0.2) < 1e-12
    assert abs(lambda_init_fn(23) - (0.8 - 0.6 * math.exp(-6.9))) < 1e-12
    cfg = GPTConfig(arch="diff", n_layer=3, n_head=4, d_model=64, mlp_hidden=176, vocab_size=97, seq_len=16)
    torch.manual_seed(0)
    a = DiffAttention(cfg, 2)
    with torch.no_grad():
        for p in (a.lambda_q1, a.lambda_k1, a.lambda_q2, a.lambda_k2):
            p.zero_()
    assert abs(a.lam().item() - lambda_init_fn(2)) < 1e-6  # exp(0)-exp(0)+init


def test_gradcheck_core():
    q, k, v = rand_qkv(B=1, H=2, T=5, hd=4)
    lam = torch.tensor(0.37, dtype=torch.float64)
    for t in (q, k, v, lam):
        t.requires_grad_(True)
    assert torch.autograd.gradcheck(diff_core, (q, k, v, lam))


def test_gradcheck_tiny_model_layer():
    cfg = GPTConfig(arch="diff", n_layer=2, n_head=4, d_model=16, mlp_hidden=32, vocab_size=11, seq_len=6)
    torch.manual_seed(0)
    m = GPT(cfg).double()
    x = torch.randn(1, 6, 16, dtype=torch.float64, requires_grad=True)
    blk = m.blocks[1]
    assert torch.autograd.gradcheck(lambda z: blk(z, m.cos.double(), m.sin.double()), (x,))
    # lambda params get gradient through the loss
    idx = torch.randint(0, 11, (2, 6))
    m(idx, idx).backward()
    assert blk.attn.lambda_q1.grad is not None and blk.attn.lambda_q1.grad.abs().sum() > 0


def test_causality():
    cfg = GPTConfig(arch="diff", n_layer=2, n_head=4, d_model=32, mlp_hidden=64, vocab_size=50, seq_len=12)
    torch.manual_seed(0)
    m = GPT(cfg).double()
    x = torch.randint(0, 50, (1, 12))
    y1 = m(x)
    x2 = x.clone()
    x2[0, 8:] = (x2[0, 8:] + 1) % 50
    y2 = m(x2)
    torch.testing.assert_close(y1[:, :8], y2[:, :8])
