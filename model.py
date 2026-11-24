"""GPT with three attention arms: 'std' (no QK-norm), 'qknorm', 'diff' (Differential Attention).

DiffAttn follows Ye et al. 2024 (microsoft/unilm Diff-Transformer, multihead_diffattn.py):
half the heads, each head has two q/k halves of head_dim and a v of 2*head_dim; output
(softmax(q1k1) - lambda*softmax(q2k2)) v, per-head RMSNorm (subln), times (1 - lambda_init).
lambda = exp(lq1.lk1) - exp(lq2.lk2) + lambda_init, lambda_init = 0.8 - 0.6*exp(-0.3*layer_idx).
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    arch: str = "qknorm"  # std | qknorm | diff
    vocab_size: int = 50304
    n_layer: int = 24
    n_head: int = 16  # baseline head count; diff uses n_head // 2
    d_model: int = 1024
    mlp_hidden: int = 2752  # ~8/3 * d_model, multiple of 64
    seq_len: int = 1024
    rope_base: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x):
        xf = x.to(torch.promote_types(x.dtype, torch.float32))  # fp32 for bf16 inputs, keeps fp64
        y = F.rms_norm(xf, (x.shape[-1],), eps=self.eps).type_as(x)
        return y * self.weight if self.weight is not None else y


def rope_cache(seq_len, head_dim, base, device):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    f = torch.outer(t, inv)
    return f.cos(), f.sin()  # (T, hd/2)


def apply_rope(x, cos, sin):
    # x: (B, H, T, hd); non-interleaved halves
    T = x.shape[2]
    cos, sin = cos[:T].to(x.dtype), sin[:T].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    """Standard causal MHA, optional QK-norm (RMSNorm over head_dim, per head)."""

    def __init__(self, cfg: GPTConfig, layer_idx: int, qk_norm: bool):
        super().__init__()
        self.h = cfg.n_head
        self.hd = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.hd)
            self.k_norm = RMSNorm(self.hd)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


def lambda_init_fn(layer_idx):
    return 0.8 - 0.6 * math.exp(-0.3 * layer_idx)


def diff_core(q, k, v, lam):
    """q, k: (B, 2H, T, hd) with heads (2h, 2h+1) paired; v: (B, H, T, 2hd); lam: scalar tensor.
    Returns (softmax(q1 k1^T) - lam * softmax(q2 k2^T)) @ v, shape (B, H, T, 2hd).
    One fused SDPA call over 4H heads: [q1,q1,q2,q2] x [k1,k1,k2,k2] x [v_a,v_b,v_a,v_b],
    because the flash kernel needs equal q/k and v head dims (same split as the official flash impl).
    """
    B, H2, T, hd = q.shape
    H = H2 // 2
    q = q.view(B, H, 2, T, hd)
    k = k.view(B, H, 2, T, hd)
    va, vb = v.chunk(2, dim=-1)  # (B, H, T, hd) each
    Q = torch.stack([q[:, :, 0], q[:, :, 0], q[:, :, 1], q[:, :, 1]], dim=2).view(B, 4 * H, T, hd)
    K = torch.stack([k[:, :, 0], k[:, :, 0], k[:, :, 1], k[:, :, 1]], dim=2).view(B, 4 * H, T, hd)
    V = torch.stack([va, vb, va, vb], dim=2).view(B, 4 * H, T, hd)
    O = F.scaled_dot_product_attention(Q, K, V, is_causal=True).view(B, H, 4, T, hd)
    a1 = torch.cat([O[:, :, 0], O[:, :, 1]], dim=-1)
    a2 = torch.cat([O[:, :, 2], O[:, :, 3]], dim=-1)
    return a1 - lam.to(a1.dtype) * a2


class DiffAttention(nn.Module):
    def __init__(self, cfg: GPTConfig, layer_idx: int):
        super().__init__()
        self.h = cfg.n_head // 2
        self.hd = cfg.d_model // self.h // 2
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.lambda_init = lambda_init_fn(layer_idx)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.hd).normal_(0, 0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.hd).normal_(0, 0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.hd).normal_(0, 0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.hd).normal_(0, 0.1))
        self.subln = RMSNorm(2 * self.hd)

    def lam(self):
        dt = torch.promote_types(self.lambda_q1.dtype, torch.float32)
        l1 = torch.exp(torch.sum(self.lambda_q1.to(dt) * self.lambda_k1.to(dt)))
        l2 = torch.exp(torch.sum(self.lambda_q2.to(dt) * self.lambda_k2.to(dt)))
        return l1 - l2 + self.lambda_init

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q(x).view(B, T, 2 * self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, 2 * self.h, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, 2 * self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = diff_core(q, k, v, self.lam())
        y = self.subln(y) * (1 - self.lambda_init)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_up = nn.Linear(cfg.d_model, 2 * cfg.mlp_hidden, bias=False)
        self.down = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


class Block(nn.Module):
    def __init__(self, cfg, i):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.n2 = RMSNorm(cfg.d_model)
        if cfg.arch == "diff":
            self.attn = DiffAttention(cfg, i)
        else:
            self.attn = Attention(cfg, i, qk_norm=(cfg.arch == "qknorm"))
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.mlp(self.n2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.arch in ("std", "qknorm", "diff")
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.d_model)
        hd = cfg.d_model // cfg.n_head  # rope dim is 64 in every arm
        cos, sin = rope_cache(cfg.seq_len, hd, cfg.rope_base, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        x = self.wte(idx)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        x = self.norm(x)
        logits = F.linear(x, self.wte.weight)  # tied
        if targets is None:
            return logits
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))

    def lambdas(self):
        if self.cfg.arch != "diff":
            return []
        return [b.attn.lam().item() for b in self.blocks]

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def flops_per_token(self):
        """PaLM-style fwd+bwd FLOPs/token: 6*N_matmul + 6*L*T*(d_qk + d_av).
        N_matmul includes the tied lm_head (V*d), excludes norms/lambdas.
        std/qknorm: d_qk = d, d_av = d. diff: d_qk = 2H*hd = d, d_av = 2 maps * H*2hd = 2d."""
        c = self.cfg
        n_mm = sum(p.numel() for n, p in self.named_parameters() if p.dim() == 2)
        d_av = 2 * c.d_model if c.arch == "diff" else c.d_model
        return 6 * n_mm + 6 * c.n_layer * c.seq_len * (c.d_model + d_av)

    def hw_flops_per_token(self):
        """FLOPs actually executed by our kernels: diff runs QK^T twice (4H heads of hd)."""
        c = self.cfg
        n_mm = sum(p.numel() for n, p in self.named_parameters() if p.dim() == 2)
        att = 4 * c.d_model if c.arch == "diff" else 2 * c.d_model
        return 6 * n_mm + 6 * c.n_layer * c.seq_len * att
