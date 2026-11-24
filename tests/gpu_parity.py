"""GPU-only check: the FLASH SDPA path of diff_core in bf16 matches an fp32 naive reference, and the
flash kernel is actually usable for these shapes (forced backend, so a silent math fallback fails)."""
import os
import sys

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import diff_core  # noqa: E402
from test_diffattn import naive_diff  # noqa: E402

torch.manual_seed(0)
B, H, T, hd = 4, 4, 1024, 64
q = torch.randn(B, 2 * H, T, hd, device="cuda")
k = torch.randn(B, 2 * H, T, hd, device="cuda")
v = torch.randn(B, H, T, 2 * hd, device="cuda")
lam = torch.tensor(0.6, device="cuda")
ref = naive_diff(q, k, v, 0.6)
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = diff_core(q.bfloat16(), k.bfloat16(), v.bfloat16(), lam).float()
err = (out - ref).abs().max().item()
scale = ref.abs().max().item()
print(f"flash bf16 vs fp32 naive: max abs err {err:.4g} (ref max {scale:.3g})", flush=True)
assert err < 0.03 * max(scale, 1.0), err
print("GPU PARITY OK")
