"""GPU-only check: the fused SDPA path of diff_core (flash/bf16 on sm80+, mem-efficient/fp16 on T4) matches an fp32 naive reference, and the
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
sm80 = torch.cuda.get_device_capability()[0] >= 8
backend = SDPBackend.FLASH_ATTENTION if sm80 else SDPBackend.EFFICIENT_ATTENTION  # T4 (sm75): no flash
half = torch.bfloat16 if sm80 else torch.float16
with sdpa_kernel(backend):
    out = diff_core(q.to(half), k.to(half), v.to(half), lam).float()
err = (out - ref).abs().max().item()
scale = ref.abs().max().item()
print(f"{backend} {half} vs fp32 naive: max abs err {err:.4g} (ref max {scale:.3g})", flush=True)
assert err < 0.03 * max(scale, 1.0), err
print("GPU PARITY OK")
