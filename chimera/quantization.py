from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def setup_cpu_runtime() -> None:
    n = os.cpu_count() or 4
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    try:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", n)))
    except Exception:
        pass


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (y.to(dtype=x.dtype) * self.weight.to(dtype=x.dtype))


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


def ternarize_weight(weight: torch.Tensor, group_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """BitNet b1.58 style abs-mean ternarization with STE-compatible math.

    Returns q in {-1,0,1} and alpha scales.  The last dimension is grouped to
    reduce quantization error; dimensions not divisible by group_size are padded
    internally and cropped back.
    """
    if weight.numel() == 0:
        return weight, weight.new_ones(())
    original_k = weight.shape[-1]
    if group_size <= 0 or group_size >= original_k:
        alpha = weight.float().abs().mean(dim=-1, keepdim=True).clamp_min(1e-6)
        scaled = weight.float() / alpha
        q = _ste_round(scaled.clamp(-1, 1)).clamp(-1, 1)
        return q.to(weight.dtype), alpha.to(weight.dtype)
    pad = (-original_k) % group_size
    w = F.pad(weight.float(), (0, pad)) if pad else weight.float()
    view = w.reshape(*w.shape[:-1], -1, group_size)
    alpha = view.abs().mean(dim=-1, keepdim=True).clamp_min(1e-6)
    q = _ste_round((view / alpha).clamp(-1, 1)).clamp(-1, 1)
    q = q.reshape(*w.shape)[..., :original_k].to(weight.dtype)
    alpha_full = alpha.expand_as(view).reshape(*w.shape)[..., :original_k].to(weight.dtype)
    return q, alpha_full


def pack_ternary(q: torch.Tensor) -> torch.Tensor:
    """Pack int/float ternary tensor (..., K) into 2-bit uint8 (4 weights/byte)."""
    q = q.detach().to(torch.int8).cpu()
    k = q.shape[-1]
    pad = (-k) % 4
    if pad:
        q = F.pad(q, (0, pad))
    codes = torch.zeros_like(q, dtype=torch.uint8)
    codes[q > 0] = 1
    codes[q < 0] = 2
    codes = codes.reshape(*codes.shape[:-1], -1, 4)
    shifts = torch.tensor([6, 4, 2, 0], dtype=torch.uint8)
    return ((codes << shifts).sum(dim=-1)).contiguous()


def unpack_ternary(packed: torch.Tensor, k: int, alpha: Optional[torch.Tensor] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    packed = packed.to(torch.uint8)
    parts = []
    for shift in (6, 4, 2, 0):
        code = ((packed >> shift) & 3).to(torch.int8)
        part = torch.zeros_like(code, dtype=dtype)
        part = torch.where(code == 1, torch.ones_like(part), part)
        part = torch.where(code == 2, -torch.ones_like(part), part)
        parts.append(part)
    out = torch.stack(parts, dim=-1).flatten(-2)[..., :k]
    if alpha is not None:
        out = out * alpha.to(out.device, out.dtype)
    return out


class BitLinear(nn.Module):
    """CPU-safe ternary linear layer.

    Training keeps a latent FP32/BF16 weight and applies STE ternarization in the
    forward pass.  `prepare_for_inference()` stores a 2-bit packed snapshot for
    low-memory serving; the pure PyTorch fallback always remains valid.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, group_size: int = 128):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.register_buffer("packed_weight", torch.empty(0, dtype=torch.uint8), persistent=False)
        self.register_buffer("packed_alpha", torch.empty(0), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(max(1, self.in_features)))
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.invalidate_packed()

    def invalidate_packed(self) -> None:
        self.packed_weight = torch.empty(0, dtype=torch.uint8, device=self.weight.device)
        self.packed_alpha = torch.empty(0, device=self.weight.device)

    @torch.no_grad()
    def prepare_for_inference(self) -> None:
        q, alpha = ternarize_weight(self.weight.detach().float(), self.group_size)
        self.packed_weight = pack_ternary(q).to(self.weight.device)
        self.packed_alpha = alpha.to(self.weight.device, torch.float32)

    def dequantized_weight(self) -> torch.Tensor:
        if (not self.training) and self.packed_weight.numel() > 0:
            return unpack_ternary(self.packed_weight, self.in_features, self.packed_alpha, dtype=self.weight.dtype).to(self.weight.device)
        q, alpha = ternarize_weight(self.weight, self.group_size)
        return q * alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantized_weight().to(dtype=x.dtype)
        return F.linear(x, w, self.bias.to(dtype=x.dtype) if self.bias is not None else None)


def apply_2_4_sparsity_(weight: torch.Tensor) -> torch.Tensor:
    """In-place N:M 2:4 pruning helper used by import/training tools."""
    with torch.no_grad():
        k = weight.shape[-1]
        pad = (-k) % 4
        w = F.pad(weight, (0, pad)) if pad else weight
        view = w.view(*w.shape[:-1], -1, 4)
        idx = view.abs().argsort(dim=-1)[..., :2]
        view.scatter_(-1, idx, 0)
        if pad:
            weight.copy_(w[..., :k])
    return weight
