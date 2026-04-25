from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantization import BitLinear, RMSNorm


class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, use_ternary: bool = True):
        super().__init__()
        Linear = BitLinear if use_ternary else (lambda i, o, **kw: nn.Linear(i, o, bias=False))
        self.gate = Linear(hidden_size, intermediate_size)
        self.up = Linear(hidden_size, intermediate_size)
        self.down = Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class _MixerBase(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, norm_eps: float = 1e-6, use_ternary: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = max(1, num_heads)
        self.head_dim = hidden_size // self.num_heads
        Linear = BitLinear if use_ternary else (lambda i, o, **kw: nn.Linear(i, o, bias=False))
        self.in_proj = Linear(hidden_size, hidden_size * 3)
        self.gate = Linear(hidden_size, hidden_size)
        self.out_proj = Linear(hidden_size, hidden_size)
        self.norm = RMSNorm(hidden_size, eps=norm_eps)

    def _split(self, x: torch.Tensor):
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        return q, F.normalize(k.float(), dim=-1).to(k.dtype), v


class GatedDeltaNetLayer(_MixerBase):
    """Vectorized causal delta-style mixer; no recurrent Python token loop."""
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, norm_eps: float = 1e-6, chunk_size: int = 256, use_ternary: bool = True):
        super().__init__(hidden_size, num_heads, head_dim, norm_eps, use_ternary)
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._split(x)
        beta = torch.sigmoid(self.gate(x))
        mixed = torch.cumsum(v * beta, dim=1)
        denom = torch.arange(1, x.size(1) + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        y = mixed / denom
        y = y * torch.sigmoid(q * k)
        return self.out_proj(self.norm(y))


class MLSTMLayer(_MixerBase):
    def __init__(self, hidden_size: int, num_heads: int, memory_size: int = 64, norm_eps: float = 1e-6, use_ternary: bool = True):
        super().__init__(hidden_size, num_heads, hidden_size // max(1, num_heads), norm_eps, use_ternary)
        Linear = BitLinear if use_ternary else (lambda i, o, **kw: nn.Linear(i, o, bias=False))
        self.ifog = Linear(hidden_size, hidden_size * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._split(x)
        i, f, o = self.ifog(x).chunk(3, dim=-1)
        i = torch.sigmoid(i)
        # stable cumulative forget approximation: recent tokens weighted more
        pos = torch.arange(x.size(1), device=x.device, dtype=x.dtype).view(1, -1, 1)
        decay = torch.exp(-F.softplus(f.float()).to(x.dtype).clamp_max(8) * (x.size(1) - 1 - pos) / max(1, x.size(1)))
        mem = torch.cumsum(i * decay * v, dim=1) / torch.cumsum(i * decay + 1e-6, dim=1)
        y = mem * torch.sigmoid(o) * torch.sigmoid(q * k)
        return self.out_proj(self.norm(y))


class TitansMACLayer(_MixerBase):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, memory_depth: int = 2, persistent_slots: int = 64, local_window: int = 1024, norm_eps: float = 1e-6, use_ternary: bool = True):
        super().__init__(hidden_size, num_heads, head_dim, norm_eps, use_ternary)
        self.memory_depth = memory_depth
        self.local_window = local_window
        self.persistent = nn.Parameter(torch.zeros(max(1, persistent_slots), hidden_size))
        nn.init.normal_(self.persistent, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._split(x)
        surprise = (v - v.mean(dim=1, keepdim=True)).abs()
        gate = torch.sigmoid(self.gate(surprise))
        global_mem = (v * gate).mean(dim=1, keepdim=True)
        p = self.persistent.mean(dim=0).view(1, 1, -1).to(x.dtype)
        y = v + global_mem + p
        return self.out_proj(self.norm(y * torch.sigmoid(q * k)))


class TSPSpanKnotLayer(_MixerBase):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, norm_eps: float = 1e-6, chunk_size: int = 256, use_ternary: bool = True):
        super().__init__(hidden_size, num_heads, head_dim, norm_eps, use_ternary)
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._split(x)
        # span knot: chunk-level summary plus causal prefix signal
        c = min(max(1, self.chunk_size), x.size(1))
        pooled = F.avg_pool1d(v.transpose(1, 2), kernel_size=c, stride=1, padding=c // 2).transpose(1, 2)[:, :x.size(1)]
        prefix = torch.cumsum(v, dim=1) / torch.arange(1, x.size(1) + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        y = 0.5 * pooled + 0.5 * prefix
        return self.out_proj(self.norm(y * torch.sigmoid(q + k)))
