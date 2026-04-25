"""
Chimera 5.1 — Multimodal Encoders (Vision + Audio) — CPU-Optimized
- GatedDeltaNet-based ternary encoders
- torch.compile friendly (no dynamic module creation in forward)
- Gradient checkpointing support per layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .quantization import BitLinear, RMSNorm
from .layers import GatedDeltaNetLayer


class PatchEmbed(nn.Module):
    __constants__ = ['patch_size']

    def __init__(self, patch_size: int = 16, in_channels: int = 3,
                 hidden_size: int = 384):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_size,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = RMSNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class _EncoderBlock(nn.Module):
    """Single encoder block — extracted as Module for checkpointing."""

    def __init__(self, hidden: int, num_heads: int, head_dim: int,
                 use_ternary: bool = True):
        super().__init__()
        self.norm = RMSNorm(hidden)
        self.attn = GatedDeltaNetLayer(hidden, num_heads, head_dim,
                                        use_ternary=use_ternary, chunk_size=64)
        self.mlp_norm = RMSNorm(hidden)
        L = BitLinear if use_ternary else lambda i, o, **kw: nn.Linear(i, o, bias=False)
        self.mlp = nn.Sequential(
            L(hidden, hidden * 4),
            nn.GELU(),
            L(hidden * 4, hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class VisionEncoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.enabled = config.get('enabled', True)
        hidden = config.get('vision', {}).get('hidden', 384)
        depth = config.get('vision', {}).get('depth', 12)
        patch = config.get('vision', {}).get('patch', 16)
        out_dim = config.get('vision', {}).get('out', 2560)
        use_ternary = config.get('vision', {}).get('quant', 'ternary') == 'ternary'

        self.patch_embed = PatchEmbed(patch_size=patch, hidden_size=hidden)
        num_heads = max(1, hidden // 64)
        head_dim = hidden // num_heads

        self.layers = nn.ModuleList([
            _EncoderBlock(hidden, num_heads, head_dim, use_ternary)
            for _ in range(depth)
        ])
        self.proj = nn.Linear(hidden, out_dim, bias=False)
        self.norm = RMSNorm(out_dim)
        self.use_checkpoint = True

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return None
        x = self.patch_embed(pixel_values)
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.norm(self.proj(x))


class AudioEncoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.enabled = config.get('enabled', True)
        hidden = config.get('audio', {}).get('hidden', 256)
        depth = config.get('audio', {}).get('depth', 6)
        out_dim = config.get('audio', {}).get('out', 2560)
        use_ternary = config.get('audio', {}).get('quant', 'ternary') == 'ternary'

        self.input_proj = nn.Linear(80, hidden, bias=False)
        num_heads = max(1, hidden // 64)
        head_dim = hidden // num_heads

        self.layers = nn.ModuleList([
            _EncoderBlock(hidden, num_heads, head_dim, use_ternary)
            for _ in range(depth)
        ])
        self.proj = nn.Linear(hidden, out_dim, bias=False)
        self.norm = RMSNorm(out_dim)
        self.use_checkpoint = True

    def forward(self, mel_features: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return None
        x = self.input_proj(mel_features)
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.norm(self.proj(x))
