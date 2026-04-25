from __future__ import annotations

import torch
import torch.nn as nn

from .quantization import RMSNorm
from .layers import GatedDeltaNetLayer


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 16, in_channels: int = 3, hidden_size: int = 384):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.norm = RMSNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class VisionEncoder(nn.Module):
    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = config or {}
        h = int(cfg.get("vision_hidden_size", cfg.get("hidden_size", 384)))
        self.embed = PatchEmbed(cfg.get("patch_size", 16), cfg.get("in_channels", 3), h)
        self.layers = nn.ModuleList([GatedDeltaNetLayer(h, max(1, h // 64), 64) for _ in range(int(cfg.get("vision_layers", 2)))])

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.embed(pixel_values)
        for layer in self.layers:
            x = x + layer(x)
        return x


class AudioEncoder(nn.Module):
    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = config or {}
        h = int(cfg.get("audio_hidden_size", cfg.get("hidden_size", 384)))
        n_mels = int(cfg.get("n_mels", 80))
        self.proj = nn.Linear(n_mels, h, bias=False)
        self.norm = RMSNorm(h)
        self.layers = nn.ModuleList([GatedDeltaNetLayer(h, max(1, h // 64), 64) for _ in range(int(cfg.get("audio_layers", 2)))])

    def forward(self, mel_features: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.proj(mel_features))
        for layer in self.layers:
            x = x + layer(x)
        return x
