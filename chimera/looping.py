from __future__ import annotations

import torch
import torch.nn as nn


class ParcaeInjection(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(hidden_size))
        self.B = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.delta = nn.Parameter(torch.ones(hidden_size) * 0.5)

    def forward(self, h_prev: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        a_bar = torch.exp(-self.delta.abs() * self.log_A.exp()).to(h_prev.dtype)
        return a_bar * h_prev + self.delta.to(h_prev.dtype) * self.B.to(h_prev.dtype) * e


class ParcaeLoopController(nn.Module):
    def __init__(self, hidden_size: int, loop_range=(1, 6), loop_default: int = 2, adaptive_exit_threshold: float = 0.01):
        super().__init__()
        self.loop_min, self.loop_max = map(int, loop_range)
        self.loop_default = int(loop_default)
        self.threshold = float(adaptive_exit_threshold)
        self.inject = ParcaeInjection(hidden_size)

    def forward(self, x: torch.Tensor, loop_fn, num_loops: int | None = None) -> torch.Tensor:
        n = self.loop_default if num_loops is None else int(num_loops)
        n = max(self.loop_min, min(self.loop_max, n))
        h = x
        for i in range(n):
            new = loop_fn(h)
            h = self.inject(h, new)
            if (not self.training) and i + 1 >= self.loop_min:
                if (new - h).float().pow(2).mean().sqrt().item() < self.threshold:
                    break
        return h
