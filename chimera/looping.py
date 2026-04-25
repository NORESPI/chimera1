"""
Chimera 5.1 — Parcae Looping (Prelude/Loop/Coda) — CPU-Optimized
- torch.compile compatible (no numpy dependency in forward)
- Deterministic loop count (compatible with gradient checkpointing)
- Stable ZOH diagonal injection with fused exp
- Backward truncation: detach early iterations to save compute
arxiv:2604.12946
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParcaeInjection(nn.Module):
    """ZOH-stable diagonal injection: h' = exp(Δ·A)·h + Δ·B·e"""
    __constants__ = ['hidden_size']

    def __init__(self, hidden_size: int):
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(hidden_size))
        self.B_raw = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.delta = nn.Parameter(torch.ones(hidden_size) * 0.5)
        self.log_A._no_weight_decay = True

    def forward(self, h_prev: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        neg_A = self.delta * self.log_A.exp().neg()
        A_bar = neg_A.exp()
        B_bar = self.delta * self.B_raw
        return A_bar * h_prev + B_bar * e


class ParcaeLoopController(nn.Module):
    """Parcae prelude/loop/coda controller.
    
    Deterministic loop count during training (fixed at loop_default)
    to ensure gradient checkpointing recomputation consistency.
    Stochastic depth is applied via the stochastic_depth flag only
    when gradient checkpointing is OFF.
    """
    __constants__ = ['loop_min', 'loop_max', 'loop_default', 'exit_threshold']

    def __init__(self, hidden_size: int, loop_range: tuple = (1, 6),
                 loop_default: int = 2, adaptive_exit_threshold: float = 0.01,
                 spectral_radius_bound: float = 1.0):
        super().__init__()
        self.injection = ParcaeInjection(hidden_size)
        self.loop_min, self.loop_max = loop_range
        self.loop_default = loop_default
        self.exit_threshold = adaptive_exit_threshold
        self.e_norm = nn.LayerNorm(hidden_size)

    def forward(self, prelude_output: torch.Tensor, loop_fn,
                num_loops=None) -> torch.Tensor:
        B, T, D = prelude_output.shape
        e = self.e_norm(prelude_output)
        h = torch.zeros_like(e)

        # Deterministic loop count (safe for gradient checkpointing recompute)
        n_loops = num_loops if num_loops is not None else self.loop_default

        if self.training:
            # Backward truncation: only backprop through last half of iterations
            n_bwd = max(1, n_loops // 2)
        else:
            n_bwd = n_loops

        for t in range(n_loops):
            h_new = self.injection(h, e)
            h_new = loop_fn(h_new)

            should_backprop = (not self.training) or (t >= n_loops - n_bwd)
            if should_backprop:
                h = h_new
            else:
                h = h_new.detach()

            # Adaptive exit (inference only)
            if not self.training and t > 0:
                delta = (h_new - h).abs().mean()
                if delta < self.exit_threshold:
                    break

        return h
