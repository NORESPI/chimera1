"""
CPU-optimized Mixture-of-Experts blocks for Chimera.

Design goals for real CPU use:
- no dense [tokens, experts, hidden] materialization;
- route with torch.topk only, then group selected token/expert pairs by expert;
- expert computation is batched per expert and scattered back with index_add_;
- duplicate/tied parameters are handled by the training script, not here;
- works with BitLinear for ternary low-memory inference/training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantization import BitLinear


class SwiGLUMLP(nn.Module):
    """Expert MLP using SwiGLU and optional ternary projections."""

    __constants__ = ["hidden_size", "intermediate_size"]

    def __init__(self, hidden_size: int, intermediate_size: int, use_ternary: bool = True):
        super().__init__()
        linear = BitLinear if use_ternary else lambda i, o, **kw: nn.Linear(i, o, bias=False)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = linear(hidden_size, intermediate_size)
        self.up_proj = linear(hidden_size, intermediate_size)
        self.down_proj = linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class NoAuxMoEGate(nn.Module):
    """No-aux-loss top-k router with group-limited optional bias correction."""

    def __init__(self, hidden_size: int, n_routed_experts: int, num_experts_per_tok: int = 2):
        super().__init__()
        self.n_routed_experts = int(n_routed_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.weight = nn.Parameter(torch.empty(self.n_routed_experts, hidden_size))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(self.n_routed_experts), requires_grad=False)
        nn.init.normal_(self.weight, mean=0.0, std=hidden_size ** -0.5)

    def forward(self, x: torch.Tensor):
        # x: [N, D]. Router stays fp32 for stable top-k decisions on CPU.
        scores = F.linear(x.float(), self.weight.float())
        scores = scores + self.e_score_correction_bias
        probs = F.softmax(scores, dim=-1)
        weights, indices = torch.topk(probs, k=self.num_experts_per_tok, dim=-1, sorted=False)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return indices, weights.to(dtype=x.dtype)


class MoELayer(nn.Module):
    """Sparse CPU MoE.

    The common naive MoE implementation loops over tokens or computes every expert.
    This implementation loops only over active experts.  Selected token/expert pairs
    are sorted by expert, processed as dense mini-batches, then accumulated with
    index_add_.  This is typically much faster for CPU batch/sequence workloads.
    """

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        n_routed_experts: int = 16,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 2,
        use_ternary: bool = True,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.n_routed_experts = int(n_routed_experts)
        self.n_shared_experts = int(n_shared_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.gate = NoAuxMoEGate(hidden_size, n_routed_experts, num_experts_per_tok)
        self.experts = nn.ModuleList([
            SwiGLUMLP(hidden_size, moe_intermediate_size, use_ternary=use_ternary)
            for _ in range(n_routed_experts)
        ])
        shared_intermediate = max(1, moe_intermediate_size * max(1, n_shared_experts))
        self.shared_experts = (SwiGLUMLP(hidden_size, shared_intermediate, use_ternary=use_ternary)
                               if n_shared_experts > 0 else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        n_tokens = x_flat.shape[0]

        topk_idx, topk_weight = self.gate(x_flat)
        pair_expert = topk_idx.reshape(-1)
        pair_token = torch.arange(n_tokens, device=x.device).repeat_interleave(self.num_experts_per_tok)
        pair_weight = topk_weight.reshape(-1, 1)

        # Group pairs by expert.  Sorting O(N log N) is cheaper than Python token loops
        # and avoids evaluating inactive experts entirely.
        order = torch.argsort(pair_expert, stable=False)
        pair_expert = pair_expert[order]
        pair_token = pair_token[order]
        pair_weight = pair_weight[order]

        out = torch.zeros_like(x_flat)
        counts = torch.bincount(pair_expert, minlength=self.n_routed_experts)
        offset = 0
        for expert_id, count_t in enumerate(counts.tolist()):
            if count_t == 0:
                continue
            sl = slice(offset, offset + count_t)
            token_ids = pair_token[sl]
            expert_out = self.experts[expert_id](x_flat.index_select(0, token_ids))
            expert_out = expert_out * pair_weight[sl].to(dtype=expert_out.dtype)
            out.index_add_(0, token_ids, expert_out)
            offset += count_t

        if self.shared_experts is not None:
            out = out + self.shared_experts(x_flat)
        return out.reshape(orig_shape)


__all__ = ["SwiGLUMLP", "NoAuxMoEGate", "MoELayer"]
