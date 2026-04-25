from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SwiGLUMLP
from .quantization import BitLinear


class NoAuxMoEGate(nn.Module):
    def __init__(self, hidden_size: int, n_experts: int, k: int = 2):
        super().__init__()
        self.k = int(k)
        self.n_experts = int(n_experts)
        self.score = nn.Linear(hidden_size, n_experts, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_experts))

    def forward(self, x: torch.Tensor):
        logits = self.score(x.float()) + self.bias
        k = min(self.k, self.n_experts)
        weights, indices = torch.topk(logits, k=k, dim=-1)
        weights = F.softmax(weights, dim=-1).to(x.dtype)
        return indices, weights, logits


class MoELayer(nn.Module):
    """Sparse expert MLP with expert-grouped CPU dispatch."""
    def __init__(self, hidden_size: int, moe_intermediate_size: int, n_routed_experts: int = 16, n_shared_experts: int = 1, num_experts_per_tok: int = 2, use_ternary: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_routed_experts = int(n_routed_experts)
        self.n_shared_experts = int(n_shared_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.gate = NoAuxMoEGate(hidden_size, self.n_routed_experts, self.num_experts_per_tok)
        self.experts = nn.ModuleList([SwiGLUMLP(hidden_size, moe_intermediate_size, use_ternary) for _ in range(self.n_routed_experts)])
        self.shared = nn.ModuleList([SwiGLUMLP(hidden_size, moe_intermediate_size, use_ternary) for _ in range(self.n_shared_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h = x.shape
        flat = x.reshape(-1, h)
        indices, weights, _ = self.gate(x)
        idx = indices.reshape(-1)
        w = weights.reshape(-1).to(flat.dtype)
        token_ids = torch.arange(flat.size(0), device=x.device).repeat_interleave(indices.size(-1))
        out = torch.zeros_like(flat)
        order = torch.argsort(idx)
        idx_s, tok_s, w_s = idx[order], token_ids[order], w[order]
        for expert_id in idx_s.unique(sorted=True).tolist():
            mask = idx_s == expert_id
            toks = tok_s[mask]
            contrib = self.experts[int(expert_id)](flat.index_select(0, toks)) * w_s[mask].unsqueeze(-1)
            out.index_add_(0, toks, contrib)
        if self.shared:
            shared = sum(expert(flat) for expert in self.shared) / len(self.shared)
            out = out + shared
        return out.view(b, t, h)
