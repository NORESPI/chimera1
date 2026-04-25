from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpanBank(nn.Module):
    def __init__(self, max_entries: int = 4096, max_tokens: int = 64, hidden_size: int = 2560, memory_mb: int = 64):
        super().__init__()
        max_by_mem = max(1, int(memory_mb * 1024 * 1024 / max(1, hidden_size * 4)))
        self.max_entries = min(max_entries, max_by_mem)
        self.register_buffer("keys", torch.zeros(self.max_entries, hidden_size), persistent=False)
        self.register_buffer("values", torch.zeros(self.max_entries, hidden_size), persistent=False)
        self.register_buffer("count", torch.zeros((), dtype=torch.long), persistent=False)

    @torch.no_grad()
    def add(self, key: torch.Tensor, value: torch.Tensor) -> None:
        n = min(key.shape[0], self.max_entries)
        start = int(self.count.item()) % self.max_entries
        slots = (torch.arange(n, device=self.keys.device) + start) % self.max_entries
        self.keys.index_copy_(0, slots, F.normalize(key[:n].float(), dim=-1))
        self.values.index_copy_(0, slots, value[:n].float())
        self.count += n

    def query(self, q: torch.Tensor, topk: int = 4) -> torch.Tensor:
        valid = min(int(self.count.item()), self.max_entries)
        if valid == 0:
            return torch.zeros_like(q)
        sims = F.normalize(q.float(), dim=-1) @ self.keys[:valid].t()
        k = min(topk, valid)
        score, idx = torch.topk(sims, k=k, dim=-1)
        val = self.values[:valid].index_select(0, idx.reshape(-1)).view(*idx.shape, -1)
        return (F.softmax(score, dim=-1).unsqueeze(-1) * val).sum(dim=-2).to(q.dtype)


class SpanInferenceEngine(nn.Module):
    def __init__(self, hidden_size: int, config: dict):
        super().__init__()
        self.enabled = config.get("enabled", True)
        self.bank = SpanBank(config.get("bank_entries", 4096), config.get("bank_max_tokens", 64), hidden_size, config.get("bank_memory_mb", 64))
        self.mix = nn.Parameter(torch.tensor(0.05))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        retrieved = self.bank.query(x.reshape(-1, x.size(-1)), topk=4).view_as(x)
        return x + self.mix.to(x.dtype).clamp(0, 0.25) * retrieved


class GrammarFST(nn.Module):
    def __init__(self, config: dict | None = None):
        super().__init__()
        self.enabled = bool((config or {}).get("enabled", True))
        self.mode = "plain_text"

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Hook point for hard constraints.  Kept side-effect free and shape safe.
        return logits


class EntropyValve(nn.Module):
    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = config or {}
        self.threshold_bits = float(cfg.get("threshold_bits", 2.0))
        levels = cfg.get("levels", {})
        self.low = int(levels.get("low", {}).get("loops", 1))
        self.medium = int(levels.get("medium", {}).get("loops", 2))
        self.high = int(levels.get("high", {}).get("loops", 4))

    @torch.no_grad()
    def get_loop_count(self, logits: torch.Tensor) -> int:
        p = F.softmax(logits[:, -1].float(), dim=-1)
        entropy = -(p * p.clamp_min(1e-12).log2()).sum(dim=-1).mean().item()
        if entropy < self.threshold_bits:
            return self.low
        if entropy < self.threshold_bits * 2:
            return self.medium
        return self.high


class DebtLedger(nn.Module):
    def __init__(self, config: dict | None = None):
        super().__init__()
        self.enabled = bool((config or {}).get("enabled", True))
        self.pressure_weight = float((config or {}).get("pressure_weight", 0.3))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits


@dataclass
class BraidState:
    continuous_hidden: torch.Tensor | None = None
    fast_hidden: torch.Tensor | None = None
    semantic_sketch: torch.Tensor | None = None
