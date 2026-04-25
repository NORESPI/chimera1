from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticMemory(nn.Module):
    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = config or {}
        self.vector_bits = int(cfg.get("vector_bits", 8192))
        self.words = max(1, self.vector_bits // 64)
        capacity = min(int(cfg.get("capacity", 4096)), 4096)
        self.register_buffer("keys", torch.zeros(capacity, self.words, dtype=torch.long), persistent=False)
        self.register_buffer("values", torch.zeros(capacity, self.words, dtype=torch.long), persistent=False)
        self.register_buffer("score", torch.zeros(capacity), persistent=False)
        self.register_buffer("count", torch.zeros((), dtype=torch.long), persistent=False)

    @staticmethod
    def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_xor(a, b)

    @torch.no_grad()
    def store(self, key: torch.Tensor, value: torch.Tensor | None = None) -> None:
        key = key.to(self.keys.device, torch.long).view(-1, self.words)
        value = key if value is None else value.to(self.keys.device, torch.long).view(-1, self.words)
        n = min(key.size(0), self.keys.size(0))
        slots = (torch.arange(n, device=self.keys.device) + int(self.count.item())) % self.keys.size(0)
        self.keys.index_copy_(0, slots, key[:n])
        self.values.index_copy_(0, slots, value[:n])
        self.score.index_fill_(0, slots, 1.0)
        self.count += n

    def query(self, key: torch.Tensor, topk: int = 4) -> torch.Tensor:
        valid = min(int(self.count.item()), self.keys.size(0))
        if valid == 0:
            return torch.zeros_like(key.view(-1, self.words))
        key = key.to(self.keys.device, torch.long).view(-1, self.words)
        # byte-level hamming approximation, vectorized and portable.
        dist = torch.bitwise_xor(key[:, None, :], self.keys[:valid][None]).ne(0).sum(dim=-1).float()
        vals, idx = torch.topk(-dist, k=min(topk, valid), dim=-1)
        out = self.values[:valid].index_select(0, idx.reshape(-1)).view(*idx.shape, self.words)
        return out[:, 0]


class InPlaceTTT(nn.Module):
    def __init__(self, hidden_size: int, lr: float = 3e-4):
        super().__init__()
        self.adapter = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.adapter(x)


class EpisodicCaseMemory(nn.Module):
    def __init__(self, max_cases: int = 4096, hidden_size: int = 2560):
        super().__init__()
        self.register_buffer("keys", torch.zeros(max_cases, hidden_size), persistent=False)
        self.register_buffer("values", torch.zeros(max_cases, hidden_size), persistent=False)
        self.register_buffer("weights", torch.zeros(max_cases), persistent=False)
        self.register_buffer("count", torch.zeros((), dtype=torch.long), persistent=False)

    @torch.no_grad()
    def add(self, key: torch.Tensor, value: torch.Tensor, weight: float = 1.0) -> None:
        n = min(key.shape[0], self.keys.shape[0])
        slots = (torch.arange(n, device=self.keys.device) + int(self.count.item())) % self.keys.shape[0]
        self.keys.index_copy_(0, slots, F.normalize(key[:n].float(), dim=-1))
        self.values.index_copy_(0, slots, value[:n].float())
        self.weights.index_fill_(0, slots, float(weight))
        self.count += n

    def retrieve(self, q: torch.Tensor, topk: int = 4) -> torch.Tensor:
        valid = min(int(self.count.item()), self.keys.size(0))
        if valid == 0:
            return torch.zeros_like(q)
        sim = F.normalize(q.float(), dim=-1) @ self.keys[:valid].t()
        score, idx = torch.topk(sim + self.weights[:valid], k=min(topk, valid), dim=-1)
        val = self.values[:valid].index_select(0, idx.reshape(-1)).view(*idx.shape, -1)
        return (F.softmax(score, -1).unsqueeze(-1) * val).sum(-2).to(q.dtype)


class SelfEvolutionEngine(nn.Module):
    def __init__(self, config: dict | None, hidden_size: int):
        super().__init__()
        cfg = config or {}
        self.semantic_memory = SemanticMemory(cfg.get("_semantic_memory_config", cfg.get("semantic_memory", {})))
        self.ttt = InPlaceTTT(hidden_size, cfg.get("tier1", {}).get("ttt", {}).get("inner_lr", 3e-4))
        self.episodic = EpisodicCaseMemory(cfg.get("tier2", {}).get("episodic_cases", {}).get("max_cases", 4096), hidden_size)
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ttt(x) if self.training else x
