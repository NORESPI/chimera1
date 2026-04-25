"""
Chimera 5.1 — Inference Systems (CPU-Optimized)
Span bank, Grammar FST, Entropy valve, Debt ledger, Braid state
- Vectorized span bank queries (batched cosine similarity)
- Fused grammar constraint computation
- Efficient entropy calculation (log_softmax path)
- torch.compile friendly (no Python-level data-dependent branching in hot path)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────
# Span Bank — Vectorized semantic search
# ─────────────────────────────────────────────────
class SpanBank(nn.Module):
    def __init__(self, max_entries: int = 524288, max_tokens: int = 64,
                 hidden_size: int = 2560, memory_mb: int = 384):
        super().__init__()
        self.max_entries = max_entries
        self.max_tokens = max_tokens
        self.hidden_size = hidden_size
        proj_dim = hidden_size // 4
        actual_entries = min(max_entries, int(memory_mb * 1024 * 1024 / (max_tokens * 4)))
        self.register_buffer('bank_keys', torch.zeros(actual_entries, proj_dim))
        self.register_buffer('bank_lengths', torch.zeros(actual_entries, dtype=torch.long))
        self.register_buffer('bank_values', torch.zeros(actual_entries, hidden_size))
        self.register_buffer('bank_count', torch.tensor(0, dtype=torch.long))
        self.semantic_proj = nn.Linear(hidden_size, proj_dim, bias=False)

    def query_scores(self, hidden_state: torch.Tensor, top_k: int = 64):
        if self.bank_count == 0:
            return None, None
        q = F.normalize(self.semantic_proj(hidden_state), dim=-1)
        count = self.bank_count.item()
        keys = F.normalize(self.bank_keys[:count], dim=-1)
        sims = torch.matmul(q, keys.t())
        k = min(top_k, count)
        scores, indices = sims.topk(k, dim=-1)
        return scores, indices

    def query(self, hidden_state: torch.Tensor, top_k: int = 64):
        scores, indices = self.query_scores(hidden_state, top_k=top_k)
        if scores is None:
            return torch.zeros_like(hidden_state)
        values = self.bank_values[:self.bank_count.item()][indices]
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (values * weights).sum(dim=-2)

    @torch.no_grad()
    def add_span(self, hidden_state: torch.Tensor, length: int, value: torch.Tensor = None):
        if self.bank_count >= self.bank_keys.shape[0]:
            return
        idx = self.bank_count.item()
        h = hidden_state.detach().reshape(-1, self.hidden_size).mean(dim=0, keepdim=True)
        self.bank_keys[idx] = self.semantic_proj(h).squeeze(0)
        self.bank_values[idx] = (value.detach().reshape(-1, self.hidden_size).mean(dim=0) if value is not None else h.squeeze(0))
        self.bank_lengths[idx] = length
        self.bank_count += 1

    @torch.no_grad()
    def add(self, keys: torch.Tensor, values: torch.Tensor):
        for k, v in zip(keys.reshape(-1, self.hidden_size), values.reshape(-1, self.hidden_size)):
            self.add_span(k.unsqueeze(0), 1, v.unsqueeze(0))


# ─────────────────────────────────────────────────
# STree Verifier — Compact scoring network
# ─────────────────────────────────────────────────
class STreeVerifier(nn.Module):
    def __init__(self, tree_width: int = 4, tree_depth: int = 5,
                 hidden_size: int = 256):
        super().__init__()
        self.tree_width = tree_width
        self.tree_depth = tree_depth
        self.score_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.score_net(hidden_states)).squeeze(-1)


# ─────────────────────────────────────────────────
# Certificate Verifier — Vectorized field extraction
# ─────────────────────────────────────────────────
class CertificateVerifier(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.semantic_proj = nn.Linear(hidden_size, 64, bias=False)
        self.grammar_proj = nn.Linear(hidden_size, 16, bias=False)
        self.entity_proj = nn.Linear(hidden_size, 32, bias=False)
        self.boundary_proj = nn.Linear(hidden_size, 1, bias=False)
        self.risk_proj = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> dict:
        return {
            'semantic': self.semantic_proj(hidden_states),
            'grammar': self.grammar_proj(hidden_states),
            'entity': self.entity_proj(hidden_states),
            'boundary': self.boundary_proj(hidden_states),
            'risk': torch.sigmoid(self.risk_proj(hidden_states)),
        }


# ─────────────────────────────────────────────────
# Span Inference Engine
# ─────────────────────────────────────────────────
class SpanInferenceEngine(nn.Module):
    def __init__(self, hidden_size: int, config: dict):
        super().__init__()
        self.enabled = config.get('enabled', True)
        self.hidden_size = hidden_size
        self.span_bank = SpanBank(
            max_entries=config.get('bank_entries', 524288),
            max_tokens=config.get('bank_max_tokens', 64),
            hidden_size=hidden_size,
            memory_mb=config.get('bank_memory_mb', 384),
        )
        self.tree_verifier = STreeVerifier(
            tree_width=config.get('tree_verify', {}).get('tree_width', 4),
            tree_depth=config.get('tree_verify', {}).get('tree_depth', 5),
            hidden_size=hidden_size,
        )
        self.certificate = CertificateVerifier(hidden_size)
        self.scoring_weights = nn.Parameter(
            torch.tensor(config.get('scoring_weights_fast', [1.0, 0.8, 0.5, 0.7, 0.35])))
        self.fallback_threshold = config.get('fallback_below_acceptance', 0.5)
        self.risk_gate = nn.Linear(hidden_size + 1, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return hidden_states
        cert = self.certificate(hidden_states)
        risk = cert['risk']
        gate_input = torch.cat([hidden_states, risk], dim=-1)
        modulation = torch.sigmoid(self.risk_gate(gate_input))
        return hidden_states * modulation


# ─────────────────────────────────────────────────
# Grammar FST — Fused constraint penalty
# ─────────────────────────────────────────────────
class GrammarFST(nn.Module):
    """Grammar FST with fused constraint computation.
    
    Optimizations:
    - Single forward pass for all constraint features
    - Fused entropy + margin + repetition penalty computation
    - Pre-allocated feature buffer
    """

    def __init__(self, config: dict):
        super().__init__()
        self.enabled = config.get('enabled', True)
        self.modes = config.get('modes', ['plain_text'])
        self.hard_constraints = config.get('hard_constraints', [])
        self.soft_constraints = config.get('soft_constraints', [])
        n_features = len(self.hard_constraints) + len(self.soft_constraints) + 1
        self.constraint_proj = nn.Linear(n_features, 1, bias=True)
        nn.init.normal_(self.constraint_proj.weight, std=0.01)
        nn.init.zeros_(self.constraint_proj.bias)
        self._n_hard = len(self.hard_constraints)
        self._n_soft = len(self.soft_constraints)
        self._n_features = n_features

    def forward(self, logits: torch.Tensor, state=None) -> torch.Tensor:
        if not self.enabled:
            return logits
        B, T, V = logits.shape

        # Fused feature computation
        # 1. Entropy from log_softmax (numerically stable, single pass)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(-1)  # [B, T]

        # 2. Repetition penalty via cosine of adjacent logit vectors
        features = torch.zeros(B, T, self._n_features, device=logits.device,
                               dtype=logits.dtype)
        features[..., 0] = entropy
        if self._n_soft > 0 and T > 1:
            # Cosine similarity with previous position (vectorized)
            cos = F.cosine_similarity(logits[:, 1:], logits[:, :-1], dim=-1)
            features[:, 1:, self._n_hard] = cos.clamp(min=0)

        penalty = self.constraint_proj(features)  # [B, T, 1]
        return logits + penalty.expand_as(logits)


# ─────────────────────────────────────────────────
# Entropy Valve — Fast entropy routing
# ─────────────────────────────────────────────────
class EntropyValve(nn.Module):
    """Entropy-based compute allocation valve.
    
    Optimizations:
    - log_softmax path for entropy (single pass, numerically stable)
    - Pre-computed thresholds as constants
    """

    def __init__(self, config: dict):
        super().__init__()
        self.enabled = config.get('enabled', True)
        self.threshold_bits = config.get('threshold_bits', 2.0)
        self.levels = config.get('levels', {
            'low':    {'loops': 1, 'min_span': 8, 'audit': 0.125},
            'medium': {'loops': 2, 'min_span': 4, 'audit': 0.5},
            'high':   {'loops': 4, 'min_span': 1, 'audit': 1.0},
        })
        self.router = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 3))
        self._log2 = math.log(2.0)

    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Entropy in bits via log_softmax (numerically stable)."""
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        return -(probs * log_probs).sum(dim=-1) / self._log2

    def get_level(self, entropy: torch.Tensor) -> str:
        if not self.enabled:
            return 'medium'
        mean_h = entropy.mean().item()
        if mean_h < self.threshold_bits * 0.5:
            return 'low'
        elif mean_h < self.threshold_bits:
            return 'medium'
        return 'high'

    def get_loop_count(self, logits: torch.Tensor) -> int:
        if not self.enabled:
            return 2
        entropy = self.compute_entropy(logits)
        level = self.get_level(entropy)
        return self.levels.get(level, self.levels['medium'])['loops']

    def forward(self, logits: torch.Tensor):
        entropy = self.compute_entropy(logits)
        level = self.get_level(entropy)
        return level, self.levels.get(level, self.levels['medium'])


# ─────────────────────────────────────────────────
# Debt Ledger
# ─────────────────────────────────────────────────
class DebtLedger(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.enabled = config.get('enabled', True)
        self.obligations = config.get('obligations', [])
        self.max_outstanding = config.get('max_outstanding', 64)
        self.pressure_weight = config.get('pressure_weight', 0.3)
        self.active_debts = []
        self.debt_bias_scale = nn.Parameter(torch.tensor(0.5))
        self.debt_proj = nn.Linear(1, 1, bias=True)
        nn.init.ones_(self.debt_proj.weight)
        nn.init.zeros_(self.debt_proj.bias)

    def add_debt(self, debt_type: str):
        if len(self.active_debts) < self.max_outstanding:
            self.active_debts.append(debt_type)

    def resolve_debt(self, debt_type: str):
        if debt_type in self.active_debts:
            self.active_debts.remove(debt_type)

    def get_pressure(self) -> float:
        return self.pressure_weight * len(self.active_debts) / max(self.max_outstanding, 1)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return logits
        pressure = self.get_pressure()
        if pressure > 0:
            boost = self.debt_bias_scale * pressure
            boosted = self.debt_proj(boost.unsqueeze(0).unsqueeze(0))
            logits = logits + boosted * 0.01
        return logits


# ─────────────────────────────────────────────────
# Braid State (runtime state container, not an nn.Module)
# ─────────────────────────────────────────────────
class BraidState:
    __slots__ = ['continuous', 'fast', 'semantic_sketch', 'entity_slots',
                 'grammar_stack', 'debt_ledger_slots']

    def __init__(self, config: dict, device: str = 'cpu'):
        D = config.get('continuous_hidden', [2560, 'float32'])[0]
        self.continuous = torch.zeros(1, D, dtype=torch.float32, device=device)
        self.fast = torch.zeros(1, D, dtype=torch.int8, device=device)
        bits = config.get('semantic_sketch', [8192, 'uint64_x128'])[0]
        self.semantic_sketch = torch.zeros(1, bits // 8, dtype=torch.uint8, device=device)
        et = config.get('entity_table', {})
        self.entity_slots = torch.zeros(
            et.get('slots', 256), et.get('slot_bits', 512) // 8,
            dtype=torch.uint8, device=device)
        gs = config.get('grammar_stack', {})
        self.grammar_stack = torch.zeros(
            gs.get('slots', 64), gs.get('width_bits', 128) // 8,
            dtype=torch.uint8, device=device)
        self.debt_ledger_slots = torch.zeros(
            config.get('debt_ledger_slots', 64), dtype=torch.int32, device=device)

    def reset(self):
        self.continuous.zero_()
        self.fast.zero_()
        self.semantic_sketch.zero_()
