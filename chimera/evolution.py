"""
Chimera 5.1 — Self-Evolution Systems (CPU-Optimized)
- Vectorized HDC ops (batch hamming, majority, XOR bind/unbind)
- Optimized In-Place TTT with fused update
- Efficient episodic case retrieval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────
# Semantic Memory — Vectorized HDC (8192-bit hypervectors)
# ─────────────────────────────────────────────────
class SemanticMemory(nn.Module):
    """HDC semantic memory with vectorized operations.
    
    Optimizations:
    - Batch hamming distance via XOR + bit unpack (vectorized, no Python loop)
    - Vectorized majority bundle
    - Efficient store with access-count eviction
    """

    def __init__(self, config: dict):
        super().__init__()
        self.vector_bits = config.get('vector_bits', 8192)
        self.capacity = config.get('capacity', 200000)
        self.pool_fixed = config.get('pool_size_fixed', True)
        self.lsh_tables = config.get('lsh_tables', 64)
        self.lsh_bits = config.get('lsh_bits_per_table', 14)

        actual_cap = min(self.capacity, 50000)
        n_bytes = self.vector_bits // 8
        self.register_buffer('memory', torch.zeros(actual_cap, n_bytes, dtype=torch.uint8))
        self.register_buffer('count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('access_counts', torch.zeros(actual_cap, dtype=torch.long))

        lsh_proj_size = self.lsh_tables * self.lsh_bits
        self.lsh_proj = nn.Linear(n_bytes, lsh_proj_size, bias=False)

    @staticmethod
    def xor_bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_xor(a, b)

    @staticmethod
    def xor_unbind(bound: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_xor(bound, key)

    @staticmethod
    def majority_bundle(hvs: torch.Tensor) -> torch.Tensor:
        """Vectorized majority rule over hypervectors.
        hvs: [N, D] uint8 tensors — returns [D] uint8
        """
        N = hvs.shape[0]
        threshold = N / 2.0
        result = torch.zeros(hvs.shape[1], dtype=torch.uint8, device=hvs.device)
        for bit in range(8):
            bit_plane = ((hvs >> bit) & 1).float()  # [N, D]
            majority = (bit_plane.sum(0) > threshold).byte()  # [D]
            result = result | (majority << bit)
        return result

    @staticmethod
    def hamming_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Vectorized batch Hamming distance.
        
        Optimization: unpack all 8 bits simultaneously via stacked shifts,
        then sum over bits and bytes in a single operation.
        """
        xor = torch.bitwise_xor(a, b)
        # Unpack all 8 bits at once: [*, D, 8]
        shifts = torch.arange(8, device=xor.device, dtype=torch.uint8)
        bits = ((xor.unsqueeze(-1) >> shifts) & 1).float()  # [*, D, 8]
        # Sum over bits (8) and bytes (D) in one step
        return bits.sum(dim=(-1, -2))

    def query(self, query_vec: torch.Tensor, top_k: int = 16):
        if self.count == 0:
            return None, None
        c = self.count.item()
        # Batch hamming distance
        dists = self.hamming_distance(
            query_vec.unsqueeze(-2),    # [*, 1, D]
            self.memory[:c].unsqueeze(0)  # [1, c, D]
        )
        k = min(top_k, c)
        values, indices = dists.topk(k, dim=-1, largest=False)
        # Update access counts
        with torch.no_grad():
            self.access_counts[indices.reshape(-1)] += 1
        return values, indices

    @torch.no_grad()
    def store(self, vec: torch.Tensor, surprise_magnitude: float = 0.0):
        vec_flat = vec.detach().squeeze(0)
        if self.pool_fixed and self.count >= self.memory.shape[0]:
            # Evict least-accessed entry
            min_idx = self.access_counts[:self.count.item()].argmin()
            self.memory[min_idx] = vec_flat
            self.access_counts[min_idx] = 0
        else:
            idx = self.count.item()
            if idx < self.memory.shape[0]:
                self.memory[idx] = vec_flat
                self.count += 1


# ─────────────────────────────────────────────────
# In-Place TTT — Optimized gradient computation
# ─────────────────────────────────────────────────
class InPlaceTTT(nn.Module):
    """In-Place Test-Time Training with fused update.
    
    Optimizations:
    - Fused conv1d + matmul for delta computation
    - Gradient clipping built-in (no separate pass)
    - Zero-init conv for stable start
    """

    def __init__(self, config: dict, hidden_size: int):
        super().__init__()
        self.enabled = config.get('enabled', True)
        self.target_layers = config.get('target_layers', [13, 23])
        self.inner_lr = config.get('inner_lr', 0.0003)
        self.momentum = config.get('momentum', 0.9)
        self.chunk_size = config.get('chunk_size', 1024)
        self.reset_decay = config.get('reset_decay', 0.95)
        self.delta_clip = 1e-5

        self.conv1d = nn.Conv1d(hidden_size, hidden_size, kernel_size=5,
                                padding=4, groups=hidden_size, bias=False)
        nn.init.zeros_(self.conv1d.weight)
        self.w_target = nn.Parameter(torch.eye(hidden_size) * 0.01)

    def compute_update(self, x_raw: torch.Tensor, z: torch.Tensor,
                       w_down: torch.Tensor) -> torch.Tensor:
        # Causal conv (fused transpose)
        x_shifted = self.conv1d(x_raw.transpose(1, 2))[:, :, :x_raw.shape[1]].transpose(1, 2)
        v_hat = x_shifted @ self.w_target
        delta = v_hat.transpose(-2, -1) @ z
        # Clip in-place
        norm = delta.norm()
        if norm > self.delta_clip:
            delta = delta * (self.delta_clip / norm)
        return delta

    def apply_update(self, w_down: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return w_down + self.inner_lr * delta

    def forward(self, x_raw: torch.Tensor, z: torch.Tensor,
                w_down: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return w_down
        delta = self.compute_update(x_raw, z, w_down)
        return self.apply_update(w_down, delta)


# ─────────────────────────────────────────────────
# Episodic Case Memory — Optimized retrieval
# ─────────────────────────────────────────────────
class EpisodicCaseMemory(nn.Module):
    """Episodic case memory with weighted soft Q-learning retrieval.
    
    Optimizations:
    - Pre-projected query (single matmul for retrieval)
    - Modular eviction (ring buffer, no reallocation)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.enabled = config.get('enabled', True)
        self.max_cases = config.get('max_cases', 4096)
        self.case_bytes = config.get('case_bytes', 2048)
        case_dim = min(self.case_bytes, 512)
        self.register_buffer('cases', torch.zeros(self.max_cases, case_dim))
        self.register_buffer('weights', torch.ones(self.max_cases))
        self.register_buffer('count', torch.tensor(0, dtype=torch.long))
        self.query_proj = nn.Linear(case_dim, case_dim, bias=False)
        self.ema_decay = 0.99

    def retrieve(self, query: torch.Tensor, top_k: int = 5):
        if self.count == 0:
            return None
        c = self.count.item()
        q = self.query_proj(query)
        # Batch cosine similarity via normalized matmul
        q_norm = F.normalize(q.reshape(-1, q.shape[-1]), dim=-1)
        c_norm = F.normalize(self.cases[:c], dim=-1)
        sims = torch.matmul(q_norm, c_norm.t())  # [N, c]
        weighted_sims = sims * self.weights[:c].unsqueeze(0)
        k = min(top_k, c)
        scores, indices = weighted_sims.topk(k, dim=-1)
        return self.cases[indices], scores

    @torch.no_grad()
    def store(self, case_vec: torch.Tensor, outcome: float = 1.0):
        idx = self.count.item() % self.max_cases
        self.cases[idx] = case_vec.detach().squeeze(0)[:self.cases.shape[-1]]
        self.weights[idx] = outcome
        if self.count < self.max_cases:
            self.count += 1

    @torch.no_grad()
    def update_weight(self, idx: int, outcome: float):
        self.weights[idx] = self.ema_decay * self.weights[idx] + (1 - self.ema_decay) * outcome


# ─────────────────────────────────────────────────
# Meta-Guideline Bank
# ─────────────────────────────────────────────────
class MetaGuidelineBank(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.enabled = config.get('enabled', True)
        self.max_guidelines = config.get('max', 256)
        bits = 8192
        self.register_buffer('guidelines',
                             torch.zeros(self.max_guidelines, bits // 8, dtype=torch.uint8))
        self.register_buffer('count', torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def add_guideline(self, vec: torch.Tensor):
        idx = self.count.item() % self.max_guidelines
        self.guidelines[idx] = vec.detach()
        if self.count < self.max_guidelines:
            self.count += 1

    def query(self, query_vec: torch.Tensor, top_k: int = 5):
        if self.count == 0:
            return None
        c = self.count.item()
        dists = SemanticMemory.hamming_distance(
            query_vec.unsqueeze(-2), self.guidelines[:c].unsqueeze(0))
        k = min(top_k, c)
        return dists.topk(k, dim=-1, largest=False)


# ─────────────────────────────────────────────────
# Self-Feedback
# ─────────────────────────────────────────────────
class SelfFeedback(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.enabled = config.get('enabled', True)
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.max_rounds = config.get('max_refinement_rounds', 1)

    def should_refine(self, confidence: float) -> bool:
        return self.enabled and confidence < self.confidence_threshold

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        return probs.amax(dim=-1).mean()


# ─────────────────────────────────────────────────
# Loop Depth Classifier
# ─────────────────────────────────────────────────
class LoopDepthClassifier(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.enabled = config.get('enabled', True)
        hidden = 256
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 6),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).argmax(dim=-1) + 1


# ─────────────────────────────────────────────────
# Self-Evolution Engine (unified controller)
# ─────────────────────────────────────────────────
class SelfEvolutionEngine(nn.Module):
    def __init__(self, config: dict, hidden_size: int):
        super().__init__()
        t1 = config.get('tier1', {})
        t2 = config.get('tier2', {})
        t3 = config.get('tier3', {})

        self.ttt = InPlaceTTT(t1.get('ttt', {}), hidden_size)
        self.semantic_memory = SemanticMemory(config.get('_semantic_memory_config', {}))
        self.episodic = EpisodicCaseMemory(t2.get('episodic_cases', {}))
        self.meta_guidelines = MetaGuidelineBank(t2.get('meta_guidelines', {}))
        self.self_feedback = SelfFeedback(t2.get('self_feedback', {}))
        self.loop_classifier = LoopDepthClassifier(t3.get('loop_depth_learning', {}))

        safety = config.get('safety', {})
        self.freeze_threshold = safety.get('freeze_threshold', 0.05)
        self.frozen = False

    def check_safety(self, cert_failure_rate: float) -> bool:
        if cert_failure_rate > self.freeze_threshold:
            self.frozen = True
        return self.frozen
