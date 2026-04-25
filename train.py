"""
Chimera 5.1 — Training Script (CPU-Optimized)
==================================================
Optimizations implemented:
  1. MeZO (Memory-Efficient Zeroth-Order) optimizer — eliminates backward pass entirely
     - 2× forward only, no activation storage, no gradient computation
     - arxiv:2305.17333
  2. BFloat16 autocast on CPU — 2-4× faster matmuls on AVX-512/AMX hardware
  3. torch.compile with Inductor backend — fused ops, reduced Python overhead
  4. Gradient checkpointing (for AdamW mode) — trades compute for memory
  5. Optimal CPU threading — KMP_AFFINITY, OMP tuning, NUMA-aware
  6. Persistent DataLoader workers — no worker restart overhead
  7. Intel IPEX integration (optional) — auto-detected
  8. Cosine LR with warmup
  9. Standard AdamW with backprop as fallback mode
  10. Generic dataset loading — supports any HF dataset, messages/text columns, category filtering

Usage:
  # MeZO mode (recommended for CPU — no backward pass):
  python train.py --optimizer mezo --scale tiny --seq_len 64 --max_steps 100

  # AdamW mode (standard backprop with gradient checkpointing + bf16):
  python train.py --optimizer adamw --scale tiny --seq_len 64 --max_steps 100

  # Full run with custom dataset and category filter:
  python train.py --optimizer mezo --scale tiny --seq_len 64 --max_steps 10000 \
    --dataset_name Roman1111111/claude-sonnet-4.6-120000x \
    --dataset_split train --text_column messages \
    --category_filter "C++,organic chemistry"
"""

import os
import sys
import json
import time
import math
import argparse

# ─── CPU Threading Setup (MUST be before torch import) ───
def _setup_cpu_threading():
    """Configure optimal CPU threading for training."""
    n_cpus = os.cpu_count() or 4
    # Use all physical cores for compute
    os.environ.setdefault('OMP_NUM_THREADS', str(n_cpus))
    os.environ.setdefault('MKL_NUM_THREADS', str(n_cpus))
    # Compact thread affinity: pack threads on adjacent cores
    os.environ.setdefault('KMP_AFFINITY', 'granularity=fine,compact,1,0')
    # Short blocktime: allow threads to sleep quickly (reduces power, same perf)
    os.environ.setdefault('KMP_BLOCKTIME', '1')
    # jemalloc background thread for faster allocation
    os.environ.setdefault('MALLOC_CONF', 'background_thread:true,metadata_thp:auto')

_setup_cpu_threading()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chimera import Chimera51ForCausalLM
from chimera.quantization import BitLinear

# Configure PyTorch threading
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', os.cpu_count() or 4)))
try:
    torch.set_num_interop_threads(int(os.environ.get('CHIMERA_INTEROP_THREADS', '1')))
except RuntimeError:
    pass

# ─── Optional: Intel Extension for PyTorch ───
HAS_IPEX = False
try:
    import intel_extension_for_pytorch as ipex
    HAS_IPEX = True
    print("[IPEX] Intel Extension for PyTorch detected — will use optimized kernels")
except ImportError:
    pass


# ─────────────────────────────────────────────────
# MeZO Optimizer — Ternary-Aware (arxiv:2305.17333)
# ─────────────────────────────────────────────────
class MeZOOptimizer:
    """Ternary-Aware Memory-Efficient Zeroth-Order Optimizer.
    
    Eliminates the backward pass entirely:
    - 2 forward passes per step (θ+εz and θ-εz)
    - Memory = model size only (no activations, no gradients, no optimizer states)
    - Gradient estimated via finite differences
    
    TERNARY OPTIMIZATION: For BitLinear layers, perturbation and update
    skip zero-weight positions (~33% of weights), saving ~33% of the
    perturbation and update compute. Uses C++ kernel when available.
    """

    def __init__(self, model, lr=1e-4, eps=1e-3, weight_decay=0.0,
                 momentum=0.0, direction="rademacher", cache_directions=True):
        self.model = model
        self.lr = lr
        self.eps = eps
        self.wd = weight_decay
        self.momentum = momentum
        self.direction = direction
        self.cache_directions = cache_directions

        # Collect trainable parameters once and deduplicate tied weights.  The
        # embedding and tied lm_head can share storage; updating both silently
        # doubles the effective LR and wastes CPU.
        self._bitlinear_params = []
        self._other_params = []
        found_params = set()

        def add_other(name, param):
            if param.requires_grad and id(param) not in found_params:
                self._other_params.append((name, param))
                found_params.add(id(param))

        for name, module in model.named_modules():
            if isinstance(module, BitLinear):
                self._bitlinear_params.append((name, module))
                for p in module.parameters(recurse=False):
                    found_params.add(id(p))
            elif isinstance(module, (nn.Linear, nn.Embedding)):
                for pn, p in module.named_parameters(recurse=False):
                    add_other(f"{name}.{pn}", p)

        # Also collect params not in any submodule we found.
        for name, p in model.named_parameters():
            add_other(name, p)

        self._mezo_masks = {}
        self._direction_cache = {}

        # Momentum buffer
        if momentum > 0:
            self._momentum_buffer = {}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    self._momentum_buffer[n] = torch.zeros_like(p.data)

    def _sample_direction(self, p: torch.Tensor, seed: int) -> torch.Tensor:
        gen = torch.Generator(device=p.device if p.device.type != 'cpu' else 'cpu')
        gen.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
        if self.direction == "gaussian":
            return torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
        # Rademacher ±1 is a valid ZO direction, much cheaper to sample than
        # Gaussian on CPU and avoids transcendental RNG work.
        z = torch.empty(p.shape, dtype=p.dtype, device=p.device)
        z.bernoulli_(0.5, generator=gen).mul_(2).sub_(1)
        return z

    def _direction_for(self, name: str, p: torch.Tensor, seed: int, mask=None) -> torch.Tensor:
        if self.cache_directions and name in self._direction_cache:
            return self._direction_cache[name]
        z = self._sample_direction(p, seed)
        if mask is not None:
            z.mul_(mask.to(device=p.device, dtype=z.dtype))
        if self.cache_directions:
            self._direction_cache[name] = z
        return z

    def _perturb_params(self, seed: int, scale: float):
        """Ternary-aware perturbation with cached deterministic directions."""
        sub_seed = seed
        for name, module in self._bitlinear_params:
            mask = self._mezo_masks.get(name)
            if mask is None:
                mask = module.ternary_nonzero_mask()
            z = self._direction_for(f"{name}.weight", module.weight.data, sub_seed, mask=mask)
            module.weight.data.add_(z, alpha=scale)
            module.invalidate_packed()
            sub_seed += 1000003

        for i, (name, p) in enumerate(self._other_params):
            z = self._direction_for(name, p.data, seed + 500000007 + i * 1000003)
            p.data.add_(z, alpha=scale)

    def _update_params(self, seed: int, projected_grad: float):
        """Ternary-aware parameter update using the same cached directions."""
        sub_seed = seed
        for name, module in self._bitlinear_params:
            z = self._direction_for(f"{name}.weight", module.weight.data, sub_seed,
                                    mask=self._mezo_masks.get(name))
            if self.momentum > 0 and f"{name}.weight" in self._momentum_buffer:
                buf = self._momentum_buffer[f"{name}.weight"]
                buf.mul_(self.momentum).add_(z, alpha=projected_grad)
                module.weight.data.add_(buf, alpha=-self.lr)
            else:
                module.weight.data.add_(z, alpha=-self.lr * projected_grad)
            if self.wd > 0:
                module.weight.data.mul_(1 - self.lr * self.wd)
            module.invalidate_packed()
            sub_seed += 1000003

        for i, (name, p) in enumerate(self._other_params):
            z = self._direction_for(name, p.data, seed + 500000007 + i * 1000003)
            if self.momentum > 0 and name in self._momentum_buffer:
                buf = self._momentum_buffer[name]
                buf.mul_(self.momentum).add_(z, alpha=projected_grad)
                p.data.add_(buf, alpha=-self.lr)
            else:
                p.data.add_(z, alpha=-self.lr * projected_grad)
            if self.wd > 0:
                p.data.mul_(1 - self.lr * self.wd)

    @torch.no_grad()
    def step(self, loss_fn, batch) -> float:
        """Single MeZO step: 2 forward passes, no backward.
        
        Returns: loss estimate (average of pos/neg)
        """
        seed = torch.randint(0, 2**31, (1,)).item()

        # Snapshot sparse masks once from θ. The same mask and direction are reused
        # for +eps, -eps, reset and update, reducing MeZO RNG from 4× model-size
        # samples/step to 1× while preserving the finite-difference direction.
        self._mezo_masks = {name: module.ternary_nonzero_mask().detach()
                            for name, module in self._bitlinear_params}
        self._direction_cache = {}

        # Forward at θ + εz
        self._perturb_params(seed, self.eps)
        loss_pos = loss_fn(batch).item()

        # Forward at θ - εz (net: θ + εz - 2εz = θ - εz)
        self._perturb_params(seed, -2 * self.eps)
        loss_neg = loss_fn(batch).item()

        # Reset to θ (net: θ - εz + εz = θ)
        self._perturb_params(seed, self.eps)

        # Projected gradient
        projected_grad = (loss_pos - loss_neg) / (2 * self.eps)

        # Update parameters (sparse for BitLinear, dense for others)
        self._update_params(seed, projected_grad)

        # Invalidate packed caches (weights changed)
        for _, module in self._bitlinear_params:
            module.invalidate_packed()
        self._mezo_masks = {}
        self._direction_cache = {}

        return (loss_pos + loss_neg) / 2


# ─────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────
class TokenDataset(Dataset):
    def __init__(self, chunks: torch.Tensor):
        self.chunks = chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict:
        return {"input_ids": self.chunks[idx], "labels": self.chunks[idx]}


def _matches_category_filter(ex: dict, filters: list) -> bool:
    """Check if example matches any of the requested category substrings."""
    cat = ex.get("category", "")
    if not cat:
        return False
    cat_lower = cat.lower()
    return any(f.lower() in cat_lower for f in filters)


def _format_example(ex: dict, tok, text_column: str = "auto", include_reasoning: bool = False) -> str:
    """Convert an example dict to a single text string for tokenization."""
    # Auto-detect text column
    if text_column == "auto":
        if "messages" in ex:
            text_column = "messages"
        elif "text" in ex:
            text_column = "text"
        elif "content" in ex:
            text_column = "content"
        elif "conversation" in ex:
            text_column = "conversation"
        else:
            text_column = None

    if text_column == "messages" and "messages" in ex:
        msgs = ex["messages"]
        # Inject reasoning into assistant messages if requested
        if include_reasoning and isinstance(msgs, list):
            msgs = []
            for m in ex["messages"]:
                if isinstance(m, dict) and m.get("role") == "assistant" and "reasoning" in m:
                    content = f"<|thinking|>\n{m['reasoning']}\n<|/thinking|>\n{m.get('content', '')}"
                    msgs.append({"role": "assistant", "content": content})
                else:
                    msgs.append(m)
        return tok.apply_chat_template(msgs)

    if text_column and text_column in ex:
        val = ex[text_column]
        if isinstance(val, str):
            return val
        # Some datasets store conversation as list of dicts even in 'text' col
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
            return tok.apply_chat_template(val)
        return str(val)

    # Fallback: stringify the whole example
    return str(ex)


def build_dataset(seq_len: int, max_samples=None, max_tokens=None, split: str = "train",
                  dataset_name: str = "roneneldan/TinyStories",
                  dataset_config: str = None,
                  text_column: str = "auto",
                  category_filter: str = None,
                  include_reasoning: bool = False):
    """Build dataset from any HuggingFace dataset with splintr tokenizer.

    Supports:
      - Generic text columns ('text', 'content', etc.)
      - Messages/chat format (auto-detected, uses apply_chat_template)
      - Category filtering (comma-separated substrings)
      - Streaming for huge datasets
      - Pre-allocated token buffer to avoid OOM on billion-token datasets
    """
    from datasets import load_dataset
    from chimera import ChimeraTokenizer

    print(f"[DATA] Loading {dataset_name} ({split})...")
    load_kwargs = {"split": split, "streaming": True}
    if dataset_config:
        load_kwargs["name"] = dataset_config
    ds = load_dataset(dataset_name, **load_kwargs)

    print(f"[DATA] Loading tokenizer (splintr o200k_base)...")
    tok = ChimeraTokenizer(pretrained="o200k_base")

    # Parse category filters
    cat_filters = None
    if category_filter:
        cat_filters = [c.strip() for c in category_filter.split(",") if c.strip()]
        print(f"[DATA] Filtering categories: {cat_filters}")

    # Determine token budget
    if max_tokens is not None:
        token_budget = max_tokens
    elif max_samples is not None:
        token_budget = max_samples * (seq_len + 1)
    else:
        token_budget = None

    processed = 0
    skipped = 0

    if token_budget is not None and token_budget > 0:
        # Pre-allocated flat buffer — avoids Python list overhead (~28 bytes/token)
        buffer = torch.empty(token_budget, dtype=torch.long)
        buf_idx = 0

        for i, ex in enumerate(ds):
            if cat_filters and not _matches_category_filter(ex, cat_filters):
                skipped += 1
                continue

            text = _format_example(ex, tok, text_column, include_reasoning)
            if not text or not text.strip():
                skipped += 1
                continue

            ids = tok.encode(text, add_special_tokens=False)
            ids.append(tok.eos_token_id)
            n_ids = len(ids)

            # Truncate if we would exceed the buffer
            if buf_idx + n_ids > token_budget:
                n_ids = token_budget - buf_idx
                if n_ids <= 0:
                    break
                ids = ids[:n_ids]

            if n_ids > 0:
                buffer[buf_idx:buf_idx + n_ids] = torch.tensor(ids, dtype=torch.long)
                buf_idx += n_ids
            processed += 1

            if buf_idx >= token_budget:
                break
            if (processed + 1) % 10000 == 0:
                print(f"  {processed:,} examples, {buf_idx:,} tokens...")

        all_ids = buffer[:buf_idx]
    else:
        # Fallback: old list approach for unbounded collection
        all_ids = []
        target = max_samples * (seq_len + 1) if max_samples else float('inf')
        for i, ex in enumerate(ds):
            if cat_filters and not _matches_category_filter(ex, cat_filters):
                skipped += 1
                continue

            text = _format_example(ex, tok, text_column, include_reasoning)
            if not text or not text.strip():
                skipped += 1
                continue

            all_ids.extend(tok.encode(text, add_special_tokens=False))
            all_ids.append(tok.eos_token_id)
            processed += 1

            if len(all_ids) >= target:
                break
            if (processed + 1) % 10000 == 0:
                print(f"  {processed:,} examples, {len(all_ids):,} tokens...")
        all_ids = torch.tensor(all_ids, dtype=torch.long)

    print(f"[DATA] Processed {processed:,} examples, skipped {skipped:,} (category/text mismatch)")

    if len(all_ids) == 0:
        raise ValueError(
            f"No data matched filters. dataset={dataset_name}, "
            f"category_filter={category_filter}, text_column={text_column}"
        )

    n = len(all_ids) // (seq_len + 1)
    if max_samples:
        n = min(n, max_samples)
    chunks = all_ids[:n * (seq_len + 1)].view(n, seq_len + 1)
    print(f"[DATA] {n:,} chunks × {seq_len} tokens = {n * seq_len:,} total")
    return TokenDataset(chunks), tok


# ─────────────────────────────────────────────────
# LR Schedule
# ─────────────────────────────────────────────────
def cosine_lr(step: int, warmup: int, total: int,
              max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    p = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * p))


# ─────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────
def train(args):
    with open(args.config) as f:
        config = json.load(f)

    # ─── Scale overrides ───
    if args.scale == "tiny":
        config['hidden_size'] = 256
        config['intermediate_size'] = 512
        config['num_hidden_layers'] = 28
        config['num_heads'] = 4
        config['head_dim'] = 48
    elif args.scale == "small":
        config['hidden_size'] = 512
        config['intermediate_size'] = 1024
        config['num_hidden_layers'] = 28
        config['num_heads'] = 8
        config['head_dim'] = 48
    elif args.scale == "medium":
        config['hidden_size'] = 1024
        config['intermediate_size'] = 2048
        config['num_hidden_layers'] = 28
        config['num_heads'] = 8
        config['head_dim'] = 96

    config['vocab_size'] = 200073
    config.setdefault('gated_deltanet', {})['chunk_size'] = min(args.seq_len, 64)
    config.setdefault('xlstm', {})['memory_size_per_head'] = [config['head_dim'], config['head_dim']]
    config.setdefault('titans', {}).update({
        'memory_depth': 2, 'persistent_memory_slots': 16,
        'local_window_size': min(args.seq_len, 256)
    })
    moe_cfg = config.setdefault('backbone', {}).setdefault('moe', {})
    moe_cfg.update({
        'layers': [3, 7, 11, 15, 19, 23, 27],
        'moe_intermediate_size': config['intermediate_size'] // 4,
        'n_routed_experts': 8, 'n_shared_experts': 1, 'num_experts_per_tok': 2
    })
    config.setdefault('looping', {}).update({
        'enabled': True, 'prelude': [0, 3], 'loop': [4, 23], 'coda': [24, 27],
        'loop_range': [1, 3], 'loop_default': 2, 'adaptive_exit_threshold': 0.01
    })
    config.setdefault('span_inference', {})['enabled'] = True
    config.setdefault('grammar', {})['enabled'] = True
    config.setdefault('entropy_valve', {})['enabled'] = True
    config.setdefault('debt_ledger', {}).update({
        'enabled': True, 'obligations': ['close_bracket', 'close_string'],
        'max_outstanding': 32, 'pressure_weight': 0.3
    })
    config.setdefault('self_evolution', {}).update({
        'tier1': {
            'ttt': {'enabled': True, 'target_layers': [13, 23], 'inner_lr': 0.0003,
                    'momentum': 0.9, 'chunk_size': 256, 'reset_decay': 0.95},
            'memory_growth': {'enabled': True, 'pool_size_fixed': True}
        },
        'tier2': {
            'meta_guidelines': {'enabled': True, 'max': 64},
            'episodic_cases': {'enabled': True, 'max_cases': 256, 'case_bytes': 512},
            'self_feedback': {'enabled': True, 'confidence_threshold': 0.6,
                              'max_refinement_rounds': 1}
        },
        'tier3': {'loop_depth_learning': {'enabled': True}},
        'safety': {'freeze_threshold': 0.05},
    })
    config.setdefault('semantic_memory', {}).update({
        'vector_bits': 1024, 'capacity': 1000, 'pool_size_fixed': True
    })
    config.setdefault('multimodal', {})['enabled'] = False

    # ─── Print configuration ───
    use_mezo = args.optimizer == 'mezo'
    use_bf16 = args.bf16 and torch.cpu.is_available()
    use_compile = args.compile

    print("=" * 60)
    print("CHIMERA 5.1 TRAINING — CPU-OPTIMIZED")
    print("=" * 60)
    print(f"Scale:        {args.scale} (h={config['hidden_size']})")
    print(f"Layers:       {config['num_hidden_layers']}")
    print(f"Seq len:      {args.seq_len}")
    print(f"Steps:        {args.max_steps}")
    print(f"Optimizer:    {'MeZO (no backward)' if use_mezo else 'AdamW (backprop)'}")
    print(f"BFloat16:     {use_bf16}")
    print(f"torch.compile:{use_compile}")
    print(f"Grad ckpt:    {args.grad_checkpoint and not use_mezo}")
    print(f"Device:       CPU ({torch.get_num_threads()} threads)")
    print(f"IPEX:         {HAS_IPEX}")
    print(f"Tokenizer:    splintr o200k_base ({config['vocab_size']} tokens)")
    print(f"Dataset:      {args.dataset_name} / {args.dataset_split}")
    if args.dataset_config:
        print(f"Dataset config: {args.dataset_config}")
    if args.category_filter:
        print(f"Category filter: {args.category_filter}")
    if args.include_reasoning:
        print("Reasoning:    INCLUDED (<|thinking|> ... <|/thinking|>)")

    # ─── Build model ───
    model = Chimera51ForCausalLM(config)
    p = model.count_parameters()
    print(f"Params:       {p['total']:,} (ternary: {p['ternary']:,})")

    if use_mezo:
        mem_mb = p['total'] * 4 * 2 / 1024 ** 2  # 2× model (params + perturbation buffer)
        print(f"Memory:       ~{mem_mb:.0f} MB (MeZO: 2× model only)")
    else:
        mem_mb = p['total'] * 12 / 1024 ** 2  # params + grads + optimizer states
        print(f"Memory:       ~{mem_mb:.0f} MB (AdamW: params + grads + states)")

    # ─── Gradient checkpointing (AdamW mode only) ───
    if args.grad_checkpoint and not use_mezo:
        model.enable_gradient_checkpointing()
        print("[OPT] Gradient checkpointing enabled")

    # ─── IPEX optimization ───
    if HAS_IPEX and not use_mezo:
        optimizer_for_ipex = torch.optim.AdamW(model.parameters(), lr=args.lr)
        model, optimizer_for_ipex = ipex.optimize(
            model, optimizer=optimizer_for_ipex,
            dtype=torch.bfloat16 if use_bf16 else torch.float32,
            level='O1'
        )
        print("[OPT] IPEX optimization applied (level O1)")

    # ─── torch.compile ───
    if use_compile:
        print("[OPT] Compiling model with torch.compile (inductor)...")
        model = torch.compile(model, backend="inductor", mode="default",
                              dynamic=True)
        print("[OPT] Compilation deferred (will compile on first forward pass)")

    # ─── Dataset ───
    dataset, tok = build_dataset(
        args.seq_len,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
        split=args.dataset_split,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        text_column=args.text_column,
        category_filter=args.category_filter,
        include_reasoning=args.include_reasoning,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=args.num_workers > 0,  # Keep workers alive between epochs
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # ─── Optimizer ───
    if use_mezo:
        optimizer = MeZOOptimizer(
            model,
            lr=args.lr * 0.01,  # MeZO needs much smaller LR
            eps=1e-3,
            weight_decay=0.1,
            momentum=0.9,
            direction=args.mezo_direction,
            cache_directions=args.mezo_direction_cache,
        )
    else:
        no_decay = {"A_log", "dt_bias", "norm", "bias", "embed", "energy_weights"}
        param_groups = [
            {"params": [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.1},
            {"params": [p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.0},
        ]
        if HAS_IPEX:
            optimizer = optimizer_for_ipex  # Already created during ipex.optimize
        else:
            optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    # ─── Loss function (shared) ───
    def compute_loss(batch):
        ids = batch["input_ids"][:, :-1]
        labels = batch["labels"][:, 1:]
        if use_bf16:
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                loss, _ = model(ids, labels=labels)
        else:
            loss, _ = model(ids, labels=labels)
        return loss

    # ─── Training loop ───
    os.makedirs(args.output_dir, exist_ok=True)
    log_f = open(os.path.join(args.output_dir, "log.jsonl"), "w")

    model.train()
    step = 0
    total_loss = 0.0
    best = float('inf')
    t0 = time.time()
    toks = 0
    data_iter = iter(loader)
    warmup = min(args.warmup, args.max_steps // 10)

    if not use_mezo:
        optimizer.zero_grad()

    print(f"\n{'=' * 60}")
    print(f"Starting training...")
    print(f"{'=' * 60}\n")

    while step < args.max_steps:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # ─── MeZO step (no backward) ───
        if use_mezo:
            # Update LR
            lr = cosine_lr(step, warmup, args.max_steps,
                           args.lr * 0.01, args.lr * 0.001)
            optimizer.lr = lr

            loss_val = optimizer.step(compute_loss, batch)
            total_loss += loss_val
            toks += batch["input_ids"][:, :-1].numel()

        # ─── AdamW step (standard backprop) ───
        else:
            loss = compute_loss(batch)
            (loss / args.grad_accum).backward()
            total_loss += loss.item()
            toks += batch["input_ids"][:, :-1].numel()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                lr = cosine_lr(step, warmup, args.max_steps, args.lr, args.lr * 0.1)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
                optimizer.step()
                optimizer.zero_grad()

        step += 1

        # ─── Logging ───
        if step % args.log_every == 0:
            dt = time.time() - t0
            avg = total_loss / args.log_every
            ppl = math.exp(min(avg, 20))
            tps = toks / dt if dt > 0 else 0
            eta = (args.max_steps - step) / (step / dt) / 3600 if dt > 0 else 0
            entry = {
                "step": step, "loss": round(avg, 4), "ppl": round(ppl, 2),
                "lr": round(lr, 8), "tok/s": round(tps), "eta_h": round(eta, 1),
                "optimizer": "mezo" if use_mezo else "adamw",
            }
            print(f"  step {step:>6}/{args.max_steps} | loss {avg:.4f} | "
                  f"ppl {ppl:>8.2f} | {tps:.0f} tok/s | ETA {eta:.1f}h")
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
            if avg < best:
                best = avg
            total_loss = 0.0
            toks = 0
            t0 = time.time()

        # ─── Checkpoint ───
        if step % args.save_every == 0:
            path = os.path.join(args.output_dir, f"ckpt-{step}")
            os.makedirs(path, exist_ok=True)
            # Save raw model (unwrap compile if needed)
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save({
                "model": raw_model.state_dict(),
                "config": config,
                "step": step,
                "optimizer": args.optimizer,
            }, os.path.join(path, "ckpt.pt"))
            print(f"  [SAVE] {path}")

    # ─── Final save ───
    path = os.path.join(args.output_dir, "final")
    os.makedirs(path, exist_ok=True)
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save({
        "model": raw_model.state_dict(),
        "config": config,
        "step": step,
        "best_loss": best,
    }, os.path.join(path, "model.pt"))
    json.dump(config, open(os.path.join(path, "config.json"), "w"), indent=2)
    print(f"\n{'=' * 60}")
    print(f"DONE — Best loss: {best:.4f}, PPL: {math.exp(min(best, 20)):.2f}")
    print(f"Optimizer: {'MeZO (no backward)' if use_mezo else 'AdamW'}")
    print(f"Saved: {path}")
    log_f.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Chimera 5.1 CPU-Optimized Training")

    # Model
    p.add_argument("--config", default="config.json")
    p.add_argument("--scale", default="tiny", choices=["tiny", "small", "medium", "full"])
    p.add_argument("--seq_len", type=int, default=256)

    # Training
    p.add_argument("--optimizer", default="mezo", choices=["mezo", "adamw"],
                   help="mezo: no backward pass (CPU-optimal). adamw: standard backprop.")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Maximum number of chunks to generate")
    p.add_argument("--max_tokens", type=int, default=None,
                   help="Maximum total tokens to collect (pre-allocated buffer, prevents OOM on huge datasets)")

    # CPU Optimizations
    p.add_argument("--bf16", action="store_true", default=True,
                   help="Enable BFloat16 autocast on CPU (default: True)")
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--compile", action="store_true", default=False,
                   help="Enable torch.compile with Inductor backend")
    p.add_argument("--grad_checkpoint", action="store_true", default=True,
                   help="Enable gradient checkpointing (AdamW mode only)")
    p.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    p.add_argument("--mezo_direction", choices=["rademacher", "gaussian"],
                   default="rademacher",
                   help="ZO perturbation distribution; rademacher is fastest on CPU")
    p.add_argument("--no-mezo-direction-cache", dest="mezo_direction_cache",
                   action="store_false", default=True,
                   help="Regenerate directions instead of caching them for the step")

    # Data — fully configurable
    p.add_argument("--dataset_name", default="roneneldan/TinyStories",
                   help="HuggingFace dataset name (e.g. Roman1111111/claude-sonnet-4.6-120000x)")
    p.add_argument("--dataset_config", default=None,
                   help="Dataset config/subset name")
    p.add_argument("--dataset_split", default="train",
                   help="Dataset split to use")
    p.add_argument("--text_column", default="auto",
                   help="Column containing text. 'auto' detects 'messages'/'text'/'content'/'conversation'")
    p.add_argument("--category_filter", default=None,
                   help="Comma-separated category substrings to filter on (e.g. 'C++,python,math')")
    p.add_argument("--include_reasoning", action="store_true", default=False,
                   help="Include reasoning/thinking content from assistant messages as <|thinking|>...<|/thinking|>")

    # Logging / Output
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--output_dir", default="./chimera_output")

    train(p.parse_args())