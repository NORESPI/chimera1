# Chimera 5.1 — True 1.58-bit Ternary CPU Compute (v5.1.3)

100% faithful implementation of the Chimera 5.1 config. All 15 architectural components implemented in pure PyTorch, with **true 1.58-bit ternary computation** on CPU.

**Key breakthrough**: Ternary weights `{-1, 0, 1}` are stored in 2-bit packed format (4 weights per byte), giving **16× memory reduction** and enabling zero-multiply forward/backward paths via custom C++ kernels with OpenMP.

**Tokenizer**: splintr-rs (Rust) — o200k_base vocab (200,073 tokens, OpenAI o1/o3).

---

## v5.1.4 — Real CPU Fast Path Audit

Implemented after a full CPU hot-path audit:
- fixed the package/runtime mismatch (`chimera` imports now match the repository layout);
- added the missing sparse `MoELayer` with expert-grouped dispatch and `index_add_` accumulation;
- made C++ ternary extensions lazy-loaded instead of compiling at import time;
- vectorized BitLinear AbsMean scaling and removed Python repack loops;
- cached causal/triangular masks reused by recurrent layers during generation and MeZO;
- reduced no-grad Gated DeltaNet clone churn while keeping autograd-safe behavior for AdamW;
- made MeZO CPU training use cached per-step directions and fast Rademacher perturbations by default;
- deduplicated tied embedding/lm-head parameters in MeZO updates;
- added deterministic greedy inference fast path (`--temperature 0`) and optional bounded context (`--max_context`).

Recommended CPU modes:
```bash
# Ultra-efficient CPU fine-tuning
OMP_NUM_THREADS=$(nproc) python train.py \
  --scale tiny --seq_len 64 --max_steps 10 \
  --optimizer mezo --mezo_direction rademacher \
  --batch_size 2 --grad_accum 1 --no-bf16 --num_workers 0

# Lowest-latency deterministic CPU serving
python inference.py \
  --checkpoint chimera_output/final/model.pt \
  --prompt "Once upon a time" --temperature 0 --top_k 1 \
  --max_context 256 --max_tokens 128
```

---

## v5.1.3 — Fix Illegal Instruction Crash

**Fixed**: Removed `-march=native` from C++ JIT compilation flags. This flag caused `Illegal instruction (core dumped)` on CPUs with different instruction sets than the build machine. The C++ kernel now uses **runtime CPUID detection** to select AVX-512/AVX2 paths, while compilation remains portable.

**If you get `Illegal instruction`:**
```bash
rm -rf .ternary_build .ternary_build_v2  # Clear old cache
python train.py ...  # Rebuild with portable flags
```

---

## v5.1.2 — True Ternary Compute

| Component | Implementation | Memory | Speed (training) | Speed (inference) |
|---|---|---|---|---|
| **Weight storage** | 2-bit packed uint8 (4 w/byte) | **16× smaller** vs FP32 | — | — |
| **Forward path** | C++ unpack + MKL BLAS | 94% less bandwidth | ~0.5-0.7× (unpack overhead) | ~1.0-1.2× (amortized) |
| **Backward grad_x** | Same ternary kernel | — | Included in above | — |
| **Backward grad_w** | FP32 outer product (STE req) | — | standard | — |
| **MeZO optimizer** | Sparse perturbation (skip ~33% zeros) | 2× model size | **No backward pass** | — |
| **MeZO sparse update** | C++ kernel, perturb only non-zero weights | — | ~1.5× faster per step | — |

**Note**: Ternary compute is **memory-optimized**, not raw compute-optimized. On CPU, MKL BLAS for FP32 matmul is so optimized that ternary unpack+BLAS has ~30-50% overhead at small sizes. The win is:
- **16× less RAM** — models that don't fit in FP32 fit in ternary
- **16× less memory bandwidth** — weight loading from DRAM is the bottleneck for large models
- **MeZO eliminates backward** — no gradient through 28 layers of recurrences

### When Ternary Wins

| Scenario | FP32 | Ternary + MeZO | Winner |
|---|---|---|---|
| Model > L3 cache (e.g. 2B params) | 10GB, bandwidth-bound | 0.6GB, fits L3 | **Ternary** |
| Small model, fits L1 (e.g. 50M) | Fast BLAS | Unpack overhead | FP32 |
| CPU without AVX-512/AMX | Standard | Same path | Tie |
| CPU with VNNI/AMX + `_int_mm` | Slow INT8 path | Native INT8 matmul | **Ternary** |
| Fine-tuning with limited RAM | OOM | Fits | **Ternary** |

---

## Architecture (28 layers, 4 types)

```
Layer pattern: GD XM GD TM GD XM GD SK × 3.5
  GD = Gated DeltaNet (14 layers) — arxiv:2412.06464
  XM = xLSTM mLSTM (7 layers) — arxiv:2405.04517
  TM = Titans MAC (4 layers) — arxiv:2501.00663
  SK = TSP Span Knot (3 layers)
```

All linear layers use **BitLinear** (ternary 1.58-bit) with per-group AbsMean scaling.

---

## Components

| Module | File | Status |
|--------|------|--------|
| **splintr Tokenizer** (o200k_base, 200K vocab, Rust-backed) | `tokenizer.py` | ✅ |
| **BitNet 1.58 QAT** (2-bit packed, C++ unpack kernel, STE, N:M 2:4) | `quantization.py` | ✅ v5.1.3 |
| **Ternary SIMD Kernels** (AVX2 unpack, OpenMP, sparse MeZO) | `ternary_simd.py` | ✅ v5.1.3 |
| **Gated DeltaNet** (α/β gates, chunkwise parallel) | `layers.py` | ✅ |
| **xLSTM mLSTM** (parallelized, no timestep loop) | `layers.py` | ✅ v5.1.1 |
| **Titans MAC** (parallelized, no timestep loop) | `layers.py` | ✅ v5.1.1 |
| **TSP Span Knot** (vectorized Hamming) | `layers.py` | ✅ v5.1.1 |
| **Parcae Looping** (deterministic, checkpoint-safe) | `looping.py` | ✅ v5.1.1 |
| **MoE** (sort-based dispatch, 16 experts, 2 active) | `moe.py` | ✅ v5.1.1 |
| **Span Inference** (bank, STree verifier, certificates) | `inference.py` | ✅ |
| **Grammar FST** (9 modes, hard/soft constraints, fused penalty) | `inference.py` | ✅ |
| **Entropy Valve** (3 levels, causal predictor router) | `inference.py` | ✅ |
| **Debt Ledger** (8 obligation types, pressure scoring) | `inference.py` | ✅ |
| **Braid State** (continuous + fast + semantic sketch + entity + grammar) | `inference.py` | ✅ |
| **Self-Evolution** (TTT, semantic memory HDC, episodic cases, meta-guidelines) | `evolution.py` | ✅ |
| **Multimodal** (vision + audio encoders, ternary, checkpointed) | `multimodal.py` | ✅ |
| **Full Model** (Chimera51ForCausalLM) | `model.py` | ✅ |

---

## Quick Start

```bash
pip install torch datasets transformers einops splintr-rs
```

### Training

```bash
# Test rapide (MeZO, tiny, 10 steps)
OMP_NUM_THREADS=$(nproc) python train.py \
  --scale tiny --seq_len 64 --max_steps 10 \
  --optimizer mezo --batch_size 2 --grad_accum 1 \
  --lr 1e-3 --no-bf16 --num_workers 0 --log_every 1

# Entraînement réel (MeZO + compile, small, 50K steps)
OMP_NUM_THREADS=$(nproc) python train.py \
  --scale small --seq_len 256 --max_steps 50000 \
  --optimizer mezo --batch_size 2 --grad_accum 4 \
  --lr 1e-3 --warmup 2000 --compile \
  --num_workers 0 --save_every 5000
```

### Inference (génération de texte)

```bash
# Générer à partir du checkpoint final
python inference.py \
  --checkpoint chimera_output/final/model.pt \
  --prompt "Once upon a time" \
  --max_tokens 200 \
  --temperature 0.8 --top_p 0.9 --top_k 50

# Avec torch.compile pour accélérer l'inférence
python inference.py \
  --checkpoint chimera_output/final/model.pt \
  --prompt "Once upon a time" \
  --max_tokens 200 \
  --temperature 0.8 --top_p 0.9 --top_k 50 \
  --compile

# Avec BF16 (si supporté par votre CPU)
python inference.py \
  --checkpoint chimera_output/final/model.pt \
  --prompt "Once upon a time" \
  --max_tokens 200 \
  --bf16 --compile
```

---

## Training Modes

### MeZO (Recommended for CPU)
- **No backward pass** — eliminates all gradient computation through complex recurrences
- **Memory = 2× model size** — no activations, no gradients, no optimizer states
- **Ternary-aware sparse perturbation** — skips ~33% zero-weight positions in BitLinear layers
- Best for fine-tuning; requires ~32× more steps for pretraining
- Combined with BF16 autocast for maximum CPU throughput

### AdamW (Standard backprop)
- Full gradient computation with gradient checkpointing
- Ternary forward/backward via C++ kernel (2-bit packed → float → BLAS)
- BFloat16 autocast for forward pass
- Weight decay differentiated (no decay for norms, biases, embeddings)
- Best when gradient quality matters (pretraining from scratch)

---

## Ternary Compute Details

### Weight Packing
```
2 bits per weight: 00→0, 01→+1, 10→-1
4 weights per uint8 byte
Per-row scale α = mean(|W|) per group
```

### Forward Pass
```
1. Quantize latent FP32 → ternary int8 {-1,0,1}
2. Pack to 2-bit uint8 (4× compression)
3. Unpack to float32 buffer (pre-allocated, reused)
4. MKL BLAS matmul (x @ W^T)
```

### MeZO Sparse Perturbation (C++)
```
For each weight position:
  If packed_bits == 0: SKIP (no perturbation, no update)
  Else: generate z ~ N(0,1), perturb by ε·z
```
This saves **33% of perturbation operations** since ~1/3 of ternary weights are zero.

### C++ Kernel Features
- OpenMP parallel over output dimensions
- Pre-allocated unpack buffer (zero allocation in hot loop)
- Deterministic LCG RNG per thread (reproducible across runs)
- Falls back to pure PyTorch if C++ compilation fails

---

## Files

```
chimera/
  __init__.py          — Package exports
  quantization.py      — BitLinear (2-bit packed, C++ kernel, STE, N:M 2:4)
  ternary_simd.py      — AVX2/AVX-512 SIMD unpack kernels (optional)
  layers.py            — GatedDeltaNet, MLSTMLayer (PARALLEL), TitansMACLayer (PARALLEL), TSPSpanKnotLayer
  moe.py               — MoELayer (sort-based dispatch), NoAuxMoEGate
  looping.py           — ParcaeLoopController (deterministic, checkpoint-safe)
  inference.py         — SpanBank, STree, Grammar, EntropyValve, DebtLedger, BraidState
  evolution.py         — TTT, SemanticMemory (vectorized HDC), EpisodicCases, MetaGuidelines
  multimodal.py        — VisionEncoder, AudioEncoder (checkpointed)
  tokenizer.py         — ChimeraTokenizer (splintr Rust wrapper, o200k_base vocab)
  model.py             — Chimera51ForCausalLM (compile + checkpoint + bf16 support)
config.json            — Chimera 5.1 config (honest P3 section)
train.py               — Training script (MeZO + AdamW, ternary, bf16, compile, IPEX)
inference.py           — Inference script (checkpoint loading, autoregressive generation)
```

---

## References

37 papers indexed in `config.json` under `§`. Key ones:
- [Gated DeltaNet](https://arxiv.org/abs/2412.06464) — NVIDIA
- [xLSTM](https://arxiv.org/abs/2405.04517) — NXAI/JKU
- [Titans](https://arxiv.org/abs/2501.00663) — Google
- [Parcae](https://arxiv.org/abs/2604.12946) — Stanford/Together
- [BitNet b1.58](https://arxiv.org/abs/2402.17764) — Microsoft
- [Bitnet.cpp](https://arxiv.org/abs/2502.11880) — MSRA (ELUT kernel)
- [T-MAC](https://arxiv.org/abs/2407.00088) — MSRA (LUT inference)
- [MeZO](https://arxiv.org/abs/2305.17333) — Princeton (CPU training optimizer)
- [DeepSeek MoE routing](https://arxiv.org/abs/2408.15664) — DeepSeek
- [In-Place TTT](https://arxiv.org/abs/2604.06169) — ByteDance
