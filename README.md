# Chimera 5.1 CPU — Rebuilt Implementation

This repository is a clean, CPU-only rebuild of the Chimera 5.1 project.  It preserves the original intent and public API while replacing fragile hot paths with deterministic, testable PyTorch code designed for commodity CPUs and 16 GB DDR5 systems.

## Preserved features

- Chimera51ForCausalLM causal LM API
- 1.58-bit ternary `BitLinear` with STE training and 2-bit packed inference snapshots
- Hybrid recurrent/no-attention blocks: Gated DeltaNet, mLSTM, Titans MAC, TSP Span Knot
- Sparse MoE with expert-grouped dispatch
- Parcae loop controller and entropy-based loop routing
- Span bank, grammar/debt hooks, braid state types
- Self-evolution scaffolding: semantic memory, in-place TTT, episodic cases
- Vision/audio encoders
- splintr tokenizer wrapper with byte fallback
- CPU training (`AdamW` or `MeZO`) and autoregressive inference CLIs
- GGUF import utility with safe shape handling

## Quick start

```bash
python -m pytest
python train.py --scale nano --seq_len 16 --max_steps 1 --optimizer adamw
python inference.py --scale nano --prompt "Hello" --max_tokens 8 --temperature 0 --top_k 1
```

Use `--scale nano|tiny|small|base` to select the practical CPU size.  `base` keeps the uploaded dimensions and is not recommended for 16 GB RAM without an imported packed checkpoint.
