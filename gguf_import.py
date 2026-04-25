#!/usr/bin/env python3
"""Import GGUF tensors into a Chimera checkpoint.

This converter is intentionally conservative: it streams tensors when the
optional `gguf` package is available, maps common LLaMA/Qwen names, auto-crops or
pads shape mismatches, and never treats token embeddings/lm_head as BitLinear.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from chimera import Chimera51ForCausalLM, load_config, scale_config
from chimera.quantization import BitLinear


def fit_tensor(src: torch.Tensor, dst: torch.Tensor, mode: str = "crop_pad") -> torch.Tensor:
    src = src.to(dtype=dst.dtype)
    if src.shape == dst.shape:
        return src
    if src.ndim == 2 and src.t().shape == dst.shape:
        return src.t().contiguous()
    if mode == "strict":
        raise ValueError(f"shape mismatch {tuple(src.shape)} -> {tuple(dst.shape)}")
    out = torch.zeros_like(dst)
    slices = tuple(slice(0, min(a, b)) for a, b in zip(src.shape, dst.shape))
    out[slices].copy_(src[slices])
    return out


def map_name(name: str) -> str | None:
    n = name.replace("blk.", "layers.").replace("token_embd.weight", "embed.weight").replace("output.weight", "lm_head.weight")
    repl = {
        ".attn_q.weight": ".attn.in_proj.weight",
        ".ffn_gate.weight": ".mlp.gate.weight",
        ".ffn_up.weight": ".mlp.up.weight",
        ".ffn_down.weight": ".mlp.down.weight",
        ".attn_norm.weight": ".attn_norm.weight",
        ".ffn_norm.weight": ".mlp_norm.weight",
    }
    for a, b in repl.items():
        n = n.replace(a, b)
    return n


def load_gguf_tensors(path: str):
    try:
        from gguf import GGUFReader
    except Exception as exc:
        raise RuntimeError("install gguf (`pip install gguf`) to read GGUF files") from exc
    reader = GGUFReader(path)
    for tensor in reader.tensors:
        yield tensor.name, torch.tensor(tensor.data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    p.add_argument("--config", default="config.json")
    p.add_argument("--scale", default="tiny", choices=["nano", "tiny", "small", "base"])
    p.add_argument("--output", default="chimera_imported/model.pt")
    p.add_argument("--resize", default="crop_pad", choices=["strict", "crop_pad"])
    p.add_argument("--pack", action="store_true")
    args = p.parse_args()
    cfg = scale_config(load_config(args.config), args.scale)
    model = Chimera51ForCausalLM(cfg)
    state = model.state_dict()
    loaded = 0
    for name, tensor in load_gguf_tensors(args.gguf):
        target = map_name(name)
        if target in state:
            state[target].copy_(fit_tensor(tensor.float(), state[target], args.resize))
            loaded += 1
    model.load_state_dict(state, strict=False)
    if args.pack:
        model.prepare_for_inference()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg, "loaded_tensors": loaded}, args.output)
    print(f"saved {args.output}; loaded_tensors={loaded}")

if __name__ == "__main__":
    main()
