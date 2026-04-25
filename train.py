#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch

from chimera import Chimera51ForCausalLM, ChimeraTokenizer, load_config, scale_config
from chimera.quantization import setup_cpu_runtime


def make_batch(tokenizer, text: str, batch_size: int, seq_len: int, vocab_size: int):
    ids = tokenizer.encode(text)
    if len(ids) < seq_len + 2:
        ids = (ids * ((seq_len + 2) // max(1, len(ids)) + 1))[: seq_len + 2]
    starts = torch.randint(0, max(1, len(ids) - seq_len - 1), (batch_size,))
    data = torch.stack([torch.tensor(ids[s:s+seq_len+1], dtype=torch.long) for s in starts]) % vocab_size
    return data[:, :-1], data[:, 1:]


@torch.no_grad()
def mezo_step(model, input_ids, labels, lr: float, eps: float, seed: int):
    params = [p for p in model.parameters() if p.requires_grad]
    gen = torch.Generator().manual_seed(seed)
    noises = [torch.empty_like(p).bernoulli_(0.5, generator=gen).mul_(2).sub_(1) for p in params]
    for p, z in zip(params, noises):
        p.add_(z, alpha=eps)
    loss_pos = model(input_ids, labels=labels).loss.detach()
    for p, z in zip(params, noises):
        p.add_(z, alpha=-2 * eps)
    loss_neg = model(input_ids, labels=labels).loss.detach()
    grad_est = (loss_pos - loss_neg) / (2 * eps)
    for p, z in zip(params, noises):
        p.add_(z, alpha=eps)      # restore original
        p.add_(z, alpha=-lr * grad_est)
    return ((loss_pos + loss_neg) / 2).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--scale", default="nano", choices=["nano", "tiny", "small", "base"])
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "mezo"])
    p.add_argument("--text", default="Chimera is a CPU first language model. ")
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mezo_eps", type=float, default=1e-3)
    p.add_argument("--output", default="chimera_output")
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--grad_checkpointing", action="store_true")
    args = p.parse_args()
    setup_cpu_runtime()
    cfg = scale_config(load_config(args.config), args.scale)
    tok = ChimeraTokenizer(vocab_size=cfg["vocab_size"])
    model = Chimera51ForCausalLM(cfg)
    if args.grad_checkpointing:
        model.enable_gradient_checkpointing()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr) if args.optimizer == "adamw" else None
    model.train()
    for step in range(1, args.max_steps + 1):
        x, y = make_batch(tok, args.text, args.batch_size, args.seq_len, cfg["vocab_size"])
        if args.optimizer == "mezo":
            loss = mezo_step(model, x, y, args.lr, args.mezo_eps, step)
        else:
            opt.zero_grad(set_to_none=True)
            out = model(x, labels=y)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss = out.loss.item()
        if step % args.log_every == 0:
            print(f"step={step} loss={loss:.4f}")
    out_dir = Path(args.output) / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg}, out_dir / "model.pt")
    print(f"saved {out_dir / 'model.pt'}")

if __name__ == "__main__":
    main()
