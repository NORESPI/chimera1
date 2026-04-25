#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import torch
import torch.nn.functional as F

from chimera import Chimera51ForCausalLM, ChimeraTokenizer, load_config, scale_config
from chimera.quantization import setup_cpu_runtime


def sample_next(logits, temperature=0.8, top_k=50, top_p=0.9):
    logits = logits.float()
    if temperature <= 0 or top_k == 1:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / max(temperature, 1e-6)
    if top_k and top_k > 0:
        vals, idx = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        mask = torch.full_like(logits, float("-inf")); logits = mask.scatter(-1, idx, vals)
    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        remove = probs.cumsum(dim=-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone(); remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return torch.multinomial(F.softmax(logits, dim=-1), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint")
    p.add_argument("--config", default="config.json")
    p.add_argument("--scale", default="tiny", choices=["nano", "tiny", "small", "base"])
    p.add_argument("--prompt", default="Hello")
    p.add_argument("--max_tokens", type=int, default=64)
    p.add_argument("--max_context", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--num_loops", type=int)
    args = p.parse_args()
    setup_cpu_runtime()
    cfg = scale_config(load_config(args.config), args.scale)
    tok = ChimeraTokenizer(vocab_size=cfg["vocab_size"])
    model = Chimera51ForCausalLM(cfg)
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.prepare_for_inference()
    ids = torch.tensor([tok.encode(args.prompt) or [tok.bos_token_id]], dtype=torch.long)
    generated = ids.clone()
    with torch.inference_mode():
        for _ in range(args.max_tokens):
            ctx = generated[:, -args.max_context:]
            out = model(ctx, logits_to_keep=1, num_loops=args.num_loops)
            nxt = sample_next(out.logits[:, -1], args.temperature, args.top_k, args.top_p)
            generated = torch.cat([generated, nxt], dim=1)
            if int(nxt) == tok.eos_token_id:
                break
    print(tok.decode(generated[0].tolist()))

if __name__ == "__main__":
    main()
