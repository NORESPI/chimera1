#!/usr/bin/env python3
"""
Chimera 5.1 — Inference Script
Load trained checkpoint and generate text autoregressively.

Usage:
    python inference.py \
        --checkpoint chimera_output/final/model.pt \
        --prompt "Once upon a time" \
        --max_tokens 100 \
        --temperature 0.8 \
        --top_p 0.9 \
        --top_k 50
"""

import argparse
import json
import os
import time

# CPU runtime defaults must be set before importing torch.
def _setup_cpu_runtime():
    n = os.cpu_count() or 4
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
    os.environ.setdefault("KMP_BLOCKTIME", "1")
    os.environ.setdefault("MALLOC_CONF", "background_thread:true,metadata_thp:auto")

_setup_cpu_runtime()

import torch
import torch.nn.functional as F

try:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 4)))
    torch.set_num_interop_threads(int(os.environ.get("CHIMERA_INTEROP_THREADS", "1")))
except RuntimeError:
    pass

from chimera import Chimera51ForCausalLM, ChimeraTokenizer

def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load model from checkpoint — config extraite du checkpoint en priorité."""
    print(f"[LOAD] Checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # ✅ Priorité au config embarqué dans le checkpoint
    config = ckpt.get("config", None)
    if config is None:
        # Fallback : chercher config.json à côté
        checkpoint_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(checkpoint_dir, "config.json") if checkpoint_dir else "config.json"
        if not os.path.exists(config_path):
            config_path = "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"[LOAD] Config from {config_path}")
    else:
        print(f"[LOAD] Config from checkpoint (embedded)")

    print(f"[LOAD] Model: {config.get('model_name', 'chimera-5.1')} "
          f"(vocab={config.get('vocab_size', '?')}, h={config.get('hidden_size', '?')})")

    model = Chimera51ForCausalLM(config)
    print(f"[LOAD] Parameters: {model.count_parameters()['total']:,}")

    state = ckpt.get("model", ckpt)

    # Handle vocab size mismatch
    model_vocab = config.get("vocab_size", 200073)
    ckpt_vocab = None
    for key, tensor in state.items():
        if key.endswith("embed.weight") or key == "embed.weight":
            ckpt_vocab = tensor.shape[0]
            break
        if key.endswith("lm_head.weight") or key == "lm_head.weight":
            ckpt_vocab = tensor.shape[0]
            break

    if ckpt_vocab and ckpt_vocab != model_vocab:
        print(f"[WARN] Vocab mismatch: checkpoint={ckpt_vocab}, config={model_vocab}")
        print(f"[WARN] Resizing model to {ckpt_vocab} tokens...")
        with torch.no_grad():
            old_embed = model.embed.weight.data
            old_vocab = old_embed.shape[0]
            new_embed = torch.zeros(ckpt_vocab, old_embed.shape[1],
                                    dtype=old_embed.dtype, device=old_embed.device)
            new_embed[:min(old_vocab, ckpt_vocab)] = old_embed[:min(old_vocab, ckpt_vocab)]
            model.embed = torch.nn.Embedding(ckpt_vocab, old_embed.shape[1])
            model.embed.weight.data = new_embed
            old_head = model.lm_head.weight.data
            new_head = torch.zeros(ckpt_vocab, old_head.shape[1],
                                   dtype=old_head.dtype, device=old_head.device)
            new_head[:min(old_vocab, ckpt_vocab)] = old_head[:min(old_vocab, ckpt_vocab)]
            model.lm_head = torch.nn.Linear(old_head.shape[1], ckpt_vocab, bias=False)
            model.lm_head.weight.data = new_head
        config["vocab_size"] = ckpt_vocab

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    model.to(device).eval()

    step = ckpt.get("step", "?")
    best_loss = ckpt.get("best_loss", None)
    print(f"[LOAD] Step {step}, best_loss={best_loss:.4f}" if best_loss is not None
          else f"[LOAD] Step {step}")

    return model, config


def generate(
    model: Chimera51ForCausalLM,
    tokenizer: ChimeraTokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    device: str = "cpu",
    bf16: bool = False,
    max_context: int = 0,
):
    """Autoregressive text generation with sampling."""
    model.eval()

    # Encode prompt and pre-allocate the growing context to avoid O(T²) cat reallocs.
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    # Recurrent layers in this architecture do not expose a KV cache, so CPU
    # generation recomputes the visible context.  Bound it explicitly for real
    # deployments to prevent quadratic latency growth during long generations.
    visible_context = max_context if max_context and max_context > 0 else len(input_ids) + max_tokens
    alloc_context = min(len(input_ids) + max_tokens, max(visible_context, 1))
    input_buffer = torch.empty((1, alloc_context), dtype=torch.long, device=device)
    prompt_ids = input_ids[-alloc_context:]
    input_buffer[0, :len(prompt_ids)] = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    cur_len = len(prompt_ids)

    print(f"\n[GEN] Prompt: {prompt!r}")
    print(f"[GEN] max_tokens={max_tokens}, temp={temperature}, top_p={top_p}, top_k={top_k}")
    print("=" * 60)

    generated = list(input_ids)
    t0 = time.time()

    with torch.inference_mode():
        for i in range(max_tokens):
            input_tensor = input_buffer[:, :cur_len]
            # Forward pass; only materialize last-token logits to avoid [B,T,V] CPU work.
            if bf16:
                with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                    _, logits = model(input_tensor, logits_to_keep=1)
            else:
                _, logits = model(input_tensor, logits_to_keep=1)

            # Get next token logits (last position)
            next_logits = logits[:, -1, :].float() / max(temperature, 1e-6)

            # Greedy path: fastest for deterministic CPU serving; avoids softmax,
            # multinomial and sort entirely.
            if temperature <= 0:
                next_token = torch.argmax(next_logits, dim=-1).item()
            # Fast sampling: restrict to top-k first so top-p never sorts the full
            # 200K vocabulary in the common case (top_k=50 by default).
            elif top_k > 0:
                k = min(top_k, next_logits.size(-1))
                cand_logits, cand_indices = torch.topk(next_logits, k, dim=-1)
                if top_p < 1.0:
                    sorted_logits, sorted_order = torch.sort(cand_logits, descending=True)
                    sorted_indices = cand_indices.gather(1, sorted_order)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cumulative_probs > top_p
                    remove[..., 0] = False
                    sorted_logits = sorted_logits.masked_fill(remove, -float('inf'))
                    probs = F.softmax(sorted_logits, dim=-1)
                    next_token = sorted_indices.gather(1, torch.multinomial(probs, 1)).item()
                else:
                    probs = F.softmax(cand_logits, dim=-1)
                    next_token = cand_indices.gather(1, torch.multinomial(probs, 1)).item()
            else:
                # Full-vocab nucleus fallback only when explicitly requested.
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cumulative_probs > top_p
                    remove[..., 0] = False
                    sorted_logits = sorted_logits.masked_fill(remove, -float('inf'))
                    probs = F.softmax(sorted_logits, dim=-1)
                    next_token = sorted_indices.gather(1, torch.multinomial(probs, 1)).item()
                else:
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()

            # Stop on EOS
            if next_token == tokenizer.eos_token_id:
                break

            generated.append(next_token)
            if cur_len >= input_buffer.shape[1]:
                # Sliding window without reallocating.  copy_ handles overlap safely
                # for this 1-row buffer and keeps generation bounded.
                input_buffer[:, :-1].copy_(input_buffer[:, 1:].clone())
                input_buffer[0, -1] = next_token
            else:
                input_buffer[0, cur_len] = next_token
                cur_len += 1

            # Print streaming
            if (i + 1) % 10 == 0:
                print(f"\r[GEN] {i+1}/{max_tokens} tokens...", end="", flush=True)

    elapsed = time.time() - t0
    n_new = len(generated) - len(input_ids)
    speed = n_new / elapsed if elapsed > 0 else 0

    print(f"\r{' ' * 50}")
    print("=" * 60)
    full_text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"\n{full_text}\n")
    print(f"[STATS] {n_new} new tokens in {elapsed:.2f}s ({speed:.1f} tok/s)")

    return full_text


def main():
    p = argparse.ArgumentParser(description="Chimera 5.1 Inference")
    p.add_argument("--checkpoint", default="chimera_output/final/model.pt",
                   help="Path to checkpoint .pt file")
    p.add_argument("--prompt", default="Once upon a time", help="Generation prompt")
    p.add_argument("--max_tokens", type=int, default=100,
                   help="Maximum new tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--max_context", type=int, default=0,
                   help="Sliding visible context limit; 0 keeps full prompt+generation")
    p.add_argument("--device", default="cpu")
    p.add_argument("--bf16", action="store_true", default=True,
                   help="Use BFloat16 autocast (CPU only, default: True)")
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--threads", type=int, default=None,
                   help="Override torch/OMP thread count")
    p.add_argument("--compile", action="store_true", default=False,
                   help="Compile model with torch.compile for faster inference")
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)

    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        print("Train first with: python train.py ...")
        return

    # Load model
    model, config = load_model(args.checkpoint, device=args.device)

    # torch.compile for inference speed
    if args.compile:
        print("[OPT] Compiling model with torch.compile...")
        model = torch.compile(model, backend="inductor", mode="reduce-overhead")

    # Load tokenizer
    print("[LOAD] Loading tokenizer (splintr o200k_base)...")
    tokenizer = ChimeraTokenizer(pretrained="o200k_base")

    # Warmup (compile + cache)
    print("[WARM] Running warmup pass...")
    dummy = torch.tensor([[tokenizer.eos_token_id]], device=args.device)
    with torch.inference_mode():
        _ = model(dummy, logits_to_keep=1)
    print("[WARM] Done.")

    # Generate
    generate(
        model, tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        device=args.device,
        bf16=args.bf16,
        max_context=args.max_context,
    )


if __name__ == "__main__":
    main()
