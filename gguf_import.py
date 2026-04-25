#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chimera GGUF Import Optimized
═════════════════════════════

Convert GGUF tensors into a Chimera-compatible checkpoint.

Améliorations vs version originale :
  - Ne garde pas tous les tensors GGUF FP32 en mémoire.
  - Corrige le bug embeddings/lm_head traités comme BitLinear.
  - Quantization ternary offline sans autograd.
  - Clipping outlier par ligne pour les matrices.
  - Auto-transpose si shape inversée.
  - Modes de stockage :
      fp32   : compatible Chimera classique, sauvegarde weight latent.
      packed : sauvegarde packed_weight + alpha uniquement pour couches linéaires.
      both   : sauvegarde weight + packed_weight + alpha.
  - Init des poids manquants pour checkpoint complet.
  - Resize configurable : strict, crop_pad, interpolate.
  - Mapping GGUF plus robuste pour LLaMA/Qwen/Mistral-like.

Usage :
    python gguf_import_optimized.py \
        --gguf model.gguf \
        --config config.json \
        --scale tiny \
        --output imported_chimera.pt \
        --storage fp32

Pour checkpoint compact expérimental :
    python gguf_import_optimized.py \
        --gguf model.gguf \
        --config config.json \
        --output imported_chimera_packed.pt \
        --storage packed

Attention :
  - storage=packed nécessite que ton loader Chimera sache lire
    *.packed_weight et *.alpha.
  - Importer un gros modèle vers tiny/small via resize détruit beaucoup
    d'information. C'est utile pour bootstrap, pas équivalent à distillation.
"""

import os
import re
import gc
import json
import math
import argparse
from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple, Optional, Iterable, Any

import numpy as np
import torch
import torch.nn.functional as F


try:
    from gguf import GGUFReader, dequantize
    HAS_GGUF = True
except Exception:
    GGUFReader = None
    dequantize = None
    HAS_GGUF = False


# ═══════════════════════════════════════════════════════════
# Config scales
# ═══════════════════════════════════════════════════════════

SCALE_OVERRIDES = {
    "tiny": {
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 28,
        "num_heads": 4,
        "head_dim": 48,
    },
    "small": {
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_hidden_layers": 28,
        "num_heads": 8,
        "head_dim": 48,
    },
    "medium": {
        "hidden_size": 1024,
        "intermediate_size": 2048,
        "num_hidden_layers": 28,
        "num_heads": 8,
        "head_dim": 96,
    },
    # full = garde config telle quelle
    "full": {},
}


# ═══════════════════════════════════════════════════════════
# Mapping GGUF -> Chimera
# ═══════════════════════════════════════════════════════════

DIRECT_NAME_MAP = {
    "token_embd": "embed.weight",
    "token_embd.weight": "embed.weight",

    "output": "lm_head.weight",
    "output.weight": "lm_head.weight",

    "output_norm": "norm.weight",
    "output_norm.weight": "norm.weight",

    # Variants parfois rencontrées
    "norm": "norm.weight",
    "norm.weight": "norm.weight",
}


BLOCK_SUFFIX_MAP = {
    # Attention norm
    "attn_norm": "attn_norm.weight",
    "attn_norm.weight": "attn_norm.weight",

    # FFN norm
    "ffn_norm": "mlp_norm.weight",
    "ffn_norm.weight": "mlp_norm.weight",

    # Attention projections
    "attn_q": "attn.q_proj.weight",
    "attn_q.weight": "attn.q_proj.weight",
    "attn_k": "attn.k_proj.weight",
    "attn_k.weight": "attn.k_proj.weight",
    "attn_v": "attn.v_proj.weight",
    "attn_v.weight": "attn.v_proj.weight",
    "attn_output": "attn.o_proj.weight",
    "attn_output.weight": "attn.o_proj.weight",

    # MLP / SwiGLU
    "ffn_gate": "mlp.gate_proj.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up": "mlp.up_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down": "mlp.down_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}


def map_gguf_name(name: str, n_layers: int) -> Optional[str]:
    """
    Convertit un nom GGUF vers une clé Chimera.
    Retourne None si non mappable.
    """
    if name in DIRECT_NAME_MAP:
        return DIRECT_NAME_MAP[name]

    m = re.match(r"^blk\.(\d+)\.(.+)$", name)
    if not m:
        return None

    bid = int(m.group(1))
    suffix = m.group(2)

    if bid >= n_layers:
        return None

    mapped_suffix = BLOCK_SUFFIX_MAP.get(suffix)
    if mapped_suffix is None:
        return None

    return f"layers.{bid}.{mapped_suffix}"


# ═══════════════════════════════════════════════════════════
# Ternary quantization + packing
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def ternary_quantize_absmean(
    w: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convertit w FP32 [M,K] -> w_q int8 {-1,0,1} + alpha [M].

    alpha = mean(abs(w), dim=1)
    w_norm = w / alpha
    q = -1 si w_norm <= -threshold
        0 si entre
        +1 si w_norm >= threshold
    """
    if w.ndim != 2:
        raise ValueError("ternary_quantize_absmean attend un tensor 2D")

    w = w.to(torch.float32)
    alpha = w.abs().mean(dim=1).clamp_min(eps)

    wn = w / alpha[:, None]
    q = torch.zeros_like(wn, dtype=torch.int8)
    q[wn >= threshold] = 1
    q[wn <= -threshold] = -1

    return q, alpha.to(torch.float32)


@torch.no_grad()
def pack_ternary_2bit(w_q: torch.Tensor) -> torch.Tensor:
    """
    Pack int8 {-1,0,+1} -> uint8, 4 poids par byte.

    Encoding :
      0  -> 00
      +1 -> 01
      -1 -> 10

    Ordre :
      weight0 bits 7..6
      weight1 bits 5..4
      weight2 bits 3..2
      weight3 bits 1..0
    """
    if w_q.ndim != 2:
        raise ValueError("pack_ternary_2bit attend un tensor 2D")

    M, K = w_q.shape
    K4 = (K + 3) // 4
    pad = K4 * 4 - K

    codes = torch.zeros_like(w_q, dtype=torch.uint8)
    codes[w_q == 1] = 1
    codes[w_q == -1] = 2

    if pad:
        codes = F.pad(codes, (0, pad), value=0)

    codes = codes.view(M, K4, 4)
    packed = (
        (codes[..., 0] << 6)
        | (codes[..., 1] << 4)
        | (codes[..., 2] << 2)
        | codes[..., 3]
    )
    return packed.contiguous()


# ═══════════════════════════════════════════════════════════
# Noise reduction
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def reduce_noise(
    w: torch.Tensor,
    method: str = "row_outlier_clip",
    sigma: float = 3.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Prétraitement avant ternarisation.

    none              : rien.
    global_clip       : clip global mean ± sigma*std.
    row_outlier_clip  : clip par ligne, meilleur pour matrices linéaires.
    median_center     : recentrage robuste global median/MAD.
    """
    if method == "none":
        return w

    w = w.to(torch.float32)

    if method == "global_clip":
        mu = w.mean()
        std = w.std(unbiased=False).clamp_min(eps)
        return w.clamp(mu - sigma * std, mu + sigma * std)

    if method == "row_outlier_clip":
        if w.ndim != 2:
            return reduce_noise(w, method="global_clip", sigma=sigma, eps=eps)

        mu = w.mean(dim=1, keepdim=True)
        std = w.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
        return w.clamp(mu - sigma * std, mu + sigma * std)

    if method == "median_center":
        med = w.median()
        mad = (w - med).abs().median().clamp_min(eps)
        return (w - med) / mad

    return w


# ═══════════════════════════════════════════════════════════
# Resize helpers
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def resize_1d(w: torch.Tensor, target: int) -> torch.Tensor:
    src = w.numel()
    if src == target:
        return w.contiguous()

    out = torch.ones(target, dtype=w.dtype)
    n = min(src, target)
    out[:n] = w[:n]
    return out.contiguous()


@torch.no_grad()
def resize_2d_crop_pad(
    w: torch.Tensor,
    target_shape: Tuple[int, int],
    fill_std: float = 0.02,
) -> torch.Tensor:
    """
    Resize rapide par crop/pad.
    Plus prévisible qu'une interpolation sur poids Transformer.
    """
    target_out, target_in = target_shape
    src_out, src_in = w.shape

    if (src_out, src_in) == (target_out, target_in):
        return w.contiguous()

    out = torch.empty((target_out, target_in), dtype=w.dtype)

    # init zones non copiées
    std = float(w.std(unbiased=False).item()) if w.numel() > 1 else fill_std
    std = max(min(std, 0.2), 1e-4)
    out.normal_(mean=0.0, std=std)

    ro = min(src_out, target_out)
    ci = min(src_in, target_in)
    out[:ro, :ci] = w[:ro, :ci]

    return out.contiguous()


@torch.no_grad()
def resize_2d_interpolate(
    w: torch.Tensor,
    target_shape: Tuple[int, int],
) -> torch.Tensor:
    target_out, target_in = target_shape
    if tuple(w.shape) == tuple(target_shape):
        return w.contiguous()

    x = w[None, None, :, :]
    y = F.interpolate(
        x,
        size=(target_out, target_in),
        mode="bilinear",
        align_corners=False,
    )
    return y[0, 0].contiguous()


@torch.no_grad()
def resize_2d(
    w: torch.Tensor,
    target_shape: Tuple[int, int],
    strategy: str = "crop_pad",
) -> torch.Tensor:
    if tuple(w.shape) == tuple(target_shape):
        return w.contiguous()

    if strategy == "strict":
        raise ValueError(f"Shape mismatch: got {tuple(w.shape)}, expected {target_shape}")

    if strategy == "crop_pad":
        return resize_2d_crop_pad(w, target_shape)

    if strategy == "interpolate":
        return resize_2d_interpolate(w, target_shape)

    raise ValueError(f"resize strategy inconnue: {strategy}")


# ═══════════════════════════════════════════════════════════
# Importer
# ═══════════════════════════════════════════════════════════

class OptimizedGGUFImporter:
    def __init__(
        self,
        config: Dict[str, Any],
        scale: str = "tiny",
        storage: str = "fp32",
        param_dtype: str = "fp32",
        noise_method: str = "row_outlier_clip",
        noise_sigma: float = 3.0,
        ternary_threshold: float = 0.5,
        resize_strategy: str = "crop_pad",
        auto_transpose: bool = True,
        init_missing: bool = True,
        verbose: bool = True,
    ):
        self.config = deepcopy(config)
        self.scale = scale
        self.storage = storage
        self.param_dtype = param_dtype
        self.noise_method = noise_method
        self.noise_sigma = noise_sigma
        self.ternary_threshold = ternary_threshold
        self.resize_strategy = resize_strategy
        self.auto_transpose = auto_transpose
        self.init_missing = init_missing
        self.verbose = verbose

        if scale not in SCALE_OVERRIDES:
            raise ValueError(f"scale invalide: {scale}")

        self.config.update(SCALE_OVERRIDES[scale])

        self.n_layers = int(self.config["num_hidden_layers"])
        self.hidden_size = int(self.config["hidden_size"])
        self.vocab_size = int(self.config["vocab_size"])
        self.num_heads = int(self.config.get("num_heads", 4))
        self.head_dim = int(self.config.get("head_dim", self.hidden_size // self.num_heads))

        inter = int(self.config["intermediate_size"])
        self.intermediate_size = 256 * ((inter + 255) // 256)
        self.config["intermediate_size"] = self.intermediate_size

        if storage not in {"fp32", "packed", "both"}:
            raise ValueError("storage doit être: fp32, packed ou both")

        if param_dtype not in {"fp32", "fp16", "bf16"}:
            raise ValueError("param_dtype doit être: fp32, fp16 ou bf16")

        if self.verbose:
            self.log(
                f"[CONFIG] scale={scale} h={self.hidden_size} "
                f"layers={self.n_layers} heads={self.num_heads} "
                f"head_dim={self.head_dim} inter={self.intermediate_size} "
                f"vocab={self.vocab_size}"
            )
            self.log(
                f"[CONFIG] storage={storage} param_dtype={param_dtype} "
                f"resize={resize_strategy} noise={noise_method}"
            )

    def log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    def target_dtype(self):
        if self.param_dtype == "fp16":
            return torch.float16
        if self.param_dtype == "bf16":
            return torch.bfloat16
        return torch.float32

    def infer_shape(self, key: str) -> Tuple[int, ...]:
        h = self.hidden_size
        attn_dim = self.num_heads * self.head_dim

        if key == "embed.weight":
            return (self.vocab_size, h)

        if key == "lm_head.weight":
            return (self.vocab_size, h)

        if key == "norm.weight":
            return (h,)

        if key.endswith("attn_norm.weight") or key.endswith("mlp_norm.weight"):
            return (h,)

        if key.endswith("attn.q_proj.weight"):
            return (attn_dim, h)
        if key.endswith("attn.k_proj.weight"):
            return (attn_dim, h)
        if key.endswith("attn.v_proj.weight"):
            return (attn_dim, h)
        if key.endswith("attn.o_proj.weight"):
            return (h, attn_dim)

        if key.endswith("mlp.gate_proj.weight"):
            return (self.intermediate_size, h)
        if key.endswith("mlp.up_proj.weight"):
            return (self.intermediate_size, h)
        if key.endswith("mlp.down_proj.weight"):
            return (h, self.intermediate_size)

        raise KeyError(f"Impossible d'inférer la shape pour {key}")

    def all_expected_keys(self) -> Iterable[str]:
        yield "embed.weight"
        yield "norm.weight"
        yield "lm_head.weight"

        for i in range(self.n_layers):
            prefix = f"layers.{i}"
            yield f"{prefix}.attn_norm.weight"
            yield f"{prefix}.mlp_norm.weight"
            yield f"{prefix}.attn.q_proj.weight"
            yield f"{prefix}.attn.k_proj.weight"
            yield f"{prefix}.attn.v_proj.weight"
            yield f"{prefix}.attn.o_proj.weight"
            yield f"{prefix}.mlp.gate_proj.weight"
            yield f"{prefix}.mlp.up_proj.weight"
            yield f"{prefix}.mlp.down_proj.weight"

    def is_linear_key(self, key: str) -> bool:
        return any(
            key.endswith(s)
            for s in (
                "attn.q_proj.weight",
                "attn.k_proj.weight",
                "attn.v_proj.weight",
                "attn.o_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            )
        )

    def is_embedding_or_head(self, key: str) -> bool:
        return key in {"embed.weight", "lm_head.weight"}

    def maybe_transpose(self, w: torch.Tensor, expected: Tuple[int, ...], key: str) -> torch.Tensor:
        if not self.auto_transpose:
            return w

        if w.ndim == 2 and len(expected) == 2:
            if tuple(w.shape) != tuple(expected) and tuple(w.t().shape) == tuple(expected):
                self.log(f"  [TRANSPOSE] {key}: {tuple(w.shape)} -> {tuple(w.t().shape)}")
                return w.t().contiguous()

        return w

    def convert_tensor(
        self,
        gguf_name: str,
        key: str,
        arr: np.ndarray,
    ) -> Optional[Dict[str, torch.Tensor]]:
        expected = self.infer_shape(key)

        w = torch.from_numpy(np.asarray(arr)).to(torch.float32)
        w = self.maybe_transpose(w, expected, key)

        result: Dict[str, torch.Tensor] = {}

        # 1D norms
        if len(expected) == 1:
            if w.ndim != 1:
                self.log(f"  [SKIP] {gguf_name}: expected 1D {expected}, got {tuple(w.shape)}")
                return None

            if tuple(w.shape) != tuple(expected):
                self.log(f"  [RESIZE-1D] {gguf_name}: {tuple(w.shape)} -> {expected}")
                w = resize_1d(w, expected[0])

            result[key] = w.to(self.target_dtype()).contiguous()
            return result

        # Embeddings/lm_head doivent rester denses, pas ternaires ici.
        if self.is_embedding_or_head(key):
            if w.ndim != 2:
                self.log(f"  [SKIP] {gguf_name}: expected 2D embedding/head, got {tuple(w.shape)}")
                return None

            if tuple(w.shape) != tuple(expected):
                self.log(f"  [RESIZE-EMB] {gguf_name}: {tuple(w.shape)} -> {expected}")
                w = resize_2d(w, expected, self.resize_strategy)

            result[key] = w.to(self.target_dtype()).contiguous()
            return result

        # Linéaires BitLinear
        if self.is_linear_key(key):
            if w.ndim != 2:
                self.log(f"  [SKIP] {gguf_name}: expected 2D linear, got {tuple(w.shape)}")
                return None

            if tuple(w.shape) != tuple(expected):
                self.log(f"  [RESIZE-2D] {gguf_name}: {tuple(w.shape)} -> {expected}")
                w = resize_2d(w, expected, self.resize_strategy)

            w = reduce_noise(w, method=self.noise_method, sigma=self.noise_sigma)

            if self.storage in {"fp32", "both"}:
                result[key] = w.to(self.target_dtype()).contiguous()

            if self.storage in {"packed", "both"}:
                q, alpha = ternary_quantize_absmean(
                    w,
                    threshold=self.ternary_threshold,
                )
                packed = pack_ternary_2bit(q)
                result[f"{key}.packed_weight"] = packed.cpu().contiguous()
                result[f"{key}.alpha"] = alpha.cpu().contiguous()
                result[f"{key}.shape"] = torch.tensor(list(expected), dtype=torch.int32)

            return result

        self.log(f"  [SKIP] {gguf_name}: key non reconnue {key}")
        return None

    def init_missing_tensor(self, key: str) -> Dict[str, torch.Tensor]:
        expected = self.infer_shape(key)
        out: Dict[str, torch.Tensor] = {}

        if len(expected) == 1:
            # Norms : init à 1.0
            w = torch.ones(expected, dtype=self.target_dtype())
            out[key] = w
            return out

        if key in {"embed.weight", "lm_head.weight"}:
            w = torch.empty(expected, dtype=torch.float32)
            w.normal_(0.0, 0.02)
            out[key] = w.to(self.target_dtype())
            return out

        if self.is_linear_key(key):
            w = torch.empty(expected, dtype=torch.float32)
            fan_in = max(1, expected[1])
            std = math.sqrt(2.0 / fan_in)
            w.normal_(0.0, std)

            if self.storage in {"fp32", "both"}:
                out[key] = w.to(self.target_dtype()).contiguous()

            if self.storage in {"packed", "both"}:
                q, alpha = ternary_quantize_absmean(w, threshold=self.ternary_threshold)
                out[f"{key}.packed_weight"] = pack_ternary_2bit(q)
                out[f"{key}.alpha"] = alpha
                out[f"{key}.shape"] = torch.tensor(list(expected), dtype=torch.int32)

            return out

        return out

    def dequantize_tensor(self, tensor) -> np.ndarray:
        """
        Dequantize GGUF tensor vers numpy float32.
        Compatible avec l'API gguf-py la plus courante.
        """
        qtype = getattr(tensor, "tensor_type", None)
        data = getattr(tensor, "data", None)

        if data is None:
            raise RuntimeError(f"Tensor GGUF sans data: {getattr(tensor, 'name', '?')}")

        try:
            arr = dequantize(data, qtype)
        except Exception:
            # Certains tensors peuvent déjà être float array
            arr = np.asarray(data)

        arr = np.asarray(arr)

        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)

        return np.ascontiguousarray(arr)

    def read_arch(self, reader) -> str:
        try:
            field = reader.fields.get("general.architecture")
            if field is None:
                return "unknown"
            # gguf-py field formats can vary.
            if hasattr(field, "parts") and field.parts:
                return str(field.parts[-1])
            return str(field)
        except Exception:
            return "unknown"

    def import_model(self, gguf_path: str, output_path: str) -> Dict[str, Any]:
        if not HAS_GGUF:
            raise ImportError("Package gguf manquant. Installe avec: pip install gguf")

        gguf_path = str(gguf_path)
        output_path = str(output_path)

        self.log("=" * 70)
        self.log("CHIMERA GGUF IMPORT OPTIMIZED")
        self.log("=" * 70)

        reader = GGUFReader(gguf_path)
        arch = self.read_arch(reader)

        self.log(f"[GGUF] file={gguf_path}")
        self.log(f"[GGUF] arch={arch}")
        self.log(f"[GGUF] tensors={len(reader.tensors)}")

        state_dict: Dict[str, torch.Tensor] = {}

        stats = {
            "mapped": 0,
            "unmapped": 0,
            "skipped": 0,
            "linear": 0,
            "dense": 0,
            "norm": 0,
            "resized_or_transposed_possible": 0,
        }

        imported_keys = set()

        for idx, tensor in enumerate(reader.tensors):
            name = str(tensor.name)
            key = map_gguf_name(name, self.n_layers)

            if key is None:
                stats["unmapped"] += 1
                if self.verbose:
                    self.log(f"  [UNMAPPED] {name}")
                continue

            try:
                arr = self.dequantize_tensor(tensor)
                converted = self.convert_tensor(name, key, arr)

                if not converted:
                    stats["skipped"] += 1
                    continue

                state_dict.update(converted)
                imported_keys.add(key)
                stats["mapped"] += 1

                if self.is_linear_key(key):
                    stats["linear"] += 1
                elif key in {"embed.weight", "lm_head.weight"}:
                    stats["dense"] += 1
                else:
                    stats["norm"] += 1

                if self.verbose:
                    qtype = getattr(tensor, "tensor_type", "?")
                    shape = tuple(arr.shape)
                    self.log(f"  [OK] {idx+1:04d} {name} -> {key} shape={shape} qtype={qtype}")

            except Exception as e:
                stats["skipped"] += 1
                self.log(f"  [ERROR] {name}: {type(e).__name__}: {e}")

            finally:
                # Libère le FP32 temporaire.
                try:
                    del arr
                except Exception:
                    pass
                gc.collect()

        # Init des clés manquantes
        missing = []
        if self.init_missing:
            for key in self.all_expected_keys():
                if key not in imported_keys:
                    missing.append(key)
                    init_tensors = self.init_missing_tensor(key)
                    state_dict.update(init_tensors)

            if missing:
                self.log(f"[MISSING] {len(missing)} tensors initialisés automatiquement")

        ckpt = {
            "model": state_dict,
            "config": self.config,
            "source": {
                "gguf_path": gguf_path,
                "gguf_arch": arch,
                "scale": self.scale,
                "storage": self.storage,
                "param_dtype": self.param_dtype,
                "noise_method": self.noise_method,
                "noise_sigma": self.noise_sigma,
                "ternary_threshold": self.ternary_threshold,
                "resize_strategy": self.resize_strategy,
                "auto_transpose": self.auto_transpose,
            },
            "stats": stats,
            "missing_keys": missing,
            "import_version": "2.0-optimized",
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, output_path)

        gguf_mb = os.path.getsize(gguf_path) / 1024 / 1024
        out_mb = os.path.getsize(output_path) / 1024 / 1024

        self.log("")
        self.log("=" * 70)
        self.log("[DONE]")
        self.log(f"[STATS] {stats}")
        self.log(f"[SIZE] GGUF={gguf_mb:.2f} MB -> checkpoint={out_mb:.2f} MB")
        self.log(f"[SAVE] {output_path}")
        self.log("=" * 70)

        return ckpt


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Optimized GGUF -> Chimera checkpoint importer"
    )

    parser.add_argument("--gguf", required=True, help="Path to input .gguf")
    parser.add_argument("--config", default="config.json", help="Chimera config.json")
    parser.add_argument("--output", required=True, help="Output .pt checkpoint")

    parser.add_argument(
        "--scale",
        default="tiny",
        choices=["tiny", "small", "medium", "full"],
        help="Chimera scale override",
    )

    parser.add_argument(
        "--storage",
        default="fp32",
        choices=["fp32", "packed", "both"],
        help=(
            "fp32=compatible Chimera classique, "
            "packed=2-bit seulement, both=les deux"
        ),
    )

    parser.add_argument(
        "--param-dtype",
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="dtype pour les tensors denses/latents sauvegardés",
    )

    parser.add_argument(
        "--noise-method",
        default="row_outlier_clip",
        choices=["none", "global_clip", "row_outlier_clip", "median_center"],
        help="Noise reduction before ternary conversion",
    )

    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=3.0,
        help="Sigma for clipping",
    )

    parser.add_argument(
        "--ternary-threshold",
        type=float,
        default=0.5,
        help="Threshold on normalized weights for ternary quantization",
    )

    parser.add_argument(
        "--resize-strategy",
        default="crop_pad",
        choices=["strict", "crop_pad", "interpolate"],
        help="Resize strategy when GGUF shape != Chimera shape",
    )

    parser.add_argument(
        "--no-auto-transpose",
        action="store_true",
        help="Disable automatic transpose when reversed shape matches",
    )

    parser.add_argument(
        "--no-init-missing",
        action="store_true",
        help="Do not initialize missing Chimera weights",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less logs",
    )

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    importer = OptimizedGGUFImporter(
        config=config,
        scale=args.scale,
        storage=args.storage,
        param_dtype=args.param_dtype,
        noise_method=args.noise_method,
        noise_sigma=args.noise_sigma,
        ternary_threshold=args.ternary_threshold,
        resize_strategy=args.resize_strategy,
        auto_transpose=not args.no_auto_transpose,
        init_missing=not args.no_init_missing,
        verbose=not args.quiet,
    )

    importer.import_model(args.gguf, args.output)


if __name__ == "__main__":
    main()