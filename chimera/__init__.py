"""Chimera 5.1 CPU-first implementation with lazy public imports."""
from .config import load_config, tiny_config, scale_config

__version__ = "5.1.4"

__all__ = [
    "load_config", "tiny_config", "scale_config",
    "Chimera51ForCausalLM", "Chimera51Block", "expand_layer_pattern",
    "BitLinear", "RMSNorm", "pack_ternary", "unpack_ternary",
    "ternarize_weight", "_quantize_weights_ternary", "apply_2_4_sparsity_",
    "ChimeraTokenizer",
]


def __getattr__(name):
    if name in {"Chimera51ForCausalLM", "Chimera51Block", "expand_layer_pattern"}:
        from .model import Chimera51ForCausalLM, Chimera51Block, expand_layer_pattern
        return {
            "Chimera51ForCausalLM": Chimera51ForCausalLM,
            "Chimera51Block": Chimera51Block,
            "expand_layer_pattern": expand_layer_pattern,
        }[name]
    if name in {"BitLinear", "RMSNorm", "pack_ternary", "unpack_ternary", "ternarize_weight", "_quantize_weights_ternary", "apply_2_4_sparsity_"}:
        from .quantization import (
            BitLinear, RMSNorm, pack_ternary, unpack_ternary,
            ternarize_weight, _quantize_weights_ternary, apply_2_4_sparsity_,
        )
        return {
            "BitLinear": BitLinear,
            "RMSNorm": RMSNorm,
            "pack_ternary": pack_ternary,
            "unpack_ternary": unpack_ternary,
            "ternarize_weight": ternarize_weight,
            "_quantize_weights_ternary": _quantize_weights_ternary,
            "apply_2_4_sparsity_": apply_2_4_sparsity_,
        }[name]
    if name == "ChimeraTokenizer":
        from .tokenizer import ChimeraTokenizer
        return ChimeraTokenizer
    raise AttributeError(name)
