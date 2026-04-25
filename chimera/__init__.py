"""Chimera 5.1 CPU-first implementation.

The package uses lazy imports for torch-backed modules so configuration utilities
remain usable before PyTorch is installed.
"""
from .config import load_config, tiny_config, scale_config

__version__ = "5.2.0"

__all__ = [
    "load_config", "tiny_config", "scale_config", "Chimera51ForCausalLM",
    "Chimera51Block", "expand_layer_pattern", "BitLinear", "RMSNorm",
    "pack_ternary", "unpack_ternary", "ternarize_weight", "ChimeraTokenizer",
]


def __getattr__(name):
    if name in {"Chimera51ForCausalLM", "Chimera51Block", "expand_layer_pattern"}:
        from .model import Chimera51ForCausalLM, Chimera51Block, expand_layer_pattern
        return {"Chimera51ForCausalLM": Chimera51ForCausalLM, "Chimera51Block": Chimera51Block, "expand_layer_pattern": expand_layer_pattern}[name]
    if name in {"BitLinear", "RMSNorm", "pack_ternary", "unpack_ternary", "ternarize_weight"}:
        from .quantization import BitLinear, RMSNorm, pack_ternary, unpack_ternary, ternarize_weight
        return {"BitLinear": BitLinear, "RMSNorm": RMSNorm, "pack_ternary": pack_ternary, "unpack_ternary": unpack_ternary, "ternarize_weight": ternarize_weight}[name]
    if name == "ChimeraTokenizer":
        from .tokenizer import ChimeraTokenizer
        return ChimeraTokenizer
    raise AttributeError(name)
