from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> dict:
    """Load a Chimera JSON config and apply shallow dotted-key overrides."""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "config.json"
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    if overrides:
        cfg = copy.deepcopy(cfg)
        for key, value in overrides.items():
            cur = cfg
            parts = str(key).split(".")
            for part in parts[:-1]:
                cur = cur.setdefault(part, {})
            cur[parts[-1]] = value
    return cfg


def scale_config(config: dict, scale: str = "base") -> dict:
    """Return a safe CPU-scaled copy while preserving feature flags.

    The uploaded Chimera config targets a large model.  These presets keep all
    modules wired but resize dimensions so tests/fine-tuning fit commodity CPU
    memory (including 16 GB DDR5 machines).
    """
    cfg = copy.deepcopy(config)
    presets = {
        "nano": dict(hidden_size=128, intermediate_size=344, num_hidden_layers=4, num_heads=4, head_dim=32, vocab_size=min(cfg.get("vocab_size", 32000), 8192)),
        "tiny": dict(hidden_size=256, intermediate_size=688, num_hidden_layers=6, num_heads=4, head_dim=64, vocab_size=min(cfg.get("vocab_size", 32000), 32768)),
        "small": dict(hidden_size=512, intermediate_size=1376, num_hidden_layers=8, num_heads=8, head_dim=64, vocab_size=min(cfg.get("vocab_size", 32000), 65536)),
        "base": {},
    }
    if scale not in presets:
        raise ValueError(f"unknown scale {scale!r}; choose {sorted(presets)}")
    cfg.update(presets[scale])
    h = cfg["hidden_size"]
    cfg["num_heads"] = max(1, min(cfg.get("num_heads", 4), h // max(1, cfg.get("head_dim", 64))))
    cfg["head_dim"] = h // cfg["num_heads"]
    cfg.setdefault("backbone", {}).setdefault("moe", {})
    moe = cfg["backbone"]["moe"]
    moe["layers"] = [i for i in moe.get("layers", []) if i < cfg["num_hidden_layers"]]
    moe["n_routed_experts"] = min(int(moe.get("n_routed_experts", 4)), 4 if scale in {"nano", "tiny"} else 8)
    moe["n_shared_experts"] = min(int(moe.get("n_shared_experts", 1)), 1)
    moe["num_experts_per_tok"] = min(int(moe.get("num_experts_per_tok", 2)), moe["n_routed_experts"])
    moe["moe_intermediate_size"] = min(int(moe.get("moe_intermediate_size", h * 2)), max(64, cfg["intermediate_size"] // 2))
    loop = cfg.setdefault("looping", {})
    if cfg["num_hidden_layers"] < 8:
        loop["enabled"] = False
    else:
        loop["prelude"] = [0, min(1, cfg["num_hidden_layers"] - 1)]
        loop["loop"] = [2, max(2, cfg["num_hidden_layers"] - 3)]
        loop["coda"] = [max(0, cfg["num_hidden_layers"] - 2), cfg["num_hidden_layers"] - 1]
    cfg.setdefault("span_inference", {})["enabled"] = bool(cfg.get("span_inference", {}).get("enabled", True))
    return cfg


def tiny_config() -> dict:
    return scale_config(load_config(), "nano")
