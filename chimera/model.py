from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .evolution import SelfEvolutionEngine
from .inference import DebtLedger, EntropyValve, GrammarFST, SpanInferenceEngine
from .layers import GatedDeltaNetLayer, MLSTMLayer, SwiGLUMLP, TSPSpanKnotLayer, TitansMACLayer
from .looping import ParcaeLoopController
from .moe import MoELayer
from .multimodal import AudioEncoder, VisionEncoder
from .quantization import BitLinear, RMSNorm


def expand_layer_pattern(config: dict) -> list[str]:
    backbone = config.get("backbone", {})
    pattern = backbone.get("layer_pattern", "GD XM GD TM GD XM GD SK").split()
    aliases = backbone.get("layer_aliases", {"GD": "gated_deltanet", "XM": "xlstm_m", "TM": "titans_mac", "SK": "tsp_span_knot"})
    n = int(config.get("num_hidden_layers", 28))
    return [aliases.get(x, x) for x in (pattern * (n // max(1, len(pattern)) + 1))[:n]]


class CausalLMOutput(dict):
    def __init__(self, loss=None, logits=None, hidden_states=None):
        super().__init__(loss=loss, logits=logits, hidden_states=hidden_states)
        self.loss = loss
        self.logits = logits
        self.hidden_states = hidden_states


class Chimera51Block(nn.Module):
    def __init__(self, config: dict, layer_type: str, layer_idx: int, use_moe: bool = False):
        super().__init__()
        h = int(config["hidden_size"])
        eps = float(config.get("rms_norm_eps", 1e-6))
        heads = int(config.get("num_heads", 1))
        head_dim = int(config.get("head_dim", max(1, h // heads)))
        self.attn_norm = RMSNorm(h, eps)
        if layer_type == "gated_deltanet":
            self.attn = GatedDeltaNetLayer(h, heads, head_dim, eps, config.get("gated_deltanet", {}).get("chunk_size", 256))
        elif layer_type == "xlstm_m":
            self.attn = MLSTMLayer(h, heads, config.get("xlstm", {}).get("memory_size_per_head", [64, 64])[0], eps)
        elif layer_type == "titans_mac":
            tc = config.get("titans", {})
            self.attn = TitansMACLayer(h, heads, head_dim, tc.get("memory_depth", 2), tc.get("persistent_memory_slots", 64), tc.get("local_window_size", 1024), eps)
        elif layer_type == "tsp_span_knot":
            self.attn = TSPSpanKnotLayer(h, heads, head_dim, eps, config.get("gated_deltanet", {}).get("chunk_size", 256))
        else:
            raise ValueError(f"unknown layer type {layer_type}")
        self.mlp_norm = RMSNorm(h, eps)
        self.use_moe = use_moe
        if use_moe:
            mc = config.get("backbone", {}).get("moe", {})
            self.mlp = MoELayer(h, mc.get("moe_intermediate_size", max(64, h * 2)), mc.get("n_routed_experts", 4), mc.get("n_shared_experts", 1), mc.get("num_experts_per_tok", 2))
        else:
            self.mlp = SwiGLUMLP(h, int(config.get("intermediate_size", h * 4)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Chimera51ForCausalLM(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        h = int(config["hidden_size"])
        vocab = int(config["vocab_size"])
        n_layers = int(config["num_hidden_layers"])
        self.embed = nn.Embedding(vocab, h)
        layer_types = expand_layer_pattern(config)
        moe_layers = set(config.get("backbone", {}).get("moe", {}).get("layers", []))
        self.layers = nn.ModuleList([Chimera51Block(config, layer_types[i], i, i in moe_layers) for i in range(n_layers)])
        self.norm = RMSNorm(h, config.get("rms_norm_eps", 1e-6))
        self.lm_head = nn.Linear(h, vocab, bias=False)
        if config.get("tie_word_embeddings", True):
            self.lm_head.weight = self.embed.weight
        loop = config.get("looping", {})
        self.looping_enabled = bool(loop.get("enabled", True)) and n_layers >= 3
        if self.looping_enabled:
            self.prelude_start, self.prelude_end = loop.get("prelude", [0, 0])
            self.loop_start, self.loop_end = loop.get("loop", [1, n_layers - 2])
            self.coda_start, self.coda_end = loop.get("coda", [n_layers - 1, n_layers - 1])
            self.loop_controller = ParcaeLoopController(h, tuple(loop.get("loop_range", [1, 6])), loop.get("loop_default", 2), loop.get("adaptive_exit_threshold", 0.01))
        self.span_engine = SpanInferenceEngine(h, config.get("span_inference", {})) if config.get("span_inference", {}).get("enabled", True) else None
        self.grammar = GrammarFST(config.get("grammar", {}))
        self.entropy_valve = EntropyValve(config.get("entropy_valve", {}))
        self.debt_ledger = DebtLedger(config.get("debt_ledger", {}))
        evo_cfg = dict(config.get("self_evolution", {}))
        evo_cfg["_semantic_memory_config"] = config.get("semantic_memory", {})
        self.evolution = SelfEvolutionEngine(evo_cfg, h)
        mm = config.get("multimodal", {})
        mm = {**mm, "hidden_size": h}
        self.vision_encoder = VisionEncoder(mm) if mm.get("enabled", False) else None
        self.audio_encoder = AudioEncoder(mm) if mm.get("enabled", False) else None
        self.gradient_checkpointing = False
        self._init_weights()

    def _init_weights(self) -> None:
        std = float(self.config.get("initializer_range", 0.02))
        for m in self.modules():
            if isinstance(m, (nn.Linear, BitLinear)):
                if getattr(m, "weight", None) is not None:
                    nn.init.normal_(m.weight, 0.0, std)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, std)

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def _run_layers(self, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        start = max(0, int(start)); end = min(len(self.layers) - 1, int(end))
        if end < start:
            return x
        for i in range(start, end + 1):
            if self.gradient_checkpointing and self.training:
                x = checkpoint(self.layers[i], x, use_reentrant=False)
            else:
                x = self.layers[i](x)
        return x

    def _loop_fn(self, x: torch.Tensor) -> torch.Tensor:
        return self._run_layers(x, self.loop_start, self.loop_end)

    def prepare_for_inference(self) -> None:
        for module in self.modules():
            if isinstance(module, BitLinear):
                module.prepare_for_inference()
        self.eval()

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, pixel_values: torch.Tensor | None = None, mel_features: torch.Tensor | None = None, num_loops: int | None = None, logits_to_keep: int = 0) -> CausalLMOutput:
        x = self.embed(input_ids)
        if pixel_values is not None and self.vision_encoder is not None:
            x = torch.cat([self.vision_encoder(pixel_values).to(x.dtype), x], dim=1)
        if mel_features is not None and self.audio_encoder is not None:
            x = torch.cat([self.audio_encoder(mel_features).to(x.dtype), x], dim=1)
        if self.looping_enabled:
            x = self._run_layers(x, self.prelude_start, self.prelude_end)
            loops = num_loops
            if loops is None and not self.training:
                with torch.no_grad():
                    probe = self.lm_head(self.norm(x[:, -1:, :]))
                    loops = self.entropy_valve.get_loop_count(probe)
            x = self.loop_controller(x, self._loop_fn, loops)
            x = self._run_layers(x, self.coda_start, self.coda_end)
        else:
            x = self._run_layers(x, 0, len(self.layers) - 1)
        x = self.norm(x)
        if self.span_engine is not None:
            x = self.span_engine(x)
        if logits_to_keep and labels is None:
            x = x[:, -int(logits_to_keep):]
        logits = self.debt_ledger(self.grammar(self.lm_head(x)))
        loss = None
        if labels is not None:
            seq = min(logits.size(1), labels.size(1))
            loss = F.cross_entropy(logits[:, :seq].reshape(-1, logits.size(-1)), labels[:, :seq].reshape(-1), ignore_index=-100)
        return CausalLMOutput(loss=loss, logits=logits, hidden_states=x)
