"""
Chimera 5.1 — Full Model Assembly (CPU-Optimized)
- torch.compile integration at block level
- BFloat16 autocast support
- Gradient checkpointing per block
- Fused forward with minimal Python overhead
"""

import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .quantization import BitLinear, RMSNorm
from .layers import GatedDeltaNetLayer, MLSTMLayer, TitansMACLayer, TSPSpanKnotLayer, SwiGLUMLP
from .moe import MoELayer, SwiGLUMLP as MoESwiGLU
from .looping import ParcaeLoopController
from .inference import SpanInferenceEngine, GrammarFST, EntropyValve, DebtLedger, BraidState
from .evolution import SelfEvolutionEngine
from .multimodal import VisionEncoder, AudioEncoder


class CausalLMOutput(dict):
    def __init__(self, loss=None, logits=None, hidden_states=None):
        super().__init__(loss=loss, logits=logits, hidden_states=hidden_states)
        self.loss = loss
        self.logits = logits
        self.hidden_states = hidden_states

    def __iter__(self):
        yield self.loss
        yield self.logits


def expand_layer_pattern(config: dict) -> list:
    """Expand the layer pattern string into a list of layer type strings."""
    backbone = config.get('backbone', {})
    pattern_str = backbone.get('layer_pattern', 'GD XM GD TM GD XM GD SK')
    aliases = backbone.get('layer_aliases', {
        'GD': 'gated_deltanet', 'XM': 'xlstm_m',
        'TM': 'titans_mac', 'SK': 'tsp_span_knot'
    })
    pattern = pattern_str.split()
    n_layers = config.get('num_hidden_layers', 28)
    full = (pattern * (n_layers // len(pattern) + 1))[:n_layers]
    return [aliases.get(p, p) for p in full]


class Chimera51Block(nn.Module):
    """Single Chimera block: LayerNorm → Attention → LayerNorm → MLP/MoE
    
    Gradient checkpointing is controlled at the model level.
    """

    def __init__(self, config: dict, layer_type: str, layer_idx: int,
                 use_moe: bool = False):
        super().__init__()
        h = config['hidden_size']
        eps = config.get('rms_norm_eps', 1e-6)
        heads = config['num_heads']
        head_dim = config['head_dim']
        ternary = True
        chunk_sz = config.get('gated_deltanet', {}).get('chunk_size', 256)

        self.attn_norm = RMSNorm(h, eps=eps)

        if layer_type == 'gated_deltanet':
            self.attn = GatedDeltaNetLayer(h, heads, head_dim, norm_eps=eps,
                                            chunk_size=chunk_sz, use_ternary=ternary)
        elif layer_type == 'xlstm_m':
            xc = config.get('xlstm', {})
            mem_h = xc.get('memory_size_per_head', [64, 64])
            self.attn = MLSTMLayer(h, heads, mem_h[0], norm_eps=eps,
                                    use_ternary=ternary)
        elif layer_type == 'titans_mac':
            tc = config.get('titans', {})
            self.attn = TitansMACLayer(h, heads, head_dim,
                                        memory_depth=tc.get('memory_depth', 2),
                                        persistent_slots=tc.get('persistent_memory_slots', 64),
                                        local_window=tc.get('local_window_size', 1024),
                                        norm_eps=eps, use_ternary=ternary)
        elif layer_type == 'tsp_span_knot':
            self.attn = TSPSpanKnotLayer(h, heads, head_dim, norm_eps=eps,
                                          chunk_size=chunk_sz, use_ternary=ternary)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        self.mlp_norm = RMSNorm(h, eps=eps)
        self.use_moe = use_moe

        if use_moe:
            moe_cfg = config.get('backbone', {}).get('moe', {})
            self.mlp = MoELayer(
                hidden_size=h,
                moe_intermediate_size=moe_cfg.get('moe_intermediate_size', 1728),
                n_routed_experts=moe_cfg.get('n_routed_experts', 16),
                n_shared_experts=moe_cfg.get('n_shared_experts', 1),
                num_experts_per_tok=moe_cfg.get('num_experts_per_tok', 2),
                use_ternary=ternary,
            )
        else:
            intermediate = config.get('intermediate_size', int(h * 4 * 2 / 3))
            intermediate = 256 * ((intermediate + 255) // 256)
            self.mlp = SwiGLUMLP(h, intermediate, use_ternary=ternary)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Chimera51ForCausalLM(nn.Module):
    """Full Chimera 5.1 model with CPU optimizations.
    
    CPU Optimizations:
    - Gradient checkpointing per block (configurable)
    - BFloat16 autocast support (forward pass)
    - torch.compile compatibility (no graph-breaking ops in hot path)
    - Efficient loss computation with fused CE
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        h = config['hidden_size']
        vocab = config['vocab_size']
        n_layers = config['num_hidden_layers']
        eps = config.get('rms_norm_eps', 1e-6)

        # Embedding + LM head
        self.embed = nn.Embedding(vocab, h)
        layer_types = expand_layer_pattern(config)
        moe_layers = set(config.get('backbone', {}).get('moe', {}).get('layers', []))

        self.layers = nn.ModuleList([
            Chimera51Block(config, layer_types[i], i, use_moe=(i in moe_layers))
            for i in range(n_layers)
        ])

        self.norm = RMSNorm(h, eps=eps)
        self.lm_head = nn.Linear(h, vocab, bias=False)

        if config.get('tie_word_embeddings', True):
            self.lm_head.weight = self.embed.weight

        # Parcae looping
        loop_cfg = config.get('looping', {})
        self.looping_enabled = loop_cfg.get('enabled', True)
        if self.looping_enabled and n_layers >= 3:
            self.prelude_start, self.prelude_end = loop_cfg.get('prelude', [0, 3])
            self.loop_start, self.loop_end = loop_cfg.get('loop', [4, 23])
            self.coda_start, self.coda_end = loop_cfg.get('coda', [24, 27])
            self.loop_controller = ParcaeLoopController(
                h,
                loop_range=tuple(loop_cfg.get('loop_range', [1, 6])),
                loop_default=loop_cfg.get('loop_default', 2),
                adaptive_exit_threshold=loop_cfg.get('adaptive_exit_threshold', 0.01),
            )

        # Inference systems
        si_cfg = config.get('span_inference', {})
        self.span_engine = SpanInferenceEngine(h, si_cfg) if si_cfg.get('enabled', True) else None
        self.grammar = GrammarFST(config.get('grammar', {}))
        self.entropy_valve = EntropyValve(config.get('entropy_valve', {}))
        self.debt_ledger = DebtLedger(config.get('debt_ledger', {}))

        # Self-evolution
        evo_cfg = config.get('self_evolution', {})
        evo_cfg['_semantic_memory_config'] = config.get('semantic_memory', {})
        self.evolution = SelfEvolutionEngine(evo_cfg, h)

        # Multimodal
        mm_cfg = config.get('multimodal', {})
        mm_cfg = {**mm_cfg, "hidden_size": h}
        self.vision_encoder = VisionEncoder(mm_cfg) if mm_cfg.get('enabled', False) else None
        self.audio_encoder = AudioEncoder(mm_cfg) if mm_cfg.get('enabled', False) else None

        # Gradient checkpointing control
        self.gradient_checkpointing = False

        self._init_weights()
        self._wire_semantic_memory()

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for all blocks."""
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

    def _wire_semantic_memory(self):
        mem = self.evolution.semantic_memory
        for layer in self.layers:
            if hasattr(layer.attn, 'set_semantic_memory'):
                layer.attn.set_semantic_memory(mem)

    def _init_weights(self):
        init_range = self.config.get('initializer_range', 0.006)
        for module in self.modules():
            if isinstance(module, (nn.Linear, BitLinear)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.normal_(module.weight, mean=0.0, std=init_range)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=init_range)

    def _run_layers(self, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        for i in range(start, min(end + 1, len(self.layers))):
            if self.gradient_checkpointing and self.training:
                # use_reentrant=True because MoE layers have data-dependent shapes
                # that can differ on recomputation (expert routing counts vary)
                x = checkpoint(self.layers[i], x, use_reentrant=True)
            else:
                x = self.layers[i](x)
        return x

    def _loop_fn(self, x: torch.Tensor) -> torch.Tensor:
        return self._run_layers(x, self.loop_start, self.loop_end)

    def forward(self, input_ids: torch.Tensor, labels=None,
                pixel_values=None, mel_features=None, num_loops=None,
                logits_to_keep: int = 0):
        x = self.embed(input_ids)

        # Multimodal prepend
        if pixel_values is not None and self.vision_encoder is not None:
            vision_embeds = self.vision_encoder(pixel_values)
            if vision_embeds is not None:
                x = torch.cat([vision_embeds, x], dim=1)

        if mel_features is not None and self.audio_encoder is not None:
            audio_embeds = self.audio_encoder(mel_features)
            if audio_embeds is not None:
                x = torch.cat([audio_embeds, x], dim=1)

        # Parcae looping: prelude → loop × N → coda
        if self.looping_enabled and hasattr(self, "loop_controller"):
            x = self._run_layers(x, self.prelude_start, self.prelude_end)
            effective_loops = num_loops
            if effective_loops is None and not self.training:
                # Route compute from the last position only; full-vocab logits for
                # every prompt token are a major CPU bottleneck during generation.
                probe_logits = self.lm_head(self.norm(x[:, -1:, :]))
                effective_loops = self.entropy_valve.get_loop_count(probe_logits)
            x = self.loop_controller(x, self._loop_fn, num_loops=effective_loops)
            x = self._run_layers(x, self.coda_start, self.coda_end)
        else:
            x = self._run_layers(x, 0, len(self.layers) - 1)

        x = self.norm(x)

        if self.span_engine is not None:
            x = self.span_engine(x)

        if logits_to_keep and labels is None:
            x = x[:, -int(logits_to_keep):, :]

        logits = self.lm_head(x)
        logits = self.grammar(logits)
        logits = self.debt_ledger(logits)

        loss = None
        if labels is not None:
            seq_len = min(logits.shape[1], labels.shape[1])
            # The training script feeds input_ids[:, :-1] and labels[:, 1:], so
            # logits and labels are already next-token aligned. Avoid a second
            # internal shift that silently drops an extra token and trains t→t+2.
            shift_logits = logits[:, :seq_len, :].contiguous()
            shift_labels = labels[:, :seq_len].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return CausalLMOutput(loss=loss, logits=logits, hidden_states=x)

    def get_mode_config(self, mode: str = 'balanced') -> dict:
        modes = self.config.get('modes', {})
        return modes.get(mode, modes.get('balanced', {}))

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        ternary = sum(p.numel() for n, m in self.named_modules()
                      if isinstance(m, BitLinear) for p in m.parameters())
        return {'total': total, 'ternary': ternary, 'fp32': total - ternary}

    @classmethod
    def from_config_file(cls, path: str):
        with open(path) as f:
            config = json.load(f)
        return cls(config)
