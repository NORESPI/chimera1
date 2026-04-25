"""
Chimera 5.1 — Splintr (Rust) Tokenizer Wrapper — o200k_base (OpenAI o1/o3)
Wraps splintr's high-performance Rust tokenizer for transformers-compatible API.
Vocab: o200k_base (200,073 tokens) — OpenAI's o1/o3 tokenizer.

Optimizations:
- __slots__ for reduced memory footprint
- Cached special token set for fast skip_special_tokens filtering
- Batch encode uses list comprehension (minimizes Python overhead)
"""

import torch
from typing import List, Union, Optional

try:
    from splintr import Tokenizer as _SplintrTokenizer, O200K_AGENT_TOKENS
    HAS_SPLINTR = True
except ImportError:
    HAS_SPLINTR = False

__all__ = ["ChimeraTokenizer"]


class ChimeraTokenizer:
    """
    High-performance Rust-backed tokenizer (splintr) with HuggingFace-like interface.
    Falls back to a basic tiktoken wrapper if splintr is not installed.
    """

    def __init__(self, pretrained: str = "o200k_base", vocab_size: int = 200073):
        if not HAS_SPLINTR:
            self._tok = None
            self.vocab_size = int(vocab_size)
            self.eos_token_id = min(self.vocab_size - 1, 199999)
            self.pad_token_id = min(self.vocab_size - 1, 200058)
            self.sep_token_id = min(self.vocab_size - 1, 200060)
            self.stop_token_id = min(self.vocab_size - 1, 200059)
            self.user_token_id = min(self.vocab_size - 1, 200020)
            self.assistant_token_id = min(self.vocab_size - 1, 200021)
            self.system_token_id = min(self.vocab_size - 1, 200019)
            self.endofprompt_token_id = min(self.vocab_size - 1, 200018)
            self.bos_token_id = self.eos_token_id
            self.eos_token = "<|endoftext|>"
            self.pad_token = "<|pad|>"
            self.model_max_length = 4194304
            self._special_ids = frozenset({self.eos_token_id, self.pad_token_id, self.sep_token_id, self.stop_token_id, self.user_token_id, self.assistant_token_id, self.system_token_id, self.endofprompt_token_id})
            self._byte_offset = 3
            return
        self._tok = _SplintrTokenizer.from_pretrained(pretrained)
        self.vocab_size = self._tok.vocab_size

        # o200k_base single-token special IDs
        self.eos_token_id = 199999
        self.pad_token_id = O200K_AGENT_TOKENS.PAD    # 200058
        self.sep_token_id = O200K_AGENT_TOKENS.SEP    # 200060
        self.stop_token_id = O200K_AGENT_TOKENS.STOP  # 200059
        self.user_token_id = O200K_AGENT_TOKENS.USER  # 200020
        self.assistant_token_id = O200K_AGENT_TOKENS.ASSISTANT  # 200021
        self.system_token_id = 200019
        self.endofprompt_token_id = 200018
        self.bos_token_id = self.eos_token_id

        self.eos_token = "<|endoftext|>"
        self.pad_token = "<|pad|>"
        self.model_max_length = 4194304

        # Cached set for fast filtering
        self._special_ids = frozenset({
            self.eos_token_id, self.pad_token_id, self.sep_token_id,
            self.stop_token_id, self.user_token_id,
            self.assistant_token_id, self.system_token_id,
            self.endofprompt_token_id,
        })

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str, add_special_tokens: bool = True,
               max_length: Optional[int] = None) -> List[int]:
        if self._tok is None:
            ids = [self._byte_offset + b for b in text.encode("utf-8", errors="replace")]
        else:
            ids = self._tok.encode(text)
        if add_special_tokens:
            ids = ids + [self.eos_token_id]
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        return ids

    def encode_batch(self, texts: List[str], add_special_tokens: bool = True,
                     max_length: Optional[int] = None,
                     padding: bool = False,
                     truncation: bool = False,
                     return_tensors: Optional[str] = None):
        all_ids = [self.encode(t, add_special_tokens=add_special_tokens,
                               max_length=max_length)
                   for t in texts]
        if padding:
            max_len = max(len(ids) for ids in all_ids)
            all_ids = [ids + [self.pad_token_id] * (max_len - len(ids))
                       for ids in all_ids]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(all_ids, dtype=torch.long)}
        return all_ids

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if skip_special_tokens:
            token_ids = [t for t in token_ids if t not in self._special_ids]
        if self._tok is None:
            data = bytes(max(0, min(255, int(t) - self._byte_offset)) for t in token_ids if int(t) >= self._byte_offset)
            return data.decode("utf-8", errors="replace")
        return self._tok.decode(token_ids)

    def decode_batch(self, token_ids_list, skip_special_tokens: bool = True) -> List[str]:
        return [self.decode(ids, skip_special_tokens=skip_special_tokens)
                for ids in token_ids_list]

    def __call__(self, text, **kwargs) -> dict:
        return_tensors = kwargs.get("return_tensors", "pt")
        padding = kwargs.get("padding", False)
        max_length = kwargs.get("max_length", None)
        add_special_tokens = kwargs.get("add_special_tokens", True)
        if isinstance(text, str):
            text = [text]
        result = self.encode_batch(
            text, add_special_tokens=add_special_tokens,
            max_length=max_length, padding=padding,
            return_tensors=return_tensors
        )
        if isinstance(result, list):
            return {"input_ids": torch.tensor(result, dtype=torch.long)}
        return result

    def get_vocab(self) -> dict:
        return {
            self.eos_token_id: self.eos_token,
            self.pad_token_id: self.pad_token,
            self.user_token_id: "<|user|>",
            self.assistant_token_id: "<|assistant|>",
            self.system_token_id: "<|system|>",
        }

    def apply_chat_template(self, messages: List[dict],
                            add_generation_prompt: bool = False) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"<|system|>\n{content}\n<|endofprompt|>")
            elif role == "user":
                parts.append(f"<|user|>\n{content}\n<|endofprompt|>")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}\n<|endofprompt|>")
        text = "\n".join(parts)
        if add_generation_prompt:
            text += "\n<|assistant|>\n"
        return text
