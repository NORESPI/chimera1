from __future__ import annotations

from typing import Iterable, List, Sequence

try:
    from splintr import Tokenizer as _SplintrTokenizer
except Exception:  # pragma: no cover
    _SplintrTokenizer = None


class ChimeraTokenizer:
    """splintr/tiktoken compatible wrapper with a deterministic byte fallback."""
    def __init__(self, pretrained: str = "o200k_base", vocab_size: int = 200073):
        self.vocab_size = int(vocab_size)
        self.eos_token_id = min(self.vocab_size - 1, 200058)
        self.bos_token_id = min(self.vocab_size - 1, 199999)
        self.pad_token_id = 0
        self._tok = _SplintrTokenizer(pretrained) if _SplintrTokenizer is not None else None

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if self._tok is not None:
            ids = list(self._tok.encode(text))
        else:
            ids = [b + 1 for b in text.encode("utf-8", errors="replace")]
        ids = [i % self.vocab_size for i in ids]
        return ([self.bos_token_id] + ids + [self.eos_token_id]) if add_special_tokens else ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        ids = [int(i) for i in ids]
        if self._tok is not None:
            if skip_special_tokens:
                ids = [i for i in ids if i not in {self.bos_token_id, self.eos_token_id, self.pad_token_id}]
            return self._tok.decode(ids)
        bs = []
        for i in ids:
            if skip_special_tokens and i in {self.bos_token_id, self.eos_token_id, self.pad_token_id}:
                continue
            if 1 <= i <= 256:
                bs.append(i - 1)
        return bytes(bs).decode("utf-8", errors="replace")

    def batch_encode(self, texts: Iterable[str], add_special_tokens: bool = False) -> List[List[int]]:
        return [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]

    def __call__(self, text: str, return_tensors: str | None = None):
        ids = self.encode(text)
        if return_tensors == "pt":
            import torch
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}
