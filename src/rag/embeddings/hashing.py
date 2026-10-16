from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np


class HashingEmbedder:
    """Deterministic, offline bag-of-tokens embedder for tests only.

    Not semantic. Two texts sharing tokens land near each other; this exists so
    the pipeline, store, and isolation tests run fast without downloading BGE-M3.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [t for t in text.lower().split() if t]

    def _one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in self._tokens(text):
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest, "big") % self.dim
            vec[idx] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._one(t) for t in texts])
