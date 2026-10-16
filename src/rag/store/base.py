from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from ..contracts import Chunk, RetrievedChunk, Scope


class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        ...

    def search(self, scope: Scope, query_vector: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        ...
