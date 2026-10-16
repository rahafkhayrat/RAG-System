from __future__ import annotations

from typing import Sequence

import numpy as np

from ..contracts import Chunk, RetrievedChunk, Scope


class InMemoryVectorStore:
    """Phase 1 store. Isolation is enforced structurally inside search()."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: list[np.ndarray] = []

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        for chunk, vector in zip(chunks, vectors):
            self._replace_or_append(chunk, np.asarray(vector, dtype=np.float32))

    def _replace_or_append(self, chunk: Chunk, vector: np.ndarray) -> None:
        for i, existing in enumerate(self._chunks):
            if existing.chunk_id == chunk.chunk_id:
                self._chunks[i] = chunk
                self._vectors[i] = vector
                return
        self._chunks.append(chunk)
        self._vectors.append(vector)

    def search(self, scope: Scope, query_vector: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        if not self._chunks:
            return []
        query = np.asarray(query_vector, dtype=np.float32).ravel()
        qnorm = float(np.linalg.norm(query))
        if qnorm == 0.0:
            return []
        query = query / qnorm

        scored: list[RetrievedChunk] = []
        for chunk, vector in zip(self._chunks, self._vectors):
            if not scope.allows(chunk):
                continue
            vnorm = float(np.linalg.norm(vector))
            if vnorm == 0.0:
                continue
            score = float(np.dot(vector / vnorm, query))
            scored.append(RetrievedChunk(chunk=chunk, score=score))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._chunks)
