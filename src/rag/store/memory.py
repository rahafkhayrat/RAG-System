from __future__ import annotations
import re
import unicodedata
from collections import Counter
from typing import Sequence

import numpy as np

from ..contracts import Chunk, RetrievedChunk, Scope
from .base import VectorStore

_GREEK_SYMBOL_NAMES = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "ο": "omicron",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "ς": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
}


class InMemoryVectorStore(VectorStore):
    """In-memory store with semantic and keyword search."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: list[np.ndarray] = []

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")

        for chunk, vector in zip(chunks, vectors):
            self._replace_or_append(
                chunk,
                np.asarray(vector, dtype=np.float32),
            )

    def _replace_or_append(self, chunk: Chunk, vector: np.ndarray) -> None:
        for i, existing in enumerate(self._chunks):
            same_chunk = (
                existing.tenant_id == chunk.tenant_id
                and existing.course_id == chunk.course_id
                and existing.chunk_id == chunk.chunk_id
            )

            if same_chunk:
                self._chunks[i] = chunk
                self._vectors[i] = vector
                return

        self._chunks.append(chunk)
        self._vectors.append(vector)

    def _allowed_indices(self, scope: Scope) -> list[int]:
        return [
            i
            for i, chunk in enumerate(self._chunks)
            if scope.allows(chunk)
        ]

    def search(
        self,
        scope: Scope,
        query_vector: np.ndarray,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Semantic search using cosine similarity."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query = np.asarray(query_vector, dtype=np.float32).ravel()
        qnorm = float(np.linalg.norm(query))

        if qnorm == 0.0:
            return []

        query = query / qnorm
        scored: list[RetrievedChunk] = []

        for i in self._allowed_indices(scope):
            chunk = self._chunks[i]
            vector = self._vectors[i]
            vnorm = float(np.linalg.norm(vector))

            if vnorm == 0.0:
                continue

            score = float(np.dot(vector / vnorm, query))
            scored.append(RetrievedChunk(chunk=chunk, score=score))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        normalized = re.sub(r"(?<=[^\W\d_])(?=\d)", " ", normalized)
        normalized = "".join(
            f" {_GREEK_SYMBOL_NAMES[character]} "
            if character in _GREEK_SYMBOL_NAMES
            else character
            for character in normalized
        )
        return re.findall(r"[^\W_]+(?:['’-][^\W_]+)*", normalized)

    def search_keywords(
        self,
        scope: Scope,
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Keyword search with a simple BM25-inspired scoring function."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_tokens = self._tokens(query)

        if not query_tokens:
            return []

        query_counts = Counter(query_tokens)
        indices = self._allowed_indices(scope)

        if not indices:
            return []

        documents: list[list[str]] = [
            self._tokens(self._chunks[i].text)
            for i in indices
        ]

        average_length = (
            sum(len(tokens) for tokens in documents) / len(documents)
        )

        if average_length == 0:
            return []

        document_frequency: Counter[str] = Counter()

        for tokens in documents:
            document_frequency.update(set(tokens))

        total_documents = len(documents)
        k1 = 1.5
        b = 0.75
        scored: list[RetrievedChunk] = []

        for index, tokens in zip(indices, documents):
            if not tokens:
                continue

            frequencies = Counter(tokens)
            document_length = len(tokens)
            score = 0.0

            for term, query_frequency in query_counts.items():
                frequency = frequencies.get(term, 0)

                if frequency == 0:
                    continue

                df = document_frequency[term]

                # BM25 inverse document frequency.
                idf = np.log(
                    1.0
                    + (total_documents - df + 0.5)
                    / (df + 0.5)
                )

                denominator = frequency + k1 * (
                    1.0
                    - b
                    + b * document_length / average_length
                )

                term_score = idf * (
                    frequency * (k1 + 1.0) / denominator
                )

                score += float(term_score * query_frequency)

            if score > 0:
                scored.append(
                    RetrievedChunk(
                        chunk=self._chunks[index],
                        score=score,
                    )
                )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._chunks)