from __future__ import annotations

from dataclasses import dataclass

from ..contracts import RetrievedChunk, Scope
from ..embeddings.base import Embedder
from ..store.base import VectorStore


@dataclass(frozen=True)
class RetrievalDiagnostics:
    scope_tenant: str
    scope_courses: tuple[str, ...]
    top_k: int
    returned: int
    top_score: float


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier

    def retrieve(
        self,
        scope: Scope,
        question: str,
        top_k: int,
    ) -> tuple[list[RetrievedChunk], RetrievalDiagnostics]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if not question.strip():
            return [], self._diagnostics(scope, top_k, 0, 0.0)

        candidate_k = max(top_k, top_k * self.candidate_multiplier)

        query_vector = self.embedder.embed([question])[0]

        semantic_results = self.store.search(
            scope,
            query_vector,
            candidate_k,
        )

        keyword_search = getattr(self.store, "search_keywords", None)

        if callable(keyword_search):
            keyword_results = keyword_search(
                scope,
                question,
                candidate_k,
            )
        else:
            # Preserve compatibility with stores that only implement search().
            keyword_results = []

        fused_results = self._fuse_results(
            semantic_results,
            keyword_results,
        )

        results = fused_results[:top_k]
        top_score = results[0].score if results else 0.0

        diagnostics = self._diagnostics(
            scope,
            top_k,
            len(results),
            top_score,
        )

        return results, diagnostics

    def _fuse_results(
        self,
        semantic_results: list[RetrievedChunk],
        keyword_results: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Reciprocal Rank Fusion.

        Each list contributes according to rank, not raw score,
        so scores from different retrieval methods are not compared directly.
        """
        fused_scores: dict[tuple[str, str, str], float] = {}
        chunks: dict[tuple[str, str, str], RetrievedChunk] = {}

        result_lists = (semantic_results, keyword_results)

        for results in result_lists:
            for rank, result in enumerate(results, start=1):
                chunk = result.chunk
                key = (
                    chunk.tenant_id,
                    chunk.course_id,
                    chunk.chunk_id,
                )

                fused_scores[key] = fused_scores.get(key, 0.0) + (
                    1.0 / (self.rrf_k + rank)
                )

                previous = chunks.get(key)

                if previous is None or result.score > previous.score:
                    chunks[key] = result

        ranked = sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            RetrievedChunk(
                chunk=chunks[key].chunk,
                score=fused_score,
            )
            for key, fused_score in ranked
        ]

    @staticmethod
    def _diagnostics(
        scope: Scope,
        top_k: int,
        returned: int,
        top_score: float,
    ) -> RetrievalDiagnostics:
        return RetrievalDiagnostics(
            scope_tenant=scope.tenant_id,
            scope_courses=tuple(sorted(scope.course_ids)),
            top_k=top_k,
            returned=returned,
            top_score=top_score,
        )
