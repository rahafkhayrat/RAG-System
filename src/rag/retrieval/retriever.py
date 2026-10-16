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
    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(
        self,
        scope: Scope,
        question: str,
        top_k: int,
    ) -> tuple[list[RetrievedChunk], RetrievalDiagnostics]:
        if not question.strip():
            return [], self._diagnostics(scope, top_k, 0, 0.0)

        query_vector = self.embedder.embed([question])[0]
        results = self.store.search(scope, query_vector, top_k)
        top_score = results[0].score if results else 0.0
        return results, self._diagnostics(scope, top_k, len(results), top_score)

    @staticmethod
    def _diagnostics(scope: Scope, top_k: int, returned: int, top_score: float) -> RetrievalDiagnostics:
        return RetrievalDiagnostics(
            scope_tenant=scope.tenant_id,
            scope_courses=tuple(sorted(scope.course_ids)),
            top_k=top_k,
            returned=returned,
            top_score=top_score,
        )
