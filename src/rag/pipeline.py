from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .contracts import AnswerResult, RetrievedChunk, Scope
from .embeddings.base import Embedder
from .generation.llm import LLMClient
from .ingestion.chunk import chunk_document
from .ingestion.extract import extract_document
from .retrieval.retriever import RetrievalDiagnostics, Retriever
from .store.base import VectorStore

SYSTEM_PROMPT = (
    "You are a course assistant. Answer using ONLY the provided course excerpts. "
    "If the excerpts do not contain enough information, reply that the course materials "
    "do not provide enough information to answer. Cite sources using their [n] labels."
)


def build_context(results: Sequence[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        location = f"{chunk.source_uri} p.{chunk.page}"
        if chunk.section:
            location += f" section '{chunk.section}'"
        blocks.append(f"[{index}] {location}\n{chunk.text}")
    return "\n\n".join(blocks)


class RagEngine:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        llm: LLMClient | None = None,
        top_k: int = 5,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.top_k = top_k
        self.retriever = Retriever(store, embedder)

    def ingest(
        self,
        path: str | Path,
        *,
        tenant_id: str,
        course_id: str,
        document_id: str,
        source_uri: str | None = None,
        chunk_size: int = 180,
        overlap: int = 30,
    ) -> int:
        document = extract_document(
            path,
            tenant_id=tenant_id,
            course_id=course_id,
            document_id=document_id,
            source_uri=source_uri,
        )
        chunks = chunk_document(document, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return 0
        vectors = self.embedder.embed([chunk.text for chunk in chunks])
        self.store.upsert(chunks, vectors)
        return len(chunks)

    def retrieve(
        self,
        scope: Scope,
        question: str,
        top_k: int | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalDiagnostics]:
        return self.retriever.retrieve(scope, question, top_k or self.top_k)

    def answer(
        self,
        question: str,
        scope: Scope,
        top_k: int | None = None,
    ) -> AnswerResult:
        results, _ = self.retrieve(scope, question, top_k)
        if not results:
            return AnswerResult(
                question=question,
                answer="",
                sources=(),
                abstained=True,
                reason="no_evidence",
            )
        if self.llm is None:
            return AnswerResult(
                question=question,
                answer="",
                sources=tuple(results),
                abstained=True,
                reason="no_llm_configured",
            )
        user = f"Course excerpts:\n{build_context(results)}\n\nQuestion: {question}\nAnswer:"
        text = self.llm.complete(SYSTEM_PROMPT, user)
        return AnswerResult(
            question=question,
            answer=text,
            sources=tuple(results),
            abstained=False,
            reason="",
        )
