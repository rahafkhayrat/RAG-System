from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from .contracts import AnswerResult, RetrievedChunk, Scope
from .embeddings.base import Embedder
from .evaluation.sufficiency import EvidenceSufficiencyChecker
from .generation.llm import LLMClient
from .ingestion.chunk import chunk_document
from .ingestion.extract import extract_document
from .retrieval.retriever import RetrievalDiagnostics, Retriever
from .store.base import VectorStore


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a course assistant. Answer using ONLY the provided course "
    "excerpts. If the excerpts do not contain enough information, explain "
    "that the course materials do not provide enough information to answer. "
    "Cite sources using their [n] labels. Preserve mathematical notation "
    "and write equations clearly using LaTeX when appropriate."
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
        sufficiency_checker: EvidenceSufficiencyChecker | None = None,
        debug: bool = False,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.top_k = top_k
        self.retriever = Retriever(store, embedder)
        self.sufficiency_checker = sufficiency_checker
        self.debug = debug

        if self.sufficiency_checker is None and self.llm is not None:
            self.sufficiency_checker = EvidenceSufficiencyChecker(self.llm)

    def _debug(self, message: str, *args) -> None:
        if self.debug:
            logger.warning("[RAG DEBUG] " + message, *args)

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

        chunks = chunk_document(
            document,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        if not chunks:
            self._debug("No chunks extracted from %s", path)
            return 0

        vectors = self.embedder.embed(
            [chunk.text for chunk in chunks]
        )

        self.store.upsert(chunks, vectors)

        self._debug(
            "Ingested document=%s course=%s chunks=%d",
            source_uri or path,
            course_id,
            len(chunks),
        )

        return len(chunks)

    def retrieve(
        self,
        scope: Scope,
        question: str,
        top_k: int | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalDiagnostics]:
        effective_top_k = self.top_k if top_k is None else top_k

        return self.retriever.retrieve(
            scope,
            question,
            effective_top_k,
        )

    def _retry_retrieval(
        self,
        scope: Scope,
        question: str,
        current_results: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retry retrieval with a larger top_k in the same permitted scope."""

        current_top_k = self.top_k if top_k is None else top_k
        retry_top_k = max(current_top_k * 2, current_top_k + 1)

        retry_results, _ = self.retrieve(
            scope=scope,
            question=question,
            top_k=retry_top_k,
        )

        self._debug(
            "Retry retrieval: initial=%d retry=%d top_k=%d",
            len(current_results),
            len(retry_results),
            retry_top_k,
        )

        # Prefer the retry when it contains more chunks. The debug output
        # allows us to inspect whether the relevant evidence was retrieved.
        if len(retry_results) > len(current_results):
            return retry_results

        return current_results

    def _log_results(
        self,
        question: str,
        results: list[RetrievedChunk],
    ) -> None:
        if not self.debug:
            return

        self._debug("Question: %s", question)
        self._debug("Retrieved chunks: %d", len(results))

        for index, result in enumerate(results, start=1):
            chunk = result.chunk

            self._debug(
                "Chunk %d | source=%s | page=%s | section=%s",
                index,
                chunk.source_uri,
                chunk.page,
                chunk.section,
            )

            self._debug("Chunk %d text: %.1200s", index, chunk.text)

    def answer(
        self,
        question: str,
        scope: Scope,
        top_k: int | None = None,
    ) -> AnswerResult:
        # Step 1: Retrieve evidence from the permitted course scope.
        results, diagnostics = self.retrieve(
            scope,
            question,
            top_k,
        )

        self._debug(
            "Initial retrieval diagnostics: %s",
            diagnostics,
        )
        self._log_results(question, results)

        # Step 2: Abstain if no evidence was retrieved.
        if not results:
            self._debug("Abstaining: no evidence retrieved.")

            return AnswerResult(
                question=question,
                answer="",
                sources=(),
                abstained=True,
                reason="no_evidence",
            )

        # Step 3: Preserve behavior when no generation LLM is configured.
        if self.llm is None:
            self._debug("Abstaining: no LLM configured.")

            return AnswerResult(
                question=question,
                answer="",
                sources=tuple(results),
                abstained=True,
                reason="no_llm_configured",
            )

        # Step 4: Check the initial evidence.
        if self.sufficiency_checker is not None:
            sufficiency = self.sufficiency_checker.evaluate(
                question,
                results,
            )

            self._debug(
                "Initial sufficiency: sufficient=%s reason=%s evidence_count=%d",
                sufficiency.sufficient,
                sufficiency.reason,
                sufficiency.evidence_count,
            )

            # Step 5: Retry once if the initial evidence is insufficient.
            if not sufficiency.sufficient:
                initial_reason = sufficiency.reason

                results = self._retry_retrieval(
                    scope=scope,
                    question=question,
                    current_results=results,
                    top_k=top_k,
                )

                self._log_results(question, results)

                sufficiency = self.sufficiency_checker.evaluate(
                    question,
                    results,
                )

                self._debug(
                    "Retry sufficiency: sufficient=%s reason=%s evidence_count=%d",
                    sufficiency.sufficient,
                    sufficiency.reason,
                    sufficiency.evidence_count,
                )

                if not sufficiency.sufficient:
                    self._debug(
                        "Abstaining after retry. Initial reason=%s final reason=%s",
                        initial_reason,
                        sufficiency.reason,
                    )

                    return AnswerResult(
                        question=question,
                        answer="",
                        sources=tuple(results),
                        abstained=True,
                        reason=sufficiency.reason or initial_reason,
                    )

        # Step 6: Generate an answer only after the evidence check passes.
        user = (
            f"Course excerpts:\n{build_context(results)}"
            f"\n\nQuestion: {question}\n"
            "Answer using the excerpts only. Cite relevant excerpts with "
            "their [n] labels. Format mathematical equations clearly."
        )

        try:
            self._debug("Generation started.")

            text = self.llm.complete(
                SYSTEM_PROMPT,
                user,
            ).strip()

        except Exception:
            logger.exception("RAG answer generation failed.")

            return AnswerResult(
                question=question,
                answer="",
                sources=tuple(results),
                abstained=True,
                reason="generation_failed",
            )

        if not text:
            self._debug("Abstaining: LLM returned an empty answer.")

            return AnswerResult(
                question=question,
                answer="",
                sources=tuple(results),
                abstained=True,
                reason="empty_generation",
            )

        self._debug("Generation succeeded.")

        return AnswerResult(
            question=question,
            answer=text,
            sources=tuple(results),
            abstained=False,
            reason="",
        )
