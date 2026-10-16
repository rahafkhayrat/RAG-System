from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedPage:
    page: int
    text: str
    section: str | None = None


@dataclass(frozen=True)
class ExtractedDocument:
    tenant_id: str
    course_id: str
    document_id: str
    filename: str
    source_uri: str
    pages: tuple[ExtractedPage, ...]

    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


@dataclass(frozen=True)
class Chunk:
    tenant_id: str
    course_id: str
    document_id: str
    chunk_id: str
    text: str
    ordinal: int = 0
    page: int | None = None
    section: str | None = None
    source_uri: str = ""


@dataclass(frozen=True)
class Scope:
    tenant_id: str
    course_ids: frozenset[str]

    def allows(self, chunk: Chunk) -> bool:
        return chunk.tenant_id == self.tenant_id and chunk.course_id in self.course_ids


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class AnswerResult:
    question: str
    answer: str
    sources: tuple[RetrievedChunk, ...]
    abstained: bool = False
    reason: str = ""
