from __future__ import annotations

from ..contracts import Chunk, ExtractedDocument


def chunk_document(
    document: ExtractedDocument,
    *,
    chunk_size: int = 180,
    overlap: int = 30,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    ordinal = 0

    for page in document.pages:
        words = page.text.split()
        if not words:
            continue
        start = 0
        while start < len(words):
            window = words[start : start + chunk_size]
            chunks.append(
                Chunk(
                    tenant_id=document.tenant_id,
                    course_id=document.course_id,
                    document_id=document.document_id,
                    chunk_id=f"{document.document_id}#{ordinal}",
                    text=" ".join(window),
                    ordinal=ordinal,
                    page=page.page,
                    section=page.section,
                    source_uri=document.source_uri,
                )
            )
            ordinal += 1
            if start + chunk_size >= len(words):
                break
            start += step

    return chunks
