from rag.contracts import ExtractedDocument, ExtractedPage
from rag.ingestion.chunk import chunk_document


def _doc(word_count: int) -> ExtractedDocument:
    text = " ".join(f"w{i}" for i in range(word_count))
    page = ExtractedPage(page=1, text=text, section="S1")
    return ExtractedDocument(
        tenant_id="t",
        course_id="c",
        document_id="d",
        filename="d.pdf",
        source_uri="d.pdf",
        pages=(page,),
    )


def test_chunk_metadata_and_sequential_ids():
    chunks = chunk_document(_doc(250), chunk_size=100, overlap=20)
    assert len(chunks) >= 3
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert chunks[0].chunk_id == "d#0"
    for chunk in chunks:
        assert chunk.tenant_id == "t"
        assert chunk.course_id == "c"
        assert chunk.document_id == "d"
        assert chunk.page == 1
        assert chunk.section == "S1"
        assert chunk.source_uri == "d.pdf"


def test_overlap_repeats_boundary_tokens():
    chunks = chunk_document(_doc(150), chunk_size=100, overlap=20)
    assert len(chunks) == 2
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]


def test_rejects_invalid_overlap():
    import pytest

    with pytest.raises(ValueError):
        chunk_document(_doc(50), chunk_size=10, overlap=10)
