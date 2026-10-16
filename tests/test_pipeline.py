import docx

from rag.contracts import Scope
from rag.embeddings.hashing import HashingEmbedder
from rag.pipeline import RagEngine
from rag.store.memory import InMemoryVectorStore


def _write_docx(tmp_path, name, heading, body):
    path = tmp_path / name
    document = docx.Document()
    document.add_heading(heading, level=1)
    document.add_paragraph(body)
    document.save(str(path))
    return path


def test_ingest_and_scoped_retrieval(tmp_path):
    engine = RagEngine(store=InMemoryVectorStore(), embedder=HashingEmbedder(dim=128))
    path_a = _write_docx(tmp_path, "a.docx", "Algebra", "linear equations a x plus b equals zero")
    path_b = _write_docx(tmp_path, "b.docx", "History", "industrial revolution steam power")

    assert engine.ingest(path_a, tenant_id="t1", course_id="math", document_id="a") >= 1
    assert engine.ingest(path_b, tenant_id="t1", course_id="hist", document_id="b") >= 1

    scope = Scope(tenant_id="t1", course_ids=frozenset({"math"}))
    results, diagnostics = engine.retrieve(scope, "linear equations", top_k=5)

    assert results
    assert all(result.chunk.document_id == "a" for result in results)
    assert diagnostics.returned == len(results)


def test_other_tenant_never_retrieved(tmp_path):
    engine = RagEngine(store=InMemoryVectorStore(), embedder=HashingEmbedder(dim=128))
    path = _write_docx(tmp_path, "a.docx", "Algebra", "linear equations a x plus b equals zero")
    engine.ingest(path, tenant_id="tenantA", course_id="math", document_id="a")

    scope_b = Scope(tenant_id="tenantB", course_ids=frozenset({"math"}))
    results, _ = engine.retrieve(scope_b, "linear equations", top_k=5)
    assert results == []


def test_answer_abstains_without_evidence():
    engine = RagEngine(store=InMemoryVectorStore(), embedder=HashingEmbedder(dim=64))
    scope = Scope(tenant_id="t1", course_ids=frozenset({"math"}))
    result = engine.answer("anything", scope)
    assert result.abstained is True
    assert result.reason == "no_evidence"
