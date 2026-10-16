from rag.contracts import Chunk, Scope
from rag.embeddings.hashing import HashingEmbedder
from rag.store.memory import InMemoryVectorStore


def _chunk(tenant: str, course: str, doc: str, cid: str, text: str) -> Chunk:
    return Chunk(
        tenant_id=tenant,
        course_id=course,
        document_id=doc,
        chunk_id=cid,
        text=text,
        ordinal=0,
        page=1,
        section=None,
        source_uri=f"{doc}.pdf",
    )


def _populated_store() -> tuple[InMemoryVectorStore, HashingEmbedder]:
    embedder = HashingEmbedder(dim=64)
    store = InMemoryVectorStore()
    chunks = [
        _chunk("tenantA", "course1", "docA", "a1", "linear equations and algebra basics"),
        _chunk("tenantA", "course1", "docA", "a2", "quadratic formula derivation"),
        _chunk("tenantA", "course2", "docA2", "a3", "unrelated history topic"),
        _chunk("tenantB", "course1", "docB", "b1", "linear equations and algebra basics"),
    ]
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    return store, embedder


def test_cross_tenant_query_returns_only_own_tenant():
    store, embedder = _populated_store()
    scope = Scope(tenant_id="tenantA", course_ids=frozenset({"course1"}))
    query = embedder.embed(["linear equations"])[0]
    results = store.search(scope, query, top_k=10)
    assert results
    assert all(r.chunk.tenant_id == "tenantA" for r in results)


def test_unauthorized_course_returns_nothing():
    store, embedder = _populated_store()
    scope = Scope(tenant_id="tenantA", course_ids=frozenset({"course_not_owned"}))
    query = embedder.embed(["linear equations"])[0]
    assert store.search(scope, query, top_k=10) == []


def test_tenant_b_identity_does_not_leak_into_tenant_a():
    store, embedder = _populated_store()
    scope = Scope(tenant_id="tenantA", course_ids=frozenset({"course1"}))
    query = embedder.embed(["linear equations"])[0]
    results = store.search(scope, query, top_k=10)
    assert all(r.chunk.chunk_id != "b1" for r in results)
