
from rag.contracts import Chunk, RetrievedChunk, Scope
from rag.retrieval.retriever import Retriever


class FakeEmbedder:
    dim = 2

    def embed(self, texts):
        # The retriever only needs a query vector in these tests.
        return [[1.0, 0.0] for _ in texts]


class FakeStore:
    def __init__(self, semantic_results, keyword_results=None):
        self.semantic_results = semantic_results
        self.keyword_results = keyword_results
        self.semantic_calls = []
        self.keyword_calls = []

    def search(self, scope, query_vector, top_k):
        self.semantic_calls.append((scope, query_vector, top_k))
        return self.semantic_results[:top_k]

    def search_keywords(self, scope, question, top_k):
        self.keyword_calls.append((scope, question, top_k))
        return (self.keyword_results or [])[:top_k]


def make_result(chunk_id, score=0.8, tenant="t1", course="math"):
    chunk = Chunk(
        tenant_id=tenant,
        course_id=course,
        document_id=f"doc-{chunk_id}",
        chunk_id=chunk_id,
        text=f"Content for {chunk_id}",
        source_uri=f"{chunk_id}.pdf",
        page=1,
    )
    return RetrievedChunk(chunk=chunk, score=score)


def make_scope():
    return Scope(
        tenant_id="t1",
        course_ids=frozenset({"math"}),
    )


def test_rrf_combines_semantic_and_keyword_results():
    semantic = [
        make_result("A"),
        make_result("B"),
    ]
    keyword = [
        make_result("B"),
        make_result("C"),
    ]

    retriever = Retriever(
        store=FakeStore(semantic, keyword),
        embedder=FakeEmbedder(),
        rrf_k=60,
    )

    results = retriever._fuse_results(semantic, keyword)

    # B appears in both lists, so it receives contributions from both.
    assert [r.chunk.chunk_id for r in results] == ["B", "A", "C"]


def test_rrf_does_not_duplicate_chunks_found_by_both_methods():
    shared_semantic = make_result("shared", score=0.7)
    shared_keyword = make_result("shared", score=0.9)

    retriever = Retriever(
        store=FakeStore([], []),
        embedder=FakeEmbedder(),
    )

    results = retriever._fuse_results(
        [shared_semantic],
        [shared_keyword],
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "shared"

    # RRF score = 1/(60+1) + 1/(60+1)
    assert abs(results[0].score - (2 / 61)) < 1e-9

    # For duplicate IDs, the chunk with the higher original score is retained.
    assert results[0].chunk.text == shared_keyword.chunk.text


def test_retrieve_calls_semantic_and_keyword_search_with_scope():
    store = FakeStore(
        semantic_results=[make_result("A")],
        keyword_results=[make_result("B")],
    )
    retriever = Retriever(store=store, embedder=FakeEmbedder())
    scope = make_scope()

    results, diagnostics = retriever.retrieve(
        scope=scope,
        question="eigenvalues",
        top_k=2,
    )

    assert store.semantic_calls
    assert store.keyword_calls
    assert store.semantic_calls[0][0] == scope
    assert store.keyword_calls[0][0] == scope
    assert store.keyword_calls[0][1] == "eigenvalues"

    assert len(results) == 2
    assert diagnostics.returned == 2
    assert diagnostics.top_k == 2


def test_retrieve_respects_top_k_after_fusion():
    store = FakeStore(
        semantic_results=[
            make_result("A"),
            make_result("B"),
            make_result("C"),
        ],
        keyword_results=[
            make_result("C"),
            make_result("D"),
            make_result("E"),
        ],
    )
    retriever = Retriever(store=store, embedder=FakeEmbedder())

    results, diagnostics = retriever.retrieve(
        scope=make_scope(),
        question="matrix eigenvalues",
        top_k=2,
    )

    assert len(results) == 2
    assert diagnostics.returned == 2


def test_retrieve_rejects_top_k_less_than_one():
    retriever = Retriever(
        store=FakeStore([], []),
        embedder=FakeEmbedder(),
    )

    try:
        retriever.retrieve(
            scope=make_scope(),
            question="test",
            top_k=0,
        )
    except ValueError as exc:
        assert "top_k must be at least 1" in str(exc)
    else:
        raise AssertionError("Expected ValueError for top_k=0")


def test_retrieve_returns_empty_for_blank_question():
    store = FakeStore([], [])
    retriever = Retriever(store=store, embedder=FakeEmbedder())

    results, diagnostics = retriever.retrieve(
        scope=make_scope(),
        question="   ",
        top_k=3,
    )

    assert results == []
    assert diagnostics.returned == 0
    assert store.semantic_calls == []
    assert store.keyword_calls == []
