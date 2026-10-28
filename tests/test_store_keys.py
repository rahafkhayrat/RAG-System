import numpy as np

from rag.contracts import Chunk, Scope
from rag.store.memory import InMemoryVectorStore


def make_chunk(
    tenant_id: str,
    course_id: str,
    chunk_id: str,
    text: str,
) -> Chunk:
    return Chunk(
        tenant_id=tenant_id,
        course_id=course_id,
        document_id="doc-1",
        chunk_id=chunk_id,
        text=text,
    )


def test_same_chunk_id_in_different_courses_is_preserved():
    store = InMemoryVectorStore()

    chunk_a = make_chunk(
        tenant_id="tenant-1",
        course_id="course-ml",
        chunk_id="chunk-1",
        text="Machine Learning content",
    )

    chunk_b = make_chunk(
        tenant_id="tenant-1",
        course_id="course-dl",
        chunk_id="chunk-1",
        text="Deep Learning content",
    )

    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    store.upsert([chunk_a, chunk_b], vectors)

    assert store.count() == 2

    results_ml = store.search(
        Scope(
            tenant_id="tenant-1",
            course_ids=frozenset({"course-ml"}),
        ),
        np.array([1.0, 0.0], dtype=np.float32),
        top_k=5,
    )

    results_dl = store.search(
        Scope(
            tenant_id="tenant-1",
            course_ids=frozenset({"course-dl"}),
        ),
        np.array([0.0, 1.0], dtype=np.float32),
        top_k=5,
    )

    assert len(results_ml) == 1
    assert results_ml[0].chunk.text == "Machine Learning content"

    assert len(results_dl) == 1
    assert results_dl[0].chunk.text == "Deep Learning content"


def test_same_chunk_key_updates_existing_chunk():
    store = InMemoryVectorStore()

    original = make_chunk(
        tenant_id="tenant-1",
        course_id="course-ml",
        chunk_id="chunk-1",
        text="Original content",
    )

    updated = make_chunk(
        tenant_id="tenant-1",
        course_id="course-ml",
        chunk_id="chunk-1",
        text="Updated content",
    )

    store.upsert(
        [original],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )

    store.upsert(
        [updated],
        np.array([[0.0, 1.0]], dtype=np.float32),
    )

    assert store.count() == 1

    results = store.search(
        Scope(
            tenant_id="tenant-1",
            course_ids=frozenset({"course-ml"}),
        ),
        np.array([0.0, 1.0], dtype=np.float32),
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].chunk.text == "Updated content"