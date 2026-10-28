
from pathlib import Path
import tempfile

from rag.contracts import Scope
from rag.embeddings.hashing import HashingEmbedder
from rag.evaluation.benchmark import QUESTIONS, TOY_DOCS, _write_docx
from rag.evaluation.metrics import hit_at_k, recall_at_k, mrr_at_k
from rag.pipeline import RagEngine
from rag.retrieval.retriever import Retriever
from rag.store.memory import InMemoryVectorStore


TOP_K = 3


class SemanticOnlyStore:
    """Expose semantic search only; hide keyword search from Retriever."""

    def __init__(self):
        self.store = InMemoryVectorStore()

    def upsert(self, chunks, vectors):
        self.store.upsert(chunks, vectors)

    def search(self, scope, query_vector, top_k):
        return self.store.search(scope, query_vector, top_k)

    def count(self):
        return self.store.count()


class KeywordOnlyStore:
    """Expose keyword search while disabling semantic candidates."""

    def __init__(self):
        self.store = InMemoryVectorStore()

    def upsert(self, chunks, vectors):
        self.store.upsert(chunks, vectors)

    def search(self, scope, query_vector, top_k):
        return []

    def search_keywords(self, scope, question, top_k):
        return self.store.search_keywords(scope, question, top_k)

    def count(self):
        return self.store.count()


def run_strategy(strategy):
    if strategy == "semantic":
        store = SemanticOnlyStore()
    elif strategy == "keyword":
        store = KeywordOnlyStore()
    elif strategy == "hybrid":
        store = InMemoryVectorStore()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    engine = RagEngine(
        store=store,
        embedder=HashingEmbedder(dim=256),
        llm=None,
        top_k=TOP_K,
    )

    rows = []

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        for toy in TOY_DOCS:
            path = tmpdir / toy.filename
            _write_docx(path, toy.blocks)

            engine.ingest(
                path,
                tenant_id="comparison_tenant",
                course_id=toy.course_id,
                document_id=toy.document_id,
            )

        for question in QUESTIONS:
            # Questions without gold evidence are not scored for retrieval.
            if not question.gold_docs and not question.gold_chunk_ids:
                continue

            scope = Scope(
                tenant_id="comparison_tenant",
                course_ids=frozenset({question.course_id}),
            )

            results, _ = engine.retrieve(
                scope,
                question.question,
                TOP_K,
            )

            retrieved_docs = list(
                dict.fromkeys(r.chunk.document_id for r in results)
            )
            retrieved_chunks = [r.chunk.chunk_id for r in results]

            if question.gold_chunk_ids:
                retrieved = retrieved_chunks
                gold = list(question.gold_chunk_ids)
            else:
                retrieved = retrieved_docs
                gold = list(question.gold_docs)

            retrieved = retrieved[:TOP_K]

            rows.append({
                "qid": question.qid,
                "hit": hit_at_k(retrieved, gold),
                "recall": recall_at_k(retrieved, gold),
                "mrr": mrr_at_k(retrieved, gold, TOP_K),
            })

    return {
        metric: round(
            sum(row[metric] for row in rows) / len(rows), 4
        )
        for metric in ("hit", "recall", "mrr")
    }


def test_compare_semantic_keyword_and_hybrid(capsys):
    results = {
        strategy: run_strategy(strategy)
        for strategy in ("semantic", "keyword", "hybrid")
    }

    print("\nRetrieval strategy comparison (top_k=3)")
    print(f"{'Strategy':<12} {'Hit@3':>10} {'Recall@3':>10} {'MRR@3':>10}")

    for strategy, metrics in results.items():
        print(
            f"{strategy:<12} "
            f"{metrics['hit']:>10.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['mrr']:>10.4f}"
        )

        assert all(
            0.0 <= metrics[name] <= 1.0
            for name in ("hit", "recall", "mrr")
        )

    # This is a comparison, not an assumption that hybrid must win.
    assert set(results) == {"semantic", "keyword", "hybrid"}
