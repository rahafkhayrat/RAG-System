import pytest

from rag.retrieval.retriever import Retriever


@pytest.mark.parametrize("top_k", [0, -1, -5])
def test_retriever_rejects_invalid_top_k(top_k):
    retriever = Retriever(store=None, embedder=None)

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        retriever.retrieve(
            scope=None,
            question="What is RAG?",
            top_k=top_k,
        )