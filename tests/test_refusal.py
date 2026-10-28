from rag.contracts import Chunk, Scope
from rag.embeddings.hashing import HashingEmbedder
from rag.pipeline import RagEngine
from rag.store.memory import InMemoryVectorStore


class RecordingLLM:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.response


class AlwaysSufficientChecker:
    """Keep generation tests focused on the generation stage."""

    def evaluate(self, question, evidence):
        from rag.evaluation.sufficiency import SufficiencyResult

        return SufficiencyResult(
            sufficient=True,
            reason="test evidence accepted",
            evidence_count=len(evidence),
        )


def _engine_with_evidence(llm, sufficiency_checker=None):
    embedder = HashingEmbedder(dim=64)
    store = InMemoryVectorStore()
    chunk = Chunk(
        tenant_id="tenant-a",
        course_id="course-a",
        document_id="doc-a",
        chunk_id="doc-a#0",
        text="Linear equations can be solved by isolating the variable.",
        page=1,
        source_uri="course.docx",
    )
    store.upsert([chunk], embedder.embed([chunk.text]))
    return RagEngine(
        store=store,
        embedder=embedder,
        llm=llm,
        sufficiency_checker=sufficiency_checker,
    ), Scope("tenant-a", frozenset({"course-a"}))


def test_no_evidence_abstains_without_calling_llm():
    llm = RecordingLLM(response="should not be called")
    engine = RagEngine(InMemoryVectorStore(), HashingEmbedder(dim=64), llm=llm)
    result = engine.answer("What is in the course?", Scope("tenant-a", frozenset({"course-a"})))

    assert result.abstained is True
    assert result.reason == "no_evidence"
    assert result.answer == ""
    assert result.sources == ()
    assert llm.calls == 0


def test_evidence_from_another_course_is_not_used():
    llm = RecordingLLM(response="should not be called")
    engine, _ = _engine_with_evidence(llm)
    result = engine.answer("How do equations work?", Scope("tenant-a", frozenset({"course-b"})))

    assert result.abstained is True
    assert result.reason == "no_evidence"
    assert result.sources == ()
    assert llm.calls == 0


def test_generation_exception_abstains_instead_of_returning_server_error():
    llm = RecordingLLM(error=RuntimeError("provider unavailable"))
    engine, scope = _engine_with_evidence(
        llm,
        sufficiency_checker=AlwaysSufficientChecker(),
    )
    result = engine.answer("How do equations work?", scope)

    assert result.abstained is True
    assert result.reason == "generation_failed"
    assert result.answer == ""
    assert result.sources


def test_empty_generation_abstains():
    llm = RecordingLLM(response="  \n ")
    engine, scope = _engine_with_evidence(
        llm,
        sufficiency_checker=AlwaysSufficientChecker(),
    )
    result = engine.answer("How do equations work?", scope)

    assert result.abstained is True
    assert result.reason == "empty_generation"
    assert result.answer == ""
    assert result.sources
