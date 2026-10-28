import json

from rag.contracts import Chunk, RetrievedChunk
from rag.evaluation.sufficiency import EvidenceSufficiencyChecker


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.last_system = ""
        self.last_user = ""

    def complete(self, system, user):
        self.calls += 1
        self.last_system = system
        self.last_user = user
        return self.response


def _evidence():
    chunk = Chunk(
        tenant_id="t1", course_id="math", document_id="doc1", chunk_id="c1",
        text="A variable represents a number.", page=1, source_uri="algebra.pdf",
    )
    return [RetrievedChunk(chunk=chunk, score=0.9)]


def test_empty_question_is_rejected_without_calling_llm():
    llm = FakeLLM('{"sufficient": true, "reason": "ok"}')
    result = EvidenceSufficiencyChecker(llm).evaluate("   ", _evidence())
    assert result.sufficient is False
    assert result.reason == "empty_question"
    assert llm.calls == 0


def test_no_evidence_is_rejected_without_calling_llm():
    llm = FakeLLM('{"sufficient": true, "reason": "ok"}')
    result = EvidenceSufficiencyChecker(llm).evaluate("What is a variable?", [])
    assert result.sufficient is False
    assert result.reason == "no_evidence"
    assert result.evidence_count == 0
    assert llm.calls == 0


def test_valid_sufficiency_json_is_parsed():
    llm = FakeLLM(json.dumps({"sufficient": True, "reason": "definition is present"}))
    result = EvidenceSufficiencyChecker(llm).evaluate("What is a variable?", _evidence())
    assert result.sufficient is True
    assert result.reason == "definition is present"
    assert result.evidence_count == 1
    assert "A variable represents a number." in llm.last_user


def test_invalid_json_fails_closed():
    llm = FakeLLM("not json")
    result = EvidenceSufficiencyChecker(llm).evaluate("What is a variable?", _evidence())
    assert result.sufficient is False
    assert result.reason == "sufficiency_evaluation_failed"


def test_non_boolean_sufficiency_fails_closed():
    llm = FakeLLM('{"sufficient": "yes", "reason": "maybe"}')
    result = EvidenceSufficiencyChecker(llm).evaluate("What is a variable?", _evidence())
    assert result.sufficient is False
    assert result.reason == "sufficiency_evaluation_failed"


def test_llm_provider_error_fails_closed():
    class BrokenLLM:
        def complete(self, system, user):
            raise RuntimeError("provider unavailable")

    result = EvidenceSufficiencyChecker(BrokenLLM()).evaluate(
        "What is a variable?", _evidence()
    )
    assert result.sufficient is False
    assert result.reason == "sufficiency_evaluation_failed"
