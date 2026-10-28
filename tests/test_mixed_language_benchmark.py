from rag.evaluation.benchmark import QUESTIONS, run_benchmark
from rag.evaluation.metrics import mrr_at_k, ndcg_at_k, precision_at_k


def test_benchmark_contains_mixed_arabic_english_cases():
    mixed = [question for question in QUESTIONS if question.case_type == "mixed_language"]
    assert len(mixed) >= 2
    assert all(question.language == "mixed_ar_en" for question in mixed)
    assert all(question.gold_docs for question in mixed)
    assert any("linear equation" in question.question.lower() for question in mixed)
    assert any("cramer" in question.question.lower() for question in mixed)


def test_missing_evidence_and_false_premise_are_marked_for_abstention():
    by_case = {question.case_type: question for question in QUESTIONS}
    assert by_case["missing_evidence"].expect_abstain is True
    assert by_case["false_premise"].expect_abstain is True


def test_math_notation_cases_have_chunk_level_gold_evidence():
    math_cases = [
        question
        for question in QUESTIONS
        if question.case_type == "math_symbol_notation"
    ]

    assert len(math_cases) == 2
    assert all(question.gold_chunk_ids == ("math_notation#0",) for question in math_cases)


def test_ranking_metrics_use_relevant_positions():
    retrieved = ["wrong", "gold", "other"]
    gold = ["gold"]

    assert precision_at_k(retrieved, gold, 2) == 0.5
    assert mrr_at_k(retrieved, gold, 2) == 0.5
    assert round(ndcg_at_k(retrieved, gold, 2), 4) == 0.6309


def test_math_notation_cases_retrieve_the_expected_chunk():
    summary = run_benchmark(embedder_name="hashing", top_k=3)
    by_qid = {row["qid"]: row for row in summary["rows"]}

    for qid in ("q11", "q12"):
        assert by_qid[qid]["metric_level"] == "chunk"
        assert by_qid[qid]["hit"] == 1.0
        assert "math_notation#0" in by_qid[qid]["retrieved_chunks"]
