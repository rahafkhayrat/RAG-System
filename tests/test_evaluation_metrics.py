import pytest

from rag.evaluation.metrics import hit_at_k, recall_at_k, refusal_metrics, token_f1


def test_hit_at_k_is_one_when_any_gold_document_is_retrieved():
    assert hit_at_k(["wrong", "gold"], ["gold"]) == 1.0


def test_recall_at_k_counts_retrieved_gold_documents():
    assert recall_at_k(["gold1", "other"], ["gold1", "gold2"]) == 0.5


def test_token_f1_is_one_for_exact_token_match():
    assert token_f1("linear equation", "linear equation") == 1.0


def test_refusal_metrics_counts_abstention_errors():
    result = refusal_metrics(
        predicted_abstain=[True, False, True, False],
        expected_abstain=[True, True, False, False],
    )
    assert result == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_negative": 1,
        "precision": 0.5,
        "recall": 0.5,
        "accuracy": 0.5,
    }


def test_refusal_metrics_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="length mismatch"):
        refusal_metrics([True], [True, False])
