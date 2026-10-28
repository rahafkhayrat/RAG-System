from __future__ import annotations

import math
import re
from collections import Counter

_WORD = re.compile(r"\w+", re.UNICODE)


def tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def hit_at_k(retrieved_doc_ids: list[str], gold_doc_ids: list[str]) -> float:
    gold = set(gold_doc_ids)
    if not gold:
        return 0.0
    return 1.0 if gold & set(retrieved_doc_ids) else 0.0


def recall_at_k(retrieved_doc_ids: list[str], gold_doc_ids: list[str]) -> float:
    gold = set(gold_doc_ids)
    if not gold:
        return 0.0
    return len(gold & set(retrieved_doc_ids)) / len(gold)


def precision_at_k(
    retrieved_ids: list[str],
    gold_ids: list[str],
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    gold = set(gold_ids)
    if not gold:
        return 0.0
    return sum(item in gold for item in retrieved_ids[:k]) / k


def mrr_at_k(
    retrieved_ids: list[str],
    gold_ids: list[str],
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    gold = set(gold_ids)
    if not gold:
        return 0.0
    for rank, item in enumerate(retrieved_ids[:k], start=1):
        if item in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    gold_ids: list[str],
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    gold = set(gold_ids)
    if not gold:
        return 0.0

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(retrieved_ids[:k], start=1)
        if item in gold
    )
    ideal_count = min(len(gold), k)
    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )
    return dcg / ideal_dcg if ideal_dcg else 0.0


def token_f1(prediction: str, reference: str) -> float:
    pred = Counter(tokens(prediction))
    ref = Counter(tokens(reference))
    if not pred or not ref:
        return 0.0
    overlap = sum((pred & ref).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def refusal_metrics(predicted_abstain: list[bool], expected_abstain: list[bool]) -> dict[str, float | int]:
    """Evaluate abstention decisions; True means the system refused to answer.

    Precision/recall are reported for the positive class (expected abstention).
    Empty denominators yield 0.0 instead of dividing by zero.
    """
    if len(predicted_abstain) != len(expected_abstain):
        raise ValueError("predicted_abstain and expected_abstain length mismatch")

    tp = sum(p and e for p, e in zip(predicted_abstain, expected_abstain))
    fp = sum(p and not e for p, e in zip(predicted_abstain, expected_abstain))
    fn = sum(not p and e for p, e in zip(predicted_abstain, expected_abstain))
    tn = sum(not p and not e for p, e in zip(predicted_abstain, expected_abstain))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / len(expected_abstain) if expected_abstain else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round(accuracy, 4),
    }
