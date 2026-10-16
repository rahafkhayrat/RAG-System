from __future__ import annotations

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
