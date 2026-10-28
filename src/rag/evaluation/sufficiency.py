from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..contracts import RetrievedChunk
from ..generation.llm import LLMClient


@dataclass(frozen=True)
class SufficiencyResult:
    sufficient: bool
    reason: str
    evidence_count: int


class EvidenceSufficiencyChecker:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def evaluate(
        self,
        question: str,
        evidence: list[RetrievedChunk],
    ) -> SufficiencyResult:
        if not question.strip():
            return SufficiencyResult(
                sufficient=False,
                reason="empty_question",
                evidence_count=len(evidence),
            )

        if not evidence:
            return SufficiencyResult(
                sufficient=False,
                reason="no_evidence",
                evidence_count=0,
            )

        # Preserve source boundaries so the checker can evaluate
        # information across multiple retrieved chunks.
        context_parts = []

        for index, result in enumerate(evidence, start=1):
            chunk = result.chunk

            source = chunk.source_uri or "Unknown source"
            page = chunk.page

            context_parts.append(
                f"[Evidence {index}]\n"
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Content:\n{chunk.text}"
            )

        context = "\n\n".join(context_parts)

        system_prompt = """
You are an evidence sufficiency evaluator for an educational RAG system.

Your task is ONLY to determine whether the supplied course excerpts contain
enough information to answer the student's question.

Rules:
1. Evaluate the actual meaning of the evidence, not exact keyword matches.
2. The answer may be supported by information distributed across multiple
   excerpts.
3. Accept definitions, formulas, worked examples, and explanations as
   evidence when they answer the question.
4. Mathematical evidence may be expressed using equations, symbols, or text.
5. Do not mark evidence insufficient merely because it is brief or uses
   terminology different from the question.
6. Mark evidence insufficient if the required information is genuinely
   missing, contradictory, or unrelated.
7. Do not use outside knowledge to fill gaps in the supplied evidence.
8. Do not answer the student's question.

Return one valid JSON object only, with this schema:
{
  "sufficient": true,
  "reason": "Brief explanation"
}

The value of "sufficient" must be a JSON boolean.
Do not wrap the JSON in Markdown fences.
""".strip()

        user_prompt = (
            f"Student question:\n{question.strip()}\n\n"
            f"Retrieved course evidence:\n{context}\n\n"
            "Does the evidence support answering this question?"
        )

        try:
            raw_result = self.llm.complete(
                system_prompt,
                user_prompt,
            )

            data = self._parse_json(raw_result)

            sufficient = data.get("sufficient")
            reason = data.get("reason", "")

            if not isinstance(sufficient, bool):
                raise ValueError(
                    "The sufficiency response must contain a boolean."
                )

            return SufficiencyResult(
                sufficient=sufficient,
                reason=str(reason),
                evidence_count=len(evidence),
            )

        except Exception:
            # Keep the conservative fallback when the evaluator fails.
            return SufficiencyResult(
                sufficient=False,
                reason="sufficiency_evaluation_failed",
                evidence_count=len(evidence),
            )

    @staticmethod
    def _parse_json(raw_result: str) -> dict:
        """Parse plain JSON or JSON surrounded by Markdown fences."""
        if not isinstance(raw_result, str) or not raw_result.strip():
            raise ValueError("Empty sufficiency response.")

        text = raw_result.strip()

        # Remove an optional Markdown JSON fence.
        text = re.sub(
            r"^\s*```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s*```\s*$", "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Recover a JSON object if the provider added surrounding text.
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)

            if match is None:
                raise ValueError("No JSON object found.")

            data = json.loads(match.group(0))

        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")

        return data
