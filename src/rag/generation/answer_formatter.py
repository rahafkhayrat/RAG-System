from __future__ import annotations

import re
from typing import Any


MATH_PATTERN = re.compile(
    r"\\\[(?P<bracket_display>.*?)\\\]"
    r"|\\\((?P<paren_inline>.*?)\\\)"
    r"|\$\$(?P<dollar_display>.*?)\$\$"
    r"|(?<!\$)\$(?P<dollar_inline>[^$\n]+?)\$(?!\$)",
    re.DOTALL,
)


def _text_block(text: str) -> dict[str, Any]:
    return {
        "type": "text",
        "text": text,
    }


def _math_block(latex: str, display: bool) -> dict[str, Any]:
    return {
        "type": "math",
        "text": "",
        "latex": latex.strip(),
        "display": display,
    }


def format_answer(answer: str) -> dict[str, Any]:
    """
    Split an answer into ordered text and math blocks.

    The original answer is not modified.
    The frontend can render math blocks using MathJax or KaTeX.
    """
    if not isinstance(answer, str) or not answer:
        return {
            "answer_format": "text",
            "content": [],
        }

    content: list[dict[str, Any]] = []
    position = 0
    has_math = False
    has_text = False

    for match in MATH_PATTERN.finditer(answer):
        if match.start() > position:
            text_part = answer[position:match.start()]

            content.append(_text_block(text_part))
            has_text = True

        if match.group("bracket_display") is not None:
            latex = match.group("bracket_display")
            display = True

        elif match.group("paren_inline") is not None:
            latex = match.group("paren_inline")
            display = False

        elif match.group("dollar_display") is not None:
            latex = match.group("dollar_display")
            display = True

        else:
            latex = match.group("dollar_inline")
            display = False

        content.append(_math_block(latex, display))

        has_math = True
        position = match.end()

    if position < len(answer):
        content.append(_text_block(answer[position:]))
        has_text = True

    if not has_math:
        return {
            "answer_format": "text",
            "content": [_text_block(answer)],
        }

    if has_text:
        answer_format = "mixed"
    else:
        answer_format = "math"

    return {
        "answer_format": answer_format,
        "content": content,
    }