import pytest

from rag.generation.answer_formatter import format_answer


def test_plain_text_answer():
    result = format_answer("The answer is correct.")

    assert result["answer_format"] == "text"
    assert result["content"] == [
        {
            "type": "text",
            "text": "The answer is correct.",
        }
    ]


def test_inline_latex():
    result = format_answer(
        r"The formula is \(x^2 + y^2 = z^2\)."
    )

    assert result["answer_format"] == "mixed"
    assert len(result["content"]) == 3

    assert result["content"][0]["type"] == "text"

    assert result["content"][1] == {
        "type": "math",
        "text": "",
        "latex": "x^2 + y^2 = z^2",
        "display": False,
    }

    assert result["content"][2]["type"] == "text"


def test_display_latex():
    result = format_answer(
        r"\[\frac{a}{b} = c\]"
    )

    assert result["answer_format"] == "math"
    assert result["content"] == [
        {
            "type": "math",
            "text": "",
            "latex": r"\frac{a}{b} = c",
            "display": True,
        }
    ]


def test_inline_dollar_math():
    result = format_answer(
        r"The result is $x + 1 = 5$."
    )

    math_blocks = [
        block
        for block in result["content"]
        if block["type"] == "math"
    ]

    assert len(math_blocks) == 1
    assert math_blocks[0]["latex"] == "x + 1 = 5"
    assert math_blocks[0]["display"] is False


def test_display_dollar_math():
    result = format_answer(
        r"$$\sum_{i=1}^{n} i$$"
    )

    assert result["answer_format"] == "math"
    assert result["content"][0]["latex"] == (
        r"\sum_{i=1}^{n} i"
    )
    assert result["content"][0]["display"] is True


def test_multiple_equations():
    result = format_answer(
        r"First \(x=2\), then \(y=3\)."
    )

    math_blocks = [
        block
        for block in result["content"]
        if block["type"] == "math"
    ]

    assert len(math_blocks) == 2
    assert math_blocks[0]["latex"] == "x=2"
    assert math_blocks[1]["latex"] == "y=3"


def test_arabic_text_with_math():
    result = format_answer(
        r"مساحة المستطيل هي \(A = l \times w\)"
    )

    assert result["answer_format"] == "mixed"
    assert result["content"][0]["text"] == (
        "مساحة المستطيل هي "
    )
    assert result["content"][1]["latex"] == (
        r"A = l \times w"
    )


def test_math_only_answer():
    result = format_answer(r"\(x=1\)")

    assert result["answer_format"] == "math"
    assert len(result["content"]) == 1


@pytest.mark.parametrize("answer", ["", None, 123])
def test_empty_or_invalid_answer(answer):
    result = format_answer(answer)

    assert result == {
        "answer_format": "text",
        "content": [],
    }