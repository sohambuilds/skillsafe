"""
CLAUDE.md section 3: "Implement it once in src/grading.py and unit-test it against 20
known pairs before using it."

Twenty-eight pairs below, chosen to cover the normalisation cases that actually appear
in WikiTableQuestions gold answers and in 8B model output: casing, articles, footnote
markers, thousands separators, currency, percentages, unicode dashes and quotes,
trailing zeros, multi-answer sets in the wrong order, and honest negatives.

The negatives matter as much as the positives. A normaliser that matches everything
scores 100% and measures nothing -- the same failure mode section 2 warns about for
judges.
"""

from __future__ import annotations

import pytest

from src.grading import exact_match, extract_answer, grade, normalize, split_multi

# (prediction, gold, expected) -- the twenty known pairs, plus extras.
PAIRS = [
    # --- identity and casing -------------------------------------------------
    ("Paris", ["Paris"], True),
    ("paris", ["Paris"], True),
    ("john smith", ["John Smith"], True),
    ("  Paris  ", ["Paris"], True),
    # --- articles ------------------------------------------------------------
    ("The Beatles", ["Beatles"], True),
    ("Beatles", ["The Beatles"], True),
    ("a", ["a"], True),  # an answer that IS an article must not normalise to empty
    ("the", ["The"], True),
    # --- punctuation ---------------------------------------------------------
    ("U.S.A.", ["USA"], True),
    ("New York, NY", ["New York NY"], True),
    ('"Hello"', ["Hello"], True),
    ("“Hello”", ["Hello"], True),  # curly quotes
    # --- footnote markers ----------------------------------------------------
    ("New York City[3]", ["New York City"], True),
    ("Smith [citation needed]", ["Smith"], True),
    # --- numbers -------------------------------------------------------------
    ("1,000", ["1000"], True),
    ("1.0", ["1"], True),
    ("2", ["2.0"], True),
    ("$1,234.50", ["1234.5"], True),
    ("50%", ["50"], True),
    ("−5", ["-5"], True),  # unicode minus sign
    # --- multi-answer as a set ----------------------------------------------
    ("apple | banana", ["banana", "apple"], True),
    ("apple, banana", ["apple", "banana"], True),
    ("apple\nbanana", ["banana", "apple"], True),
    # --- honest negatives ----------------------------------------------------
    ("3", ["4"], False),
    ("1st", ["first"], False),
    ("", ["Paris"], False),
    ("apple", ["apple", "banana"], False),  # missing one of two
    ("apple | banana | cherry", ["apple", "banana"], False),  # one too many
]


@pytest.mark.parametrize("prediction,gold,expected", PAIRS)
def test_known_pairs(prediction, gold, expected):
    assert exact_match(prediction, gold) is expected


def test_at_least_twenty_pairs():
    """The brief asks for twenty. Fail loudly if someone trims the list."""
    assert len(PAIRS) >= 20


def test_normalizer_is_not_vacuous():
    """A normaliser that collapses everything onto the same string would pass every
    positive above. Check that distinct answers stay distinct."""
    distinct = {normalize(p) for p, _, _ in PAIRS if p.strip()}
    assert len(distinct) > 12


# ---------------------------------------------------------------------------
# Answer extraction -- format compliance is measured separately from correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "completion,answer,format_ok",
    [
        ("Answer: Paris", "Paris", True),
        ("Let me look.\nAnswer: Paris", "Paris", True),
        ("**Final Answer:** 42", "42", True),
        ("answer - 42", "42", True),
        ("Answer: 5\nSome trailing note", "5", True),  # last matching line wins
        ("The answer is Paris.", "The answer is Paris", False),
        ("- Paris", "Paris", False),
        ("", "", False),
        ("   \n  \n", "", False),
    ],
)
def test_extract_answer(completion, answer, format_ok):
    got, ok = extract_answer(completion)
    assert got == answer
    assert ok is format_ok


def test_split_multi_does_not_break_thousands_separator():
    """The trap this guards: gold arity 1, prediction '1,000'. Splitting on commas
    would produce two answers and score a correct response wrong."""
    assert split_multi("1,000", 1) == ["1,000"]
    assert exact_match("1,000", ["1000"]) is True


def test_split_multi_prefers_pipe_over_comma():
    assert split_multi("1,000 | 2,000", 2) == ["1,000", "2,000"]


def test_grade_record_shape():
    record = grade("Answer: 1,000", ["1000"])
    assert record["extracted_answer"] == "1,000"
    assert record["format_ok"] is True
    assert record["correct"] is True
    assert record["gold"] == ["1000"]


def test_grade_separates_format_from_correctness():
    """Right answer, wrong shape: correct, but format_ok False. This decomposition is
    what tells us whether the optimizer improved reasoning or only taught the model to
    emit a parseable line."""
    record = grade("Paris", ["Paris"])
    assert record["correct"] is True
    assert record["format_ok"] is False
