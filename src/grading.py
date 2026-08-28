"""
WikiTableQuestions exact-match grading. CLAUDE.md section 3.

Two things happen here and they are deliberately kept apart:

  extract_answer()  -- pull a candidate answer string out of a free-text completion,
                       and report whether the completion followed the skill's stated
                       output format.
  exact_match()     -- normalise and compare against the reference answers.

Keeping them apart matters for the positive control. An optimizer can raise measured
accuracy purely by teaching the model to emit a parseable "Answer:" line, without the
model reasoning any better. That is still the optimizer optimising, but it is a
different story from better table reading, and section 6 should be able to tell them
apart. `format_ok` is logged on every task rollout so accuracy can be decomposed into
format compliance and accuracy-given-format.

The normalisation follows CLAUDE.md section 3 -- "lowercase, strip punctuation and
articles, handle multi-answer as a set" -- plus the parts of the official WTQ evaluator
that are cheap and matter in practice: footnote-marker removal, unicode punctuation
folding, and numeric canonicalisation. It is not a bit-for-bit reimplementation of
Stanford's `evaluator.py`; the deviations are listed in DEVIATIONS below and belong in
the write-up.
"""

from __future__ import annotations

import re
import unicodedata

DEVIATIONS = [
    "Articles (a/an/the) are stripped. The official WTQ evaluator does not do this; "
    "CLAUDE.md section 3 asks for it.",
    "Dates are compared as normalised strings, not as DateValue objects with "
    "partial-date semantics.",
    "Multi-answer splitting is heuristic: the model emits free text, whereas official "
    "WTQ submissions are already lists. See split_multi().",
]

# Unicode characters that show up in Wikipedia tables and in model output for the same
# logical character.
_UNICODE_FOLD = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", "​": "",
    "­": "",
}

# [3], [note 1], [citation needed], [a] -- footnote markers, never part of the answer.
_FOOTNOTE_RE = re.compile(r"\[[^\]]{0,24}\]")

_ARTICLE_RE = re.compile(r"\b(?:a|an|the)\b")

# Punctuation to drop in the string path. The number path runs first, so decimal points
# and minus signs inside numbers are already handled by then. Deleted rather than
# replaced with a space, SQuAD-style, so "U.S.A." folds onto "USA"; the whitespace
# collapse afterwards keeps "New York, NY" as two words.
_PUNCT_RE = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]")

_WS_RE = re.compile(r"\s+")

# "Answer: x", "**Final Answer:** x", "answer - x"
_ANSWER_LINE_RE = re.compile(
    r"^\s*[*_`>\s]*(?:final\s+)?answer[*_`\s]*\s*[:\-–]\s*(.+?)\s*$",
    re.IGNORECASE,
)

_STRIP_WRAPPERS = "`*_\"' \t"


def _fold_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "".join(_UNICODE_FOLD.get(ch, ch) for ch in text)


def _as_number(text: str) -> float | None:
    """Parse a cell that is semantically a number, tolerating the decorations that
    appear in Wikipedia tables and in model output: currency, thousands separators,
    percent signs, plus signs, and surrounding whitespace.

    Returns None if the text is not a bare number, so the caller falls back to the
    string path. "1,000" -> 1000.0, "$3.50" -> 3.5, "45%" -> 45.0, "-2" -> -2.0.
    """
    s = text.strip()
    if not s:
        return None
    s = s.replace(",", "")
    s = re.sub(r"^[£€¥$+]+", "", s)
    s = re.sub(r"%$", "", s)
    s = s.strip()
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _canonical_number(value: float) -> str:
    """1.0 -> '1', 1.50 -> '1.5', -0.0 -> '0'."""
    if value == int(value):
        return str(int(value))
    return repr(round(value, 6)).rstrip("0").rstrip(".")


def normalize(text: str) -> str:
    """Normalise a single answer value to its comparison form."""
    if text is None:
        return ""
    text = _fold_unicode(str(text))
    text = _FOOTNOTE_RE.sub(" ", text)
    text = text.strip().strip(_STRIP_WRAPPERS)

    number = _as_number(text)
    if number is not None:
        return _canonical_number(number)

    text = text.lower()
    # Punctuation first, then articles. The other order is wrong: "u.s.a." has a word
    # boundary around its final "a", so stripping articles first deletes it and the
    # answer no longer matches "USA".
    text = _PUNCT_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()

    stripped = _WS_RE.sub(" ", _ARTICLE_RE.sub(" ", text)).strip()
    # Guard: an answer that IS an article ("the", "a") must not normalise to empty.
    if stripped:
        text = stripped
    return text


def split_multi(prediction: str, n_gold: int) -> list[str]:
    """Split a free-text prediction into `n_gold` values.

    Conditioned on the gold arity, which is known at grading time. This avoids the
    thousands-separator trap: "1,000" must not become two answers, and it cannot,
    because comma splitting is only attempted when the gold answer has more than one
    value AND the split lands on exactly the right count.
    """
    text = prediction.strip()
    if n_gold <= 1:
        return [text]

    for pattern in (r"\|", r"\n", r";"):
        parts = [p.strip() for p in re.split(pattern, text) if p.strip()]
        if len(parts) == n_gold:
            return parts

    for pattern in (r",", r"\band\b", r"\s{2,}"):
        parts = [p.strip() for p in re.split(pattern, text) if p.strip()]
        if len(parts) == n_gold:
            return parts

    # No split recovers the right arity. Return the best-effort split so the comparison
    # fails honestly rather than accidentally matching on a single blob.
    parts = [p.strip() for p in re.split(r"\||\n|;|,", text) if p.strip()]
    return parts or [text]


def extract_answer(completion: str) -> tuple[str, bool]:
    """Return (answer_text, format_ok).

    format_ok is True when the completion ended with the output format the skills ask
    for -- an "Answer: ..." line. When it is False the last non-empty line is used
    instead, so a well-reasoned answer in the wrong shape is not scored zero, but the
    format failure is still recorded.
    """
    if not completion:
        return "", False

    lines = [ln for ln in completion.splitlines() if ln.strip()]
    if not lines:
        return "", False

    for line in reversed(lines):
        match = _ANSWER_LINE_RE.match(line)
        if match:
            return match.group(1).strip().strip(_STRIP_WRAPPERS), True

    last = lines[-1].strip()
    last = re.sub(r"^[-*•]\s+", "", last)
    last = last.strip().strip(_STRIP_WRAPPERS)
    last = re.sub(r"\.$", "", last).strip()
    return last, False


def exact_match(prediction: str, gold: list[str] | str) -> bool:
    """WTQ exact match. `prediction` is an already-extracted answer string."""
    gold_list = [gold] if isinstance(gold, str) else list(gold)
    gold_norm = {normalize(g) for g in gold_list}
    gold_norm.discard("")
    if not gold_norm:
        return False

    parts = split_multi(prediction, len(gold_norm))
    pred_norm = {normalize(p) for p in parts}
    pred_norm.discard("")
    return pred_norm == gold_norm


def grade(completion: str, gold: list[str] | str) -> dict:
    """Grade a raw completion. Returns the fields that go onto the rollout record."""
    answer, format_ok = extract_answer(completion)
    return {
        "extracted_answer": answer,
        "format_ok": format_ok,
        "correct": exact_match(answer, gold),
        "gold": [gold] if isinstance(gold, str) else list(gold),
    }
