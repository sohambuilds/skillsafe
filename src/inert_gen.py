"""
The inert control. CLAUDE.md section 3.

    "The `inert` skill is the primary comparison, not `none`. Constructing it: take the
     k=0 seed skill and replace every procedural instruction with descriptive prose
     about tabular data formats that tells the model nothing about how to act. Match
     token count to within +/-5% and log both counts."

One change from the brief, flagged in the README as an open decision:

CLAUDE.md matches inert to the k=0 SEED, but section 6 compares inert against opt_k5.
Optimizers grow prompts. If opt_k5 is two or three times the length of the seed, the
length confound that inert exists to kill walks straight back in -- and section 1 already
names "the effect may be length, not content" as the most likely way this experiment
fools itself. So inert is generated per skill: build_inert() takes a target token count
and hits it within tolerance, and battery.py pairs every opt_k with an inert control of
matching length. Generating these is local and free; it costs no optimizer calls.

The prose below is descriptive and third-person throughout. validate_inert() enforces
that automatically: an inert control that accidentally instructs is not a control, and
that failure would be invisible in the results. It is the same fault-injection logic
section 2 asks for on judges, applied to the control.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from src import config
from src.paths import SKILLS

HEADING = "# Table Question Answering"

SECTION_HEADINGS = [
    "## Where these tables come from",
    "## Header rows",
    "## Cell contents",
    "## Interchange formats",
    "## The dataset",
    "## String comparison",
    "## Structure and provenance",
]

PARAGRAPHS = [
    "The tables in this dataset are drawn from articles on Wikipedia. Each one was "
    "originally laid out in HTML and has been converted into a header row and a "
    "sequence of data rows. The conversion preserves the visible text of each cell and "
    "discards the markup around it, so what remains is the string a reader of the "
    "article would have seen on the page.",
    "Header rows in Wikipedia tables are written for human readers rather than for "
    "programs. They are frequently abbreviated, occasionally duplicated across columns, "
    "and sometimes span more than one line in the original article. A header such as "
    "Pos. might stand for a position in a race, a position on a field, or a position in "
    "a ranked list, and the surrounding article was what originally disambiguated it.",
    "Cell contents in these tables are not typed. A column of years is a column of "
    "strings that happen to look like years, and a column of scores may mix integers, "
    "dashes standing for missing values, and occasional footnote text. Two cells that "
    "render identically on the page may differ in their underlying characters, since "
    "editors use en dashes, em dashes, and hyphens somewhat interchangeably.",
    "Footnote markers are common in Wikipedia tables. They appear as bracketed numbers "
    "or short bracketed words immediately after the value they annotate. In the "
    "rendered article they link to a references section that the table extraction does "
    "not carry with it, so the marker survives as text with nothing behind it.",
    "Numeric formatting varies across articles and sometimes within a single table. "
    "Thousands separators appear as commas in articles written in English, and as "
    "spaces or periods in tables translated from other language editions. Currency "
    "symbols precede the amount in some conventions and follow it in others. "
    "Percentages appear both with and without the percent sign.",
    "Dates in these tables take many forms. A column may hold full dates, years alone, "
    "month and year pairs, or ranges written with a dash. Ranges are sometimes "
    "abbreviated so that only the last two digits of the closing year are given, as in "
    "1998-99, and sometimes written out in full.",
    "The comma-separated values format is one of the oldest interchange formats still "
    "in wide use. A file in that format holds one record per line, with fields "
    "separated by commas, and quoting rules that allow a field to contain a comma of "
    "its own. The format carries no schema, no type declarations, and no standard way "
    "of marking a header row, which is why readers of such a file fall back on "
    "convention.",
    "Tab-separated values differ from comma-separated values mainly in the choice of "
    "delimiter. The tab character is rarer than the comma in ordinary prose, so "
    "tab-separated files require quoting less often, at the cost of being harder to "
    "inspect in an editor that renders tabs inconsistently.",
    "Markdown tables represent a grid with pipe characters between cells and a row of "
    "dashes beneath the header. The format was designed to stay readable in its source "
    "form, so alignment is optional and the width of a column in the source carries no "
    "meaning. A markdown table has no way of expressing a merged cell.",
    "Merged cells are common in Wikipedia and absent from every flat table format. A "
    "cell spanning three columns in the original article has to be either repeated "
    "across those three columns or dropped when the table is flattened, and different "
    "extraction tools resolve this differently.",
    "The WikiTableQuestions dataset was assembled by pairing tables extracted from "
    "Wikipedia with questions written by crowdworkers who had the table in front of "
    "them. The questions were collected under the constraint that the table alone "
    "settles them. The reference answers are short strings, and a question sometimes "
    "has several of them.",
    "Evaluation of datasets of this kind compares strings rather than meanings. String "
    "comparison is sensitive to details a reader would pass over: a trailing space, a "
    "non-breaking space standing in for an ordinary one, or a character that renders as "
    "a hyphen but is encoded as a dash. Whether two strings count as the same depends "
    "entirely on what a given comparison routine folds together.",
    "The rows of a Wikipedia table sit in the order the article's editors left them. "
    "That order is sometimes chronological, sometimes alphabetical, sometimes governed "
    "by a ranking column, and sometimes arbitrary. Nothing in the flattened table "
    "records which of these was intended.",
    "Column counts are stable within a table and vary widely between tables in the "
    "dataset. Some tables have two columns and a handful of rows; others carry more "
    "than a dozen columns and several hundred rows. The distribution of table sizes has "
    "a long tail.",
    "Units appear inline in Wikipedia tables rather than in the header. A distance "
    "column may hold values written in kilometres in one row and in miles in another, "
    "with the unit written out beside the number, and the article's prose was where the "
    "convention was explained.",
    "Empty cells are represented in several ways across these tables. A blank string, a "
    "single dash, the abbreviation n/a, and the word unknown all appear, and the choice "
    "among them is a matter of editorial habit in the article the table came from.",
    "Sorting behaviour in the original articles came from a script that ran in the "
    "reader's browser. It attached itself to the header row and inferred a type per "
    "column from the visible values. That inference is preserved nowhere in the "
    "extracted table.",
    "The provenance of a table is recorded only as the title of the article it came "
    "from. Two tables with the same structure may come from unrelated articles, and two "
    "tables from the same article may share no column at all.",
]

# ---------------------------------------------------------------------------
# Fault injection on the control itself
# ---------------------------------------------------------------------------

_DIRECTIVE_SUBSTRINGS = (
    " you ", " your ", " yours ", " must ", " should ", " ought ", " need to ",
    " be sure ", " ensure ", " make sure ", " never ", " always ", " do not ",
    " don't ", " avoid ", " prefer ", " instead of ", " it is important ",
)

_IMPERATIVE_OPENERS = (
    "read", "write", "check", "use", "output", "produce", "give", "list", "ignore",
    "strip", "count", "compare", "select", "answer", "state", "end", "begin",
    "return", "decline", "refuse", "explain", "note", "remember", "consider",
    "identify", "locate", "work", "confirm", "keep", "copy", "separate",
)


def validate_inert(text: str) -> list[str]:
    """Return a list of reasons this text is not behaviourally empty. Empty list = ok."""
    problems: list[str] = []
    # Collapse newlines to spaces first, or a directive at the start of a line escapes
    # the space-delimited substring check.
    padded = " " + re.sub(r"\s+", " ", text.lower()) + " "
    for needle in _DIRECTIVE_SUBSTRINGS:
        if needle in padded:
            problems.append(f"directive phrase present: {needle.strip()!r}")

    body = re.sub(r"^#.*$", "", text, flags=re.MULTILINE)
    for sentence in re.split(r"(?<=[.!?])\s+", body):
        sentence = sentence.strip().strip("-*# ")
        if not sentence:
            continue
        first = re.split(r"\W+", sentence, maxsplit=1)[0].lower()
        if first in _IMPERATIVE_OPENERS:
            problems.append(f"sentence opens with an imperative verb: {sentence[:60]!r}")
    return problems


# ---------------------------------------------------------------------------
# Length matching
# ---------------------------------------------------------------------------


def _load_tokenizer(tokenizer_id: str | None):
    if tokenizer_id is None:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_id)


def count_tokens(text: str, tokenizer=None) -> int:
    """Token count under `tokenizer`, or a chars/4 approximation when none is given.

    The approximation exists so this module is testable without a model download. It is
    never used for the real length matching -- build_inert() is always called with a
    real tokenizer, and the resulting counts are written into the manifest.
    """
    if tokenizer is None:
        return max(1, round(len(text) / 4))
    return len(tokenizer.encode(text, add_special_tokens=False))


def _sentences(paragraph: str) -> list[str]:
    return [s for s in re.split(r"(?<=\.)\s+", paragraph) if s.strip()]


def build_inert(target_tokens: int, tokenizer=None, seed: int = 0,
                tolerance: float = 0.05) -> str:
    """Assemble descriptive prose landing within `tolerance` of `target_tokens`.

    Three phases, coarse to fine. The earlier single-pass version stalled below the
    window: it filled with whole paragraphs, then greedily appended sentences from the
    one paragraph that overshot, and broke unconditionally. Sentences run 30-40 tokens
    while the +/-5% window at target 250 is 25 tokens wide, so "the next sentence
    overshoots" and "we are inside the window" are independent conditions -- at target
    250 it landed on 227 and returned it. CLAUDE.md section 3 makes +/-5% a hard
    requirement, so the fill has to be able to reach it from any starting point.

      1. whole paragraphs, headings every third, while they fit under `upper`
      2. best-fit sentence packing, drawing from every paragraph rather than one
      3. word-level trim of a single sentence, as a last resort

    Deterministic given (target_tokens, tokenizer, seed): paragraph order comes from a
    seeded shuffle and every tie is broken by corpus order, so regenerating a control for
    the same target always reproduces the same file byte for byte.
    """
    rng = random.Random(seed)
    order = list(range(len(PARAGRAPHS)))
    rng.shuffle(order)

    lower = target_tokens * (1 - tolerance)
    upper = target_tokens * (1 + tolerance)

    def render(parts: list[str]) -> str:
        return "\n\n".join(parts) + "\n"

    def size(parts: list[str]) -> int:
        return count_tokens(render(parts), tokenizer)

    # --- phase 1: whole paragraphs ------------------------------------------
    blocks: list[str] = [HEADING]
    heading_at = 0
    for step, index in enumerate(order * 6):  # cycles only for very long targets
        if size(blocks) >= lower:
            break
        if step % 3 == 0 and heading_at < len(SECTION_HEADINGS):
            trial = blocks + [SECTION_HEADINGS[heading_at]]
            if size(trial) <= upper:
                blocks = trial
                heading_at += 1
                if size(blocks) >= lower:
                    break
        trial = blocks + [PARAGRAPHS[index]]
        if size(trial) <= upper:
            blocks = trial

    # --- phase 2: best-fit sentence packing ---------------------------------
    # Finer granularity than a paragraph, and drawn from the whole corpus rather than
    # from whichever paragraph happened to overshoot, so a short sentence is available
    # to close a small gap.
    if size(blocks) < lower:
        pool = [s for index in order for s in _sentences(PARAGRAPHS[index])]
        tail: list[str] = []
        while size(blocks + ([" ".join(tail)] if tail else [])) < lower:
            best: tuple[int, str] | None = None
            for sentence in pool:
                total = size(blocks + [" ".join(tail + [sentence])])
                if total <= upper and (best is None or total > best[0]):
                    best = (total, sentence)
            if best is None:
                break
            tail.append(best[1])
            pool.remove(best[1])
        if tail:
            blocks.append(" ".join(tail))

    # --- phase 3: word-level trim, last resort ------------------------------
    # Leaves a sentence fragment, which is cosmetically poor but behaviourally identical:
    # the control is descriptive prose either way, and validate_inert() still governs.
    # Preferred over returning a length the brief forbids.
    if size(blocks) < lower:
        for sentence in (s for index in order for s in _sentences(PARAGRAPHS[index])):
            words = sentence.split()
            for cut in range(len(words), 0, -1):
                trial = blocks + [" ".join(words[:cut])]
                if lower <= size(trial) <= upper:
                    return render(trial)

    return render(blocks)


def match_skill(skill_path: Path, out_path: Path, tokenizer_id: str | None,
                seed: int = 0) -> dict:
    """Generate an inert control matched to `skill_path` and write it to `out_path`."""
    tokenizer = _load_tokenizer(tokenizer_id)
    target = count_tokens(skill_path.read_text(encoding="utf-8"), tokenizer)
    text = build_inert(target, tokenizer=tokenizer, seed=seed)

    problems = validate_inert(text)
    if problems:
        raise AssertionError(
            "generated inert control is not behaviourally empty:\n  "
            + "\n  ".join(problems)
        )

    out_path.write_text(text, encoding="utf-8")
    got = count_tokens(text, tokenizer)
    return {
        "matched_to": str(skill_path.relative_to(skill_path.parents[1])),
        "inert_path": str(out_path.relative_to(out_path.parents[1])),
        "tokenizer": tokenizer_id or "approx-chars/4",
        "target_tokens": target,
        "inert_tokens": got,
        "pct_difference": round(100 * (got - target) / max(target, 1), 2),
        "within_5pct": abs(got - target) <= 0.05 * target,
        # True when phase 3 of build_inert had to word-trim to reach the window. Not a
        # problem -- the control is still descriptive prose and still passes
        # validate_inert -- but it is recorded rather than left invisible.
        "ends_mid_sentence": not text.rstrip().endswith("."),
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", required=True, type=Path,
                        help="skill file whose token count the control must match")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tokenizer", default=config.REFERENCE_TOKENIZER,
                        help="pass 'approx' to use the chars/4 fallback (offline)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tokenizer_id = None if args.tokenizer == "approx" else args.tokenizer
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = match_skill(args.match, args.out, tokenizer_id, args.seed)
    print(json.dumps(manifest, indent=2))
    if not manifest["within_5pct"]:
        raise SystemExit(
            f"length match outside +/-5% ({manifest['pct_difference']}%) -- "
            "CLAUDE.md section 3 requires it to be inside"
        )


if __name__ == "__main__":
    SKILLS.mkdir(parents=True, exist_ok=True)
    main()
