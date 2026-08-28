"""
Freeze every input to disk, once. CLAUDE.md section 3: "Fixed random sample, seed 0,
saved to disk. Never resample."

Enforced, not just documented: an existing frozen file is never silently overwritten,
and data/frozen_manifest.json records the SHA256 of each file. If a downstream result
cannot be traced to a specific digest here, it did not happen.

Every item carries a content-derived `item_id`. The paired analysis in section 6 joins
on that id, not on row index, so re-ordering or re-reading a file cannot silently
mis-pair a prompt with another prompt's verdict.

Run on the GPU server (it needs network):

    uv run python -m src.data_prep --all
    uv run python -m src.data_prep --safe-source orbench_hard --force   # if swapping
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from src import config
from src.paths import (
    BATTERY_HARMFUL,
    BATTERY_SAFE,
    DATA,
    FROZEN_MANIFEST,
    WTQ_HELDOUT,
    WTQ_TRAIN,
    WTQ_VAL,
    ensure_dirs,
)


def item_id(text: str, prefix: str) -> str:
    return f"{prefix}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict], force: bool) -> None:
    """Idempotent freeze. CLAUDE.md section 3: "Never resample."

    Regenerating byte-identical content is not resampling, so it is allowed silently --
    every draw here is seeded, so a re-run of a step that already succeeded reproduces
    exactly. Regenerating DIFFERENT content is resampling, and that stops the run.

    This is a stronger guard than refusing on mere existence: bare existence blocks a
    harmless retry after an unrelated step failed, while saying nothing about whether the
    upstream source shifted underneath a file that already exists.
    """
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == payload:
            print(f"unchanged {len(rows):>5} rows == {path}")
            return
        if not force:
            raise SystemExit(
                f"{path} exists and the regenerated content DIFFERS from it "
                f"({len(existing.splitlines())} rows on disk vs {len(rows)} now). The "
                "upstream source has changed under a frozen file. Section 3 says never "
                "resample: investigate the difference, then pass --force only if you "
                "intend to re-freeze, and record why in FINDINGS.md."
            )
        print(f"OVERWRITING {path} (--force)")
    path.write_text(payload, encoding="utf-8")
    print(f"wrote {len(rows):>5} rows -> {path}")


# ---------------------------------------------------------------------------
# WikiTableQuestions
# ---------------------------------------------------------------------------


def _load_wtq():
    """Load WTQ, working around the deprecation of script-based dataset loading.

    `Stanford/wikitablequestions` originally shipped a loading script. datasets>=3
    refuses to execute those, so the fallback reads the Hub's auto-converted parquet
    branch, which exists for every public dataset.
    """
    from datasets import load_dataset

    try:
        # trust_remote_code is explicit rather than left to the interactive [y/N] prompt.
        # The prompt blocks a re-run under nohup or in CI, and answering it by hand makes
        # the freeze depend on a keystroke that nothing records. The repo is Stanford's
        # own and its loading script is what produced the 11321/2831/4344 splits.
        return load_dataset(config.WTQ_HF_ID, trust_remote_code=True)
    except Exception as first_error:  # noqa: BLE001 - we re-raise with both causes
        print(f"  default load failed ({type(first_error).__name__}), trying parquet branch")
        try:
            return load_dataset(config.WTQ_HF_ID, revision="refs/convert/parquet")
        except Exception as second_error:  # noqa: BLE001
            raise SystemExit(
                "Could not load WikiTableQuestions.\n"
                f"  direct:  {first_error}\n"
                f"  parquet: {second_error}\n"
                "Check `hf auth whoami`, then try "
                "`uv run python -c \"from datasets import load_dataset; "
                f"print(load_dataset('{config.WTQ_HF_ID}'))\"` to see the raw error."
            ) from second_error


def _normalise_table(table) -> dict:
    if isinstance(table, dict) and "header" in table and "rows" in table:
        return {
            "header": [str(c) for c in table["header"]],
            "rows": [[str(c) for c in row] for row in table["rows"]],
            "name": str(table.get("name", "")),
        }
    raise SystemExit(
        f"Unexpected WTQ table schema: {type(table)} with keys "
        f"{list(table) if hasattr(table, 'keys') else 'n/a'}. "
        "Inspect one example before continuing -- do not guess the mapping."
    )


def _content_key(row: dict) -> str:
    """Split-independent identity of a WTQ item, for the disjointness assertion."""
    return f"{row['source_id']}|{row['question'].strip().lower()}"


def _extract(example, index: int, split_name: str, tag: str) -> dict:
    question = str(example["question"])
    return {
        "item_id": item_id(f"{example.get('id', index)}|{question}", tag),
        "source_id": str(example.get("id", index)),
        "split": split_name,
        "question": question,
        "table": _normalise_table(example["table"]),
        "answers": [str(a) for a in example["answers"]],
    }


def prepare_wtq(force: bool) -> list[dict]:
    """Three splits, disjoint by construction and then asserted disjoint anyway.

      train    the optimizer reads failure traces from here
      val      accept/reject only; the optimizer never sees these items or their failures
      heldout  the section 6 positive control; nothing in the loop touches it

    CLAUDE.md section 3 has two splits, which would score the hillclimb's accept/reject
    on the same 200 examples the optimizer reads failures from -- selection on the
    training set, with the optimism flowing straight into the positive control.
    """
    print("WikiTableQuestions")
    dataset = _load_wtq()
    print(f"  splits available: { {k: len(v) for k, v in dataset.items()} }")

    for required in ("train", "test"):
        if required not in dataset:
            raise SystemExit(f"WTQ split {required!r} missing; got {list(dataset)}")

    manifest_entries = []
    written: dict[str, set[str]] = {}

    # train and val are carved out of one draw from WTQ `train`, so they cannot overlap.
    rng = random.Random(config.WTQ_SAMPLE_SEED)
    train_split = dataset["train"]
    need = config.WTQ_N_TRAIN + config.WTQ_N_VAL
    if need > len(train_split):
        raise SystemExit(f"WTQ train has {len(train_split)} rows; need {need}")
    drawn = rng.sample(range(len(train_split)), k=need)

    for out_path, tag, indices in (
        (WTQ_TRAIN, "wtqtr", drawn[: config.WTQ_N_TRAIN]),
        (WTQ_VAL, "wtqva", drawn[config.WTQ_N_TRAIN:]),
    ):
        rows = [_extract(train_split[i], i, "train", tag) for i in indices]
        write_jsonl(out_path, rows, force)
        # Keyed on CONTENT, not on item_id: item_id carries a per-split tag prefix, so
        # comparing item_ids across splits intersects the empty set every time and the
        # assertion below could never fail. That is the same blind-check shape this
        # project keeps finding in its own instruments.
        written[out_path.stem] = {_content_key(r) for r in rows}
        manifest_entries.append({
            "file": out_path.name,
            "source": f"{config.WTQ_HF_ID}:train",
            "n": len(rows),
            "seed": config.WTQ_SAMPLE_SEED,
            "sha256": sha256_file(out_path),
        })

    test_split = dataset["test"]
    indices = random.Random(config.WTQ_SAMPLE_SEED).sample(
        range(len(test_split)), k=min(config.WTQ_N_HELDOUT, len(test_split))
    )
    rows = [_extract(test_split[i], i, "test", "wtqho") for i in indices]
    write_jsonl(WTQ_HELDOUT, rows, force)
    written[WTQ_HELDOUT.stem] = {_content_key(r) for r in rows}
    manifest_entries.append({
        "file": WTQ_HELDOUT.name,
        "source": f"{config.WTQ_HF_ID}:test",
        "n": len(rows),
        "seed": config.WTQ_SAMPLE_SEED,
        "sha256": sha256_file(WTQ_HELDOUT),
    })

    # Compared on content keys (source id + question text), so an item reaching two
    # splits by different routes is caught. All three pairs, not just train/heldout.
    overlaps = {}
    names = sorted(written)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            shared = written[left] & written[right]
            overlaps[f"{left}/{right}"] = len(shared)
            if shared:
                raise SystemExit(
                    f"{len(shared)} items appear in both {left} and {right}. "
                    "val must be unseen by the optimizer and heldout unseen by the "
                    "loop; the section 6 positive control is worthless otherwise."
                )
    print(f"  pairwise overlap, all three pairs: {overlaps}  (clean)")
    manifest_entries.append({
        "file": "_wtq_split_disjointness",
        "pairwise_overlap": overlaps,
        "n_train": config.WTQ_N_TRAIN,
        "n_val": config.WTQ_N_VAL,
        "n_heldout": config.WTQ_N_HELDOUT,
    })
    return manifest_entries


# ---------------------------------------------------------------------------
# Battery arms
# ---------------------------------------------------------------------------

STRONGREJECT_CSV = (
    "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/"
    "strongreject_dataset/strongreject_dataset.csv"
)

# How many rows each source is expected to yield AFTER filtering. Checked, not assumed.
# The failure this guards is quiet and severe: the safe arm is built by keeping rows with
# label == "safe". If a mirror renames or inverts its label vocabulary, the filter can
# silently pass every row through, and the "over-refusal arm" ends up ~44% genuinely
# harmful prompts (XSTest is 250 safe + 200 unsafe). Nothing downstream would reveal it.
EXPECTED_ROWS = {"strongreject": 313, "xstest_safe": 250, "orbench_hard": 1319}


def _try_sources(candidates: list[tuple[str, object]]) -> tuple[list[dict], str]:
    """Try each source in order; return the first that yields rows, plus its name.

    Reports every failure together at the end rather than chaining one exception inside
    another's handler, which is what made the first XSTest failure unreadable.
    """
    errors = []
    for name, loader in candidates:
        try:
            rows = loader()
        except Exception as error:  # noqa: BLE001 - collected and reported below
            errors.append((name, f"{type(error).__name__}: {error}"))
            continue
        if rows:
            print(f"  resolved via {name}  ({len(rows)} rows)")
            return rows, name
        errors.append((name, "returned 0 rows"))
    raise SystemExit(
        "every candidate source failed:\n"
        + "\n".join(f"  {name}: {why}" for name, why in errors)
    )


def _hf(repo: str, *, config_name: str | None = None, split: str = "train"):
    def load():
        from datasets import load_dataset

        data = (load_dataset(repo, config_name, split=split) if config_name
                else load_dataset(repo, split=split))
        return list(data)
    return load


def _csv(url: str):
    def load():
        import pandas as pd

        return pd.read_csv(url).to_dict("records")
    return load


def _load_harmful(source: str) -> tuple[list[dict], str]:
    if source == "strongreject":
        rows, resolved = _try_sources([
            # The CSV is the canonical release and is what actually resolved on first
            # run; the HF mirrors are kept as fallbacks, not as the primary.
            ("csv:alexandrasouly/strongreject", _csv(STRONGREJECT_CSV)),
            ("hf:walledai/StrongREJECT", _hf("walledai/StrongREJECT")),
            ("hf:csinva/strongreject", _hf("csinva/strongreject")),
        ])
        key = next((c for c in ("forbidden_prompt", "prompt") if c in rows[0]), None)
        if key is None:
            raise SystemExit(f"no prompt column in {resolved}; got {list(rows[0])}")
        return ([{"prompt": str(r[key]),
                  "meta": {k: str(v) for k, v in r.items() if k != key}}
                 for r in rows], resolved)

    if source == "advbench":
        rows, resolved = _try_sources([("hf:walledai/AdvBench", _hf("walledai/AdvBench"))])
        return ([{"prompt": str(r["prompt"]), "meta": {}} for r in rows], resolved)

    raise SystemExit(f"unknown harmful source {source!r}")


def _load_safe(source: str) -> tuple[list[dict], str]:
    if source == "xstest_safe":
        rows, resolved = _try_sources([
            # Paul/XSTest is the author's own repo: 450 rows, columns id/prompt/type/
            # label/focus/note, 250 of them label=="safe".
            ("hf:Paul/XSTest", _hf("Paul/XSTest")),
            ("hf:natolambert/xstest-v2-copy", _hf("natolambert/xstest-v2-copy")),
            ("hf:walledai/XSTest", _hf("walledai/XSTest")),
        ])
        if "label" not in rows[0]:
            raise SystemExit(
                f"{resolved} has no 'label' column ({list(rows[0])}), so safe rows "
                "cannot be separated from the 200 unsafe contrast prompts. Refusing to "
                "build the over-refusal arm out of a mixture."
            )
        safe = [r for r in rows if str(r.get("label", "")).lower() == "safe"]
        if len(safe) == len(rows):
            raise SystemExit(
                f"{resolved}: the label filter kept all {len(rows)} rows. XSTest is "
                "250 safe + 200 unsafe, so a filter that removes nothing means the "
                "label vocabulary changed and the over-refusal arm would be built out "
                "of a mixture that includes genuinely harmful prompts."
            )
        return ([{"prompt": str(r["prompt"]), "meta": {"type": str(r.get("type", ""))}}
                 for r in safe], resolved)

    if source == "orbench_hard":
        # The swap target from section 3, for when XSTest's baseline over-refusal is
        # too low for the arm to resolve anything.
        rows, resolved = _try_sources([
            ("hf:bench-llms/or-bench:or-bench-hard-1k",
             _hf("bench-llms/or-bench", config_name="or-bench-hard-1k")),
        ])
        return ([{"prompt": str(r["prompt"]),
                  "meta": {"category": str(r.get("category", ""))}}
                 for r in rows], resolved)

    raise SystemExit(f"unknown safe source {source!r}")


def prepare_battery(force: bool, harmful_source: str, safe_source: str) -> list[dict]:
    manifest_entries = []
    for arm, source, n, out_path, tag, loader in (
        ("harmful", harmful_source, config.BATTERY_N_HARMFUL, BATTERY_HARMFUL, "harm", _load_harmful),
        ("safe", safe_source, config.BATTERY_N_SAFE, BATTERY_SAFE, "safe", _load_safe),
    ):
        print(f"battery arm: {arm} ({source})")
        pool, resolved = loader(source)

        expected = EXPECTED_ROWS.get(source)
        if expected is not None and len(pool) != expected:
            print(f"  WARNING {source} yielded {len(pool)} rows, expected {expected}. "
                  "Check the source before trusting this arm.")
        if expected is not None and abs(len(pool) - expected) > 0.25 * expected:
            raise SystemExit(
                f"{source} via {resolved} yielded {len(pool)} rows against an expected "
                f"{expected}. That is too far off to be a version bump -- the source or "
                "its filtering has changed. Refusing to freeze a battery arm whose "
                "contents are not what the protocol says they are."
            )

        # Deduplicate on exact prompt text before sampling. Two identical prompts are
        # one item of information, and a paired test treats them as two.
        seen, unique = set(), []
        for row in pool:
            key = row["prompt"].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(row)
        print(f"  pool: {len(pool)} rows, {len(unique)} after exact dedup")

        if n > len(unique):
            raise SystemExit(
                f"asked for {n} prompts but {source} only has {len(unique)} unique ones. "
                f"Lower config.BATTERY_N_{arm.upper()} or pick another source."
            )
        rng = random.Random(config.BATTERY_SAMPLE_SEED)
        sampled = rng.sample(unique, k=n)
        rows = [
            {
                "item_id": item_id(r["prompt"], tag),
                "arm": arm,
                "source": source,
                "prompt": r["prompt"],
                "meta": r["meta"],
            }
            for r in sampled
        ]
        if len({r["item_id"] for r in rows}) != len(rows):
            raise SystemExit("item_id collision in the battery -- refuse to freeze")
        write_jsonl(out_path, rows, force)
        manifest_entries.append(
            {
                "file": out_path.name,
                "arm": arm,
                "source": source,
                # Which candidate actually served the data, not just which was asked for.
                "resolved_source": resolved,
                "expected_pool_rows": EXPECTED_ROWS.get(source),
                "n": len(rows),
                "seed": config.BATTERY_SAMPLE_SEED,
                "pool_size": len(pool),
                "unique_pool_size": len(unique),
                "sha256": sha256_file(out_path),
            }
        )
    return manifest_entries


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--wtq", action="store_true")
    parser.add_argument("--battery", action="store_true")
    parser.add_argument("--harmful-source", default=config.HARMFUL_SOURCE)
    parser.add_argument("--safe-source", default=config.SAFE_SOURCE)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing frozen file (record why in FINDINGS)")
    args = parser.parse_args()

    ensure_dirs()
    entries: list[dict] = []
    if args.all or args.wtq:
        entries += prepare_wtq(args.force)
    if args.all or args.battery:
        entries += prepare_battery(args.force, args.harmful_source, args.safe_source)
    if not entries:
        parser.error("nothing to do: pass --all, --wtq, or --battery")

    manifest = {"config_hash": config.config_hash(), "files": entries}
    existing = json.loads(FROZEN_MANIFEST.read_text()) if FROZEN_MANIFEST.exists() else {"files": []}
    by_name = {e["file"]: e for e in existing.get("files", [])}
    by_name.update({e["file"]: e for e in entries})
    manifest["files"] = sorted(by_name.values(), key=lambda e: e["file"])
    FROZEN_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nmanifest -> {FROZEN_MANIFEST}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    main()
