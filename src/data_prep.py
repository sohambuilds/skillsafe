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
    if path.exists() and not force:
        raise SystemExit(
            f"{path} already exists. CLAUDE.md section 3 says never resample. "
            f"Pass --force only if you are deliberately re-freezing, and record why."
        )
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
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
        return load_dataset(config.WTQ_HF_ID)
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
XSTEST_CSV = (
    "https://raw.githubusercontent.com/paul-rottger/exaggerated-safety/main/"
    "xstest_v2_prompts.csv"
)


def _load_harmful(source: str) -> list[dict]:
    from datasets import load_dataset

    if source == "strongreject":
        try:
            data = load_dataset("walledai/StrongREJECT", split="train")
            return [
                {"prompt": str(r["prompt"]), "meta": {k: str(v) for k, v in r.items() if k != "prompt"}}
                for r in data
            ]
        except Exception as error:  # noqa: BLE001
            print(f"  HF load failed ({type(error).__name__}); falling back to the CSV")
            import pandas as pd

            frame = pd.read_csv(STRONGREJECT_CSV)
            column = "forbidden_prompt" if "forbidden_prompt" in frame else frame.columns[-1]
            return [
                {"prompt": str(row[column]),
                 "meta": {k: str(row[k]) for k in frame.columns if k != column}}
                for _, row in frame.iterrows()
            ]

    if source == "advbench":
        data = load_dataset("walledai/AdvBench", split="train")
        return [{"prompt": str(r["prompt"]), "meta": {}} for r in data]

    raise SystemExit(f"unknown harmful source {source!r}")


def _load_safe(source: str) -> list[dict]:
    from datasets import load_dataset

    if source == "xstest_safe":
        try:
            data = load_dataset("walledai/XSTest", split="train")
            rows = [r for r in data if str(r.get("label", "")).lower() == "safe"]
            if not rows:
                raise ValueError("no rows with label=='safe'")
            return [
                {"prompt": str(r["prompt"]), "meta": {"type": str(r.get("type", ""))}}
                for r in rows
            ]
        except Exception as error:  # noqa: BLE001
            print(f"  HF load failed ({type(error).__name__}); falling back to the CSV")
            import pandas as pd

            frame = pd.read_csv(XSTEST_CSV)
            frame = frame[~frame["type"].astype(str).str.startswith("contrast")]
            return [
                {"prompt": str(row["prompt"]), "meta": {"type": str(row["type"])}}
                for _, row in frame.iterrows()
            ]

    if source == "orbench_hard":
        # The swap target from section 3, for when XSTest's baseline over-refusal is
        # too low for the arm to resolve anything.
        data = load_dataset("bench-llms/or-bench", "or-bench-hard-1k", split="train")
        return [
            {"prompt": str(r["prompt"]), "meta": {"category": str(r.get("category", ""))}}
            for r in data
        ]

    raise SystemExit(f"unknown safe source {source!r}")


def prepare_battery(force: bool, harmful_source: str, safe_source: str) -> list[dict]:
    manifest_entries = []
    for arm, source, n, out_path, tag, loader in (
        ("harmful", harmful_source, config.BATTERY_N_HARMFUL, BATTERY_HARMFUL, "harm", _load_harmful),
        ("safe", safe_source, config.BATTERY_N_SAFE, BATTERY_SAFE, "safe", _load_safe),
    ):
        print(f"battery arm: {arm} ({source})")
        pool = loader(source)
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
