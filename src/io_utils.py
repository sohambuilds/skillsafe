"""JSONL plumbing and run identity. CLAUDE.md section 9 logging requirements.

    "Every rollout writes a JSONL record containing: model, quantization, condition, k,
     seed, temperature, batch size, full prompt, full completion, judge verdict,
     timestamp. No exceptions. If a result cannot be traced back to its exact prompt, it
     does not exist."

base_record() below is the single place those fields are assembled, so "no exceptions"
is structural rather than a habit. Runs are resumable by item_id: a killed job restarted
does not re-roll work it already has, and never appends a duplicate.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.paths import RUNS


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def run_path(kind: str, model_key: str, condition: str, **extra) -> Path:
    """logs/runs/<kind>/<model>__<condition>[__k=..][__seed=..].jsonl"""
    parts = [model_key, condition]
    for key in ("k", "seed", "arm", "rep", "temp"):
        if extra.get(key) is not None:
            parts.append(f"{key}={extra[key]}")
    return RUNS / kind / ("__".join(parts) + ".jsonl")


def done_ids(path: Path) -> set[str]:
    """item_ids already present, so a restarted run resumes instead of duplicating."""
    return {r["item_id"] for r in read_jsonl(path) if "item_id" in r}


def base_record(*, model_spec, condition: str, skill_text: str | None, prompt_messages,
                completion: str, item_id: str, k=None, seed=None, arm=None,
                temperature: float, max_tokens: int, rep: int | None = None) -> dict:
    """The section 9 record. Everything needed to reproduce the call is on it."""
    return {
        "item_id": item_id,
        "timestamp": now_iso(),
        "config_hash": config.config_hash(),
        "model": model_spec.hf_id,
        "model_key": model_spec.key,
        "quantization": model_spec.quantization or "bf16",
        "condition": condition,
        "k": k,
        "seed": seed,
        "arm": arm,
        "rep": rep,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_num_seqs": config.MAX_NUM_SEQS,
        "client_concurrency": config.CLIENT_CONCURRENCY,
        "skill_sha256": sha256_text(skill_text) if skill_text is not None else None,
        "skill_tokens_approx": (len(skill_text) // 4) if skill_text else 0,
        "prompt_messages": prompt_messages,
        "completion": completion,
        # filled in by judge.py; present as None so the schema never varies
        "judge": None,
    }
