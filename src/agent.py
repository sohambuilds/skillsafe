"""
Skill injection and single-turn rollout. CLAUDE.md section 9.

Injection site: the skill file goes into the system message VERBATIM, with no wrapper
text of any kind. Adding a wrapper would put words in the context that appear in no
condition's file; the verbatim form is the most direct reading of "a plain text file
injected into the model's context".

The `none` condition carries config.NONE_SYSTEM_MESSAGE, one neutral line, rather than
no system message at all, so it differs from `inert` in skill content rather than in
whether a system turn exists. Section 3 makes `inert` the primary comparison regardless.

The same injection applies on battery prompts. Section 3: "This is the strong form: if
it does not erode when always present, it will not erode when conditionally loaded."

Determinism (section 5): items are processed in a fixed order sorted by item_id, with a
pinned client concurrency, against a server pinned to config.MAX_NUM_SEQS. That does not
make vLLM bit-exact -- nothing does -- which is why the noise floor is measured rather
than assumed. Both numbers land on every record so a determinism drift is detectable.

    uv run python -m src.agent --model llama31_8b --skill skills/seed.md \
        --dataset data/wtq_heldout.jsonl --kind task --condition opt --k 0 --seed 0
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from src import config, grading
from src.config import ModelSpec
from src.io_utils import append_jsonl, base_record, done_ids, read_jsonl, run_path


def render_table(table: dict) -> tuple[str, dict]:
    """Markdown pipe table, truncated to keep the prompt inside the context window.

    A table that does not fit is a guaranteed failure that has nothing to do with the
    skill under test, so truncation is deliberate and its rate is logged as a covariate
    rather than left to chance.
    """
    header = [str(c).replace("|", "\\|") for c in table["header"]]
    rows = [[str(c).replace("|", "\\|") for c in row] for row in table["rows"]]

    total_rows = len(rows)
    truncated_rows = False
    if total_rows > config.MAX_TABLE_ROWS:
        rows = rows[: config.MAX_TABLE_ROWS]
        truncated_rows = True

    def build(body: list[list[str]]) -> str:
        lines = ["| " + " | ".join(header) + " |",
                 "| " + " | ".join("---" for _ in header) + " |"]
        lines += ["| " + " | ".join(r) + " |" for r in body]
        return "\n".join(lines)

    text = build(rows)
    truncated_chars = False
    while len(text) > config.MAX_TABLE_CHARS and len(rows) > 1:
        rows = rows[: max(1, len(rows) // 2)]
        truncated_chars = True
        text = build(rows)

    if truncated_rows or truncated_chars:
        text += f"\n\n[table truncated: showing {len(rows)} of {total_rows} rows]"

    return text, {
        "table_rows_total": total_rows,
        "table_rows_shown": len(rows),
        "table_truncated": truncated_rows or truncated_chars,
    }


def task_user_message(item: dict) -> tuple[str, dict]:
    table_text, info = render_table(item["table"])
    name = item["table"].get("name") or ""
    title = f"Table: {name}\n\n" if name else ""
    return f"{title}{table_text}\n\nQuestion: {item['question']}", info


def build_messages(skill_text: str | None, user_content: str) -> list[dict]:
    messages: list[dict] = []
    if skill_text is not None:
        messages.append({"role": "system", "content": skill_text})
    messages.append({"role": "user", "content": user_content})
    return messages


def load_skill(path_or_none: str | None) -> tuple[str | None, str]:
    """Returns (skill_text, label).

    'none' carries config.NONE_SYSTEM_MESSAGE -- a minimal neutral system turn -- rather
    than no system message at all, so it differs from `inert` in skill CONTENT rather
    than in whether a system turn exists. `none` is only the secondary baseline (section
    3 makes `inert` the primary comparison), so this is cheap insurance.
    """
    if path_or_none in (None, "none", "None", ""):
        return config.NONE_SYSTEM_MESSAGE, "none"
    path = Path(path_or_none)
    return path.read_text(encoding="utf-8"), path.stem


async def _one(client, spec: ModelSpec, messages: list[dict], temperature: float,
               max_tokens: int, semaphore, attempts: int = 4) -> str:
    delay = 2.0
    async with semaphore:
        for attempt in range(attempts):
            try:
                response = await client.chat.completions.create(
                    model=spec.key,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=0,
                )
                return response.choices[0].message.content or ""
            except Exception as error:  # noqa: BLE001 - transport errors are retryable
                if attempt == attempts - 1:
                    raise RuntimeError(
                        f"rollout failed after {attempts} attempts against "
                        f"{spec.key}:{spec.port} -- {type(error).__name__}: {error}"
                    ) from error
                await asyncio.sleep(delay)
                delay *= 2
    return ""


async def run_rollouts(
    *,
    model_key: str,
    items: list[dict],
    skill_text: str | None,
    condition: str,
    kind: str,
    out_path: Path,
    k=None,
    seed=None,
    arm=None,
    rep: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    base_url: str | None = None,
) -> list[dict]:
    from openai import AsyncOpenAI

    spec = config.AGENT_MODELS[model_key]
    temperature = config.TEMPERATURE if temperature is None else temperature
    if max_tokens is None:
        max_tokens = config.MAX_TOKENS_TASK if kind == "task" else config.MAX_TOKENS_BATTERY

    already = done_ids(out_path) if rep is None else set()
    pending = sorted((i for i in items if i["item_id"] not in already),
                     key=lambda i: i["item_id"])
    if already:
        print(f"  resuming: {len(already)} done, {len(pending)} to go")
    if not pending:
        return read_jsonl(out_path)

    client = AsyncOpenAI(
        base_url=base_url or f"http://localhost:{spec.port}/v1",
        api_key="EMPTY",
        timeout=600,
    )
    semaphore = asyncio.Semaphore(config.CLIENT_CONCURRENCY)

    built = []
    for item in pending:
        if kind == "task":
            user_content, extra = task_user_message(item)
        else:
            user_content, extra = item["prompt"], {}
        built.append((item, build_messages(skill_text, user_content), extra))

    completions = await asyncio.gather(
        *(_one(client, spec, messages, temperature, max_tokens, semaphore)
          for _, messages, _ in built)
    )

    records = []
    for (item, messages, extra), completion in zip(built, completions):
        record = base_record(
            model_spec=spec,
            condition=condition,
            skill_text=skill_text,
            prompt_messages=messages,
            completion=completion,
            item_id=item["item_id"],
            k=k,
            seed=seed,
            arm=arm or item.get("arm"),
            temperature=temperature,
            max_tokens=max_tokens,
            rep=rep,
        )
        record.update(extra)
        record["kind"] = kind
        if kind == "task":
            record["question"] = item["question"]
            record.update(grading.grade(completion, item["answers"]))
        else:
            record["prompt_text"] = item["prompt"]
            record["prompt_meta"] = item.get("meta", {})
        records.append(record)

    append_jsonl(out_path, records)
    return read_jsonl(out_path)


def summarise_task(records: list[dict]) -> dict:
    graded = [r for r in records if r.get("kind") == "task"]
    if not graded:
        return {"n": 0}
    correct = sum(1 for r in graded if r.get("correct"))
    formatted = sum(1 for r in graded if r.get("format_ok"))
    correct_given_format = [r for r in graded if r.get("format_ok")]
    return {
        "n": len(graded),
        "accuracy": round(100 * correct / len(graded), 2),
        # Decomposition: an optimizer can lift accuracy purely by teaching the model to
        # emit a parseable Answer line. Still optimising, but a different story from
        # better table reading, and section 6 should be able to tell them apart.
        "format_compliance": round(100 * formatted / len(graded), 2),
        "accuracy_given_format": (
            round(100 * sum(1 for r in correct_given_format if r["correct"])
                  / len(correct_given_format), 2)
            if correct_given_format else None
        ),
        "table_truncation_rate": round(
            100 * sum(1 for r in graded if r.get("table_truncated")) / len(graded), 2
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(config.AGENT_MODELS))
    parser.add_argument("--skill", default="none", help="path to a skill file, or 'none'")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--kind", choices=["task", "battery"], required=True)
    parser.add_argument("--condition", default=None,
                        help="defaults to the skill file's stem")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--arm", default=None)
    parser.add_argument("--rep", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    skill_text, label = load_skill(args.skill)
    condition = args.condition or label
    items = read_jsonl(args.dataset)
    out_path = args.out or run_path(
        args.kind, args.model, condition, k=args.k, seed=args.seed,
        arm=args.arm, rep=args.rep,
    )

    print(f"{args.model}  condition={condition}  n={len(items)}  -> {out_path}")
    records = asyncio.run(run_rollouts(
        model_key=args.model, items=items, skill_text=skill_text, condition=condition,
        kind=args.kind, out_path=out_path, k=args.k, seed=args.seed, arm=args.arm,
        rep=args.rep, temperature=args.temperature,
    ))
    if args.kind == "task":
        print(summarise_task(records))
    else:
        print(f"{len(records)} completions written; judge them with src.judge")


if __name__ == "__main__":
    main()
