"""
CLAUDE.md section 2: "Verify that composing pieces hasn't destroyed the thing under
test", and "Never report a clean reading from an unvalidated detector."

The rubric judge depends on the server constraining its decode to three labels. That
dependency failed silently once: vLLM removed the flat guided_* request fields in
v0.12.0, and because vLLM accepts unknown request fields and still returns 200, the
judge kept working -- as a free-text generator. Nothing in the status code, the client,
or the server log distinguished that from a working judge.

These tests pin the two properties that make the failure loud:

  1. the request carries the CURRENT structured-output field, and
  2. a server that ignores it is rejected before any verdict is recorded.

The stub server is deliberately dumb. It is not a vLLM emulator -- it only answers the
question "did this code notice that its constraint was dropped?", which is the only
question a unit test can settle without a GPU.
"""

from __future__ import annotations

import asyncio
import types

import openai
import pytest

import src.judge as judge_mod
from src.judge import (
    _PROBE_CHOICES,
    ConstraintNotApplied,
    JudgeParseError,
    RUBRIC_CHOICES,
    RubricJudge,
)

# What an ignored-field server actually returned on 2026-08-29, verbatim from the
# traceback: the prompt's trailing "CLASS:" echoed back, then free prose, cut off at
# max_tokens. Kept exact so this test fails for the real reason, not a stylised one.
UNCONSTRAINED_OUTPUT = "CLASS: 2_full_refusal\n\nThe response directly states"


def _stub(mode: str, seen: list, calls: list):
    """mode='honours' obeys extra_body; mode='ignores' free-runs like vLLM >=0.12."""

    class Completions:
        async def create(self, **kw):
            calls.append(1)
            seen.append(kw.get("extra_body"))
            await asyncio.sleep(0)
            choices = ((kw.get("extra_body") or {}).get("structured_outputs") or {}).get("choice")
            content = choices[0] if (mode == "honours" and choices) else UNCONSTRAINED_OUTPUT
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
            )

    class Client:
        def __init__(self, **kw):
            self.base_url = "http://localhost:8011/v1/"
            self.chat = types.SimpleNamespace(completions=Completions())

    return Client


@pytest.fixture
def stub(monkeypatch):
    async def _ok(value):
        return value

    def build(mode):
        seen, calls = [], []
        monkeypatch.setattr(openai, "AsyncOpenAI", _stub(mode, seen, calls))
        # _server_version is a diagnostic-only lookup against a server that does not
        # exist here; short-circuit it so the test does not wait on a real socket.
        monkeypatch.setattr(RubricJudge, "_server_version", lambda self: _ok("stub"))
        return RubricJudge(), seen, calls

    return build


def test_request_uses_current_structured_output_field(stub):
    """The field name is the whole bug. Pin it."""
    judge, seen, _ = stub("honours")
    asyncio.run(judge.assert_constrained())
    assert seen[0] == {"structured_outputs": {"choice": list(_PROBE_CHOICES)}}
    assert "guided_choice" not in seen[0]


def test_ignored_constraint_is_rejected(stub):
    """A server that free-runs must not produce verdicts."""
    judge, _, _ = stub("ignores")
    with pytest.raises(ConstraintNotApplied) as excinfo:
        asyncio.run(judge.classify("q", "a"))
    # The message has to name the field and the version, because the person reading it
    # is looking at a 200 OK in the server log and has no other clue.
    assert "structured_outputs" in str(excinfo.value)
    assert "v0.12.0" in str(excinfo.value)


def test_constraint_failure_is_not_swallowable(stub):
    """judge_file() catches JudgeParseError for secondary judges and records it on the
    record instead of stopping. A dropped constraint must never take that path: it is
    not one bad parse, it is every verdict in the pass."""
    judge, _, _ = stub("ignores")
    with pytest.raises(ConstraintNotApplied) as excinfo:
        asyncio.run(judge.assert_constrained())
    assert not isinstance(excinfo.value, JudgeParseError)


def test_probe_runs_once_under_concurrency(stub):
    """judge_file() fans out over ~120 records. One probe, not 120."""
    judge, _, calls = stub("honours")

    async def go():
        return await asyncio.gather(*(judge.classify("q", "a") for _ in range(20)))

    out = asyncio.run(go())
    assert len(calls) == 20 * 2 + 1          # rubric + attempt per record, plus one probe
    assert out[0]["label"] == judge_mod.RUBRIC_TO_LABEL[RUBRIC_CHOICES[0]]


def test_probe_choices_are_not_the_real_labels():
    """Probing with the real label set would pass whenever the model guessed the format
    on its own -- which Qwen frequently does. The sentinels have to be unguessable."""
    assert not set(_PROBE_CHOICES) & set(RUBRIC_CHOICES)
