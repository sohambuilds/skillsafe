"""
Server preflight, shared by src/agent.py and src/judge.py.

Both talk to vLLM over the OpenAI client and both used to report a dead port as a bare
"APIConnectionError: Connection error." naming neither host nor port. agent.py made it
worse: four retries with exponential backoff first, then eighty lines of httpx traceback
with the useful sentence at the very bottom.

Two properties are pinned here:

  1. a failure names the model, the port, and the command that starts it
  2. agent.py checks BEFORE rolling out, so a dead port costs one request, not four
     retries per record

Property 2 is the one that actually cost time, and it is the one a unit test can prove
without a GPU: if the server is down, zero completion requests should be attempted.
"""

from __future__ import annotations

import asyncio
import types

import pytest

from src import config
from src.preflight import ServerUnavailable, check_server, serve_hint

SPEC = config.AGENT_MODELS["llama31_8b"]
URL = f"http://localhost:{SPEC.port}/v1"


def _client(served=("llama31_8b",), up=True, calls=None):
    class Models:
        async def list(self):
            if not up:
                raise ConnectionError("All connection attempts failed")
            return types.SimpleNamespace(
                data=[types.SimpleNamespace(id=name) for name in served])

    class Completions:
        async def create(self, **kw):
            if calls is not None:
                calls.append(kw)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="x"))])

    class Client:
        def __init__(self, **kw):
            self.base_url = URL
            self.models = Models()
            self.chat = types.SimpleNamespace(completions=Completions())

        async def close(self):
            pass

    return Client


def test_dead_port_names_model_port_and_start_command():
    with pytest.raises(ServerUnavailable) as excinfo:
        asyncio.run(check_server(_client(up=False)(), SPEC, URL))
    message = str(excinfo.value)
    assert SPEC.key in message
    assert str(SPEC.port) in message
    assert "vllm serve" in message          # the fix, not just the symptom
    assert "Application startup complete" in message


def test_wrong_model_on_the_port_is_caught():
    """The quiet fault. Two servers started with the same --served-model-name would
    otherwise answer confidently for the wrong model, and nothing downstream checks."""
    with pytest.raises(ServerUnavailable) as excinfo:
        asyncio.run(check_server(_client(served=("qwen25_7b",))(), SPEC, URL))
    assert "qwen25_7b" in str(excinfo.value)
    assert SPEC.key in str(excinfo.value)


def test_healthy_server_returns_the_listing():
    out = asyncio.run(check_server(_client()(), SPEC, URL))
    assert out["served"] == ["llama31_8b"]


def test_serve_hint_is_generated_from_the_real_config():
    """Hard-coding the command in the error string would let it drift from serve.py."""
    hint = serve_hint(SPEC)
    assert SPEC.hf_id in hint
    assert f"--port {SPEC.port}" in hint
    assert f"--max-num-seqs {config.MAX_NUM_SEQS}" in hint


def test_agent_does_not_roll_out_against_a_dead_port(tmp_path, monkeypatch):
    """The property that cost the time: zero completion requests when the port is dead,
    rather than four retries with backoff for every one of 250 records."""
    import openai

    from src import agent

    calls: list = []
    monkeypatch.setattr(openai, "AsyncOpenAI", _client(up=False, calls=calls))

    items = [{"item_id": f"i{i}", "prompt": "hello", "arm": "safe"} for i in range(5)]
    with pytest.raises(ServerUnavailable):
        asyncio.run(agent.run_rollouts(
            model_key="llama31_8b", items=items, skill_text=None,
            condition="opt_k0", kind="battery", out_path=tmp_path / "out.jsonl",
            arm="safe",
        ))
    assert calls == [], f"attempted {len(calls)} rollouts against a dead server"


def test_agent_rolls_out_when_the_server_is_healthy(tmp_path, monkeypatch):
    """The preflight must not become a second failure mode of its own."""
    import openai

    from src import agent

    calls: list = []
    monkeypatch.setattr(openai, "AsyncOpenAI", _client(calls=calls))

    items = [{"item_id": f"i{i}", "prompt": "hello", "arm": "safe"} for i in range(5)]
    records = asyncio.run(agent.run_rollouts(
        model_key="llama31_8b", items=items, skill_text=None,
        condition="opt_k0", kind="battery", out_path=tmp_path / "out.jsonl",
        arm="safe",
    ))
    assert len(calls) == 5
    assert len(records) == 5
