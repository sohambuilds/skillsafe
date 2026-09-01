"""
Is the right server actually on that port?

Shared by src/judge.py and src/agent.py. Both talk to vLLM over the OpenAI client, and
both had the same problem: the client reports a dead port as a bare
"APIConnectionError: Connection error." naming neither host nor port. With five models
across five ports -- two agents on their own cards and three judges rotating on GPU 2 --
that is not enough to act on, and agent.py made it worse by retrying four times with
exponential backoff before saying so, behind eighty lines of httpx traceback.

Two distinct faults, and the second is the dangerous one:

  dead port    nothing listening. Loud, obvious once the port is named.
  wrong model  a LIVE server answering for something else. If two servers were ever
               started with the same --served-model-name, or a serve command's name
               disagreed with config, requests would be answered confidently by the
               wrong model and nothing downstream would notice.

So the check is "which model is served here", not "did something reply".
"""

from __future__ import annotations


class ServerUnavailable(RuntimeError):
    """No usable server on this port.

    Never swallowed anywhere. It means the instrument is absent, not that one request
    failed -- retrying it is pointless and recording a result past it is worse.
    """


def serve_hint(spec) -> str:
    """The exact command that starts this model, generated from src/serve.py so it
    cannot drift from the real config. Imported late: src.serve pulls huggingface_hub
    for its provenance helper, and this is only ever needed on an error path."""
    try:
        from src.serve import serve_command

        return serve_command(spec)
    except Exception:  # a broken hint must never replace the real error
        return f"(see: uv run python -m src.serve --print {spec.key})"


async def check_server(client, spec, base_url: str) -> dict:
    """Confirm a server is up at base_url AND is serving spec.key.

    Returns the listing on success; raises ServerUnavailable otherwise. Callers cache
    the result -- this is one request, but it should not run once per record.
    """
    try:
        served = [m.id for m in (await client.models.list()).data]
    except Exception as error:
        raise ServerUnavailable(
            f"no server answering at {base_url} for {spec.key!r} ({spec.hf_id}).\n"
            f"  {type(error).__name__}: {error}\n\n"
            f"Start it with:\n\n{serve_hint(spec)}\n\n"
            "and wait for 'Application startup complete' -- a large checkpoint takes "
            "minutes to load, and the port refuses connections until it does."
        ) from error

    if spec.key not in served:
        raise ServerUnavailable(
            f"{base_url} is serving {served}, which does not include {spec.key!r}. "
            f"Something is on this port, but it is not {spec.key} -- most likely "
            f"another model from the rotation, or a serve command whose "
            f"--served-model-name does not match config.\n\n"
            f"Expected command:\n\n{serve_hint(spec)}"
        )

    return {"base_url": base_url, "served": served}


async def server_version(client, base_url: str) -> str:
    """Best-effort vLLM version, for diagnostics and the run record. Never raises."""
    try:
        import httpx  # transitive via openai; diagnostics only

        root = str(base_url).rstrip("/").removesuffix("/v1")
        async with httpx.AsyncClient(timeout=10) as http:
            return (await http.get(f"{root}/version")).json().get("version", "?")
    except Exception:  # a failed version lookup must never mask the real error
        return "unknown"
