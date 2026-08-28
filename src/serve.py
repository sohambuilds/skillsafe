"""
vLLM server configuration. CLAUDE.md section 9.

This module PRINTS commands; it does not launch anything. Two reasons: the human runs
the jobs (section 2), and every flag that affects determinism has to be visible in the
transcript rather than buried in a subprocess call.

Section 5 requires max_num_seqs to be pinned across every run, because vLLM is not
bit-exact across batch sizes. It is set from config.MAX_NUM_SEQS here and written onto
every rollout record by agent.py, so a mismatch between what was served and what was
recorded is detectable after the fact.

Default layout for three cards: llama on 0, qwen on 1, judges on 2, so both agent
models roll out in parallel while a judge works on whatever is already finished.
Co-locating two models on one card is awkward at max_num_seqs=64 -- the KV cache, not
the weights, is the binding constraint. Override with --gpu.

    uv run python -m src.serve --all                 # print every command
    uv run python -m src.serve --print llama31_8b    # print one
    uv run python -m src.serve --script scripts/serve_all.sh
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import config
from src.config import ModelSpec
from src.paths import LOGS

ALL_MODELS: dict[str, ModelSpec] = {**config.AGENT_MODELS, **config.JUDGE_MODELS}

# Models that need a gated-repo licence accepted on huggingface.co before they download.
GATED = {
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4": "derived from meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-Guard-3-8B": "Meta licence",
    "allenai/wildguard": "AI2 terms of use",
}


def serve_command(spec: ModelSpec, gpu: int | None = None) -> str:
    gpu = spec.gpu if gpu is None else gpu
    parts = [
        f"CUDA_VISIBLE_DEVICES={gpu}",
        # `uv run` is load-bearing, not decoration. A bare `vllm serve` resolves against
        # PATH, which on a machine with a system anaconda picks that interpreter's vLLM
        # instead of the project venv's -- and a vLLM compiled against a different
        # libtorch than the torch beside it fails with an undefined-symbol ImportError
        # that names a C++ mangled symbol and nothing else. Every other command in this
        # project goes through uv; this one has to as well.
        "uv run vllm serve",
        spec.hf_id,
        f"--served-model-name {spec.key}",
        f"--port {spec.port}",
        f"--max-model-len {spec.max_model_len}",
        # section 5: pinned across every run, and recorded on every rollout
        f"--max-num-seqs {config.MAX_NUM_SEQS}",
        "--gpu-memory-utilization 0.90",
        # No --disable-log-requests. vLLM flipped per-request logging from opt-out to
        # opt-in around 0.11 and removed the disable flag, so passing it is a hard
        # argparse error on current versions. Omitting it gives the quiet default; add
        # --enable-log-requests if you ever want the opposite.
    ]
    if spec.quantization:
        parts.append(f"--quantization {spec.quantization}")
    return " \\\n    ".join(parts)


def quantization_provenance() -> dict:
    """Read each checkpoint's real quantization_config off the Hub. Never asserted here.

    CLAUDE.md section 3 says "quantize both identically" and these two AWQ checkpoints
    come from different quantizers (hugging-quants vs Qwen). The damage is bounded:
    quantizer is constant within a model, so every within-model k-sweep is clean. It only
    becomes a candidate explanation if the result splits by family -- and a split result
    is a null under section 8.5 anyway. So: record bits, group size, version, and
    whatever calibration metadata the checkpoint carries, proceed, and re-quantize with a
    common toolchain only if the families disagree.

    The values are read from the checkpoint rather than written down from memory, because
    a group size recalled wrong is worse than no record at all.
    """
    from huggingface_hub import hf_hub_download

    out: dict = {}
    for key, spec in ALL_MODELS.items():
        entry = {"hf_id": spec.hf_id, "served_quantization": spec.quantization}
        try:
            path = hf_hub_download(spec.hf_id, "config.json")
            cfg = json.loads(Path(path).read_text(encoding="utf-8"))
            entry["quantization_config"] = cfg.get("quantization_config")
            entry["torch_dtype"] = cfg.get("torch_dtype")
            if cfg.get("quantization_config") is None:
                entry["note"] = "no quantization_config -- served unquantized (bf16)"
        except Exception as error:  # noqa: BLE001 - reported, not raised
            entry["error"] = f"{type(error).__name__}: {error}"
            entry["note"] = ("could not read config.json; accept the repo licence and "
                             "`hf auth login`, then re-run")
        out[key] = entry

    agents = [out[k].get("quantization_config") for k in config.AGENT_MODELS]
    out["_identical_across_agent_models"] = (
        len({json.dumps(a, sort_keys=True) for a in agents}) == 1
    )
    out["_note"] = (
        "If _identical_across_agent_models is false, quantizer provenance is a candidate "
        "explanation ONLY for a result that splits by model family. Record it, proceed, "
        "and re-quantize with a common toolchain only if the families disagree."
    )
    return out


def doctor() -> dict:
    """Which vLLM and torch are actually being used, and do they match.

    Exists because the failure it diagnoses surfaces as
    `ImportError: vllm/_C.abi3.so: undefined symbol: _ZNR5torch7Library4_def...`,
    which names a mangled C++ symbol and nothing an operator can act on. The cause is
    almost always that vLLM's compiled extension was built against a different libtorch
    than the torch sitting beside it -- typically because a bare `vllm` resolved to a
    system anaconda rather than the project venv.
    """
    import sys

    report: dict = {
        "python": sys.executable,
        "prefix": sys.prefix,
        "in_project_venv": str(Path(sys.prefix).resolve())
                           == str((Path(__file__).resolve().parent.parent / ".venv")),
    }
    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "file": torch.__file__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "devices": [
                {"index": i,
                 "name": torch.cuda.get_device_name(i),
                 "total_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)}
                for i in range(torch.cuda.device_count())
            ] if torch.cuda.is_available() else [],
        }
    except Exception as error:  # noqa: BLE001
        report["torch"] = {"error": f"{type(error).__name__}: {error}"}

    try:
        import vllm

        report["vllm"] = {"version": vllm.__version__, "file": vllm.__file__}
    except ImportError as error:
        report["vllm"] = {
            "error": f"{type(error).__name__}: {error}",
            "diagnosis": (
                "undefined symbol -> vLLM's compiled extension does not match the "
                "installed torch. Do not patch the system environment; install into the "
                "project venv where vLLM pins its own torch:\n"
                "    uv sync --extra gpu\n"
                "and launch every server through `uv run vllm serve ...`."
                if "undefined symbol" in str(error) else
                "vLLM is not importable here. Run `uv sync --extra gpu`."
            ),
        }
    except Exception as error:  # noqa: BLE001
        report["vllm"] = {"error": f"{type(error).__name__}: {error}"}

    report["_ok"] = ("error" not in report.get("vllm", {})
                     and "error" not in report.get("torch", {})
                     and report["torch"].get("cuda_available") is True)
    return report


def access_check() -> dict:
    """Ask the Hub whether THIS token can actually reach each repo.

    The previous version of this printed a static reminder for every repo in GATED,
    unconditionally, without consulting the Hub or the token. It printed the same text
    whether access was granted or refused -- a check that could not fail, which is the
    failure mode this project keeps finding in its own instruments. It is harmless to the
    results and it still had to go.

    A gated repo without access raises GatedRepoError (403) or, when the repo is hidden
    from the caller, RepositoryNotFoundError (404). Both are treated as BLOCKED.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    api = HfApi()
    out: dict = {}
    try:
        out["_token"] = {"status": "OK", "user": api.whoami().get("name")}
    except Exception as error:  # noqa: BLE001
        out["_token"] = {"status": "NO_TOKEN", "error": f"{type(error).__name__}: {error}",
                         "fix": "hf auth login"}

    for key, spec in ALL_MODELS.items():
        entry: dict = {"repo": spec.hf_id, "gated_because": GATED.get(spec.hf_id)}
        try:
            info = api.model_info(spec.hf_id)
            entry["status"] = "OK"
            entry["gated"] = getattr(info, "gated", None)
            entry["revision"] = getattr(info, "sha", None)
        except GatedRepoError:
            entry["status"] = "BLOCKED"
            entry["fix"] = f"accept the licence at https://huggingface.co/{spec.hf_id}"
        except RepositoryNotFoundError:
            entry["status"] = "BLOCKED"
            entry["fix"] = (f"https://huggingface.co/{spec.hf_id} is not visible to this "
                            "token -- accept the licence, or check the repo id")
        except Exception as error:  # noqa: BLE001
            entry["status"] = "ERROR"
            entry["error"] = f"{type(error).__name__}: {error}"
        out[key] = entry

    blocked = [k for k, v in out.items()
               if k != "_token" and v.get("status") != "OK"]
    out["_blocked"] = blocked
    out["_all_reachable"] = not blocked and out["_token"]["status"] == "OK"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print", dest="key", choices=sorted(ALL_MODELS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--script", type=Path, help="write a launcher shell script")
    parser.add_argument("--provenance", action="store_true",
                        help="read every checkpoint's quantization_config off the Hub "
                             "and write logs/quantization_provenance.json")
    parser.add_argument("--check-access", action="store_true",
                        help="ask the Hub whether the current token can reach every repo")
    parser.add_argument("--doctor", action="store_true",
                        help="report which vllm/torch are resolved and whether they match")
    args = parser.parse_args()

    if args.doctor:
        report = doctor()
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["_ok"] else 1)

    if args.check_access:
        report = access_check()
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["_all_reachable"] else 1)

    if args.provenance:
        report = quantization_provenance()
        out = LOGS / "quantization_provenance.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"\n-> {out}")
        return

    if args.key:
        print(serve_command(ALL_MODELS[args.key], args.gpu))
        return

    if not (args.all or args.script):
        parser.error("pass --print KEY, --all, or --script PATH")

    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    lines.append("# Generated by src/serve.py. Each server runs in its own terminal or")
    lines.append("# under tmux; they are long-lived and this script does not wait.")
    lines.append("")
    for key, spec in ALL_MODELS.items():
        role = "agent" if key in config.AGENT_MODELS else "judge"
        quant = spec.quantization or "bf16"
        lines.append(f"# {key}  ({role}, {quant}, port {spec.port}, gpu {spec.gpu})")
        lines.append(serve_command(spec, args.gpu))
        lines.append("")

    text = "\n".join(lines)
    if args.script:
        args.script.parent.mkdir(parents=True, exist_ok=True)
        args.script.write_text(text, encoding="utf-8")
        print(f"wrote {args.script}")
    else:
        print(text)

    print("\n# Repo access (live check against the Hub with your current token):")
    report = access_check()
    print(f"  token: {report['_token']['status']}"
          + (f"  ({report['_token'].get('user')})" if report["_token"].get("user") else ""))
    for key, entry in report.items():
        if key.startswith("_"):
            continue
        line = f"  {entry['status']:<8} {entry['repo']}"
        if entry.get("fix"):
            line += f"\n           -> {entry['fix']}"
        print(line)
    if report["_all_reachable"]:
        print("  all reachable -- nothing to do")


if __name__ == "__main__":
    main()
