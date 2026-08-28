"""
vLLM server configuration. CLAUDE.md section 9.

This module PRINTS commands; it does not launch anything. Two reasons: the human runs
the jobs (section 2), and every flag that affects determinism has to be visible in the
transcript rather than buried in a subprocess call.

Section 5 requires max_num_seqs to be pinned across every run, because vLLM is not
bit-exact across batch sizes. It is set from config.MAX_NUM_SEQS here and written onto
every rollout record by agent.py, so a mismatch between what was served and what was
recorded is detectable after the fact.

Layout assumes two GPUs: agent models on GPU 0, judges on GPU 1, so an agent rollout and
its judging can run concurrently. Override with --gpu.

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
        "vllm serve",
        spec.hf_id,
        f"--served-model-name {spec.key}",
        f"--port {spec.port}",
        f"--max-model-len {spec.max_model_len}",
        # section 5: pinned across every run, and recorded on every rollout
        f"--max-num-seqs {config.MAX_NUM_SEQS}",
        "--gpu-memory-utilization 0.90",
        "--disable-log-requests",
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


def preflight_notes() -> list[str]:
    notes = []
    for spec in ALL_MODELS.values():
        if spec.hf_id in GATED:
            notes.append(f"  {spec.hf_id}  --  gated: {GATED[spec.hf_id]}")
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print", dest="key", choices=sorted(ALL_MODELS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--script", type=Path, help="write a launcher shell script")
    parser.add_argument("--provenance", action="store_true",
                        help="read every checkpoint's quantization_config off the Hub "
                             "and write logs/quantization_provenance.json")
    args = parser.parse_args()

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

    notes = preflight_notes()
    if notes:
        print("\n# Gated repos -- accept the licence on huggingface.co and run "
              "`hf auth login` before the first download:")
        print("\n".join(notes))


if __name__ == "__main__":
    main()
