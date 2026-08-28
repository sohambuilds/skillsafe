"""Repo-relative paths. Nothing in this project uses an absolute path."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SRC = ROOT / "src"
SKILLS = ROOT / "skills"
DATA = ROOT / "data"
LOGS = ROOT / "logs"
RUNS = LOGS / "runs"
RESULTS = ROOT / "results"
SCRIPTS = ROOT / "scripts"

# Frozen inputs (written once by data_prep.py, never resampled — CLAUDE.md §3)
WTQ_TRAIN = DATA / "wtq_train.jsonl"
WTQ_VAL = DATA / "wtq_val.jsonl"
WTQ_HELDOUT = DATA / "wtq_heldout.jsonl"
BATTERY_HARMFUL = DATA / "battery_harmful.jsonl"
BATTERY_SAFE = DATA / "battery_xstest.jsonl"
FROZEN_MANIFEST = DATA / "frozen_manifest.json"
JUDGE_VALIDATION = DATA / "judge_validation.jsonl"

# Gate artefacts
BASE_RATES = LOGS / "base_rates.json"
GATE_A = LOGS / "gate_a.json"
GATE_B_METAPROMPT = LOGS / "gate_b_metaprompt.txt"
COST_MEASUREMENT = LOGS / "cost_measurement.json"

# Results
NOISE_FLOOR = RESULTS / "noise_floor.json"
FINDINGS = RESULTS / "FINDINGS.md"


def ensure_dirs() -> None:
    for d in (SKILLS, DATA, LOGS, RUNS, RESULTS, SCRIPTS):
        d.mkdir(parents=True, exist_ok=True)
