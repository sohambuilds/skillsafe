# RUNBOOK

Operational state and the exact command sequence. `CLAUDE.md` is the standing brief —
what the experiment is and how to reason about it. This file is what to type, in what
order, and what is already done.

Last updated 2026-08-29, after the base-rate gate passed.

---

## 0. State board

| § | Item | State |
|---|---|---|
| 7.2 | `src/grading.py` unit-tested (28 pairs) | done |
| 7.3 | Battery frozen, n=250/arm, `config_hash 09ba20105037` | done |
| 7.1 | Rubric judge smoke — 6/6 decisive, 3/3 safe arm | done |
| 7.1 | WildGuard smoke | **not run** (never served) |
| 7.1 | Llama-Guard smoke | **not run** (never served) |
| 7.4 | Base rates, both arms, both models | **PASS** — see `results/day1_base_rates.md` |
| 7.6 | Gate B — meta-prompt, zero safety language | **PASS** — `claude-sonnet-4.5`, T=1.0, scanned clean |
| 7.7 | Noise floor | **PASS** — numbers not yet read out of `results/noise_floor.json` |
| — | Scope-deflection decomposition | **done** — deflection route carries almost none of the shift |
| — | Scope-deflection hand read (50 rows) | **sampled, unlabelled** |
| 7.5 | Gate A — planted permissive/restrictive | **BLOCKED** — no judged `opt_k0` baseline; see §3.3 |
| 7.7 | Judge validation, 60 hand labels per arm | **BLOCKED** — sheet not filled |
| 7.8 | One optimizer iteration priced with a real call | **not run** — PAID |

Day 2 is blocked until every row above is resolved. §7: "Do not proceed to Day 2 with
any gate unresolved." BLOCKED is not a pass.

---

## 1. Hardware and environment

Three A6000-class cards, 47 GB each, on `murari-Precision-7960-Tower`. Repo lives at
`/mnt/vaultb/Soham/skillsafe`.

Default layout — both agent models roll out in parallel while one judge works on
whatever is already finished:

| GPU | Model | Port |
|---|---|---|
| 0 | `llama31_8b` — Meta-Llama-3.1-8B-Instruct-AWQ-INT4 | 8001 |
| 1 | `qwen25_7b` — Qwen2.5-7B-Instruct-AWQ | 8002 |
| 2 | judges, **one at a time** — `wildguard` / `rubric` / `llamaguard` | 8010 / 8011 / 8012 |

Do not co-locate two models on one card. At `max_num_seqs=64` the binding constraint is
KV cache, not weights: an 8B model at ~1,600-token prompts needs roughly 13 GB of KV for
a full batch. Two models on one 47 GB card starve each other's cache and silently change
throughput, and `max_num_seqs` is pinned across every run for the determinism reasons in
§5.

### Installing

```bash
uv sync --extra gpu
```

**`--extra gpu` is not optional.** `gpu` is an optional extra, and `uv sync` is exact —
a bare `uv sync` removes everything not in the default resolution, which uninstalls vLLM
and torch out of the venv. This has already cost one debugging session.

vLLM is pinned **exactly** at `0.28.0`, not floored. The inference engine is part of the
apparatus: §5 requires decoding, batch size and `max_num_seqs` fixed across every run,
and a `uv sync` that moves vLLM between the Day 1 gates and the Day 2 sweep invalidates
the gates exactly as changing `max_num_seqs` would. To move it: change the number,
`uv sync --extra gpu`, **re-run every gate**, and say so in FINDINGS.md.

---

## 2. Serving

Generate the launcher, or copy a command out of it:

```bash
uv run python -m src.serve --all                  # print every command
uv run python -m src.serve --script scripts/serve_all.sh
```

Agent models (leave both up for all of Day 1):

```bash
CUDA_VISIBLE_DEVICES=0 uv run vllm serve hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
    --served-model-name llama31_8b --port 8001 --max-model-len 8192 \
    --max-num-seqs 64 --gpu-memory-utilization 0.90 --quantization awq_marlin

CUDA_VISIBLE_DEVICES=1 uv run vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
    --served-model-name qwen25_7b --port 8002 --max-model-len 8192 \
    --max-num-seqs 64 --gpu-memory-utilization 0.90 --quantization awq_marlin
```

Judges, one at a time on GPU 2:

```bash
CUDA_VISIBLE_DEVICES=2 uv run vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ \
    --served-model-name rubric --port 8011 --max-model-len 8192 \
    --max-num-seqs 64 --gpu-memory-utilization 0.90 --quantization awq_marlin

CUDA_VISIBLE_DEVICES=2 uv run vllm serve allenai/wildguard \
    --served-model-name wildguard --port 8010 --max-model-len 8192 \
    --max-num-seqs 64 --gpu-memory-utilization 0.90

CUDA_VISIBLE_DEVICES=2 uv run vllm serve meta-llama/Llama-Guard-3-8B \
    --served-model-name llamaguard --port 8012 --max-model-len 8192 \
    --max-num-seqs 64 --gpu-memory-utilization 0.90
```

`uv run` in front of `vllm serve` is load-bearing. A bare `vllm serve` resolves against
PATH and picks the system anaconda's vLLM instead of the project venv's.

### The rotation protocol

Roll out with `--no-judge`, then judge in passes, one judge at a time:

```bash
uv run python -m src.battery ... --no-judge          # GPUs 0 and 1 only
uv run python -m src.judge --judge-all --judges rubric
# swap GPU 2 to wildguard, then:
uv run python -m src.judge --smoke     --judges wildguard
uv run python -m src.judge --judge-all --judges wildguard
```

`--judge-all` skips records that already carry that judge's verdict, so it is safe and
idempotent to re-run per judge. Inline judging (the default, without `--no-judge`) calls
rubric **and** WildGuard, and the preflight will stop the run if either is not served.

Smoke-test each judge the first time it is served. It has never been done for WildGuard
or Llama-Guard.

---

## 3. Next steps, in order

### 3.1 Free, no GPU, do first

```bash
uv run python -m src.optimizer --gate-b        # §7.6, hard Day 2 blocker
uv run python -m src.data_prep --all           # expect "unchanged" x5
```

### 3.2 The scope-deflection check — highest value, ~30 minutes

This is not in CLAUDE.md §7. It was raised by the base-rate result and it decides
whether the over-refusal arm measures what §3 assumes it measures.

The `inert_k0` skill raises Qwen's benign-arm refusal rate from 6.8% to 75.2%. The
leading explanation is scope deflection — a table-QA skill in the system message makes
unrelated benign requests read as out-of-scope, and the model declines. Under the §3
schema that is correctly labelled REFUSAL, and it is indistinguishable from a safety
over-refusal in the label.

`src/judge.py` already records a separate `attempt` head on every record for this exact
reason, and `battery.behaviour_label()` already uses it: a rubric `1_full_compliance`
with `attempt=False` is reclassified to REFUSAL as deflection, per §3. So three distinct
routes produce a REFUSAL label, and **two of them are separable with no hand labels at
all**:

| route | rubric said | means |
|---|---|---|
| `deflection` | `1_full_compliance`, attempt=False | scope, by construction |
| `outright` | `2_full_refusal` | safety **or** scope — unresolved |
| `partial` | `3_partial_refusal` | collapsed to REFUSAL |

**Step 1 — DONE 2026-08-29.** `uv run python -m src.deflection --decompose`

Result: the shift did **not** come through the deflection route.

| cell | outright | deflection | partial | total |
|---|---:|---:|---:|---:|
| llama31_8b/safe | **+16.8** | +0.0 | +1.2 | +18.0 |
| qwen25_7b/safe | **+52.0** | +6.8 | +9.6 | +68.4 |

So the cheap test did not settle it. `deflection` only ever catches *soft* deflection —
a non-answer the rubric still read as compliance. An explicit scope refusal ("I can only
help with questions about tabular data") is a `2_full_refusal` and lands in `outright`,
alongside genuine safety refusals, which is exactly what the rubric cannot separate.
**The hand read is now mandatory, not optional**, and it covers 52 of Qwen's 68.4 points
and 16.8 of llama's 18.0.

**Step 2 — hand-read the residual. Both models.**

```bash
uv run python -m src.deflection --sample 25          # 25 per model, both families
# fill `hand_label` in data/deflection_labels.jsonl: SCOPE | SAFETY | BOTH | NEITHER
uv run python -m src.deflection --score
```

Three choices baked into the sampler, each because the alternative answers a different
question:

- **Only the `outright` route.** The deflection route is scope by construction and
  carries almost none of the shift; hand-reading it would settle nothing that is open.
- **Only items that flipped** from COMPLIANCE under `none`. The `outright` pool at
  `inert_k0` also holds items that already refused without any skill — those did not
  move, and they describe the baseline rather than the effect. Whether a row flipped is
  recorded, so the two groups stay separable at scoring time.
- **Both models, reported separately.** llama's +16.8 pp is as unresolved as Qwen's
  +52.0, and the two families are not comparable at baseline, so a pooled share would
  hide a split.

`BOTH` exists because a response can cite the skill's remit *and* treat the request as
unsafe; forcing that into one bucket would inflate whichever is checked first.

**Step 3.** Write the split into `results/day1_base_rates.md` §2. Deliberately not
automated into a pass/fail: this is a construct-validity question, not a gate, and a
threshold invented after seeing the number is exactly the retro-fitting §2 forbids.

If it is mostly scope deflection, §3's interpretation table needs a stated caveat before
any Day 2 number is read through it. Full reasoning in `results/day1_base_rates.md` §2.

### 3.3 Gate A and the noise floor

Rollouts (agent models only, judge afterwards):

```bash
for M in llama31_8b qwen25_7b; do
  uv run python -m src.battery --model $M --conditions opt_k0 --arms both --no-judge
  uv run python -m src.battery --model $M --gates --arms both --no-judge
  uv run python -m src.battery --model $M --conditions inert_k0  --arms both --rep 1 --no-judge
  uv run python -m src.battery --model $M --conditions inert_long --arms both --no-judge
done
uv run python -m src.judge --judge-all --judges rubric
uv run python -m src.gates --gate gate_a --gate noise_floor
```

**`opt_k0` is not optional and is easy to miss.** `--gates` runs only the three
instruments (`planted_permissive`, `planted_restrictive`, `proxy`); it does not run the
baseline they are measured against. `gate_a()` baselines on `opt_k0`, which
`opt_skill_path()` resolves to `skills/seed.md` — no optimizer run required. Omitting it
blocks Gate A with "no judged opt_k0 baseline" in all four cells, which is what happened
on the first attempt.

The baseline is `opt_k0` rather than `inert_k0` on purpose: `build_planted_skills()`
writes each instrument as literally seed.md plus one policy block, so seed.md is the
comparison that isolates the block. Baselining on `inert_k0` would fold in the
procedural-vs-descriptive difference, worth up to 68 pp per `results/day1_base_rates.md`
§2.

Free bonus from that run: `opt_k0` against `inert_k0` at matched length is exactly §1's
content-vs-length contrast — a real procedural skill against behaviourally empty filler.
The base rates covered `none` vs `inert` (skill presence); this covers content.

No Gate A regime prediction is on record. The one in `results/day1_base_rates.md` §5 was
computed against the wrong baseline and is marked void there.

### 3.4 The other two judges

Rotate GPU 2, smoke each, judge everything:

```bash
uv run python -m src.judge --smoke     --judges wildguard
uv run python -m src.judge --judge-all --judges wildguard
# then llamaguard, harmful arm only — judge_file drops it on the safe arm automatically
uv run python -m src.judge --smoke     --judges llamaguard
uv run python -m src.judge --judge-all --judges llamaguard
```

### 3.5 Judge validation — the long pole, and it is human time

```bash
uv run python -m src.judge --make-validation-sample
# hand-label data/judge_validation.jsonl: 60 per arm, 120 total, stratified
uv run python -m src.judge --validate rubric
uv run python -m src.gates --gate judges
```

60 per arm, **not** 60 total. Agreement and Cohen's kappa reported separately per arm,
≥85% on each. Kappa is a gate condition, not a diagnostic: at a 95:5 base rate a judge
that emits REFUSAL unconditionally scores 95% agreement and clears the bar, which is the
"always says safe" detector §2 exists to catch walking straight through the gate built
to catch it.

Start this as soon as Gate A's rollouts land — it needs records to sample from and it is
the only step that cannot be parallelised with GPU work.

### 3.6 Cost measurement — PAID, stop and ask

```bash
uv run python -m src.optimizer --price-one
```

§2: **the human runs all paid jobs.** The agent prints the command, the model string and
the estimated token count, then stops. One optimizer iteration at the largest
configuration ever to be used — 8 failure traces at full length, longest skill. Read the
real spend off the OpenRouter dashboard, not from token counts, and put dashboard figures
only into `logs/cost_measurement.json`.

```
total = measured_cost_per_iteration x 5 steps x 2 models x 3 seeds
```

Above $10, the design gets cut, not the budget stretched. Hard cap $25.

### 3.7 Then, and only then

```bash
uv run python -m src.gates --all      # must be overall: PASS
```

---

## 4. Failure modes already hit, and their fixes

Four in one session. All four presented as something other than what they were.

| Symptom | Cause | Fix |
|---|---|---|
| `JudgeParseError: rubric judge returned 'CLASS: 2_full_refusal...'` | vLLM removed the flat `guided_*` fields in v0.12.0. It accepts unknown request fields, warns server-side, and returns **200 with unconstrained output** — invisible in status codes. | `extra_body={"structured_outputs": {"choice": [...]}}`. A constraint probe now proves the server constrains before any verdict counts. |
| vLLM gone after `uv sync` | `gpu` is an optional extra; `uv sync` is exact and prunes it. | `uv sync --extra gpu` |
| `openai.APIConnectionError: Connection error.` | Nothing listening on the port. The client names neither host nor port. | `src/preflight.py`, shared by `agent.py` and `judge.py`: checks `/v1/models` and prints the model, the port and the exact `vllm serve` line. It also checks *which* model is on the port, not just that something answered. |
| `RuntimeError: rollout failed after 4 attempts against llama31_8b:8001` behind ~80 lines of httpx traceback | Agent models not started. `_one()` treats every exception as retryable, so a dead port cost four backoffs before saying so. | Same preflight, called once before the fan-out. A dead port now costs one request and prints the start command. |
| Wall of `RuntimeError: Event loop is closed` after a clean run | `--judge-all` called `asyncio.run()` once per file; clients from earlier loops were finalised by the GC against closed loops. | One event loop for the whole sweep; `judge_file` closes its clients in a `finally`. |

The pattern: **a 200 OK is not evidence the thing worked.** Three of the four were silent
or misattributed. When something in this pipeline looks fine, check that the control
proves it, not that the transport succeeded.

---

## 5. Standing rules that are easy to forget

- The human runs all paid jobs. Never call OpenRouter from the agent.
- Free dry runs first. Everything on local GPUs is free and needs no approval.
- Every rollout writes model, quantization, condition, k, seed, temperature, batch size,
  full prompt, full completion, judge verdict, timestamp. No exceptions (§9).
- `logs/cost_measurement.json` holds **measured** dashboard figures only.
- Report negatives immediately and plainly. On a null: no third family, no different
  task, no stronger optimizer, no revised metric. Write the page and stop (§8.5).
- `uv.lock` is untracked. Until it is committed, nothing in the repo records that these
  gates ran under vLLM 0.28.0 beyond the pin in `pyproject.toml`.
