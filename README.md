# Internalizing Agent Reliability

Research code for measuring how much verifiable-reward post-training can close
the tool-execution reliability gap between a small language model and a larger,
runtime-scaffolded comparator.

> Status: implementation has started; no benchmark or model-quality results
> have been produced. One retained Qwen3-1.7B artifact records a negative
> compatibility attempt. H1–H3 in [`BLUEPRINT_v2.md`](BLUEPRINT_v2.md) remain
> open.

## Current scope

The first milestone is a CPU-only reliability kernel:

```text
model completion -> parse -> validate/dispatch -> event log
                                           |-> gate replay -> reward
                                           `-> trajectory -> pass^k metrics
```

This slice deliberately excludes model downloads, training, benchmark claims,
and paid APIs. The runtime gate and the training reward use the same predicate
engine so their semantics can be tested before GPU work begins.

## Quick start

Python 3.11 or 3.12 is supported for the initial kernel.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.lock
.venv\Scripts\python -m pip install --no-deps --editable .
.venv\Scripts\python -m unittest discover -s tests -v
```

Or run the local check wrapper:

```powershell
./scripts/check.ps1 -Python .venv\Scripts\python.exe
```

The kernel lock above covers only the reliability kernel. The Phase-A/M0 model
smoke uses a separate Windows stack: Unsloth 2026.8.18, TRL 0.24.0, and
Transformers 5.5.0. Recreate that environment in the ignored `.venv` directory
from the generated smoke lock:

```powershell
if (Test-Path -LiteralPath .venv) { Remove-Item -LiteralPath .venv -Recurse -Force }
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install uv==0.12.5
.venv\Scripts\uv.exe pip sync requirements-smoke.lock --python .venv\Scripts\python.exe
```

`requirements-smoke.lock` already includes the repository as an editable
project, so the sync command is the complete install. Do not add a second
editable-install step. Before any measured comparison, the provisional lock
and an immutable environment manifest must be frozen after compatibility
checks and recreated cleanly.

After committing the exact source used by the smoke, record the model-free
dependency, offline-import, and CUDA manifest:

```powershell
.venv\Scripts\python.exe scripts\probe_smoke_environment.py --output results\smoke_environment.json
```

The manifest fails when the Git tree is dirty, the configured lock hash has
changed, or an installed locked distribution has a different version.

The later M6 live multi-turn lane uses TRL 1.8 `environment_factory` without
Unsloth. It needs a separate lock and manifest. Compatibility evidence does
not transfer between the two lanes.

## M0 evidence controls

- [`RELATED_WORK.md`](RELATED_WORK.md) is a 15-source, primary-source-only
  review. It records what is adopted, what may be distinguishing, and which
  claims remain unsupported.
- [`data/LICENSES.md`](data/LICENSES.md) records model and dataset access terms,
  caveats, and immutable revisions. The machine-readable registry is
  [`configs/model_candidates.json`](configs/model_candidates.json).
- [`MODEL_SMOKE_PROTOCOL.md`](MODEL_SMOKE_PROTOCOL.md) pre-registers the Qwen
  comparison. No model has won until comparable measured artifacts exist.
- [`SELF_CORRECTION_SPEC.md`](SELF_CORRECTION_SPEC.md) separates same-model
  diagnostic repair from retry luck, gate prevention, and 8B escalation.
- [`RUNG_PROTOCOL.md`](RUNG_PROTOCOL.md) fixes R0/R1/R2 turn, retry, gate,
  escalation, parity, and cost-accounting semantics.
- [`HYPOTHESIS_PROTOCOL.md`](HYPOTHESIS_PROTOCOL.md) makes H1–H3 executable
  with exact arms, denominators, zero handling, and paired inference.

The model smoke command is offline by default. It validates the plan and
records local environment facts without importing the optional ML stack or
accessing a model repository:

```powershell
.venv\Scripts\python.exe scripts\smoke_models.py --output results/model_smoke-plan.json
```

Tokenizer/model access requires both `--run-load` and `--allow-download`.
The current runner implements P0-P6. P5 checks multi-message serialization and
assistant-token masking, not a live multi-turn environment. It preserves the
native inference template and derives a project-owned, training-only Qwen
wrapper whose rendered text and token IDs must match the native template
exactly. Ambiguous template structures fail closed. Immutable model identity
comes from the exact local Hugging Face snapshot commit, with any exposed
loader metadata required to agree. P6 reuses the exact P5 mask through the TRL
collator, attaches a rank-4 `q_proj`/`v_proj` LoRA adapter, obtains a same-model
reference with the PEFT adapter disabled, and runs one ephemeral SGD microstep
without writing a checkpoint. This code path has mock-only test coverage; it
has not run on any model checkpoint and makes no quality claim. P6 must execute
for all four checkpoints before selection.

The first Qwen3-1.7B attempt is retained at
[`results/model_smoke-qwen3-1.7b-6824196.json`](results/model_smoke-qwen3-1.7b-6824196.json).
It records the private-revision-metadata and native-template masking failures
that motivated the corrected probes. The corrected rerun remains pending, and
the artifact does not select a model or establish quality.

The machine-readable release gate is still pending. It pins the candidate
registry by SHA-256 and derives eligible bundles from each checkpoint's
registry state. Selection stays disabled until all four checkpoints pass
P1-P6 and a recorded release decision resolves at least one complete bundle.
Do not treat the offline plan artifact as a benchmark result.

## Source of truth

- [`BLUEPRINT_v2.md`](BLUEPRINT_v2.md): canonical research and experiment plan.
- [`DECISIONS.md`](DECISIONS.md): append-only architectural decision log.
- [`PLAN.md`](PLAN.md): implementation checklist and current work boundary.

## Reporting policy

No value is presented as a measured result unless it is generated from a
linked run artifact. Negative results will be reported in the same format as
positive results.
