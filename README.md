# Internalizing Agent Reliability

Research code for measuring how much verifiable-reward post-training can close
the tool-execution reliability gap between a small language model and a larger,
runtime-scaffolded comparator.

> Status: implementation has started; no experiments or benchmark results have
> been produced. H1–H3 in [`BLUEPRINT_v2.md`](BLUEPRINT_v2.md) remain open.

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

The current lock covers only the reliability kernel. Before any measured model
comparison, compatibility reconnaissance must produce a provisional ML smoke
lock and immutable environment manifest. The later training lock is finalized
after the Qwen decision without rewriting the smoke environment.

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
python scripts/smoke_models.py --output results/model_smoke-plan.json
```

Tokenizer/model access requires both `--run-load` and `--allow-download`.
Do not treat the offline plan artifact as a benchmark result.

## Source of truth

- [`BLUEPRINT_v2.md`](BLUEPRINT_v2.md): canonical research and experiment plan.
- [`DECISIONS.md`](DECISIONS.md): append-only architectural decision log.
- [`PLAN.md`](PLAN.md): implementation checklist and current work boundary.

## Reporting policy

No value is presented as a measured result unless it is generated from a
linked run artifact. Negative results will be reported in the same format as
positive results.
