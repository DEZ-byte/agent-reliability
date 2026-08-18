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

The ML training environment will receive a separate compatibility lock after
the M0 Qwen/Unsloth/TRL smoke test; the current lock covers only the reliability
kernel.

## Source of truth

- [`BLUEPRINT_v2.md`](BLUEPRINT_v2.md): canonical research and experiment plan.
- [`DECISIONS.md`](DECISIONS.md): append-only architectural decision log.
- [`PLAN.md`](PLAN.md): implementation checklist and current work boundary.

## Reporting policy

No value is presented as a measured result unless it is generated from a
linked run artifact. Negative results will be reported in the same format as
positive results.
