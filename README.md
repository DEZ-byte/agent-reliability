# agent-reliability

There are two ways to make a tool-using AI agent reliable. You can wrap a big
model in runtime machinery that retries, reflects, and escalates. Or you can
train a small model until it behaves correctly on its own.

The first costs you tokens and latency on every single request, forever. The
second costs GPU time once.

This repository measures which one actually wins, and by how much.

> **Status: no reliability results yet.** The measurement harness is built and
> tested, and the model stack is chosen from real measurements. The experiment
> itself has not run. Every hypothesis in
> [`BLUEPRINT_v2.md`](BLUEPRINT_v2.md) is still open.

## The question

> How much of the `pass^k` reliability gap between a small (≤4B) tool-calling
> model and a runtime-scaffolded 8B model can verifiable-reward post-training
> close, and at what token, latency, and GPU cost?

`pass^k` is the fraction of tasks where **all k** independent attempts succeed.
It is a harsh metric on purpose. A model that solves a task once in four tries
is not reliable, and averaging hides that.

Both answers get published. "Training loses to scaffolding, but runs far
cheaper" is a real finding, and it ships in the same format as the flattering
version.

## What is actually built

A CPU-only reliability kernel, fully tested:

```text
model output -> parse -> validate -> dispatch -> event log
                                         |-> gate replay -> reward
                                         `-> trajectory -> pass^k
```

One idea holds this together. The deterministic gate that blocks an unsafe
action at runtime is the **same code** that computes the gate penalty during
training. They cannot drift apart, because there is only one of them.

Every reward is computed from what the environment actually executed. Nothing
is scored by reading the model's prose. Writing "I authenticated the user" earns
nothing; calling `authenticate_user` and having it succeed is the only thing
that counts.

## What has been measured

Four checkpoints were run through a seven-stage compatibility probe on one
frozen software stack. All four load in 4-bit on a single 8 GB GPU, emit
strictly valid tool calls, and complete a real LoRA training step.

| Checkpoint | Tool-call validity | Training step |
| :-- | :-- | :-- |
| `Qwen/Qwen2.5-3B-Instruct` | 11 of 11 checks | passed |
| `Qwen/Qwen2.5-1.5B-Instruct` | 11 of 11 checks | passed |
| `Qwen/Qwen3-4B` | 10 of 11 checks | passed |
| `Qwen/Qwen3-1.7B` | 10 of 11 checks | passed |

**Selected: Qwen3 {4B, 1.7B}.** Not because it measured better. Qwen2.5 did.
Qwen3 won on licence: `Qwen/Qwen2.5-3B-Instruct` is non-commercial, and this is
a public repository. That reasoning is written down in full, including the
awkward part, in [`DECISIONS.md`](DECISIONS.md) D-048.

The eleventh check both Qwen3 models fail is tokenized-prefix stability when a
tool reply is appended. It is a multi-turn property, and this phase is
single-turn, so it was re-scoped to a recorded diagnostic (D-046). That
decision was made *after* seeing the results, and the artifacts say so: those
runs record `passed_with_demoted_gates`, never plain `passed`. The gate is
still enforced for any multi-turn work.

These are compatibility measurements. They say nothing about how good any model
is at the task.

### Contamination: measured, and mostly reassuring

GSM8K is public web text and these models were trained on public web text, so a
good score could be recall rather than reasoning. Measured over the frozen
150-task test split, two conditions:

| Checkpoint | Solves it unaided | Answers with no room to think |
| :-- | :-- | :-- |
| `Qwen/Qwen3-4B` | 70.7% [62.9-77.4] | 4.0% [1.8-8.5] |
| `Qwen/Qwen3-1.7B` | 64.0% [56.1-71.2] | 1.3% [0.4-4.7] |

The second column is the memorisation signal: 12 tokens is not enough to do
multi-step arithmetic, so a correct answer there was recalled. Shuffling the
model's own guesses against random tasks produces 1.5% and 1.2%, so both sit at
or barely above chance. **GSM8K memorisation does not threaten this study.**

The first column is the finding that matters. Both models already solve most of
this split by reasoning in prose, with no calculator. Phase A therefore does not
test arithmetic; it tests whether a model uses a tool correctly and repeatably,
which is the actual subject. It also makes one risk concrete: a model that can
reach the answer in its head may compute mentally and pass the result to the
tool as `calculator("391")`, scoring correct having computed nothing. That is
detected, reported alongside every accuracy figure, and recorded in D-062 and
D-064.

## Quick start

Python 3.11 or 3.12. If `python` on your PATH is newer, use the `py` launcher —
the editable install enforces the version bound.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.lock
.venv\Scripts\python -m pip install --no-deps --editable .
.venv\Scripts\python -m unittest discover -s tests -v
```

That runs the whole kernel on one direct dependency, pydantic. No GPU, no
model downloads, no API keys.

For the GPU stack used by the model probe, see
[`MODEL_SMOKE_PROTOCOL.md`](MODEL_SMOKE_PROTOCOL.md). It is a separate frozen
environment (Unsloth 2026.8.18, TRL 0.24.0, Transformers 5.5.0) installed from
`requirements-smoke.lock`, and its evidence does not transfer to the later
multi-turn lane.

## How this repo defends its own numbers

Most of the engineering here exists to make dishonesty difficult:

Every committed measurement is frozen by SHA-256 in
[`results/artifact_manifest.json`](results/artifact_manifest.json). Editing a
recorded failure into a pass fails four tests. Line-ending translation is
disabled for hashed files, because it silently broke every digest off-machine
until CI caught it.

Rewards are tested against deliberate cheating: authentication mentioned in a
comment, empty tool tags, out-of-order calls. A relaxed gate cannot be quietly
relaxed — it has to be declared in a config, bound to a dated decision by exact
text, and it changes the recorded probe status.

`pass^k` and `pass@k` are computed with the unbiased estimators from a single
run array, and were checked against an independent implementation across 7,312
cases.

## Reading order

Start here, then go deeper as needed:

- [`BLUEPRINT_v2.md`](BLUEPRINT_v2.md) — the experiment: hypotheses, arms,
  rewards, compute budget, milestones, and what happens when something fails.
- [`PLAN.md`](PLAN.md) — what is done, what is next, what is blocked.
- [`DECISIONS.md`](DECISIONS.md) — every decision with a date and a reason.
  Append-only. Some entries are parsed by the runner, so it is not edited in
  place.
- [`RELATED_WORK.md`](RELATED_WORK.md) — 15 verified primary sources, plus an
  explicit list of claims this project has **not** earned.

The normative specs, once you need exact definitions:
[`HYPOTHESIS_PROTOCOL.md`](HYPOTHESIS_PROTOCOL.md) (how each hypothesis is
scored), [`RUNG_PROTOCOL.md`](RUNG_PROTOCOL.md) (what each scaffolding level
may do), [`SELF_CORRECTION_SPEC.md`](SELF_CORRECTION_SPEC.md), and
[`data/LICENSES.md`](data/LICENSES.md).

## Reporting policy

No number appears here unless a linked run artifact produced it. Negative
results are reported in the same format as positive ones. Where a standard was
relaxed, the relaxation is recorded with its date, its motive, and what it
costs.

## Licence

Apache-2.0 for this repository's own code and documentation. Model weights,
adapters, and third-party datasets keep their own terms — see
[`NOTICE`](NOTICE) and [`data/LICENSES.md`](data/LICENSES.md).
