# agent-reliability

There are two ways to make a tool-using AI agent reliable. You can wrap a big
model in runtime machinery that retries, reflects, and escalates. Or you can
train a small model until it behaves correctly on its own.

The first costs you tokens and latency on every single request, forever. The
second costs GPU time once.

This repository measures which one actually wins, and by how much.

> **Status: baselines measured, one trained arm.** Untrained baselines are in
> [the first results](#first-results-untrained-baselines); the first trained
> arm is [below that](#the-first-trained-arm). Training raised capability
> considerably and strict reliability much less, widening the very gap this
> project studies. Every hypothesis in [`BLUEPRINT_v2.md`](BLUEPRINT_v2.md)
> is still open: H1 needs arms that do not exist yet.

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

## First results: untrained baselines

Both selected checkpoints, both rungs, the frozen 150-task test split, four
runs per task. 2,400 episodes. No training yet, so this is the floor everything
later gets compared against.

`R0` is one model decision per turn. `R1` is one extra decision that sees a
factual description of why the first one failed. Nothing else differs.

| Model | Rung | pass^1 | pass^4 | pass@4 | Answered without computing |
| :-- | :-- | :-- | :-- | :-- | :-- |
| `Qwen3-4B` | R0 | 0.598 [0.522-0.673] | 0.547 [0.467-0.627] | 0.647 | 0.0% |
| `Qwen3-4B` | R1 | 0.608 [0.532-0.683] | 0.567 [0.487-0.647] | 0.647 | 0.3% |
| `Qwen3-1.7B` | R0 | 0.303 [0.237-0.373] | 0.247 [0.180-0.320] | 0.353 | 1.5% |
| `Qwen3-1.7B` | R1 | 0.333 [0.262-0.405] | 0.287 [0.213-0.360] | 0.380 | 1.7% |

**The reliability gap is real and it is about 10 points.** `pass@4` counts a
task as solved if any of four attempts worked. `pass^4` counts it only if all
four worked. The distance between those columns is the band of tasks these
models solve *sometimes* - 10.0 points for the 4B, 10.7 for the 1.7B. A
benchmark reporting only `pass@4` would describe these models as a tenth better
than they are when every attempt has to hold. Closing that gap is what this
project is testing, and the first trained arm widened it rather than closing
it - see below.

**One caveat that belongs beside every `pass^4` figure here.** The untrained
1.7B emitted four byte-identical completions on 82 of these 150 tasks, so on
more than half the split its four attempts were one attempt repeated and
`pass^4` collapses to `pass^1`. Sampling at temperature 0.7 is supposed to
prevent exactly that. It is measured, reported, and it shrinks after training
to 7 of 150.

**The retry rung barely helps, and the reason is the interesting part.** Paired
permutation test on per-task `pass^4`, same tasks and seeds: the 4B gains 0.020
(p=0.25) and the 1.7B gains 0.040 (p=0.031). Only 3 and 6 tasks out of 150
actually disagree between the rungs, so even the nominally significant one is
too thin to lean on. Read it as: one extra decision buys little here.

Why it buys little is a measurement, not a guess. `R1`'s second decision only
fires when the first produced no tool call at all - 3.7% of episodes for the 4B,
10.3% for the 1.7B. When it does fire it works, converting about 28% of those
into correct answers. But it cannot touch the dominant failure. Between **85%
and 91% of all failures are a perfectly well-formed calculator call that
computes the wrong thing.**

That is a bound on what runtime scaffolding of this kind can do, derived from
data rather than argument: a retry told only that nothing parsed has nothing to
work with when the parse was fine and the reasoning was not. It predicts the
training arms have room the scaffolding arms do not. It does not demonstrate
that - no arm has been trained. It just makes the claim checkable.

**Nothing is being laundered yet.** The last column is the D-062 failure mode:
solving in your head and passing the answer to the calculator as
`calculator("391")`. It sits between 0.0% and 1.7%, so it is not inflating these
numbers. It gets reported after every training run too, because a reward for a
passing tool call is exactly the pressure that would push it up.

Full numbers in [`results/baseline-phase_a-3cc174f.json`](results/baseline-phase_a-3cc174f.json),
frozen by hash. Reasoning in [`DECISIONS.md`](DECISIONS.md) D-068.

## The first trained arm

`Qwen3-1.7B` fine-tuned on 684 verified trajectories generated by `Qwen3-4B`,
one row per training task, kept only where the deterministic grader passed the
episode and the laundering filter cleared the expression. Checkpoint chosen on
the dev split by a rule written down before any dev number existed. The test
split was touched exactly once.

| Metric (rung R0) | Base | SFT run 1 | SFT run 2 |
| :-- | :-- | :-- | :-- |
| `pass^1` | 0.303 | 0.525 | 0.515 |
| `pass^4` | 0.247 | 0.393 | 0.413 |
| `pass@4` | 0.353 | 0.680 | 0.627 |
| Generated tokens per episode | 45.9 | — | **35.8** |

Paired against the same baseline on the same tasks:

| | Run 1 | Run 2 |
| :-- | :-- | :-- |
| `pass^1` gain | **+0.222** [+0.155, +0.290] | **+0.212** [+0.143, +0.280] |
| `pass^4` gain | +0.147 [+0.067, +0.227] | +0.167 [+0.087, +0.247] |

Two runs differing in both initialisation and data order land in the same
place, so this is not one lucky run. **Two runs are a range, not a variance
estimate**, and no interval on the effect size itself is claimed from n=2.

**It is also cheaper.** The trained model spends **0.78x** the generated tokens
of the untrained one - 35.8 per episode against 45.9 - while being 21 points
more accurate. Capability moved into the weights is not paid for again at
inference. That is one leg of the argument H1 exists to test, not the claim
itself: H1 compares a trained small model against a *scaffolded larger* one,
and the 8B comparator is registered and still unmeasured.

**The improvement is tool use, not arithmetic.** Probed without any tool, the
trained model solves 66.0% against the untrained 64.0% - barely moved. It did
not get better at maths; it got better at writing the right calculator
expression.

**Capability rose about twice as much as reliability, and that is the result.**
`pass@4` nearly doubled while `pass^4` rose less than half as much, so the band
of tasks solved *sometimes but not always* went from 0.107 to **0.287**. A
report quoting only `pass@4` would claim +33 points of progress on a model that
gained +15 where every attempt has to hold.

Restricted to the 65 tasks where both arms genuinely varied their outputs, the
`pass^4` gap **widens** to +0.215, so the finding survives the degeneracy
caveat above rather than depending on it.

**What did not improve, because it could not.** Tool-call schema validity was
**1.000 in both arms**. There was no headroom in tool use; roughly +0.199 of
the `pass^1` gain is getting the arithmetic right inside a call that was
already well-formed. The honest name for what moved is execution-graded task
accuracy, not tool-use reliability.

**The retry rung is now completely inert.** `R1` and `R0` reach an identical
`pass^4` of 0.393. The untrained baseline had already shown that retry only
repairs formatting; training removed the formatting failures, so there is
nothing left for it to repair.

**What this is not.** The teacher is `Qwen3-4B`, not the larger model the plan
pre-registered, so only the smaller checkpoint was trained and nothing here
speaks to the 4B. H1 stays open: it needs a GRPO arm, a primary-model arm, and
an 8B scaffolded comparator that has been registered but never measured.

Reasoning and full caveats in [`DECISIONS.md`](DECISIONS.md) D-072, D-073 and
D-074.

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
