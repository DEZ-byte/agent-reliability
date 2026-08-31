# Agent Reliability

**Does fine-tuning a small model beat wrapping a bigger one in retry logic?**

A model that solves a task once is not reliable. This project measures `pass^k` —
the share of tasks where *all k* independent attempts succeed — because that is
what "it works" has to mean when something downstream depends on it.

On 150 held-out tool-use tasks, the answer is yes. A fine-tuned 1.7B beats a
scaffolded 8B on reliability and costs about a third as much to run.

---

## Result

All arms, 150 held-out tasks, 4 attempts per task.

| Arm | Params | `pass^1` | `pass^4` | Cost / task † | Serve @ 4-bit |
| :-- | :-- | --: | --: | --: | --: |
| Qwen3-1.7B, untrained | 1.7B | 0.303 | 0.247 | not recorded | ~1.5 GB |
| Llama-3.1-8B + retry scaffolding | 8B | 0.415 | 0.293 | 236 | ~6 GB |
| **Qwen3-1.7B, fine-tuned** | **1.7B** | **0.515 – 0.552** | **0.393 – 0.460** | **72** | **~1.5 GB** |
| Qwen3-1.7B, fine-tuned + GRPO | 1.7B | 0.553 – 0.562 | 0.460 – 0.480 | 72 | ~1.5 GB |

† Billion-parameter-tokens: generated tokens per task, weighted by the model's
actual parameter count. Qwen3-1.7B has 2.03B parameters and Llama-3.1-8B has
8.03B, so the sums are 35.4 × 2.03 ≈ 72 and 29.4 × 8.03 ≈ 236. The 8B emits
*fewer* tokens per task, but a token from an 8B model is not a token from a
1.7B one, and on raw token counts the conclusion reverses. Token counts come
from the `generated_tokens_per_episode` field in each test artifact. The GRPO
arms are slightly dearer than the fine-tuned one, at 73 and 74.

The fine-tuned range spans three independent training runs. The GRPO range spans
two learning rates, one run each, both starting from the same SFT checkpoint.
The 8B row uses its retry rung, since retry is what defines that arm.

---

## The arms

| Arm | What it is | Why it is here |
| :-- | :-- | :-- |
| Untrained 1.7B | Qwen3-1.7B, no changes | The floor |
| Scaffolded 8B | Llama-3.1-8B-Instruct with a retry rung | The "just use a bigger model" answer |
| Fine-tuned 1.7B | Same 1.7B, LoRA on 684 verified trajectories | The hypothesis |
| + GRPO | Reinforcement learning on top, execution-backed reward | Does RL add anything after SFT? |

The teacher was Qwen3-4B. Training and test tasks are disjoint by ID and by
content hash.

---

## 1. Fine-tuning beat the bigger model

Paired on the same tasks, against the scaffolded 8B:

| Training run | `pass^1` gain | 95% interval | `pass^4` gain | 95% interval |
| :-- | --: | :--: | --: | :--: |
| Run 1 | +0.115 | 0.045 – 0.185 | +0.100 | 0.027 – 0.173 |
| Run 2 | +0.102 | 0.027 – 0.178 | +0.120 | 0.033 – 0.207 |
| Run 3 | +0.138 | 0.063 – 0.215 | +0.167 | 0.080 – 0.253 |

Every interval excludes zero, on both metrics, in all three runs.

Retry buys the 8B almost nothing. Its single-attempt score moves by three tenths
of a point when the retry rung is switched on, and `pass@4` does not move at all.
A retry that fires only when nothing parses cannot fix a well-formed call that
computed the wrong thing — and that is nearly every failure it has.

---

## 2. What fine-tuning actually fixed

Failures out of 600 episodes, single-attempt rung, counted from the episode
logs. The fine-tuned column is run 3; the other two runs total 285 and 291
failures, so the shape holds but the exact counts move.

| Failure mode | Untrained 1.7B | Fine-tuned 1.7B (run 3) | Scaffolded 8B |
| :-- | --: | --: | --: |
| Emitted no tool call at all | 43 | 1 | 0 |
| Emitted a call the tool rejected | 19 | 5 | 23 |
| Called the tool, computed the wrong value | 356 | 263 | 330 |
| **Total failures** | **418** | **269** | **353** |

Two of the three rows collapse to near zero. The model learned to reach for the
tool and to format the call. That is what fine-tuning is good at, and it was most
of the untrained gap.

The third row barely moves, and it is now the whole problem. What remains is a
model that calls the calculator correctly and asks it the wrong question. That is
a reasoning limit, not a formatting one.

---

## 3. Reinforcement learning added nothing

400 GRPO steps on an execution-backed reward, starting from the fine-tuned model.

| Learning rate | `pass^1` change | 95% interval | Adapter moved | What the model sees |
| :-- | --: | :--: | --: | --: |
| 1e-6 | +0.002 | −0.010 – 0.013 | 0.45% | 1.5% |
| 1e-5 | +0.010 | −0.020 – 0.040 | 3.82% | 13.8% |

The last two columns are recomputed from the adapters in
[`results/weight-change-7e33eb5.json`](results/weight-change-7e33eb5.json)
rather than quoted from a private note. "Adapter moved" is the relative
Frobenius change across every adapter tensor; "what the model sees" is the same
measure applied to the per-module LoRA product, which is what actually reaches
the base weights. The higher rate moved the policy roughly nine times further on
both.

Both intervals contain zero. The tight one rules out any effect larger than about
a point, rather than merely failing to find one.

The obvious objection was that the run barely moved the model. Measuring the
weight shift confirmed it, so the run was repeated at ten times the rate. That
moved the weights nearly ten times as far and produced an identical dev peak. Two
nulls across a tenfold rate range are harder to dismiss than one.

**Three measurements explain why:**

| Cause | Measurement | What it means |
| :-- | :-- | :-- |
| Dead groups | ~1 step in 4 carried no gradient | GRPO learns by comparing 8 attempts at one problem. When all 8 score alike there is nothing to compare, and a policy that already works usually agrees with itself. |
| Inert reward terms | Spread: accuracy 0.339, format 0.006, efficiency 0.002, gate 0.000 | Three of the four reward terms never varied. This optimised accuracy alone, whatever the config says. |
| Nothing left to reach | See §2 | Fine-tuning had already taken the failures a preference signal can fix. |

This is a null **at this budget, on this task, from this starting point**. It is
not evidence that reinforcement learning cannot help tool-calling models. The
known fix — discarding zero-variance groups and refilling the batch — was not
implemented here.

One direction is worth chasing but is not yet a result. The higher rate raised
`pass^4` and lowered `pass@4`, narrowing the band of sometimes-solved tasks. That
is what a policy-gradient method concentrating probability mass looks like, and it
is the trade this project cares about. A paired permutation test on the per-task
band width gives p = 0.24, so it is a hint. Note that this is a different test
from the `pass^k` comparisons recorded in
[`results/grpo-lr1e5-vs-sft-8182e7e.json`](results/grpo-lr1e5-vs-sft-8182e7e.json),
which report p = 0.60; the band comparison is not yet in that artifact.

---

## 4. Capability outran reliability

This is the headline, and the second half matters more than the first.

| Metric | Untrained | Fine-tuned | Gain |
| :-- | --: | --: | --: |
| Solves it at least once in 4 (`pass@4`) | 0.353 | 0.627 – 0.680 | **+0.27 to +0.33** |
| Solves it 4 times out of 4 (`pass^4`) | 0.247 | 0.393 – 0.460 | **+0.15 to +0.21** |
| Band solved *sometimes* but not always | 0.107 | 0.213 – 0.287 | **wider** |

Training was meant to close the reliability gap. It widened it.

Reported as `pass@4` this project could claim about +33 points. Reported strictly,
where every attempt has to land, it is +15. Most of what looks like progress is
capability. Reliability is the part that did not keep up, and it is the part the
project set out to measure.

---

## 5. What did *not* change

| Claim someone might make | What the measurement says |
| :-- | :-- |
| "It got better at arithmetic." | No. Probed with no calculator at all: 64.0% before, 66.0% after. It got better at *writing the expression*. |
| "It learned to cheat the grader." | No. The reward pays the same for restating a remembered answer as for real work — deliberately, so the behaviour is measured rather than hidden. The rate fell from 3.0% untrained to 1.2% fine-tuned to 1.0% after RL. |
| "More seeds would sharpen this." | No. Between-run standard deviation is 0.019; a single run's interval is ~0.069 either side. Training is steadier than 150 tasks can resolve, so a bigger test split would buy more than more seeds. |
| **"It forgot things."** | **No.** 400 held-out MMLU questions, no tool offered: 53.5% untrained, 54.3% fine-tuned. Paired difference +0.005, 95% interval −0.038 to +0.049. 38 questions improved, 36 got worse. |
| **"It now calls tools at everything."** | **No.** On a benchmark offering no tools, every arm emitted a tool call on **0.0%** of questions. The habit is tied to being offered a tool, not to being asked a question. |

The knowledge result is the one worth stating plainly, because it is the first
thing anyone asks and the project could not answer it until now. Fine-tuning a
1.7B on a thousand calculator trajectories did not measurably cost it general
knowledge, and did not leak the tool-calling habit into contexts with no tools.

One real behavioural change did show up. The fine-tuned model answers far more
briefly — 227 characters on average against the untrained model's 628 — and is
cut off by the token budget a quarter as often. Terser, not worse.

---

## 6. Did it generalise, or specialise? Both, in different places

Every number above was measured on the task the model was trained for, which
cannot tell the two apart. So the checkpoints were run on a second environment
they had never seen: an order-support agent with three tools, no arithmetic
anywhere, and no calculator. Half the requests should be completed and half
should be refused, because a model that learned "always call the writing tool"
scores 50% on a balanced set and 100% on a one-sided one.

| | Untrained 1.7B | After SFT | After GRPO |
| :-- | --: | --: | --: |
| `pass^1` | 0.493 | 0.528 | 0.542 |
| Completes a legitimate request | **0.000** | **0.947** | **0.957** |
| Correctly refuses an unverified one | **1.000** | **0.098** | **0.115** |
| Calls any tool | 0.753 | 1.000 | 1.000 |
| Writes without the right to | 0.000 | 0.920 | 0.918 |
| Mean reward | +0.286 | **−0.394** | **−0.387** |

The headline metric barely moves. Everything underneath it inverts.

**The capability transferred.** An untrained 1.7B completes none of these
requests. After fine-tuning on GSM8K and a calculator, it completes 95% of them,
using three tools it has never seen in a domain with no maths in it. That is not
a small thing and it is the strongest evidence here that the training taught
something general.

**The judgement did not.** On requests it should refuse, it writes anyway 90% of
the time. It calls the verification tool, receives `authenticated: false`, and
changes the record regardless. It never once calls the lookup tool, so even its
legitimate writes are unauthorised. Both models score about half by acting
indiscriminately; the untrained model scores about half by never acting at all.

The mean reward is the honest summary: the fine-tuned model is **worse** in this
environment than the one that does nothing, because the gate penalty finally has
something to fire on.

This follows from how the training data was built. Every example was one call
and done, and none of them had "do not call the tool" as the right answer, so an
unconditional policy fit the data perfectly. It is the same structural limit
recorded earlier: an environment that ends the episode on the first successful
call never teaches a model to read a result and decide.

GRPO again changed nothing it did not already do.

## How the numbers are kept honest

| Rule | How it is enforced |
| :-- | :-- |
| Accuracy comes from executing the tool | Never parsed from the model's prose |
| Results cannot be edited after the fact | Every result file is frozen by SHA-256 in [`results/artifact_manifest.json`](results/artifact_manifest.json); a test fails if one changes |
| No train/test leakage | Splits are disjoint by task ID *and* by content hash |
| No checkpoint cherry-picking | The selection rule was written into the config before any dev number existed; the winner runs on test exactly once |
| Comparisons are paired | Task-level bootstrap intervals, paired permutation tests, exact sign test |

```bash
pip install -e . && python -m unittest discover -s tests
```

The editable install is not optional: most test modules import `agent` and `env`
directly, so discovery fails without it.

---

## Limits

- **Only one model has a trained arm.** The teacher was Qwen3-4B rather than the
  larger model originally planned, so nothing here says anything about the 4B.
- **Size and pretraining are tangled.** The comparator is a Llama, not an 8B from
  the Qwen family. Some of the gap may be pretraining rather than scale.
- **The untrained baseline barely varies.** It produced four identical answers on
  82 of 150 tasks, so its `pass^4` partly collapses into `pass^1`. Restricted to
  tasks where both arms genuinely varied, the trained model's lead *widens*. The
  finding survives; the caveat is real.
- **The knowledge probe is 400 questions, not 14,042.** A stratified sample of
  MMLU, which bounds how small a change it could detect: the paired interval is
  about ±4 points, so a loss smaller than that would not show up here.
- **The untrained model is cut off more often.** It was truncated on 40% of
  questions against the fine-tuned model's 16%, because it answers at length.
  Most truncated answers still named a choice, and restricting to readable
  answers moves the comparison by about a point in the other direction, so the
  null holds either way — but it is a real asymmetry and it is recorded.
- **The base model barely varies.** On the transfer environment it produced four
  identical answers on 75% of tasks, so its `pass^4` largely collapses into
  `pass^1`. It is a floor, not a competitor.
- **No arm ever called the lookup tool.** Some of the gate violations in section
  6 are a missing step rather than a defied one, and the two are separated in
  the artifacts.

[`FINDINGS.md`](FINDINGS.md) is the short version of what came out of this,
including the parts that went badly.

Licensed Apache-2.0. Models are Qwen3 (Apache-2.0) and Llama-3.1 (Llama Community
Licence); task data is GSM8K (MIT).
