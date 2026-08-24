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

| Learning rate | `pass^1` change | 95% interval | Weights moved |
| :-- | --: | :--: | --: |
| 1e-6 | +0.002 | −0.010 – 0.013 | 0.41% |
| 1e-5 | +0.010 | −0.020 – 0.040 | 3.82% |

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

---

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
- **General capability was not measured.** There is no before/after on anything
  outside this task. Nine tasks regressed and two the untrained model always
  solved now score zero, so the damage is real but small — and unquantified
  off-task.
- **One reward term never fired.** The gate term measured exactly 0.000, because
  this environment has one harmless tool and nothing to refuse.

[`FINDINGS.md`](FINDINGS.md) is the short version of what came out of this,
including the parts that went badly.

Licensed Apache-2.0. Models are Qwen3 (Apache-2.0) and Llama-3.1 (Llama Community
Licence); task data is GSM8K (MIT).
