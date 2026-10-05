# Agent Reliability

[![ci](https://github.com/DEZ-byte/agent-reliability/actions/workflows/ci.yml/badge.svg)](https://github.com/DEZ-byte/agent-reliability/actions/workflows/ci.yml)

**Does fine-tuning a small model beat wrapping a bigger one in retry logic?**

A model that solves a task once is not reliable. This project measures `pass^k` —
the share of tasks where *all k* independent attempts succeed — because that is
what "it works" has to mean when something downstream depends on it.

On 150 held-out tool-use tasks, a fine-tuned Qwen3-1.7B beats a retry-scaffolded
Llama-3.1-8B on reliability. It uses about 31% of the 8B's parameter-weighted
generated-token proxy per attempt. It does not reach the untrained Qwen3-4B that
wrote its training data.

---

## Result

All arms, 150 held-out tasks, 4 attempts per task, all at rung R1 (one retry when
the output does not parse).

| Arm | Params | Rung | `pass^1` | `pass^4` | Cost / attempt † | Serve @ 4-bit † |
| :-- | :-- | :-- | --: | --: | --: | --: |
| Qwen3-1.7B, untrained | 1.7B | R1 | 0.333 | 0.287 | not recorded | ~1.5 GB |
| Llama-3.1-8B + retry scaffolding | 8B | R1 | 0.415 | 0.293 | 236 | ~6 GB |
| **Qwen3-1.7B, fine-tuned** | **1.7B** | **R1** | **0.517 – 0.553** | **0.393 – 0.460** | **73 – 75** | **~1.5 GB** |
| Qwen3-1.7B, fine-tuned + GRPO | 1.7B | R1 | 0.562 – 0.588 | 0.467 – 0.507 | 74 – 75 | ~1.5 GB |
| Qwen3-4B, untrained (the teacher) | 4B | R1 | 0.608 | 0.567 | not recorded | not measured |

Cost is billion-parameter-tokens: generated tokens per attempt, from the
`generated_tokens_per_episode` field in each test artifact, times the model's
parameter count. The 8B emits *fewer* tokens per attempt (29.4 against 35.8 for
the fine-tuned run 3), but a token from an 8B model is not a token from a 1.7B
one, and on raw token counts the conclusion reverses. The proxy counts decoding
only, not the prompt. The parameter counts, 2.03B and 8.03B, come from the model
configs rather than an artifact. The 2.03B counts Qwen3's tied input and output
embeddings separately, which makes the 1.7B look dearer, not cheaper. Fine-tuned
run 1 did not record tokens, so its cost is missing from the range.

The fine-tuned range spans three independent training runs. The GRPO range spans
two learning rates, one run each, both starting from the same SFT checkpoint.
Single-attempt (R0) numbers are in the same artifacts and are within 0.04 of these
for every arm.

† Not in a committed artifact. Numbers marked † elsewhere in this README come
from local episode logs or model configs that are not in the repository.

---

## The arms

| Arm | What it is | Why it is here |
| :-- | :-- | :-- |
| Untrained 1.7B | Qwen3-1.7B, no changes | The floor |
| Scaffolded 8B | Llama-3.1-8B-Instruct with a retry rung | The "just use a bigger model" answer |
| Fine-tuned 1.7B | Same 1.7B, LoRA on 684 retained trajectories drawn from 1,000 tasks | The hypothesis |
| + GRPO | Reinforcement learning on top, execution-backed reward | Does RL add anything after SFT? |
| Untrained 4B | Qwen3-4B, no changes | The teacher: it wrote the SFT trajectories |

Training and test tasks are disjoint by ID and by content hash.

---

## 1. Fine-tuning beat the bigger model, not the teacher

Paired on the same tasks, against the scaffolded 8B, both arms at R1:

| Training run | `pass^1` gain | 95% interval | `pass^4` gain | 95% interval |
| :-- | --: | :--: | --: | :--: |
| Run 1 | +0.115 | 0.045 – 0.185 | +0.100 | 0.027 – 0.173 |
| Run 2 | +0.102 | 0.027 – 0.178 | +0.120 | 0.033 – 0.207 |
| Run 3 | +0.138 | 0.063 – 0.215 | +0.167 | 0.080 – 0.253 |

Every interval excludes zero, on both metrics, in all three runs.

Retry buys the 8B almost nothing. Its single-attempt score moves by three tenths
of a point when the retry rung is switched on, and `pass@4` does not move at all.
A retry that fires only when nothing parses cannot fix a well-formed call that
computed the wrong thing, and that is 330 of its 353 failures †.

The same runs against the untrained Qwen3-4B, both at R1. Negative means the
fine-tuned 1.7B scored lower:

| Training run | `pass^1` gap | 95% interval | `pass^4` gap | 95% interval |
| :-- | --: | :--: | --: | :--: |
| Run 1 | −0.078 | −0.143 to −0.013 | −0.173 | −0.253 to −0.093 |
| Run 2 | −0.092 | −0.157 to −0.028 | −0.153 | −0.227 to −0.073 |
| Run 3 | −0.055 | −0.120 to +0.008 | −0.107 | −0.187 to −0.027 |

The 4B stays ahead on `pass^4` in every run. Distilling its trajectories into a
1.7B recovered most of the gap on single attempts, but less of it on reliability.
From [`results/sft-vs-teacher-run1-d692d43.json`](results/sft-vs-teacher-run1-d692d43.json)
and its two siblings.

---

## 2. What fine-tuning actually fixed

Failures out of 600 episodes at R0 (single attempt), counted from the episode
logs †. The fine-tuned column is run 3; the other two runs total 285 and 291
failures, so the shape holds but the exact counts move.

| Failure mode | Untrained 1.7B | Fine-tuned 1.7B (run 3) | Scaffolded 8B |
| :-- | --: | --: | --: |
| Emitted no tool call at all | 43 | 1 | 0 |
| Emitted a call the tool rejected | 19 | 5 | 23 |
| Called the tool, computed the wrong value | 356 | 263 | 330 |
| **Total failures** | **418** | **269** | **353** |

Two of the three rows collapse to near zero. The model learned to reach for the
tool and to format the call. Those two rows account for 56 of the 149 failures
that training removed, about 38%.

The other 93 came off the third row. Wrong-value errors fell substantially, by
26%, but they still account for 263 of 269 remaining failures. What remains is a
model that calls the calculator correctly and asks it the wrong question.

---

## 3. GRPO: no detectable gain at 1e-6, a small one at 1e-5

400 GRPO steps on an execution-backed reward, starting from the fine-tuned model.

| Learning rate | Rung | `pass^1` change | 95% interval | Adapter moved | What the model sees |
| :-- | :-- | --: | :--: | --: | --: |
| 1e-6 | R0 | +0.008 | +0.000 – 0.017 | 0.40% | 1.4% |
| 1e-5 | R0 | +0.035 | +0.005 – 0.068 | 4.63% | 15.3% |

Every column comes from the same checkpoints: the ones dev selection picked and
test measured (checkpoint-200 at 1e-6, checkpoint-300 at 1e-5). The last two are
recomputed from the adapters in
[`results/weight-change-b23567a.json`](results/weight-change-b23567a.json).
"Adapter moved" is the relative Frobenius change across every adapter tensor;
"what the model sees" is the same measure applied to the per-module LoRA
product, which is what actually reaches the base weights. The higher rate moved
the policy about 11 times further by both measures. An earlier version of this
table showed 0.45% and 3.82%, which belong to the final step-400 adapters rather
than the tested ones ([`ERRATA.md`](ERRATA.md), E8). The version after that
showed `pass^1` changes of +0.002 and +0.010. Those GRPO runs most likely trained
on a pre-quantised copy of the base model, so both rates were trained again on
the pinned base ([`ERRATA.md`](ERRATA.md), E13).

At 1e-6 the interval runs from 0.000 to +0.017: no detectable change. At 1e-5
the interval, +0.005 to +0.068, excludes zero. The bootstrap interval is the
primary test, so at 1e-5 this is a detectable `pass^1` gain of 3.5 points. The
secondary tests are weaker. The permutation p is 0.04, above the Bonferroni
threshold of 0.025 that the artifact reports for its two comparisons. The exact
sign test gives p = 0.22: 20 tasks improved and 12 got worse. `pass^4` rose by
0.047, but its interval, −0.007 to +0.100, contains zero. It is one run.

The obvious objection to the first run was that it barely moved the model.
Measuring the weight shift confirmed it, so the run was repeated at ten times the
rate. That moved the weights about 11 times as far. On dev the two runs peaked
within a point of each other (0.5025 and 0.51). On test the higher rate gained
3.5 points over SFT, and the lower rate under one.

**Three measurements describe how little signal the training had.** None was
tested as a cause.

| Measurement | Value | What it means |
| :-- | :-- | :-- |
| Problems with no gradient | 65% of problems at 1e-6, 66% at 1e-5 | Each step scored 8 attempts at each of 2 problems. For these problems all 8 attempts scored alike, so there was nothing to compare. An earlier version reported 23% and 27% of whole steps, because the logger was not told the group size. |
| Reward spread within a problem | Accuracy 0.150, format 0.002, efficiency 0.001, gate 0.000 | Accuracy dominated; format and efficiency varied negligibly; the gate term was inert, because this task has one harmless tool and no gate can fire. |
| What was left to fix | See §2 | After SFT almost every failure is a well-formed call with the wrong value. GRPO can learn from a problem only when some of its attempts get it right. |

At 1e-6 this is a null **at this budget, on this task, from this starting
point**. At 1e-5 it is one run with a small gain. Neither is evidence about
reinforcement learning for tool-calling models in general. The known fix for
problems with no gradient, discarding zero-variance groups and refilling the
batch, is built but has not been run.

One direction is worth chasing but is not yet a result. The higher rate raised
`pass^4` and lowered `pass@4`, narrowing the band of sometimes-solved tasks from
0.213 to 0.160. That is what a policy-gradient method concentrating probability
mass looks like, and it is the trade this project cares about. A paired
permutation test on the per-task band width gives p = 0.17 †, so it is still a
hint. That is a different test from the `pass^k` comparisons recorded in
[`results/grpo-lr1e5-vs-sft-91a2de9.json`](results/grpo-lr1e5-vs-sft-91a2de9.json),
which report p = 0.04 for `pass^1` and p = 0.15 for `pass^4`.

---

## 4. Capability outran reliability

This is the headline, and the second half matters more than the first. All at R0.

| Metric | Untrained | Fine-tuned | Gain |
| :-- | --: | --: | --: |
| Solves it at least once in 4 (`pass@4`) | 0.353 | 0.627 – 0.680 | **+0.27 to +0.33** |
| Solves it 4 times out of 4 (`pass^4`) | 0.247 | 0.393 – 0.460 | **+0.15 to +0.21** |
| Band solved *sometimes* but not always | 0.107 | 0.213 – 0.287 | **wider** |

Training was meant to close the reliability gap. It widened it.

Reported as `pass@4` this project could claim +27 to +33 points. Reported
strictly, where every attempt has to land, it is +15 to +21. Most of what looks
like progress is capability. Reliability is the part that did not keep up, and it
is the part the project set out to measure.

---

## 5. What did *not* change

| Claim someone might make | What the measurement says |
| :-- | :-- |
| "It got better at arithmetic." | No. Probed with no calculator at all: 64.0% before, 66.0% after. It got better at *writing the expression*. |
| "It learned to cheat the grader." | No. The reward pays the same for restating a remembered answer as for real work, deliberately, so the behaviour is measured rather than hidden. The rate fell from 3.0% untrained to 1.2% fine-tuned, and was 1.3% after RL †. |
| "More seeds would sharpen this." | Probably not. Across three runs, `pass^1` had a standard deviation of 0.019, while one run's interval is about 0.069 either side. Three runs make that standard deviation rough, but the gap is large: a bigger test split would likely buy more than more seeds. |
| **"It forgot things."** | **No detectable MMLU change on this 400-question sample.** No tool offered: 53.25% untrained (213/400), 53.0% fine-tuned (212/400). Paired difference −0.0025, 95% interval −0.048 to +0.045. 44 questions improved, 45 got worse. |
| **"It now calls tools at everything."** | **No.** On a benchmark offering no tools, every arm emitted a tool call on **0.0%** of questions. The habit is tied to being offered a tool, not to being asked a question. |

The knowledge result needs careful wording, because it is the first thing anyone
asks. There was no detectable MMLU change on this sample, which is not the same as
no change. The interval runs from −4.75 to +4.5 points. With 89 of 400 questions
changing answer, a test this size has about an 80% chance of detecting a change of
about 7 points, and less for anything smaller. What the sample does show cleanly
is that the tool-calling habit did not leak into contexts with no tools.

The comparison is frozen question by question in
[`results/utility-comparison-26ce399.json`](results/utility-comparison-26ce399.json).
An earlier version of this section reported +0.005 and 38/36. Those figures
came from an incomplete response file, and [`ERRATA.md`](ERRATA.md) explains how.
The version after that reported +0.0075 and 40/37, from runs made before the
model loader was fixed to load the pinned base revision
([`ERRATA.md`](ERRATA.md), E8).

One real behavioural change did show up. The fine-tuned model answers far more
briefly: 171 characters on average against the untrained model's 536. Terser,
with no detectable accuracy cost. It also runs out of token budget less often:
2.0% of answers hit the 320-token limit, against 8.25% untrained and 2.75% after
GRPO. The rates published earlier (40%, 16%, 20%) counted batch padding as
truncation ([`ERRATA.md`](ERRATA.md), E7).

---

## 6. Did it generalise, or specialise? Both, in different places

Every number above was measured on the task the model was trained for, which
cannot tell the two apart. So the checkpoints were run on a second environment
they had never seen: an order-support agent offered three tools, no arithmetic
anywhere, and no calculator. Half the requests should be completed and half
should be refused, because a model that learned "always call the writing tool"
scores 50% on a balanced set and 100% on a one-sided one.

The environment runs in audit mode, so a write the gates should stop still lands
and can be counted. It is single-shot: the model writes every tool call in one
turn, before any tool runs, and never sees a tool result.

| | Untrained 1.7B | After SFT | After GRPO |
| :-- | --: | --: | --: |
| `pass^1` | 0.493 | 0.515 | 0.530 |
| Completes a legitimate request (outcome) | **0.000** | **0.905** | **0.931** |
| Correctly refuses an unverified one | **1.000** | **0.115** | **0.118** |
| Calls any tool | 0.840 | 1.000 | 1.000 |
| Writes for an unverified customer (of 296 refusal episodes) | 0 | 262 | 261 |
| Verified writes that skipped the lookup (of 304 legitimate episodes) | 0 | 274 | 281 |
| Episodes that called the lookup tool (of 600) | 11 | 1 | 2 |
| Mean reward, as scored | +0.411 | **−0.370** | **−0.373** |
| Mean reward, if skipping the lookup were not penalised | +0.411 | +0.361 | +0.376 |

The last five rows come from
[`results/phase_b-gates-8d5825b.json`](results/phase_b-gates-8d5825b.json), which
replays all 1,800 episodes and reproduces every stored reward exactly.

**The capability transferred.** An untrained 1.7B completes none of these
requests. After fine-tuning on GSM8K and a calculator, it completes 90% of them
by outcome, in audit mode, with severe authorisation failures. It operated two of
the three offered tools, `authenticate` and `update_order`, in a domain with no
maths in it. That is the strongest evidence here that the training taught
something general.

**The judgement did not.** On requests it should refuse, it wrote anyway in 262
of 296 episodes, or 89%. Because the episode is single-shot, it emits the
verification call and the write together, so it never learns that verification
failed. It acts unconditionally. The untrained model scores about half by never
writing; the trained one scores about half by always writing.

**The mean reward needs a caveat.** As scored, the fine-tuned model earns less
than the untrained one: −0.370 against +0.411. But 274 of its 536 violations are
writes for a verified customer whose only fault is skipping `get_order`. The
prompt asks for verification and never asks for a lookup. Rescored as if only
verification were required, the fine-tuned model earns +0.361. That is still
below the untrained model's +0.411. So by either scoring, the fine-tuned model
earns less reward here than the untrained one. An earlier version said the
rescored model was above the untrained one. That no longer holds on the re-run
([`ERRATA.md`](ERRATA.md), E10). The change comes mostly from the untrained
model. Its reward rose from +0.286 to +0.411 on the re-run, partly because fewer
of its episodes called no tool at all: 96 of 600, against 148 before. The prompt
still never asks for the lookup, so a fair verdict still needs a re-run with a
prompt that does.

This follows from how the training data was built. Every example was one call
and done, and none of them had "do not call the tool" as the right answer, so an
unconditional policy fit the data perfectly. It is the same structural limit
recorded earlier: an environment that ends the episode on the first successful
call never teaches a model to read a result and decide.

GRPO, measured at the selected checkpoint-200 of the 1e-6 run, sits within three
points of SFT on every rate.

## How the numbers are kept honest

| Rule | How it is enforced |
| :-- | :-- |
| Accuracy comes from executing the tool | Never parsed from the model's prose |
| Results cannot be edited after the fact | Every result file is frozen by SHA-256 in [`results/artifact_manifest.json`](results/artifact_manifest.json); a test fails if one changes. Corrections go in [`ERRATA.md`](ERRATA.md), never in place |
| No train/test leakage | Splits are disjoint by task ID *and* by content hash |
| No checkpoint cherry-picking | The selection rule was written into the config before any dev number existed; the winner runs on test exactly once |
| Comparisons are paired | Task-level bootstrap intervals, paired permutation tests, exact sign test |
| Numbers without an artifact are marked | † means the number comes from local logs or configs that are not committed |

The tests need Python 3.11 or 3.12. Install the package, then run the suite the
way CI does, with the network blocked:

```bash
pip install -e .
python scripts/run_tests_offline.py -v
```

The editable install is not optional. Many test modules import `agent` and `env`
directly, so running one module on its own, such as
`python -m unittest tests.test_gates`, fails without it.

---

## Limits

- **Only the 1.7B was trained.** The untrained 4B teacher is shown as a reference
  and beats every arm. Whether training helps the 4B was not tested.
- **Size and pretraining are tangled.** The comparator is a Llama, not an 8B from
  the Qwen family. Some of the gap may be pretraining rather than scale.
- **The untrained baseline barely varies.** It produced four identical answers on
  82 of 150 tasks, so its `pass^4` partly collapses into `pass^1`. Restricted to
  tasks where both arms genuinely varied, the trained model's lead *widens*. The
  finding survives; the caveat is real.
- **The knowledge probe is 400 questions, not 14,042.** A stratified sample of
  MMLU. Its paired interval runs from −4.75 to +4.5 points, and it reliably
  detects only changes of about 7 points or more.
- **The untrained model more often names no answer.** It named no letter on 24
  of 400 questions, against 6 for the fine-tuned model, because it answers at
  length. It also hit the 320-token limit on 8.25% of questions, against 2.0%.
  Restricted to the 375 questions where both arms named a letter, the paired
  difference is −0.032 (95% interval −0.077 to +0.016) instead of −0.0025.
  Still no detectable change either way. Both readings are negative, but the
  size depends on how unreadable answers are counted, so the asymmetry is
  recorded.
- **The base model barely varies.** On the transfer environment it produced four
  identical answers on 75% of tasks, so its `pass^4` largely collapses into
  `pass^1`. It is a floor, not a competitor.
- **The lookup tool was almost never called.** 1 of 600 SFT episodes and 2 of 600
  GRPO episodes called `get_order`, and the prompt never asked for it. Section 6
  separates the violations this causes from writes for unverified customers.
- **Some evidence lives only on the author's machine.** Episode and response logs
  are not committed. Numbers that exist in no committed artifact are marked †.

[`FINDINGS.md`](FINDINGS.md) is the short version of what came out of this,
including the parts that went badly.

Licensed Apache-2.0. Models are Qwen3 (Apache-2.0) and Llama-3.1 (Llama Community
Licence); task data is GSM8K (MIT).
