# Agent Reliability

Does fine-tuning a small model beat wrapping a bigger one in retry logic?

A model that solves a task once isn't reliable. This project measures `pass^k`,
the fraction of tasks where all k independent attempts succeed, because that is
what "it works" has to mean when something downstream depends on it.

## The headline

A fine-tuned 1.7B beats an 8B with retry scaffolding, on the same 150 held-out
tasks, and costs about a third as much to run.

| | Llama-3.1-8B, scaffolded | Qwen3-1.7B, fine-tuned |
| :-- | --: | --: |
| Solves it once (`pass^1`) | 0.415 | **0.517–0.553** |
| Solves it 4 times out of 4 (`pass^4`) | 0.293 | **0.393–0.460** |
| Generation cost per task | 236 | **72** |
| Memory to serve, 4-bit | ~6 GB | **~1.5 GB** |

Reinforcement learning on top of the fine-tuned model added nothing: `pass^1`
0.5517 to 0.5533, a difference of 0.0017 with a 95% interval of ±0.013, and
`pass^4` identical to four decimal places. Details and the three measured
reasons are below.

Paired per training run, the trained model leads by 0.102 to 0.138 on `pass^1`
and 0.100 to 0.167 on `pass^4`. Every interval excludes zero.

Cost is parameter-weighted, in billion-parameter-tokens. The 8B actually emits
*fewer* tokens per task (29.4 against 35.6), but a token from an 8.03B model is
not a token from a 2.03B one. Compared on raw token counts the conclusion
reverses, which is why it is measured this way.

Retry buys the 8B almost nothing either: `pass^1` moves 0.412 to 0.415 and
`pass@4` does not move at all. A retry told only that nothing parsed cannot fix
a well-formed call that computed the wrong thing.

## How the small model got there

Qwen3-1.7B, fine-tuned on 684 verified tool-use trajectories written by
Qwen3-4B. Evaluated on 150 held-out tasks, touched once per run. Three
independent training runs.

| Metric | Base | Trained | Change |
| :-- | --: | --: | --: |
| Solves it once (`pass^1`) | 0.303 | 0.515–0.552 | +0.227 (SD 0.019) |
| Solves it 4 times out of 4 (`pass^4`) | 0.247 | 0.393–0.460 | +0.176 (SD 0.034) |
| Solves it at least once in 4 (`pass@4`) | 0.353 | 0.627–0.680 | +0.27 to +0.33 |
| Tokens generated per task | 45.9 | 35.4–35.8 | 0.78× |

Paired against the same tasks, `pass^1` rose by 0.222, 0.212 and 0.248 on the
three runs, with 95% intervals of roughly ±0.07 each.

Worth knowing: the spread between runs is small next to the uncertainty within
one. The standard deviation across seeds is 0.019, while a single run's interval
is about 0.069 wide either side. Training is more stable than 150 tasks can
resolve, which means more test tasks would buy more than more seeds.

It also got cheaper. The trained model spends about a fifth fewer tokens per
task while answering 21 points more of them correctly, which is roughly what
you would expect once the capability lives in the weights instead of being
bought back at inference time.

## Reinforcement learning added nothing here

400 steps of GRPO on an execution-backed reward, starting from the fine-tuned
model, moved `pass^1` by 0.0017 and `pass^4` by nothing at all. The interval is
±0.013, so this rules out anything larger than about a point rather than merely
failing to find an effect.

Three measurements say why, and they are more interesting than the null.

Nearly a quarter of the training steps carried no gradient. GRPO learns by
comparing eight attempts at the same problem; when all eight score alike the
comparison is empty. Starting from a policy that already works means it usually
agrees with itself.

Only one of the four reward terms varied. Within-group spread was 0.339 for
accuracy against 0.006 for format, 0.002 for efficiency and exactly 0.000 for
the gate. Three quarters of the reward was inert, so this optimised accuracy
alone whatever the config says.

And fine-tuning had already taken the reachable failures. It drove
never-called-the-tool from 36 episodes to 0 and malformed calls from 7 to 1.
What remains is a well-formed call that computes the wrong thing, which is a
reasoning limit rather than something a preference among eight samples reaches.

The obvious objection was that the run barely moved the model, and measuring it
confirmed that: the weights shifted 0.41%. Rerunning at ten times the learning
rate moved them 3.82% and produced an identical dev peak, with test `pass^1`
moving +0.010 on an interval spanning zero. Two nulls across a tenfold rate
range is a good deal harder to dismiss than one.

The higher rate did leave one direction worth chasing. It raised `pass^4` and
lowered `pass@4`, narrowing the sometimes-solved band from 0.213 to 0.167, which
is what a policy-gradient method concentrating probability mass looks like. A
paired test gives p=0.24, so it is a hint rather than a result.

This is a null at this budget, on this task, from this starting point. It is not
evidence that reinforcement learning cannot help tool-calling models.

**It also did not learn to cheat.** The reward pays exactly the same for a call
that restates a remembered answer as for real work, which was a deliberate
choice and is pinned by a test. Reinforcement learning is the sharpest possible
version of that pressure and had 400 steps to find the shortcut. The rate went
*down*: 1.00% against the fine-tuned model's 1.17% and the untrained model's
3.00%.

## The gap got wider, not narrower

Training was supposed to close the reliability gap. It widened it.

The band of tasks the model solves sometimes but not always went from 0.107 to
0.287. It now gets far more problems right on some attempt, and repeating that
performance every time lagged well behind.

Reporting `pass@4` would let me claim +33 points here. The number where all four
attempts have to land is +15. Most of what looks like progress is capability,
and reliability is the part that didn't keep up.

## Two things the measurements ruled out

It did not get better at arithmetic. Probed with no calculator at all, the model
scored 64.0% before training and 66.0% after. What improved is writing the right
calculator expression, not doing the maths.

It also didn't learn to cheat. A model can work out an answer in its head and
hand it to the calculator, scoring correct while computing nothing. Under a
filter that catches the decorated version of this trick (`391 + 0` parses as
arithmetic and computes nothing), the rate fell from 3.0% to 1.0%.

## What's still missing

This doesn't answer the headline question yet. That comparison needs an 8B
scaffolded model, which is registered in the config and has never been run.

The teacher was Qwen3-4B rather than the larger model originally planned, so
only the smaller model has a trained arm. Nothing here says anything about the
4B. Two training runs give a range rather than a variance estimate, so I'm not
quoting an interval on the effect size itself.

One caveat is worth stating plainly: the untrained model produced four identical
answers on 82 of the 150 tasks, which means its `pass^4` partly collapses into
`pass^1`. Restricting the comparison to tasks where both models genuinely varied
their output, the gap widens to +0.215. The finding survives, but the caveat is
real and it applies to the baseline numbers too.

## How the numbers are kept honest

Accuracy comes from executing the tool and reading the result, never from
reading the model's prose. Every result file is frozen by SHA-256 in
[`results/artifact_manifest.json`](results/artifact_manifest.json), so editing a
measurement after the fact fails a test. Training and evaluation tasks are
disjoint by ID and by content hash. The checkpoint was picked on a dev split
using a rule written down before any dev number existed.

```bash
python -m unittest discover -s tests
```

[`FINDINGS.md`](FINDINGS.md) is the short version of what came out of this,
including the parts that went badly.

Licensed Apache-2.0. Models are Qwen3 (Apache-2.0) and the task data is GSM8K
(MIT).
