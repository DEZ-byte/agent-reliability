# Agent Reliability

Does fine-tuning a small model beat wrapping a bigger one in retry logic?

A model that solves a task once isn't reliable. This project measures `pass^k`,
the fraction of tasks where all k independent attempts succeed, because that is
what "it works" has to mean when something downstream depends on it.

## Result

Qwen3-1.7B, fine-tuned on 684 verified tool-use trajectories written by
Qwen3-4B. Evaluated on 150 held-out tasks, touched once. Two independent
training runs.

| Metric | Base | Trained | Change |
| :-- | --: | --: | --: |
| Solves it once (`pass^1`) | 0.303 | 0.515–0.525 | +0.21 to +0.22 |
| Solves it 4 times out of 4 (`pass^4`) | 0.247 | 0.393–0.413 | +0.15 to +0.17 |
| Solves it at least once in 4 (`pass@4`) | 0.353 | 0.627–0.680 | +0.27 to +0.33 |
| Tokens generated per task | 45.9 | 35.8 | 0.78× |

Paired against the same tasks, `pass^1` rose by 0.222 (95% CI 0.155 to 0.290)
on the first run and 0.212 (0.143 to 0.280) on the second.

It also got cheaper. The trained model spends about a fifth fewer tokens per
task while answering 21 points more of them correctly, which is roughly what
you would expect once the capability lives in the weights instead of being
bought back at inference time.

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
