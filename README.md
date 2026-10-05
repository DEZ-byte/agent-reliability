# Agent Reliability

[![ci](https://github.com/DEZ-byte/agent-reliability/actions/workflows/ci.yml/badge.svg)](https://github.com/DEZ-byte/agent-reliability/actions/workflows/ci.yml)

**Can fine-tuning make a small model call tools more reliably than a bigger
model wrapped in retry logic?**

Qwen3-1.7B is fine-tuned to solve GSM8K word problems by calling a calculator.
Every call is executed, and an attempt counts only if the executed answer is
right. Reliability is `pass^k`: the share of tasks where all k independent
attempts succeed. Solving a task once is not enough.

## Result

150 held-out test tasks, 4 attempts per task. If the first output does not give
a working calculator call, the model sees the error and gets one retry (rung R1).

| Model | `pass^1` | `pass^4` | Cost per attempt |
| :-- | --: | --: | --: |
| Qwen3-1.7B, untrained | 0.333 | 0.287 | not recorded |
| Llama-3.1-8B + retry | 0.415 | 0.293 | 236 |
| **Qwen3-1.7B, SFT** | **0.517 – 0.553** | **0.393 – 0.460** | **73 – 75** |
| Qwen3-1.7B, SFT + GRPO | 0.562 – 0.588 | 0.467 – 0.507 | 74 – 75 |
| Qwen3-4B, untrained (the teacher) | 0.608 | 0.567 | not recorded |

The SFT range spans three training runs with different seeds. The GRPO range
spans two learning rates, one run each, both from SFT run 3.

Cost is generated tokens per attempt times parameters in billions, from the
model configs: 2.03 for Qwen3-1.7B and 8.03 for Llama-3.1-8B. The 2.03 counts
Qwen3's tied embeddings twice, which makes the 1.7B look dearer. By that measure
the fine-tuned model uses about 31% of the 8B's cost. On raw token counts the
8B is cheaper, because it writes fewer tokens per attempt. SFT run 1 did not
record token counts.

## Key results

**1. SFT beats the scaffolded 8B.** Paired on the same tasks, at R1:

| SFT run | `pass^1` gain | 95% interval | `pass^4` gain | 95% interval |
| :-- | --: | :--: | --: | :--: |
| Run 1 | +0.115 | 0.045 – 0.185 | +0.100 | 0.027 – 0.173 |
| Run 2 | +0.102 | 0.027 – 0.178 | +0.120 | 0.033 – 0.207 |
| Run 3 | +0.138 | 0.063 – 0.215 | +0.167 | 0.080 – 0.253 |

Every interval excludes zero.

**2. SFT does not reach its teacher.** Against the untrained Qwen3-4B that wrote
its training data, the `pass^4` gap is −0.173 to −0.107 across the three runs,
and every interval excludes zero. The `pass^1` gap is smaller, −0.092 to
−0.055, and run 3's interval includes zero.

**3. Capability grew faster than reliability.** Single attempts (R0):

| Metric | Untrained | SFT | Gain |
| :-- | --: | --: | --: |
| Solved at least once in 4 (`pass@4`) | 0.353 | 0.627 – 0.680 | +0.27 to +0.33 |
| Solved all 4 times (`pass^4`) | 0.247 | 0.393 – 0.460 | +0.15 to +0.21 |
| Solved only sometimes | 0.107 | 0.213 – 0.287 | wider |

Reported as `pass@4`, training gained up to 33 points. Reported strictly, it
gained 15 to 21.

**4. GRPO added little.** 400 steps from SFT run 3, with rewards from executed
calls. Dev selection picked step 200 at 1e-6 and step 300 at 1e-5. Change
against SFT, at R0:

| Learning rate | `pass^1` change | 95% interval | Problems with no gradient |
| :-- | --: | :--: | --: |
| 1e-6 | +0.008 | +0.000 to +0.017 | 65% |
| 1e-5 | +0.035 | +0.005 to +0.068 | 66% |

At 1e-5 the interval excludes zero, but the sign test does not agree
(p = 0.22), and the `pass^4` interval (−0.007 to +0.100) includes zero. It is
one run. A problem gives no gradient when all 8 of its attempts score the same,
so GRPO has nothing to compare.

**5. No detectable forgetting.** On 400 MMLU questions with no tool offered:
53.25% untrained (213/400), 53.0% after SFT (212/400). Paired difference
−0.0025, 95% interval −0.048 to +0.045. None of the three 1.7B checkpoints
called a tool on any question. The sample is small: it cannot rule out a change
of up to about 5 points either way.

**6. The skill transferred; the judgement did not.** The same checkpoints ran
an order-support agent: three tools that were not in training, and no
arithmetic. There are 150 requests, run 4 times each. About half should be
completed and half refused. The grader runs in audit mode: a write that breaks
a tool rule still happens, and is counted.

| | Untrained 1.7B | SFT run 3 | GRPO 1e-6 |
| :-- | --: | --: | --: |
| `pass^1` | 0.493 | 0.515 | 0.530 |
| Completes a legitimate request | 0.000 | 0.905 | 0.931 |
| Correctly refuses an unverified one | 1.000 | 0.115 | 0.118 |

After SFT the model completes most legitimate requests in a domain with no
maths. But 274 of its 275 completions skipped the order lookup that the tool
rules require. The prompt never asked for that lookup. It also writes for
unverified customers in 262 of 296 refusal episodes. Every training example was
one call and done, and none had "do not call" as the right answer.

## How it works

- **Tasks.** GSM8K: 1,000 train, 100 dev and 150 test tasks, disjoint by ID and
  by content hash. One tool, a calculator that runs in a sandbox.
- **SFT.** The untrained Qwen3-4B wrote 4,000 attempts at the train tasks.
  2,572 executed to the right answer. 21 of those only restated the answer
  instead of computing it, and were dropped. One per task was kept: 684
  examples. LoRA on Qwen3-1.7B, with the loss on assistant tokens only.
- **GRPO.** 8 attempts per problem. The reward comes from executing each call:
  accuracy, format, efficiency, and a penalty for breaking a tool rule.
- **Selection.** The checkpoint rule was fixed in the config before any
  checkpoint was scored. Checkpoints are scored on dev only. The winner runs on
  test once.
- **Statistics.** Every comparison is paired by task: a 95% task-level
  bootstrap interval, plus permutation and exact sign tests.
- **Frozen results.** Every file in [`results/`](results/README.md) is pinned by
  SHA-256 in [`results/artifact_manifest.json`](results/artifact_manifest.json).
  CI fails if one changes. A test checks the result numbers on this page
  against those files.

## How to run

The tests run on CPU with Python 3.11 or 3.12, as CI runs them:

```bash
pip install -r requirements.lock
pip install --no-deps -e .
python scripts/build_artifact_manifest.py --check
python scripts/run_tests_offline.py -v
```

The GPU runs used Windows, CUDA 12.8 and `requirements-gpu.lock`. The order is
`generate_sft_trajectories.py`, `build_sft_dataset.py`, `train_sft.py`,
`select_checkpoint.py`, `run_phase_a_baseline.py`, then `compare_arms.py`.
GRPO repeats the last three steps after `train_grpo.py`.
`run_phase_b_eval.py` and `analyse_phase_b_gates.py` give the transfer results.
`run_utility_eval.py` and `compare_utility.py` give the MMLU results. Each script lists its flags with `--help`. A measured run refuses to
start on a dirty Git tree, so every result names the exact code that made it.

## Limits

- Only the 1.7B was trained. The untrained 4B teacher still has the best
  `pass^1` and `pass^4` of any model here.
- The comparator is a Llama, not a larger Qwen, so some of the gap may come
  from pretraining rather than size.
- One task family and 150 test tasks. GRPO has one run per learning rate.
- The untrained 1.7B gave four identical answers on 82 of 150 test tasks, so
  its `pass^4` sits close to its `pass^1`.
- MMLU is a 400-question sample. On the 375 questions where both models named
  an answer, the difference is −0.032 (95% interval −0.077 to +0.016). That is
  still no detectable change, but the interval reaches almost 8 points down.
- The transfer test is single-shot: the model never sees a tool result. Its
  prompt never asks for the lookup tool, and the models almost never call it.
- Episode logs are not committed. They are large.

## Licence

Code: Apache-2.0 ([`LICENSE`](LICENSE)). Models: Qwen3-1.7B and Qwen3-4B
(Apache-2.0); Llama-3.1-8B-Instruct (Llama 3.1 Community License), used for
evaluation only. Data: GSM8K (MIT) and MMLU (MIT).
