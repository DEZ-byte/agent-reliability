# Results

The results in the [README](../README.md) come from the files in this folder.
Nothing here is edited after it is written.
`tests/test_docs_match_artifacts.py` checks the README's result numbers against
these files and fails if they disagree.

`artifact_manifest.json` records a SHA-256, a byte length and the recording
commit for every file. A test fails if a file changes, or if a file is missing
from the index. A file that a later run replaced is removed only with
`scripts/build_artifact_manifest.py --retire`, and Git history keeps it.

| Files | What they hold |
| :-- | :-- |
| `baseline-phase_a-*.json` | Untrained Qwen3-1.7B and Qwen3-4B, on the dev and test splits |
| `sft-candidates-*.json`, `sft-dataset-*.json` | Qwen3-4B teacher trajectories, and the SFT set kept from them |
| `masking-verification-*.json` | Proof that the training loss covered assistant tokens only |
| `sft-run-*.json`, `sft-selection-*.json` | The three SFT runs, and the dev score of every checkpoint |
| `sft-test-*.json`, `sft-comparison-*.json` | Each dev-selected SFT checkpoint on test, paired against the untrained 1.7B |
| `sft-vs-teacher-*.json` | Each SFT run paired against the Qwen3-4B teacher |
| `comparator-8b-*.json` | Llama-3.1-8B with retry scaffolding, on test |
| `h1-comparison-*.json` | The headline: each SFT run against the scaffolded 8B |
| `failure-modes-*.json` | Every failed test attempt, by how it failed, and how often the R1 retry rescued one |
| `grpo-run-*.json`, `grpo-selection-*.json` | GRPO on top of SFT run 3, at learning rates 1e-6 and 1e-5 |
| `grpo-test-*.json`, `grpo-vs-sft-*.json`, `grpo-lr1e5-vs-sft-*.json` | Both GRPO arms on test, paired against SFT |
| `weight-change-*.json` | How far GRPO moved the adapter from its SFT start |
| `contamination-*.json` | The same tasks with no calculator, before and after training |
| `utility-{base,sft,grpo}-*.json`, `utility-comparison-*.json` | MMLU with no tool offered, and the paired comparison question by question |
| `phase_b-*.json` | Transfer to an order-support agent with three tools that were not in training, its gate failures, and a split by request template |

Episode logs (`*.jsonl`) hold one row per attempt. They are not in git. The
logs these files rest on are attached to the
[v1.0.0 release](https://github.com/DEZ-byte/agent-reliability/releases/tag/v1.0.0),
with their SHA-256. `failure-modes-*.json`, `phase_b-gates-*.json`,
`phase_b-templates-*.json` and `utility-comparison-*.json` record the SHA-256 of
every log they read.

The two analysis files were made at commit `2f70608`, on a clean tree, with the
release logs unpacked into `results/`. Only `created_at_utc` differs on a re-run.
A new file makes the tree dirty, so move the first output aside before running
the second command.

```bash
python scripts/analyse_phase_a_failures.py --output results/failure-modes-2f70608.json \
  --arm base-1.7b results/baseline-phase_a-3cc174f.json results/episodes-phase_a-3cc174f.jsonl Qwen/Qwen3-1.7B \
  --arm teacher-4b results/baseline-phase_a-3cc174f.json results/episodes-phase_a-3cc174f.jsonl Qwen/Qwen3-4B \
  --arm sft-run1 results/sft-test-qwen3-1.7b-c16530e.json results/episodes-sft-test-qwen3-1.7b-c16530e.jsonl Qwen/Qwen3-1.7B \
  --arm sft-run2 results/sft-test-qwen3-1.7b-seed20260823-67488ea.json results/episodes-sft-test-qwen3-1.7b-seed20260823-67488ea.jsonl Qwen/Qwen3-1.7B \
  --arm sft-run3 results/sft-test-qwen3-1.7b-seed20260824-51f16bf.json results/episodes-sft-test-qwen3-1.7b-seed20260824-51f16bf.jsonl Qwen/Qwen3-1.7B \
  --arm grpo-1e-6 results/grpo-test-qwen3-1.7b-f6138ee.json results/episodes-grpo-test-qwen3-1.7b-f6138ee.jsonl Qwen/Qwen3-1.7B \
  --arm grpo-1e-5 results/grpo-test-lr1e5-ce5f2ca.json results/episodes-grpo-test-lr1e5-ce5f2ca.jsonl Qwen/Qwen3-1.7B \
  --arm llama-8b results/comparator-8b-2889b6d.json results/episodes-comparator-8b-2889b6d.jsonl meta-llama/Llama-3.1-8B-Instruct
```

```bash
python scripts/analyse_phase_b_templates.py --output results/phase_b-templates-2f70608.json \
  --gates results/phase_b-gates-8d5825b.json \
  --arm base results/phase_b-base-10f9d97.json results/episodes-phase_b-base-10f9d97.jsonl \
  --arm sft results/phase_b-sft-aa59330.json results/episodes-phase_b-sft-aa59330.jsonl \
  --arm grpo results/phase_b-grpo-42e347c.json results/episodes-phase_b-grpo-42e347c.jsonl
```

Some older artifacts hold absolute paths from the machine that wrote them.
Those files are frozen, so the paths stay. Read them as labels: the checkpoint
name at the end of each path is the part that matters. Newer scripts record
paths relative to the repository.
