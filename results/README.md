# Results

Every number in the [README](../README.md) comes from a file in this folder.
Nothing here is edited after it is written.
`tests/test_docs_match_artifacts.py` recomputes the README tables from these
files and fails if they disagree.

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
| `grpo-run-*.json`, `grpo-selection-*.json` | GRPO on top of SFT run 3, at learning rates 1e-6 and 1e-5 |
| `grpo-test-*.json`, `grpo-vs-sft-*.json`, `grpo-lr1e5-vs-sft-*.json` | Both GRPO arms on test, paired against SFT |
| `weight-change-*.json` | How far GRPO moved the adapter from its SFT start |
| `contamination-*.json` | The same tasks with no calculator, before and after training |
| `utility-{base,sft,grpo}-*.json`, `utility-comparison-*.json` | MMLU with no tool offered, and the paired comparison question by question |
| `phase_b-*.json` | Transfer to an order-support agent with three unseen tools, and its gate failures |

Episode logs (`*.jsonl`) hold one row per attempt. They are large and are not
committed.

Some older artifacts hold absolute paths from the machine that wrote them.
Those files are frozen, so the paths stay. Read them as labels: the checkpoint
name at the end of each path is the part that matters. Newer scripts record
paths relative to the repository.
