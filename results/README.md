# Results

The results in the [README](../README.md) come from the files in this folder.
Files here are not edited in place. A file that has to change is retired and
added again under a new manifest entry.
`tests/test_docs_match_artifacts.py` rebuilds the README's tables and numbers
from these files and fails if they disagree.

`artifact_manifest.json` records a SHA-256, a byte length and the recording
commit for every file. A test fails if a file changes, or if a file is missing
from the index; `python scripts/analyze.py verify-manifest` runs the same
check. A file that a later run replaced is removed only with
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
| `phase_b-*.json` | Single-turn transfer to an order-support agent with three tools that were not in training, its gate failures, and a split by request template |
| `token-counts-*.json` | Generated tokens and cost per attempt for every model row in the README, the parameter counts behind the cost, and a check of the token rule against every recorded count |
| `order-support-multiturn-*.json` | The closed-loop multi-turn order-support run. No such run exists yet. |

Episode logs (`*.jsonl`) hold one row per attempt. They are not in git. The
logs these files rest on are attached to the
[v1.0.0 release](https://github.com/DEZ-byte/agent-reliability/releases/tag/v1.0.0),
with their SHA-256. `failure-modes-*.json`, `phase_b-gates-*.json`,
`phase_b-templates-*.json`, `utility-comparison-*.json` and
`token-counts-*.json` record the SHA-256 of every log they read.

## Commands

Every frozen artifact was produced by a script under its old name, at the
commit the artifact records. Those scripts now run as subcommands of four CLIs
in `scripts/`. Each subcommand takes the same flags as the script it replaces;
`order-support` adds `--mode` and `--max-turns`, and their defaults give the
single-turn run.

| Old script | Now |
| :-- | :-- |
| `generate_sft_trajectories.py` | `data.py generate-teacher` |
| `build_sft_dataset.py` | `data.py build-sft` |
| `build_phase_a_splits.py` | `data.py split-gsm8k` |
| `build_phase_b_split.py` | `data.py split-order-support` |
| `build_utility_split.py` | `data.py split-mmlu` |
| `train_sft.py` | `train.py sft` |
| `train_grpo.py` | `train.py grpo` |
| `select_checkpoint.py` | `train.py select-checkpoint` |
| `verify_masking.py` | `train.py verify-masking` |
| `measure_weight_change.py` | `train.py weight-change` |
| `run_phase_a_baseline.py` | `evaluate.py gsm8k` |
| `run_phase_b_eval.py` | `evaluate.py order-support` |
| `run_utility_eval.py` | `evaluate.py mmlu` |
| `probe_contamination.py` | `evaluate.py contamination` |
| `compare_arms.py` | `analyze.py compare-models` |
| `compare_utility.py` | `analyze.py compare-mmlu` |
| `analyse_phase_a_failures.py` | `analyze.py failure-taxonomy` |
| `analyse_phase_b_gates.py` | `analyze.py order-support-gates` |
| `analyse_phase_b_templates.py` | `analyze.py order-support-templates` |

The two CPU analysis files were made at commit `2f70608` by the old scripts
`analyse_phase_a_failures.py` and `analyse_phase_b_templates.py`, on a clean
tree, with the release logs unpacked into `results/`. Re-running those scripts
at `2f70608` changes only `created_at_utc`. The commands below are the current
equivalents. They record the commit they run at as `source_commit`, so a run
at a later commit also differs there. A new file makes the tree dirty, so move
the first output aside before running the second command.

```bash
python scripts/analyze.py failure-taxonomy --output results/failure-modes-2f70608.json \
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
python scripts/analyze.py order-support-templates --output results/phase_b-templates-2f70608.json \
  --gates results/phase_b-gates-8d5825b.json \
  --arm base results/phase_b-base-10f9d97.json results/episodes-phase_b-base-10f9d97.jsonl \
  --arm sft results/phase_b-sft-aa59330.json results/episodes-phase_b-sft-aa59330.jsonl \
  --arm grpo results/phase_b-grpo-42e347c.json results/episodes-phase_b-grpo-42e347c.jsonl
```

The token counts need a clean tree, the cached tokenizers at their pinned
revisions, and the release logs in `results/`. The figures in `assets/` are
drawn from the frozen files and the token counts, and need matplotlib.

```bash
HF_HUB_OFFLINE=1 python scripts/analyze.py backfill-tokens
python scripts/analyze.py plots
```

## Paths inside artifacts

Five checkpoint-selection records,
`sft-selection-qwen3-1.7b-72ff036.json`,
`sft-selection-qwen3-1.7b-seed20260823-0abb401.json`,
`sft-selection-qwen3-1.7b-seed20260824-fbb9cd4.json`,
`grpo-selection-qwen3-1.7b-7be7eab.json` and
`grpo-selection-lr1e5-cc1841e.json`, were retired and added again with their
`artifact` fields changed from absolute local temporary paths to relative
labels such as `selection\select-checkpoint-75.json`. No other byte changed,
and Git history holds the original bytes.

The SFT and GRPO run records still hold the absolute adapter directory of the
machine that wrote them. Those files are frozen, so the paths stay. Read them
as labels: the checkpoint name at the end of each path is the part that
matters. Newer commands record paths relative to the repository.
