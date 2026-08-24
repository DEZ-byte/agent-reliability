# Results

Every number in [`README.md`](../README.md) and [`FINDINGS.md`](../FINDINGS.md)
comes from a file in this folder. Nothing is retyped by hand, and nothing is
edited after it is written.

## What is here

**The index.** `artifact_manifest.json` records a SHA-256, a byte length and the
recording commit for every measurement file. If a file is missing from it, or a
hash has moved, the test suite fails.

**The measurements**, roughly in the order they were made:

| File pattern | What it holds |
| :-- | :-- |
| `baseline-phase_a-*.json` | Untrained Qwen3-1.7B and Qwen3-4B on the frozen splits |
| `sft-dataset-*.json`, `sft-run-*.json` | The training set that was built, and each of the three training runs |
| `sft-selection-*.json` | Every checkpoint's dev score, and which one the pinned rule picked |
| `sft-test-*.json`, `sft-comparison-*.json` | The dev winner on test, and the paired comparison against base |
| `grpo-run-*.json`, `grpo-test-*.json`, `grpo-vs-sft-*.json` | Both GRPO arms, at 1e-6 and at 1e-5 |
| `comparator-8b-*.json` | Llama-3.1-8B with retry scaffolding |
| `h1-comparison-*.json` | The headline comparison: trained 1.7B against the scaffolded 8B |
| `contamination-*.json` | The no-calculator probe, before and after training |
| `masking-verification-*.json` | Proof the training loss covered assistant tokens only |

**The stack checks.** `smoke_environment.json` records the installed packages,
the CUDA device and a hash of every source file the probe depends on. It refuses
to write anything while the Git tree is dirty, so a record always corresponds to
committed code. The `model_smoke-*.json` files are compatibility runs from
before any measurement existed; several are failures and they are kept on
purpose.

Episode logs (`*.jsonl`) hold one row per attempt and are not committed. They
are large, and they are reproducible from the artifact that references them.

## Why the failures are still here

A measurement record is never edited or deleted, including when it is
unflattering. Three examples:

`model_smoke-qwen3-1.7b-6824196.json` is the first real attempt. Revision
validation and assistant masking both failed. Keeping it is what makes the later
success meaningful.

`model_smoke-qwen3-1.7b-3e2522f.json` recorded a *false* failure: the placement
check rejected an empty Unsloth device map even though every parameter was on
`cuda:0` with no offload. That artifact is the evidence that motivated fixing the
check, and it stays as it was written.

`grpo-run-qwen3-1.7b-d5b5c6d.json` is a null result. GRPO on top of SFT moved
`pass^1` by 0.002. The run at ten times the learning rate next to it moved it by
0.010, on an interval spanning zero. Both are kept, and both are reported.

## Reading a status honestly

A P5 probe can report `passed_with_demoted_gates`. That is **not** a pass under
the pre-registered rule. It means the probe cleared its hard gates only because
`prefix_preserved_after_tool_observation` was re-scoped to a diagnostic by D-046
(2026-08-18), and `passed_under_preregistered_p5_rule` in the same file says
`false`.

Artifacts written before D-046 recorded a genuine hard failure under the stronger
rule. They are never reinterpreted as passes. Their `config_sha256` differs from
every later artifact, so the two evidence regimes can be told apart by hash
alone.

Never write "Qwen3 passed P1–P6" without the qualifier.

## Rule for anything generated from these files

Every table or plot must be produced from the versioned logs, and every number
shown must link back to the artifact it came from.
