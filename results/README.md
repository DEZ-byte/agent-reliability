# Results

**No reliability experiment has run yet.** Nothing in this folder says how good
any model is at anything. These files record whether the software stack works,
which is a different question.

## What is here

`artifact_manifest.json` is the index. It records a SHA-256, a byte length, and
the recording commit for every measurement file. It is the authoritative list;
if a file is not in it, or its hash has moved, the test suite fails.

`smoke_environment.json` records the installed packages, the CUDA device, and a
hash of every source file the probe depends on. It refuses to record anything
while the Git tree is dirty, so a manifest always corresponds to committed code.

`model_smoke-<candidate>-<commit>.json` files are compatibility runs. The commit
suffix ties each one to the exact source that produced it. Several are failures,
kept on purpose.

## Why the failures are still here

A measurement record is never edited or deleted, including when it is
unflattering. Two examples:

`model_smoke-qwen3-1.7b-6824196.json` is the first real attempt. Revision
validation and assistant masking both failed. Keeping it is what makes the later
success meaningful.

`model_smoke-qwen3-1.7b-3e2522f.json` recorded a *false* failure: the placement
check rejected an empty Unsloth device map even though every parameter was on
`cuda:0` with no offload. That artifact is the evidence that motivated fixing
the check, and it stays as it was written.

`model_smoke-all-05b6450.json` is the run that matters most so far. All four
checkpoints, one lane, one lock, gate resolved.

## Reading a status honestly

A P5 probe can report `passed_with_demoted_gates`. That is **not** a pass under
the pre-registered rule. It means the probe cleared its hard gates only because
`prefix_preserved_after_tool_observation` was re-scoped to a diagnostic by D-046
(2026-08-18), and `passed_under_preregistered_p5_rule` in the same file says
`false`.

Artifacts written before D-046 recorded a genuine hard failure under the
stronger rule. They are never reinterpreted as passes. Their `config_sha256`
differs from every later artifact, so the two evidence regimes can be told apart
by hash alone.

Never write "Qwen3 passed P1-P6" without the qualifier.

## Rule for anything generated from these files

Every table or plot must be produced from the versioned logs, and every number
shown must link back to the artifact it came from.
