# Results

No benchmark or model-quality experiment has run. `smoke_environment.json` is
a model-free dependency, import, and CUDA compatibility record. It is not model
quality evidence.

`model_smoke-qwen3-1.7b-6824196.json` is the retained negative artifact from
the first real Qwen3-1.7B compatibility attempt. It records why immutable
revision validation and P5 masking did not pass under the original probes. It
does not select a model. Future attempts must use separate artifacts rather
than overwrite this failure.

`model_smoke-qwen3-1.7b-3e2522f.json` is the retained diagnostic from the next
attempt. Exact snapshot evidence and assistant masking passed, while the old
placement check rejected an empty Unsloth device map even though every actual
parameter was on `cuda:0`, no offload target was present, and runtime NF4
evidence passed. It is preserved as runner-debugging evidence; a corrected
placement rerun remains separate and pending.

Generated experiment tables must be derived from versioned trajectory logs,
and every displayed measurement must link to its artifact.

## Demoted gates

From D-046 (2026-08-18), a P5 probe may report `passed_with_demoted_gates`. That
status means the probe cleared its hard gates only because a pre-registered
check — `prefix_preserved_after_tool_observation` — was re-scoped to a Phase-A
diagnostic. It is not a P5 pass under the pre-registered eleven-check rule, and
`passed_under_preregistered_p5_rule` in the same artifact says so directly.

Artifacts written before D-046 recorded a genuine hard failure under the stronger
rule. They are retained unmodified and are never reinterpreted as passes; their
`config_sha256` differs from every post-D-046 artifact, so the two evidence
regimes are distinguishable by hash alone. Never write "Qwen3 passed P1-P6"
without the qualifier.
