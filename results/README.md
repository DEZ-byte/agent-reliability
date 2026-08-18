# Results

No benchmark or model-quality experiment has run. `smoke_environment.json` is
a model-free dependency, import, and CUDA compatibility record. It is not model
quality evidence.

`model_smoke-qwen3-1.7b-6824196.json` is the retained negative artifact from
the first real Qwen3-1.7B compatibility attempt. It records why immutable
revision validation and P5 masking did not pass under the original probes. It
does not select a model, and the corrected rerun remains pending. Future
attempts must use separate artifacts rather than overwrite this failure.

Generated experiment tables must be derived from versioned trajectory logs,
and every displayed measurement must link to its artifact.
