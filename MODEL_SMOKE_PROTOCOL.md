# Qwen model-selection smoke protocol

**Status:** pre-measurement protocol. No candidate has been selected and no
performance number has been recorded.

## Purpose

Choose the primary and scale-check Qwen generation using comparable evidence,
not model age or vendor claims. The same harness, prompts, decoding settings,
software lock, and named GPU must be used for every candidate.

## Current support evidence

- The [TRL quickstart](https://huggingface.co/docs/trl/quickstart) demonstrates
  GRPO with a Qwen2.5 Instruct checkpoint.
- The [TRL GRPO documentation](https://huggingface.co/docs/trl/v1.8.0/en/grpo_trainer)
  documents tool functions and multi-turn `environment_factory` rollouts. It
  lists Qwen2.5 and Qwen3 among tested families and requires
  `transformers>=5.2.0` for `environment_factory`.
- The [Unsloth repository](https://github.com/unslothai/unsloth) links an
  advanced Qwen3 GRPO notebook.

These links establish plausible support paths only. They are not project
measurements and cannot decide the winner.

## Candidate bundles and scored mode

Selection compares two generation bundles, not four independent winners:

- Qwen2.5: 3B primary plus 1.5B scale check;
- Qwen3: 4B primary plus 1.7B scale check.

One generation must pass for both roles so the scale arm remains a within-
generation check. Mixing Qwen2.5 and Qwen3 sizes requires a new dated decision.

The scored Qwen3 condition sets `enable_thinking=false`. This matches the
direct tool-call condition available to Qwen2.5 and avoids charging hidden
reasoning tokens to only one bundle. Qwen3 thinking mode may be reported as a
separate diagnostic, but it cannot decide this selection.

## Environment lifecycle

Compatibility reconnaissance may iterate dependencies without producing a
selection result. Once every checkpoint can reach the planned import/template
probes, freeze a provisional `requirements-smoke.lock` and environment
manifest. Recreate that environment from scratch, then run every measured
candidate under the same lock. The later training lock may extend or replace
the provisional smoke lock only after the model decision; it cannot rewrite
the recorded smoke environment.

## Reproducibility rules

1. Resolve and record the exact Hugging Face commit for each candidate.
2. Record Python, PyTorch, CUDA, Transformers, TRL, Unsloth, Accelerate, and
   bitsandbytes versions before loading a tokenizer or model.
3. Use one immutable config and one fixed set of prompts for every candidate.
4. Run two warm-up generations before timed generations.
5. Record raw JSON results even when a probe fails.
6. Never replace a failed metric with an estimate or vendor-published value.
7. Record the intended public-release scope and accepted upstream license
   constraints before naming a selected bundle.
8. Append the measured decision and artifact paths to `DECISIONS.md`.

## Probe ladder

### P0 — offline validation

- Validate the config schema and candidate roles.
- Print the operations that would require network or GPU access.
- Confirm that the default command performs no download.

### P1 — repository and tokenizer

- Resolve the repository and immutable commit.
- Load the tokenizer at that commit.
- Record tokenizer class, vocabulary size, special tokens, and chat-template
  hash. Record the native inference template and any TRL-patched training
  template as separate artifacts; never overwrite one with the other.
- Apply the native chat template to the same system/user messages and tool
  schema.
- Require a non-empty generation prompt and deterministic tokenization.
- Record rendered prompt text and token count for inspection.
- For a fixed assistant/tool/observation trajectory, record whether the native
  inference template preserves the tokenized prefix when the tool observation
  is appended. This is diagnostic: Qwen3's training-specific fix is evaluated
  separately in P5.

### P2 — tool-call template behavior

- Render at least one read-only and one mutative tool schema.
- Generate fixed tool-required prompts with identical decoding settings.
- Parse output through the repository's normalized parser adapter.
- Require exactly one parsed, registered, schema-valid call and require its
  name to match the prompt's expected tool.
- Record JSON parse rate, registered-schema validity, side-effect-free
  dispatchable-call rate, and zero-tool-call rate. “Dispatchable” means the
  smoke registry would accept the call; no handler runs and no gate event is
  claimed. Do not score prose as tool use.

### P3 — 4-bit load and memory

- Load each model with the same 4-bit quantization settings on the single
  configured CUDA device; multi-GPU sharding is a failure.
- Record GPU name, total VRAM, quantization settings, actual parameter dtype,
  peak allocated VRAM, peak reserved VRAM, and load duration.
- Require runtime evidence that NF4 4-bit quantization was actually applied,
  not merely requested in the loader arguments.
- A candidate fails this probe if it cannot complete on the named target GPU
  without CPU offload under the common settings.

### P4 — deterministic generation throughput

- Use the same prompts, seed, temperature, `top_p`, and maximum new tokens.
- Record prompt tokens and generated tokens separately.
- Report generated tokens per second after warm-up and the raw durations.
- Keep failures and partial outputs in the artifact.

### P5 — training-stack import smoke

- Import the pinned TRL and Unsloth paths used by the planned GRPO recipe.
- Construct configuration objects without starting a training run.
- Obtain the TRL training template through its public helper and record its
  hash separately from the model's native inference template.
- Render a multi-turn assistant/tool trajectory with
  `return_assistant_tokens_mask=True`. Compare the complete returned mask with
  independently derived `{% generation %}` spans; sentinel spot checks alone
  are insufficient.
- Verify prefix preservation before and after appending a tool observation.

### P6 — minimal training execution

- Run one bounded forward/backward or tiny GRPO step per checkpoint with the
  frozen stack; an import or config construction alone does not establish
  Unsloth/TRL training compatibility.
- Verify assistant-only loss masking and reference-policy handling from the
  executed batch before any full training run.
- Save loss finiteness, peak VRAM, effective batch/token counts, and any
  failure as raw result data. Do not use this microstep as a quality metric.

## Selection rule

First apply the release-license constraint recorded for the intended public
artifact. An ineligible bundle may retain its technical measurements but
cannot be selected. Each remaining generation bundle must pass P1–P6 for both
its primary and scale checkpoint. Rank eligible passing bundles in this order:

1. higher primary strict tool-call validity;
2. lower primary zero-tool-call rate on tool-required prompts;
3. higher scale-check strict validity, then lower scale zero-call rate;
4. lower primary, then scale-check, peak reserved VRAM;
5. higher primary, then scale-check, generated tokens per second.

Quality rates tie only when their exact numerator and denominator match.
Resource values within 2% for peak reserved VRAM or 5% for throughput are
treated as tied because this smoke is not a performance benchmark. If the rule
still ties, expand the fixed prompt set on development-only cases and rerun
both bundles; do not choose by release date or vendor claim.
