# Qwen model-selection smoke protocol

**Status:** pre-measurement protocol. The current runner implements P0-P6. The
first Qwen3-1.7B attempt produced a retained negative compatibility artifact;
its corrected rerun remains pending, and P6 has not executed on any checkpoint.
All four checkpoint executions remain required before selection. No candidate
has been selected and no performance number has been recorded.

## Purpose

Choose the primary and scale-check Qwen generation using comparable evidence,
not model age or vendor claims. The same harness, prompts, decoding settings,
software lock, and named GPU must be used for every candidate.

## Current support evidence

- The Phase-A/M0 smoke lane is the Windows stack pinned by
  `requirements-smoke.in` and `requirements-smoke.lock`: Unsloth 2026.8.18,
  TRL 0.24.0, and Transformers 5.5.0. Its evidence applies only to that lane.
- The [TRL quickstart](https://huggingface.co/docs/trl/quickstart) demonstrates
  GRPO with a Qwen2.5 Instruct checkpoint.
- The [TRL GRPO documentation](https://huggingface.co/docs/trl/v1.8.0/en/grpo_trainer)
  documents tool functions and multi-turn `environment_factory` rollouts. It
  lists Qwen2.5 and Qwen3 among tested families and requires
  `transformers>=5.2.0` for `environment_factory`. That support belongs to a
  separate M6 environment with TRL 1.8 and no Unsloth; it is not evidence for
  the Phase-A/M0 stack.
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
selection result. Phase-A/M0 compatibility uses the Windows Unsloth 2026.8.18,
TRL 0.24.0, and Transformers 5.5.0 lane pinned by `requirements-smoke.in` and
`requirements-smoke.lock`. Once every checkpoint can reach the planned
import/template probes, freeze the provisional lock and environment manifest.
Recreate that environment from scratch, then run every measured candidate
under the same lock.

M6 `environment_factory` work uses a separate TRL 1.8 environment without
Unsloth. It requires its own input requirements, lock, manifest, and executed
evidence. Phase-A/M0 support cannot be transferred to M6, and M6 support cannot
be transferred back to the smoke selection. Neither lane may rewrite the
other lane's recorded environment.

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

- Resolve the repository and immutable commit. Prove the revision from the
  exact local Hugging Face `snapshots/<commit>/...` identity, using an offline
  cache lookup for a known repository file. If the loaded object also exposes
  revision metadata, require it to agree; do not require a private
  `_commit_hash` attribute to exist.
- Load the tokenizer at that commit.
- Record tokenizer class, vocabulary size, special tokens, and the resolved
  native chat-template hash. If a later lane introduces a separate training
  template, record it as a separate artifact; never overwrite either source.
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
- Score a prompt as a strict tool-call success only when it produces exactly
  one parsed, registered, schema-valid call whose name matches the expected
  tool. This is a measured success definition, not a compatibility hard gate.
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

### P5 — multi-message serialization and masking smoke

- Import the pinned TRL 0.24 and Unsloth 2026.8.18 paths used by the Phase-A/M0
  recipe.
- Construct configuration objects without starting a training run.
- Keep the exact native template resolved by the tokenizer returned from
  `FastLanguageModel.from_pretrained` as the unchanged inference template.
  Derive a separate project-owned, training-only template by wrapping the one
  unambiguous Qwen assistant branch in `{% generation %}` and
  `{% endgeneration %}`. Record both template hashes. Fail closed when the
  branch structure is missing, duplicated, already instrumented, or otherwise
  ambiguous.
- Require the native and training-only templates to render exactly the same
  text and exactly the same token IDs for the probe trajectory. The wrapper
  may change only assistant-mask attribution, never serialized model input.
- Render a multi-turn assistant/tool trajectory with
  `return_assistant_tokens_mask=True`. Compare the complete returned mask with
  independently derived `{% generation %}` spans; sentinel spot checks alone
  are insufficient.
- Verify prefix preservation before and after appending a tool observation.
- Import TRL's pinned `DataCollatorForLanguageModeling`; P6 must prove that it
  converts this exact P5 mask into the expected assistant-only labels.
- Treat this as serialization and masking evidence only. It does not execute a
  live multi-turn environment and does not validate the separate TRL 1.8 M6
  `environment_factory` lane.

### P6 — minimal training execution

- Run one bounded rank-4 LoRA microstep targeting `q_proj` and `v_proj` for
  each of the four checkpoints with the frozen Phase-A/M0 stack; an import or
  config construction alone does not establish Unsloth/TRL training
  compatibility.
- Reuse the exact P5 input IDs and assistant-token mask. Require the TRL
  collator to produce the expected assistant-only labels before the forward
  pass.
- Use the same PEFT model with its adapter disabled for the reference-policy
  calculation, and verify that this reference remains invariant across the
  adapter update.
- Execute one ephemeral SGD forward, backward, and optimizer step. Write no
  checkpoint and make no training-quality claim from this microstep.
- Save loss finiteness, peak VRAM, effective batch/token counts, and any
  failure as raw result data. Do not use this microstep as a quality metric.
- The P6 code path is implemented and covered by mock-only focused tests, but
  no checkpoint has executed it. All four real P6 artifacts are mandatory
  before a bundle can be selected.

The retained first Qwen3-1.7B artifact records the pre-correction P1/P3 private
metadata and P5 native-template failures. It is not overwritten by the
corrected rerun and is not model-quality evidence.

## Selection rule

First apply the release-license constraint recorded for the intended public
artifact. An ineligible bundle may retain its technical measurements but
cannot be selected. Each remaining generation bundle must pass P1–P6 for both
its primary and scale checkpoint.

The machine-readable `release_gate` remains pending with
`eligible_bundles=[]`. It pins the model registry by SHA-256. A future resolved
gate must match each bundle's registry-backed `release_eligibility`, cite one
recorded `D-###` decision, and reproduce that decision's exact release-scope
and eligible-bundle markers. Top-level selection stays ineligible even if all
four candidates pass technically until at least one complete bundle is
release-eligible.

Every P1-P5 probe result must complete successfully and every P6 must execute
and pass before the four-checkpoint comparison is selection-ready. The P2
strict tool-call and zero-call rates remain ranking observations inside the
successful deterministic-generation probe; they are not converted into an
unregistered threshold.

Rank eligible passing bundles in this order:

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
