# Implementation plan

This plan follows the milestones in `BLUEPRINT_v2.md`. Checked items must have
tests or an artifact in the repository; a design note alone does not count.

## M0 — reproducible reliability kernel

- [x] Canonical blueprint and append-only decision log
- [x] Initial repository/package skeleton
- [x] Normalized tool-call parser and Pydantic validation
- [x] Typed, JSON-serializable execution events
- [x] Shared gate engine with explicit audit and enforce modes
- [x] Execution-backed composite reward and adversarial gaming tests
- [x] Spawn-safe best-effort subprocess sandbox and escape/resource tests
- [x] Exact `pass^k` and `pass@k` estimators with known-answer tests
- [x] Versioned trajectory JSONL round trip
- [x] End-to-end CPU fixture and local check wrapper
- [ ] First GitHub Actions run green on Windows and Linux
- [x] Ten-to-fifteen-source verified literature scan
- [x] Immutable model/dataset registry and license provenance table
- [ ] Freeze the provisional Phase-A/M0 Windows smoke lock after all four
  checkpoints pass the import/template compatibility probes
- [ ] Record an immutable Phase-A/M0 environment manifest from a clean `.venv`
  recreation
- [x] Implement the bounded P6 rank-4 `q_proj`/`v_proj` LoRA microstep with
  exact P5-mask collation and adapter-disabled same-model reference checks
- [x] Retain the first negative Qwen3-1.7B compatibility artifact without
  treating it as model-quality or selection evidence
- [x] Rerun Qwen3-1.7B with exact local snapshot revision evidence and the
  project-owned training-only assistant-mask wrapper; retain the placement
  diagnostic artifact
- [x] Rerun Qwen3-1.7B after correcting empty device-map handling so P4 can
  execute while the independent P5 prefix result remains fail-closed
- [x] Execute P6 training compatibility on all four checkpoints. Both Qwen2.5
  checkpoints execute and pass P6; both Qwen3 checkpoints stop at the P5
  prefix gate, so P6 is correctly not reached for them
- [x] Run the comparable Qwen2.5-versus-Qwen3 P0-P6 smoke on one frozen lane
  and lock. Qwen2.5 {3B, 1.5B} is technically eligible on every probe; Qwen3
  {4B, 1.7B} fails only `prefix_preserved_after_tool_observation`, identically
  at both sizes
- [ ] Declare the intended release scope, then apply the license gate and
  record the bundle decision. Blocked on that declaration, not on measurement:
  the technically eligible bundle contains the non-commercial
  `Qwen/Qwen2.5-3B-Instruct`, while the Apache-2.0 Qwen3 bundle is currently
  ineligible
- [ ] Function-calling dataset decision after license caveats are resolved
- [ ] Repository license decision after the license table exists
- [ ] File Meta gated-model access requests

## M6 environment lane

- [ ] Create a separate TRL 1.8, no-Unsloth requirements input and lock for
  `environment_factory`
- [ ] Record a separate immutable M6 environment manifest
- [ ] Execute live multi-turn `environment_factory` compatibility checks; do
  not reuse Phase-A/M0 P5 serialization evidence

## M1 onward

See `BLUEPRINT_v2.md` for the full experimental matrix. Work does not proceed
to paid compute or headline evaluation until M0's reward-gaming tests are green.

## Open specification issues

These are tracked rather than silently guessed:

- Implement the correction event sidecar and matched branches specified in
  `SELF_CORRECTION_SPEC.md`.
- Implement the framework-neutral R0/R1 core and LangGraph parity fixtures in
  `RUNG_PROTOCOL.md`.
- Implement the frozen H1–H3 manifests, estimators, and paired inference in
  `HYPOTHESIS_PROTOCOL.md`.
- Validate that the Phase B model combination fits the stated GPU budget.
- Correct the reference-policy implementation after the SFT/LoRA smoke test.
- Decide whether the public release may depend on non-commercial model terms;
  apply that license filter before naming a Qwen bundle winner.
- Reconcile current TRL OpenEnv multi-turn support with the M6 backend choice
  inside the separate TRL 1.8 environment lane.
- Pin one tau2 repository, immutable revision, task set, simulator, and reward
  basis; never mix Sierra and Amazon-verified scores in one comparison.
