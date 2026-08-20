# Implementation plan

Milestones come from `BLUEPRINT_v2.md`. A box is only ticked when there is a
test or a committed artifact behind it. Writing a design note does not count as
doing the work.

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
- [x] First GitHub Actions run green on Windows and Linux. All four jobs pass
  (ubuntu/windows x 3.11/3.12) at `8f8b086`. It took three runs: the first two
  exposed defects a green local suite had hidden — a lone surrogate compiled
  into `contains_surrogate`'s own docstring (D-056), and end-of-line translation
  breaking every recorded content hash away from the machine that wrote it
  (D-057). The POSIX rlimit test now executes on Linux for the first time
- [x] Ten-to-fifteen-source verified literature scan
- [x] Immutable model/dataset registry and license provenance table
- [x] Freeze the provisional Phase-A/M0 Windows smoke lock after all four
  checkpoints pass the import/template compatibility probes. All ten committed
  artifacts record one lock state (`c81df3a3`, expected == actual), and all four
  candidates were measured under it
- [x] Verify the lock resolves into a clean Python 3.12 environment. `uv pip
  install --dry-run` against `requirements-smoke.lock` resolves every pin,
  including the four direct-URL CUDA wheels, in a throwaway environment
- [ ] Record an immutable Phase-A/M0 environment manifest from a full clean
  `.venv` recreation and re-measure all four candidates under it. Deliberately
  not done unattended: it means deleting the working environment,
  re-downloading several GB, and rerunning every candidate. It should be a
  watched operation
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
  checkpoints execute and pass P6; both Qwen3 checkpoints stopped at the P5
  prefix gate under the pre-D-046 rule, so P6 was correctly not reached for them
- [x] Rerun both Qwen3 checkpoints under D-046 so P6 executes for the first time
  on Qwen3; write NEW artifacts and never overwrite the pre-D-046 ones
- [x] Run the comparable Qwen2.5-versus-Qwen3 P0-P6 smoke on one frozen lane
  and lock. Qwen2.5 {3B, 1.5B} is technically eligible on every probe; Qwen3
  {4B, 1.7B} fails only `prefix_preserved_after_tool_observation`, identically
  at both sizes
- [x] Declare the intended release scope, then apply the license gate and
  record the bundle decision. Scope is `public-portfolio-permissive`; the
  Apache-2.0 Qwen3 {4B, 1.7B} bundle is selected on licence, not on technical
  merit, with the D-046 demotion disclosed (D-048)
- [x] Record a single four-candidate artifact on one lane with the gate
  resolved. Top-level `selection_eligible` is true for the first time; the two
  Qwen3 candidates carry `passed_with_demoted_gates`
- [x] Accept the xLAM access gate (done by the owner)
- [ ] Function-calling dataset decision — **still open (D-055)**: access is
  granted, but xLAM's card declares CC BY 4.0 while its ethical section says
  "research purposes only", which conflicts with the declared
  `public-portfolio-permissive` scope for any published adapter trained on it.
  Resolve by publisher clarification, by not publishing derivatives, or by
  generating grounding data from an Apache-2.0 teacher
- [x] Repository license decision: Apache-2.0 (D-054). `LICENSE` is the
  canonical text; `NOTICE` records that it covers this repository only and that
  every upstream artifact keeps its own terms
- [x] File Meta gated-model access requests (submitted by the owner)
- [ ] Meta approval received for `meta-llama/Llama-3.2-3B-Instruct` and
  `meta-llama/Llama-3.1-8B-Instruct` — **waiting on Meta**, not on work here.
  Both are required before M1 baselines; the Qwen-only arms can proceed
  meanwhile

## M6 environment lane

- [ ] Re-arm and re-verify `prefix_preserved_after_tool_observation` as a HARD
  gate before any multi-turn work. The D-046 demotion is Stage-1-scoped and does
  not carry over, including to BLUEPRINT_v2 7.1 Stage 2, which sits inside
  Phase A but is multi-turn
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
