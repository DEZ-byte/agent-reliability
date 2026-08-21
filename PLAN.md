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
- [x] Function-calling dataset decision: `Salesforce/xlam-function-calling-60k`
  adopted, Glaive rejected (D-058). The licence conflict is recorded as accepted
  rather than resolved, and the CC BY 4.0 attribution duties are in the
  `data/LICENSES.md` release checklist
- [x] Confirm at M1 whether any format-grounding data is needed at all. It is
  not (D-070). Measured on the dev split, 800 first decisions per model, the
  schema failure rate is 0.00% on both selected checkpoints, so
  `format_grounding.fraction` is pinned to 0.0 and no xLAM rows are mixed in
- [x] Repository license decision: Apache-2.0 (D-054). `LICENSE` is the
  canonical text; `NOTICE` records that it covers this repository only and that
  every upstream artifact keeps its own terms
- [x] File Meta gated-model access requests (submitted by the owner)
- [x] Meta approval received for `meta-llama/Llama-3.2-3B-Instruct` and
  `meta-llama/Llama-3.1-8B-Instruct`. Verified 2026-08-20 through the
  authenticated Hub API: both repositories resolve, and each pinned revision in
  `configs/model_candidates.json` matches the resolved commit exactly. Weights
  are not downloaded yet; that is M1 work

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

## M1 — Phase A baselines

- [x] Phase A calculator environment with execution-backed grading (D-060)
- [x] GSM8K pinned and train/dev/test manifests frozen (D-061)
- [x] Memorisation and laundered-recall detection (D-062, D-063)
- [x] Contamination measured on the selected bundle (D-064). Recall is at
  chance; unaided solve rate is 64-71%, so Phase A measures tool discipline
  rather than arithmetic
- [ ] Contamination probe on the two Llama checkpoints, once their weights are
  downloaded
- [x] R0 and R1 baseline runner producing pass^k with confidence intervals
- [ ] Build the Stage 2 gate-bearing environment (M3b). Until it exists, the
  §7.4 ablation ladder and both H2 configurations have no data on which the
  gate term can vary, and H2 is unanswerable rather than negative
- [x] First baseline table, every accuracy figure reported beside its
  no-arithmetic rate (D-064). Recorded in D-068: the R1 rung is not
  distinguishable from R0 on the 4B, because 85-91% of failures are a
  well-formed call with wrong arithmetic, which a format retry cannot repair

## M2 — Phase A SFT

The measurement that shapes this stage: 85-91% of failures are a well-formed,
schema-valid tool call that computes the wrong thing (D-068). Format repair
cannot reach that, which is why the retained set is about the content of the
call rather than its shape.

- [x] Assistant-token-only loss masking, with guards for the failures that are
  silent (D-046 template patch shared into `src/training/`). An unmarked
  template returns an all-zero mask rather than raising, so the module refuses
  to return labels it cannot prove
- [x] Masking verified on the real checkpoints and frozen as an artifact. Both
  selected checkpoints train 24 tokens of 293, and a training row begins with
  exactly the token sequence the evaluator generates from
- [x] Laundering filter that decoration cannot pass (D-071). `391 + 0` parses
  as arithmetic and computes nothing
- [x] Split loader fixed so train tasks load from upstream train, and shared
  with the evaluator
- [x] Dev-split diagnostic config, so no training decision is derived from the
  split the checkpoint is scored on
- [x] `configs/train_config.yaml` pinned: pre-registered values from section
  7.4, measured values from dev, none left null (D-069, D-070, D-071)
- [x] Trajectory generation, dataset selection with a dev/test leakage guard,
  and a trainer that asserts the stack's silent traps shut
- [x] End-to-end pilot on a small train slice, so rented GPU hours are not
  spent debugging the pipeline. 40 train tasks through generation, selection,
  masking and training, producing a real adapter. It caught two defects before
  any rented hour: generation was using a function default instead of the
  pinned laundering threshold, and the first-batch check reported a mismatch on
  labels that were correct
- [x] Pilot evidence that the teacher is doing real work, not ceremony. The
  pilot trained the 1.7B on its own retained outputs and the loss started at
  0.045, because a model already assigns high probability to text it generated.
  Self-distillation has almost nothing to teach here, which is the measured
  version of the section 5.2 argument for a larger teacher
- [x] Teacher decision (D-072). Qwen3-4B teaches Qwen3-1.7B, both already
  pinned, both local, nothing rented. This deviates from §5.2's "32B class"
  teacher, so it trains the scale-check model only and does not answer H1 for
  the primary model
- [x] Generation over the 1,000-task train split with the 4B. 4,000 episodes,
  684 tasks yielding a usable trajectory, one row each
- [x] SFT run on the 1.7B, checkpoint selection on dev, then test exactly once
- [x] Report SFT against its own baseline, naming the teacher, with the
  no-arithmetic rate beside every accuracy figure (D-073)
- [ ] Re-run the token-starved recall probe on checkpoint-75. D-064 covers
  base weights only, so no post-training recall number exists and the word
  "recall" cannot be used of the trained arm
- [ ] A second SFT training seed. Until one exists, +0.222 is one run rather
  than an effect size, and this is the only open item that could change how
  the result is interpreted
- [ ] Per-episode generated-token accounting in the rung evaluator. Without
  it `TokenRatio_H1` is unmeasurable, so no cost side of H1 can be reported
- [ ] Report the strict laundering rate beside the harness one. The evaluator
  implements only the first of the project's three retention rules, so every
  published `no_arithmetic_rate` understates laundering by roughly half

Still open for the primary model: the pre-registered larger teacher.
`Qwen/Qwen3-32B` at `9216db5781bf21249d130ec9da846c4624c16137` and
`Qwen/Qwen3-14B` at `40c069824f4251a91eefaf281ebe4c544efd3e18` both resolve as
Apache-2.0 and ungated. They stay out of `configs/model_candidates.json` until
one is adopted, because `configs/model_smoke.json` pins the registry by SHA-256
as part of the resolved D-048 release gate, and re-pinning a resolved gate for
an unadopted model would weaken it for nothing. Until then a teacher runs with
an explicit `--revision`, which the generator requires rather than guessing.

Not in M2, and why: `r1_recovery` trajectories. Phase A ends an episode as soon
as a calculator call succeeds, so the dominant failure never reaches a second
decision, and revealing wrongness at runtime would leak the grader. Measured
yield is one task in 100 for the 4B. Self-correction data comes from Stage 2
(D-069).

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
- Reconcile current TRL OpenEnv multi-turn support with the M6 backend choice
  inside the separate TRL 1.8 environment lane.
- Pin one tau2 repository, immutable revision, task set, simulator, and reward
  basis; never mix Sierra and Amazon-verified scores in one comparison.
