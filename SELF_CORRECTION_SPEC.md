# Self-correction measurement specification

**What this is for.** When an agent fails and then succeeds, it is tempting to
call that self-correction. Usually it is not: it may be retry luck, a gate that
blocked the mistake, or a bigger model taking over. This document defines what
would have to be true to call it self-correction, and keeps the other three
explanations separate.

**Status:** pre-registered design; no correction measurements exist yet.
**Scope:** secondary diagnostic analysis for existing small-model R1/R2
evaluation arms. An 8B parent cannot enter the 8B-handoff branch.
This document does not add or modify H1–H3, select checkpoints, or change any
headline `pass^k` result.

## 1. What “self-correction” means in this project

Self-correction is not an apology, a reflection string, or success on a later
sample. It is an observable behavior with four required parts:

1. The designated small model produces an action that fails for a
   runtime-observable reason.
2. The harness returns the exact structured failure observation to that same
   small model.
3. The same checkpoint and adapter produce a materially corrective next
   action from the shared failed prefix.
4. The event log proves whether the local action was repaired and whether the
   episode ultimately succeeded.

The project reports local action repair separately from end-to-end task
recovery. A valid next tool call can still lead to a wrong final state, and a
task can succeed on a second sample without using the error information.

The matched branch experiment in §3 is required before language such as
“feedback caused recovery” or “the model self-corrected more often” is used.
Production R1/R2 trajectories alone support only descriptive attribution.

## 2. Unit of analysis and correction opportunity

### 2.1 Unit

One unit is an existing evaluation episode identified by:

```text
(arm_id, task_id, run_idx, eval_seed)
```

The checkpoint, adapter, native inference template, prompt hashes, sampling
configuration, gate-policy fingerprint, environment revision, and the effective
`environment_turn_cap` defined in `RUNG_PROTOCOL.md` §1.1 (reference 20, or
`min(20, upstream_cap)` when an upstream benchmark binds lower)
are frozen by `arm_id`. A unit enters the correction analysis at most once.

### 2.2 Earliest eligible correction opportunity

For each valid episode, select the **earliest** event that meets every rule
below. Later failures remain in the trajectory but never create extra primary
units. This prevents a failure-prone episode from receiving more weight merely
because it failed many times.

An event is an eligible correction opportunity (`A_i = 1`) only when:

- the failing actor is the designated small model, before any 8B handoff;
- the failure is detected without consulting the final task grader;
- the model-facing error observation is deterministic, structured, and saved;
- the failing decision commits no state change between its decision-boundary
  `state_before` and `state_after` digests;
- at least one diagnostic-branch model decision and its permitted tool attempt
  remain under the branch's frozen model-decision and generation-token
  budgets; the common 20 environment-turn cap is checked separately and is
  never treated as a model-decision budget; and
- the exact history, environment state, budgets, and policy fingerprint can
  be restored for all matched branches.

Eligible runtime trigger classes are:

| `trigger_class` | Required evidence |
|---|---|
| `parse_error` | At least one normalized `ParseIssue`; no call from the failing decision dispatched. |
| `unknown_or_invalid_call` | A `ToolEvent` with `error_code` equal to `unknown_tool` or `schema_validation_error` and `dispatched=false`. |
| `tool_failure` | A dispatched `ToolEvent` with `succeeded=false`, including `tool_exception`, `invalid_tool_output`, or `invalid_tool_state`. A model-caused sandbox violation/timeout belongs here when surfaced as a tool error. |
| `gate_block` | A `GateEvent` has `action=enforce_block`, `blocked=true`, and the corresponding mutative call did not dispatch. |
| `loop_detected` | The frozen harness detects three identical consecutive normalized call signatures, as specified for R2. |

The serialized branch point contains:

- `H_i^-`: model-visible history immediately before the failed small-model
  decision;
- `a_i^0`: the exact raw and normalized failed action;
- `z_i`: the exact normalized error observation;
- `H_i^+ = H_i^- + a_i^0 + z_i`;
- `S_i`: the restorable environment state after the failure;
- remaining environment and token budgets; and
- digests of all items above.

The state digest at the decision boundary before `a_i^0` must equal the state
digest after failure handling. A completion that partially commits another
call before failing is excluded from the matched analysis because its clean
counterfactual branch is ambiguous.

### 2.3 What is not an opportunity

The following do not set `A_i = 1`:

- an incorrect final answer or semantically wrong successful call discovered
  only by the end-of-episode grader;
- an unsafe call that executed and irreversibly changed state;
- an error produced by the user simulator or the 8B model;
- a server crash, OOM, transport failure, evaluator bug, or simulator outage;
- a step-cap termination without an earlier eligible trigger; or
- prose such as “I was wrong” without a verified failed action and corrective
  action.

Infrastructure failures are rerun under the same seed according to the normal
evaluation policy and are never relabeled as model correction opportunities.
Model-authored code that deterministically times out or violates the sandbox
is a model/tool failure, not infrastructure failure.

## 3. Matched branch/control protocol

### 3.1 Lock the first attempt once

The failed prefix is generated and executed once. It is never regenerated per
branch. Every branch consumes the same `H_i^-`, `a_i^0`, `z_i`, `S_i`, and
remaining budgets. This makes the first attempt, initial sampling seed, tool
result, and environment state exactly identical.

For each opportunity, derive one continuation seed from a stable hash of:

```text
(experiment_id, arm_id, task_id, run_idx, eval_seed, "continuation")
```

Use that numeric continuation seed for both small-model branches when the
inference backend supports seeded sampling. The seed must differ from the
first-attempt seed; otherwise the neutral retry could reproduce the first
sample by construction. The 8B branch logs the same numeric seed, but it is
not treated as common random noise because its tokenizer and distribution
differ. When a backend cannot guarantee deterministic seeded decoding, log
that fact and describe the design as prefix-matched, not token-randomness-
matched.

Run branch order from a second deterministic hash so one condition is not
always first on the GPU. Restore and verify the state/history digests before
every branch. A mismatch invalidates the complete branch bundle, not just one
condition.

### 3.2 Branches

Each eligible small-model opportunity produces three diagnostic
continuations:

#### C — neutral resample control (`retry_control`)

- Actor: the same small checkpoint and adapter.
- Input: `H_i^-` plus a pinned neutral retry instruction; it does **not**
  include `a_i^0`, `z_i`, or any statement identifying the error.
- Purpose: estimate success from taking a second stochastic draw without
  error-specific information.
- Escalation: disabled.

This is the generic retry-luck control. Its prompt and hash are frozen before
test evaluation. It receives the same maximum generated-token allowance as
the repair branch. Prompt-token counts are not padded to equality because
semantic padding would be another intervention; actual prompt and generated
tokens are reported per branch.

#### R — diagnostic-aware same-model repair (`small_repair`)

- Actor: the identical small checkpoint and adapter used for `a_i^0`.
- Input: `H_i^+` plus the pinned reflection/correction instruction and the
  exact structured diagnostic `z_i`.
- Purpose: measure action repair and task recovery when the same model can use
  its observed failure.
- Escalation: disabled.

Changing the checkpoint, adapter, quantization, or system prompt beyond the
pre-registered correction instruction makes the branch ineligible for the
same-model label.

#### E — immediate 8B handoff (`escalate_8b`)

- Actor: the frozen local 8B comparator.
- Input: `H_i^+` plus a pinned handoff instruction identifying the failure
  observation; the 8B does not regenerate `a_i^0`.
- Purpose: distinguish recovery supplied by a larger model from repair by the
  small model.
- Further model switches: disabled.

The target 8B checkpoint must be the same one used by the production R2 arm.
If that target changes, the experiment ID and all affected bundles change.
Episodes whose parent actor is already that 8B checkpoint are ineligible for
this matched C/R/E analysis; they are never described as escalating to
themselves.

### 3.3 Common continuation rules

The three branches inherit the same environment-turn cap and remaining
environment-turn count, task deadline, tool schemas, policy-manual condition,
and remaining generated-token budget from the failed prefix. Each branch also
uses its separately frozen model-decision budget. The branch-specific first
decision is the only recovery intervention. After an accepted corrective
decision, the current branch actor continues under the common R1 act/observe
policy. If another eligible runtime failure occurs, the diagnostic branch
terminates as failure rather than starting a second correction ladder. This
isolates one pre-registered opportunity.

For Phase B, simulator state and seed are restored at the branch point. Future
user turns may legitimately differ after the agent branches; the bundle is
paired through the opportunity, not forced to share counterfactual future
messages. Scripted user turns or deterministic simulator decoding are
preferred where they are already part of the arm.

Branch outcomes are diagnostic records. They are not substituted into the
production arm's rollout array and cannot improve its `pass^k`.

## 4. Observable outcomes

For opportunity `i` and branch `b in {C, R, E}`, record:

- `L_i^b`: local resolution indicator for the first branch-specific decision;
- `Y_i^b`: deterministic end-to-end task success;
- `X_i^b`: whether any 8B model was invoked;
- `T_i^b`: generated tokens after the shared prefix;
- `P_i^b`: prompt tokens after the shared prefix; and
- `G_i^b`: GPU-seconds after the shared prefix on the named hardware.

By protocol, `X_i^C = X_i^R = 0` and `X_i^E = 1`.

### 4.1 Local resolution (`L`)

`L_i^b = 1` only if the branch-specific first decision resolves the trigger
according to its class:

- `parse_error` or `unknown_or_invalid_call`: every new block parses, every
  emitted call is registered and schema-valid, and at least one call
  dispatches successfully;
- `tool_failure`: a materially changed normalized call sequence dispatches
  and succeeds without the prior error class;
- `gate_block`: the next ordered calls successfully establish the failed
  predicate or perform an allowed alternative; merely repeating the blocked
  mutation is not resolution; or
- `loop_detected`: the new normalized call signature differs from the looped
  signature and at least one new call dispatches successfully.

For this classification, a normalized call sequence is materially changed if
its tool name, strict JSON arguments, or ordered prerequisite calls differ.
Whitespace, prose, or a different `call_id` alone is not material.

An exact repeat of the same normalized call that succeeds after a transient
failure is tagged `exact_retry_success`, not self-repair. It may contribute to
end-to-end retry recovery but has `L_i^b = 0` for the correction metric.

### 4.2 End-to-end recovery (`Y`)

`Y_i^b = 1` only when the normal deterministic environment grader marks the
final task successful: sandbox result for Phase A or the pinned benchmark
reward basis for Phase B. Valid syntax, a successful tool call, or positive
answer prose is insufficient.

This deliberate separation permits four cases: local repair and task success,
local repair but later task failure, no verified local repair but eventual
success, and neither.

## 5. Exact estimands and denominators

Compute metrics separately for every arm and trigger class, with a pooled
all-trigger row only as an additional summary. Do not pool arms with different
checkpoints, policy-manual conditions, or benchmark revisions.

Let `U_a` be all valid production evaluation episodes in arm `a`, and
`M_a = |U_a|`. Let:

```text
A_i = 1 if episode i has an eligible earliest opportunity, else 0
```

Let `B_i = 1` only when the complete C/R/E branch bundle passes all restore,
provenance, and evidence checks in §9. Define:

```text
O_a = sum over i in U_a of A_i
I_i = A_i * B_i
N_a = sum over i in U_a of I_i
```

`M_a` is the denominator for natural opportunity prevalence, `O_a` is the
number of natural opportunities, and `N_a` is the number of complete matched
bundles. Every primary branch comparison uses `N_a`. A branch failure or
restore mismatch therefore cannot erase a natural opportunity from the
prevalence estimate, but it also cannot create an unpaired outcome row.

### 5.1 Opportunity prevalence

```text
OpportunityRate_a = (sum A_i) / M_a = O_a / M_a
BundleRetention_a = (sum I_i) / O_a = N_a / O_a
```

This must accompany all conditional correction rates. A high repair rate on a
rare failure mode must not be presented as a large overall effect.
`BundleRetention` is `NA` when `O_a=0`.

### 5.2 Same-small-model local repair

```text
LocalRepair_R,a = (sum I_i * L_i^R) / N_a
LocalRepair_C,a = (sum I_i * L_i^C) / N_a
DeltaLocalRepair_a = (sum I_i * (L_i^R - L_i^C)) / N_a
```

`LocalRepair_R` is the observable same-model repair rate. `DeltaLocalRepair`
is the paired effect of supplying the diagnostic-aware repair context rather
than a neutral second draw.

### 5.3 End-to-end small-model recovery and retry luck

```text
SmallRecovery_a = (sum I_i * Y_i^R) / N_a
RetryRecovery_a = (sum I_i * Y_i^C) / N_a
DeltaRepair_a = (sum I_i * (Y_i^R - Y_i^C)) / N_a
```

Because escalation is disabled in C and R, `SmallRecovery` cannot contain 8B
rescues. `RetryRecovery` is the generic second-sample baseline.

The conversion from verified local repair to final success is secondary:

```text
RepairToSuccess_a = (sum I_i * L_i^R * Y_i^R)
                    / (sum I_i * L_i^R)
```

It is `NA`, not zero, when no R branch has `L_i^R = 1`.

### 5.4 8B escalation recovery

```text
EscalationRecovery_a = (sum I_i * Y_i^E) / N_a
DeltaEscalationVsRepair_a = (sum I_i * (Y_i^E - Y_i^R)) / N_a
```

Also report paired four-cell counts for `(Y^E, Y^R)`. The share

```text
EightBOnlyWin_a = (sum I_i * 1[Y_i^E = 1 and Y_i^R = 0]) / N_a
```

is called an **8B-only branch win**, not an individually proven causal rescue.
Individual causal attribution would require assumptions such as monotonicity
that this project does not make.

### 5.5 Paired outcome table for repair versus retry

Always report the following counts among units with `I_i=1`:

| Pair | Count | Interpretation |
|---|---:|---|
| `Y^R=1, Y^C=0` | `n10` | repair-branch-only win |
| `Y^R=0, Y^C=1` | `n01` | retry-control-only win / possible repair harm |
| `Y^R=1, Y^C=1` | `n11` | both recover; not uniquely attributable to feedback |
| `Y^R=0, Y^C=0` | `n00` | neither small-model continuation recovers |

`N_a = n10 + n01 + n11 + n00` and
`DeltaRepair_a = (n10 - n01) / N_a`.

### 5.6 Cost of recovery

Report branch means over the same `N_a` opportunities:

```text
MeanGeneratedTokens_b,a = (sum I_i * T_i^b) / N_a
MeanPromptTokens_b,a    = (sum I_i * P_i^b) / N_a
MeanGPUSeconds_b,a      = (sum I_i * G_i^b) / N_a
```

The common failed prefix cost is reported once and excluded from these
incremental branch means. Full production episode cost continues to include
the prefix and all later work for H1.

## 6. Production-trajectory attribution

The production R1/R2 trajectory receives one descriptive `recovery_mode` at
its earliest opportunity:

| Mode | Rule |
|---|---|
| `none` | No later accepted recovery action before termination. |
| `exact_retry` | The same normalized call is repeated and later succeeds; no material correction. |
| `generic_resample` | A small-model retry without the structured diagnostic produces the first accepted action. |
| `small_feedback_repair` | The same small model sees the diagnostic and meets the local-resolution rule before any switch. |
| `gate_prevention_only` | The runtime gate blocks the unsafe action and no model repair is verified. |
| `escalation_8b` | The 8B produces the first accepted recovery action after handoff. |

For each mode `m`, the descriptive end-to-end rate is:

```text
ObservedRecovery_m,a = count(A_i=1, Y_i^prod=1, recovery_mode_i=m) / O_a
```

These mutually exclusive rows sum to the production recovery rate among
opportunities. A gate block is credited to the scaffold, not the model. A
production success after 8B handoff is called “success after escalation”
unless the prefix-matched R branch also failed; only then may it be paired
with the label “8B-only branch win.”

## 7. Uncertainty and paired inference

- Report 95% two-sided intervals for opportunity prevalence, each branch
  rate, each paired difference, and each cost mean.
- Use a hierarchical paired bootstrap with 10,000 valid replicates and a
  stored bootstrap seed: resample task IDs, then resample `run_idx` values
  within each selected task. Keep `A_i` and the complete C/R/E branch bundle
  together on every draw.
- Recompute numerator and denominator inside every replicate. Never bootstrap
  branch outcomes independently.
- For `DeltaRepair`, also report the exact McNemar test on discordant pairs
  (`n10`, `n01`) or the pre-existing paired permutation implementation. The
  bootstrap interval, not the p-value alone, is the primary uncertainty
  summary.
- A branch-comparison bootstrap replicate with `N_a=0` is invalid. Draw until 10,000 valid
  replicates or 100,000 total draws. If that cap is reached, report the raw
  counts and `CI=NA`; do not make a comparative claim.
- Always report `M_a`, `O_a`, `N_a`, bundle-retention/exclusion counts, the
  number of opportunity-bearing task IDs, and all paired cell counts. Sparse
  opportunities are a result, not a reason to change the trigger definition
  after evaluation.

No correction metric is used for checkpoint selection, early stopping,
prompt tuning, or choosing which test rows to report.

## 8. Required event-log fields

The current trajectory schema remains the source for prompts, raw completion,
parsed calls, sandbox/tool traces, gates, ground truth, and reward. A future
schema revision or a strictly versioned sidecar must add the fields below
before correction measurements are run.

### Episode and provenance

```text
experiment_id
episode_id
parent_episode_id
arm_id
branch_id
branch_condition          # production | retry_control | small_repair | escalate_8b
task_id
run_idx
eval_seed
continuation_seed
model_id
adapter_id
actor_role               # small | escalation_8b | user
rung                      # R0 | R1 | R2
benchmark_revision
prompt_hash
template_hash
sampling_config_hash
harness_config_hash
gate_policy_fingerprint
hardware_id
inference_backend_version
seed_determinism_supported
```

### Opportunity and branch-point evidence

```text
opportunity_id
opportunity_index         # always 0 for primary analysis
eligible
exclusion_reason
bundle_valid
bundle_exclusion_reason
trigger_class
trigger_error_codes
failure_actor_model_id
failure_observation
failure_observation_hash
history_before_hash       # H_i^-
failed_history_hash       # H_i^+
failed_raw_completion
failed_raw_completion_hash
failed_normalized_calls
state_before_decision_hash
state_after_failure_hash  # must equal state_before_decision_hash
remaining_env_steps
remaining_generation_tokens
branch_restore_verified
```

### Recovery and outcome evidence

```text
recovery_group_id
recovery_attempt_index
recovery_prompt_hash
recovery_actor_model_id
recovery_raw_completion
recovery_normalized_calls
diagnostic_visible
material_action_change
exact_retry_success
local_resolution
second_failure_trigger
model_switch_count
escalation_invoked
escalation_target_model_id
recovery_mode
final_success
outcome_source
final_state_hash
termination_reason
prompt_tokens_after_prefix
generated_tokens_after_prefix
gpu_seconds_after_prefix
```

All branch records reference the same `parent_episode_id` and
`opportunity_id`. Raw values and hashes are both retained where the existing
trajectory policy already permits raw logging. The event viewer must be able
to render the failed action, structured observation, branch action, local
classification, final outcome, and model-switch boundary without inferring
them from prose.

## 9. Exclusions and fail-closed rules

Set `B_i=0` for the complete matched bundle and record the reason when:

- any branch restores a different history, state, policy, or budget digest;
- the failed action partially changed state;
- branch sampling/template/harness configuration differs beyond the declared
  condition;
- a branch receives a different task, initial state, first attempt, tool
  result, or simulator state at the fork;
- an infrastructure failure prevents a branch from completing after the
  allowed same-seed rerun policy; or
- log evidence is missing or its digest fails verification.

Do not exclude a branch because the model repeats the error, produces invalid
JSON, hits the model-caused sandbox timeout, refuses, loops, or reaches the
step cap. Those are outcome failures.

The natural opportunity remains in `O_a` and the production descriptive
analysis even when `B_i=0`; only matched branch comparisons exclude it.
Publish an exclusion table by reason. If exclusions differ by branch, the
bundle rule prevents a favorable condition from retaining an easier sample.

## 10. Fit with R0, R1, R2, and project milestones

- **R0:** remains one generation per natural agent turn with no same-state
  replacement generation. A conversational episode may generate again only
  after a genuine environment or user observation. R0 may log an observable
  failure, but no production self-correction is credited. Diagnostic forks
  from an R0 prefix are labeled separately and never alter R0 `pass^k`.
- **R1:** naturally exposes action/observation continuations from the same
  model. M1 can implement the opportunity detector, branch serializer, and
  same-model C/R diagnostic in the existing Python loop; this does not require
  LangGraph. Production R1 uses the descriptive modes in §6.
- **R2:** keeps the frozen retry → reflect → 8B escalation ladder and runtime
  gates. M4 adds the E branch and consumes the same opportunity/event schema.
  A blocked gate is scaffold prevention. A later same-model compliant action
  can qualify as repair only in the diagnostic R branch unless the frozen
  production rung protocol separately grants that opportunity.

The diagnostic branch bundle is not a 24th experimental arm. It is a paired
analysis nested inside existing evaluation units. If compute requires a
subset, select opportunity bundles by a deterministic hash threshold fixed in
the eval configuration before any branch outcome is read, and report the
selection fraction. Never select examples because a repair branch succeeded.

## 11. Relationship to H1–H3

- **H1 is unchanged.** It uses only production-arm end-to-end outcomes,
  production generated tokens, and the `pass^4` gap comparison frozen in
  `HYPOTHESIS_PROTOCOL.md`.
  Branch diagnostics explain where recovery came from but do not enter H1's
  numerator or cost denominator.
- **H2 is unchanged.** Its attempted-action denominator and matched
  reward-ablation protocol are frozen in `HYPOTHESIS_PROTOCOL.md`. Gate
  prevention and
  post-block model repair are logged separately so one cannot masquerade as
  the other.
- **H3 is unchanged.** Run the same correction protocol separately under the
  policy-manual-present and policy-manual-removed contexts only if those
  diagnostic rows were fixed before test. H3 itself remains the R0 production
  `pass^1` retention test in `HYPOTHESIS_PROTOCOL.md`.

The self-correction report is secondary and pre-measurement. Every table starts
with `M_a`, `O_a`, `N_a`, trigger distribution, and exclusion counts, followed by
local repair, end-to-end branch recovery, paired differences, escalation, and
incremental costs. Until valid logs exist, every result cell is `TBD`.
