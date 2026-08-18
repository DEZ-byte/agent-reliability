# Runtime scaffold rung protocol

**Status:** pre-measurement specification. No rung result, scaffold winner, or
cost/quality trade-off has been measured. All numeric result cells remain
`TBD` until they are generated from versioned trajectory artifacts.

This document fixes the operational meaning of R0, R1, and R2 for Phase A and
the conversational tau-family evaluation. It resolves ambiguous shorthand in
the blueprint; a framework, graph node, or retry library does not define a
rung. Any change after test evaluation begins requires a dated decision and a
new harness-config hash.

## 1. Units and counters

The harness MUST keep these units separate.

### 1.1 Environment turn

An **environment turn** is one accepted policy action that is submitted to the
benchmark state machine and produces the next state, observation, or terminal
outcome. A successful tool action and an accepted user-facing communication are
environment turns when the pinned adapter submits them through the benchmark's
step interface.

The following do **not** consume an environment turn:

- parsing or schema validation;
- a gate block before dispatch;
- a model generation by itself;
- an exact redispatch after a confirmed no-commit transient failure;
- a reflection prompt; or
- a LangGraph node transition that does not call the model or environment.

`environment_turn_cap` is common to matched R0/R1/R2 arms. The reference cap is
20. If an upstream benchmark imposes a lower cap, the effective cap is
`min(20, upstream_cap)` and that value is frozen for every matched arm. A rung
MUST NOT gain more environment turns because it has retry or escalation logic.

### 1.2 Agent turn

An **agent turn** begins when a frozen environment state and model-visible
observation are presented to the policy for its next action. It ends when an
action is accepted as an environment turn, the episode terminates, or the
rung's same-turn decision budget is exhausted.

An accepted tool result starts a new agent turn if the benchmark requests
another policy action. A retry against the same state and observation remains
inside the current agent turn.

### 1.3 Model decision

A **model decision** is one model generation, whether it yields a valid call,
invalid JSON, a refusal, prose, or a final response. Every generation consumes
one model-decision unit and its actual prompt and generated tokens are charged.
Changing the seed, adding structured failure feedback, reflecting, or switching
to the 8B actor always creates a new model decision.

### 1.4 Tool attempt and exact redispatch

A **tool attempt** is one dispatch to a registered handler or upstream tool.
An exact redispatch is another tool attempt, but it is not another model
decision and does not by itself advance the environment. Gate-blocked calls are
action attempts, not tool dispatches.

Every episode records at least:

```text
environment_turn_count
agent_turn_count
policy_model_decision_count
escalation_model_decision_count
tool_dispatch_attempt_count
exact_transient_redispatch_count
gate_block_count
model_switch_count
```

No one counter may be reported as another counter's proxy.

## 2. Phase mapping

The rung state machine is shared across phases. Only the environment adapter
changes.

### 2.1 Phase A

The initial math/function-calling task creates the first agent turn. Each
accepted calculator, REPL, or registered-tool action advances the Phase A
environment and returns a deterministic observation. If another policy action
is needed, that observation creates a new agent turn. A final submitted answer
terminates through the deterministic Phase A grader.

Therefore R0 means one generation at each Phase A agent turn. It does not mean
that the harness may silently regenerate malformed output, and it does not
force every possible Phase A environment to have only one generation for the
entire episode.

### 2.2 Conversational tau-family evaluation

The pinned upstream adapter decides which accepted tool actions and assistant
communications advance the benchmark. Each resulting user, tool, database, or
task observation creates the next agent turn. User-simulator generations are
logged under a separate actor and token counter; they never count as policy
model decisions.

The adapter MUST publish its mapping from upstream `step` calls to
`environment_turn_count`. Internal simulator calls, graph hops, and parser
passes cannot be used to give one rung additional benchmark turns. At a
same-turn retry branch, environment state, simulator state, and simulator seed
remain unchanged.

## 3. Common envelope and rung-specific budgets

All matched arms share the following environment envelope:

- task manifest and benchmark/environment revision;
- effective `environment_turn_cap`;
- episode wall-clock deadline and per-tool timeout;
- tool schemas, handler revisions, and sandbox limits;
- deterministic grader and terminal conditions; and
- user-simulator checkpoint, prompt, decoding settings, and seed in Phase B.

Model-decision budgets are separate rung treatments. The reference per-agent-
turn ladder is:

| Rung / parent actor | Initial decision | Feedback decision | Reflection decision | Escalation decision | Maximum model decisions in one agent turn |
|---|---:|---:|---:|---:|---:|
| R0, any actor | 1 | 0 | 0 | 0 | 1 |
| R1, any actor | 1 | at most 1 | 0 | 0 | 2 |
| R2, small-model parent | 1 | at most 1 | at most 1 | at most 1 frozen-8B decision | 4 |
| R2, 8B parent | 1 | at most 1 | at most 1 | 0 | 3 |

Unused decisions do not carry into another agent turn. An accepted action ends
the current agent turn. A new observation starts a new turn with a fresh rung
budget, subject to the common environment and episode caps. The complete
episode decision ceiling is derived from these frozen per-turn budgets and the
common environment-turn cap; implementations MUST also enforce and log that
ceiling explicitly.

The exact values above are protocol defaults. Dev-split tuning may reduce them,
but cannot increase one model or training regime independently. The values and
all prompts are frozen before test and included in the harness-config hash.

## 4. Normative rung definitions

### 4.1 R0 — direct, no same-turn model retry

R0 performs exactly one model generation per agent turn. It may execute an
accepted action and later receive another natural environment observation,
which begins another agent turn. It MUST NOT:

- regenerate at the same state because parsing or validation failed;
- add failure feedback and ask the model again in the same turn;
- invoke a reflection prompt;
- switch policy models; or
- convert invalid output into a call by coercion or repair code.

If the sole decision is invalid or blocked, the pinned phase adapter applies
its frozen no-action/turn-failure rule and records the outcome. It cannot grant
a replacement generation. This is the meaning of “no retry”; it is not a claim
that a conversational episode contains only one model generation in total.
The common confirmed-transient tool redispatch policy in §5.1 is infrastructure
handling, not another model generation, and does not change the R0 label.

### 4.2 R1 — framework-neutral act/observe core

R1 uses the same checkpoint throughout the episode. Its core loop is:

```text
frozen state + observation
  -> model decision
  -> strict parse/schema/policy audit
  -> accepted dispatch or structured failure observation
  -> next decision or environment turn
```

After a model-correctable failure at an unchanged environment state, R1 may use
one fresh same-model feedback decision. That decision sees the exact structured
failure observation. It is a new model decision, not an exact redispatch and
not a hidden parser repair. R1 has no reflection-specific prompt, production
policy-gate enforcement treatment, or model escalation.

The M1 implementation is a plain Python state machine behind framework-neutral
interfaces for model generation, validation, tool dispatch, environment step,
and event emission. M1 R1 does not depend on LangGraph. This core is the
reference semantics for later adapters.

### 4.3 R2 — R1 plus reflection, policy gates, and bounded cascade

R2 retains the R1 feedback decision and adds:

- deterministic policy gates in `enforce` mode;
- one pinned same-model reflection decision after the feedback decision fails;
- loop detection at three identical consecutive normalized call signatures;
- for a **small-model parent only**, one final handoff decision by the frozen
  local 8B comparator; and
- explicit accounting for every blocked action, retry, reflection, and switch.

The grader is never consulted to choose a transition. Triggers use only parser,
schema, tool, gate, loop, budget, and environment events available at runtime.
The ladder does not reset after reflection or escalation. Further failures end
the agent turn or episode under the pinned termination rule.

A small-to-8B production handoff is one-way for the remainder of the episode.
The accepted escalation action may advance the environment; subsequent agent
turns keep the 8B actor and use the 8B R2-no-escalation budget. The harness never
switches back to the small parent, and `model_switch_count` cannot exceed one.

## 5. Failure and retry state machine

### 5.1 Exact redispatch is infrastructure handling, not model repair

An implementation may redispatch the identical tool name, strict JSON
arguments, and idempotency key only when **all** of these conditions hold:

1. the versioned failure classifier marks the error explicitly transient;
2. the handler or upstream adapter confirms `committed=false`;
3. `state_before` and `state_after` digests are identical;
4. the operation is declared idempotent or uses a stable idempotency key; and
5. the common `transient_redispatch_limit` has not been exhausted.

The reference limit is one exact redispatch and is identical across rungs so
infrastructure noise is not a scaffold advantage. Eligible examples are a
pre-dispatch connection failure, an explicit rate limit, or an upstream result
that declares both `retryable=true` and `committed=false`.

Timeouts with unknown commit state, tool exceptions without a transient code,
invalid tool output/state, authentication or business-rule failures, parse
errors, schema errors, unknown tools, and gate blocks are not eligible. When
eligibility is uncertain, fail closed and require a new model decision where
the rung permits one.

An exact redispatch success is logged as `exact_retry_success`. It is never
credited as model self-correction.

### 5.2 Parse, unknown-tool, and schema failures

These failures never redispatch a call and never receive mechanical JSON or
argument repair. The harness emits a deterministic model-visible diagnostic
containing the failure class, stable public error code, relevant tool name when
known, and schema-safe field paths. It omits a suggested answer and hidden
grader state.

- R0 has no remaining same-turn decision, so the adapter records turn failure.
- R1 may spend its one feedback decision on a new action.
- R2 may spend its feedback decision, then its reflection decision if needed,
  and only then may a small-model parent hand off to the 8B.

Every such generation increments the model-decision count. Whitespace changes,
a new call ID, or prose around the same invalid call do not create a material
repair.

### 5.3 Non-transient tool failures and loops

A confirmed non-transient failure becomes a structured observation and follows
the same fresh-decision ladder. A third identical consecutive normalized call
signature emits `loop_detected` without performing an additional identical
dispatch. The loop event consumes the current decision but not another
environment turn.

Failures after possible state mutation are never rollback-assumed. If the
environment cannot prove a restorable state, no diagnostic fork or automatic
retry is eligible and the episode terminates with an explicit integrity reason.

## 6. Gate modes and production gate-block behavior

Registry checks, strict parsing/schema validation, sandbox limits, handler
output/state validation, and transactional commit are common integrity controls
in every rung. They are not the R2 policy-gate treatment.

For isolated benchmark comparisons, business-policy predicates run as follows:

| Rung | Policy-gate mode |
|---|---|
| R0 | `audit` |
| R1 | `audit` |
| R2 | `enforce` |

R0/R1 audit runs MUST use sandboxed benchmark state with no real external side
effects. Any real deployment enables enforcement and is labeled as gated; the
pure R0/R1 configurations are experimental ablations, not deployment advice.

In an R2 production trajectory, a gate block:

1. occurs before dispatch;
2. commits no state change and preserves the state digest;
3. records the attempted normalized call and gate-policy fingerprint;
4. returns a structured, remediation-safe observation with a stable public
   reason code, without exposing secrets or hidden predicate values;
5. consumes the current model decision but no tool dispatch or environment
   turn; and
6. permits the next fresh same-model decision only if the rung budget remains.

A blocked call is never exactly redispatched. The first block does not jump
directly to the 8B. A later compliant same-model action can be described as
repair; a block without such an action is `gate_prevention_only`. Repeated
blocks advance the same feedback/reflect/exhaustion ladder. For a small parent,
8B handoff is available only after same-model correction is exhausted. For an
8B parent, exhaustion terminates without a model switch.

## 7. The 8B R2 arm and diagnostic escalation

The registered 8B comparator is already the largest policy actor in scope. It
MUST NOT escalate to itself, to the user simulator, to an API, or to an
unregistered larger model.

The matrix's 8B×R2 arm therefore means:

```text
8B R2-no-escalation = same-model feedback retry + reflection + enforce gates
                      + loop/budget controls + escalation_target=none
```

Tables, filenames, and plot legends MUST label it `R2-no-escalation` or the
equivalent explicit configuration string. It is not described as a full
cascade. After its reflection budget is exhausted, it gives up gracefully and
records `retry_reflect_exhausted`; it does not invent a larger actor.

Small-parent R2 arms may use the frozen 8B comparator as their one escalation
target. “Small parent” means the registered primary, scale-check, or
Llama-3.2-3B policy, not the 8B comparator or the user simulator.

The matched diagnostic E branch in `SELF_CORRECTION_SPEC.md` is eligible only
when the failed parent is one of those small policies and the production arm
has the frozen 8B target. For an 8B-parent episode:

- the primary C/R/E bundle in `SELF_CORRECTION_SPEC.md` is inapplicable;
- an optional C/R-only diagnostic requires a separately labeled denominator;
- no E branch is created;
- `EscalationRecovery` and related E contrasts are `NA`, not zero; and
- the missing E branch is structural, not an exclusion or failed outcome.

Diagnostic E branches never alter the production trajectory or headline
`pass^k` arrays.

## 8. M1 core, M4 LangGraph adapter, and parity

### 8.1 M1

M1 implements and tests the framework-neutral R0/R1 transition core for Phase
A. The state object owns the counters, remaining budgets, state/policy digests,
failure class, actor identity, and termination reason. The model caller and
environment are injected interfaces. No framework may add an implicit retry.

### 8.2 M4

M4 may expose the same transition core through LangGraph and add the R2 nodes.
LangGraph is an adapter and orchestration implementation, not an experimental
treatment. Before any M4 result is compared with M1, scripted parity fixtures
MUST prove that the plain-Python and LangGraph paths produce identical:

- accepted actions and model-visible observations;
- environment, agent-turn, decision, dispatch, block, and switch counters;
- budget transitions and termination reasons;
- state digests, policy fingerprints, and ordered event payloads; and
- final environment state for R0, R1, small-parent R2, and 8B
  R2-no-escalation fixtures.

Automatic LangGraph/node retries and checkpoint replays are disabled unless
they implement the explicit state machine and appear in the counters. A graph
node re-entry does not become a free model decision or environment turn.

The M4 R1 adapter reruns a frozen Phase A bridge set. R1 results from the two
implementations may share a table only after event-digest parity passes. The
same parity suite is required before the conversational tau adapter is used.

## 9. Cost and accounting

Every result row reports outcome and resource use together. At minimum, retain:

- prompt, generated, and total policy tokens by model ID;
- user-simulator tokens in a separate column;
- model decisions by reason: `initial`, `post_observation`, `feedback`,
  `reflect`, and `escalate_8b`;
- environment turns, tool dispatch attempts, exact redispatches, gate blocks,
  reflections, loops, and model switches;
- end-to-end latency plus model, tool, and simulator latency components;
- peak allocated/reserved VRAM and actor residency configuration;
- measured GPU-seconds by actor; and
- actual rental cost from GPU-seconds and the recorded provider rate when a
  paid instance is used.

Do not assign an imaginary API price to local generations. Local runs report
GPU-seconds and energy only when directly measured. Failed generations,
blocked calls, exact redispatches, reflections, and unsuccessful escalations
remain in the cost denominator.

The full production episode charges its prefix once and all later work to its
arm. Matched C/R/E diagnostic continuations report incremental branch cost from
the frozen branch point, as specified in `SELF_CORRECTION_SPEC.md`; diagnostic
cost is not silently added to production-arm headline cost. Report at least
tokens per episode, GPU-seconds per episode, cost per successful episode, and
escalation frequency × measured incremental 8B cost.

## 10. Frozen parity knobs

The following values are tuned on development data only, frozen before test,
serialized in the arm manifest, and included in `harness_config_hash`:

1. model/revision, adapter, quantization, tokenizer, native template, and
   system/policy prompt hashes;
2. decoding configuration, generation-token caps, and per-run seeds;
3. benchmark/environment revision, task manifest, adapter revision, effective
   environment-turn cap, episode deadline, and tool timeout;
4. tool schemas/order, handler revisions, sandbox limits, transactional-state
   policy, and transient-failure classifier version;
5. transient allowlist, idempotency rules, and exact-redispatch limit;
6. per-rung decision budgets, feedback schema/prompt, reflection prompt,
   loop-signature definition and threshold, and termination rules;
7. gate policy/fingerprint, audit/enforce mode, public gate-feedback schema,
   and repeated-block transition;
8. escalation trigger, frozen target or explicit `none`, handoff prompt, actor
   residency, and model-switch limit;
9. user-simulator checkpoint/revision, prompt, decoding configuration, seed,
   and upstream tau tuple; and
10. implementation ID and versions for the reference loop, LangGraph adapter,
    inference server, and dependency lock.

Matched arms share every knob except the declared rung treatment and the model
or training regime named by the arm. If a framework limitation forces a
different prompt, cap, retry, or observation, that configuration is a separate
arm or bridge study, not a parity result.

## 11. Required termination labels

At minimum, use distinct machine-readable reasons for:

```text
success
environment_terminal_failure
environment_turn_cap
episode_deadline
r0_invalid_decision
feedback_exhausted
retry_reflect_exhausted
gate_prevention_only
loop_detected
tool_failure_non_transient
tool_failure_commit_unknown
escalation_unavailable
sandbox_or_integrity_failure
```

Never fold cap exhaustion, a gate block, a malformed decision, an 8B-only
recovery, and a deterministic task failure into one generic `failed` label.

## 12. Pre-measurement acceptance checklist

Before any rung result is recorded:

- the Phase A and conversational adapters document their environment-turn
  mapping;
- R0 tests prove one generation per agent turn and no same-turn model retry;
- parse/schema tests prove that only a fresh model decision can change an
  invalid call;
- transient tests prove no exact redispatch after a possible commit;
- gate tests prove block-before-dispatch and unchanged state;
- 8B R2 tests prove `escalation_target=none` and no self-switch;
- diagnostic tests prove E branches exist only for small-model parents;
- plain-Python/LangGraph golden traces pass event-digest parity; and
- all cap, prompt, policy, actor, and cost-accounting fields are present in the
  immutable run manifest.

Until those checks and real run artifacts exist, R0/R1/R2 comparisons and all
cost-quality claims remain pre-registered hypotheses.
