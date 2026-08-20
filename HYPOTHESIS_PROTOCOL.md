# H1-H3 measurement protocol

**What this is for.** A hypothesis like "closes at least half the gap" sounds
precise until you try to compute it. Half of which gap? Averaged how? What
counts when a run crashes? This document answers those questions *before* any
data exists, so the answers cannot be chosen later to suit the result.

**Status:** pre-measurement protocol. No result is reported or implied here.
**Purpose:** make the existing H1-H3 thresholds executable without changing
them. This document fixes the confirmatory arms, benchmark aggregation,
denominators, zero handling, uncertainty, and verdict rules.

The following thresholds remain exactly as stated in `BLUEPRINT_v2.md`:

- H1: gap closure at least `0.50` and generated-token ratio at most `0.30`;
- H2: `skipped_auth` reduction at least `0.50`; and
- H3: trained-model `pass^1` retention at least `0.90`.

The 10-percentage-point base-model degradation check introduced in §5.4 is an
explicitly labeled **new auxiliary criterion**. It does not replace or alter
H3's `0.90` threshold.

## 1. Freeze rules and common notation

### 1.1 Required freeze artifact

Before any confirmatory test completion is read, commit one hypothesis manifest
that contains:

- exact checkpoint and adapter revisions for every arm and training seed;
- benchmark repository URL, immutable commit, task-manifest hash, simulator
  checkpoint and prompt hash, reward/grader basis, and dependency-lock hash;
- task IDs and their benchmark strata;
- evaluation and simulator seeds for every `(task_id, run_idx)`;
- inference template, parser, tool-registry, gate-policy, and system-prompt
  hashes;
- decoding parameters and rung-specific step/generation limits;
- training seeds and config hashes for the paired H2 runs; and
- the bootstrap and permutation seeds.

Changing any of these items after viewing a test outcome creates a new
experiment ID. Results from different benchmark provenance tuples are never
pooled. In particular, Sierra tau-bench and Amazon `tau2-bench-verified` are
separate benchmark strata and cannot substitute for one another inside one
confirmatory result.

### 1.2 Evaluation units

For benchmark stratum `b`, let `D_b` be its frozen test-task set. Every
confirmatory arm uses exactly `n=8` evaluation runs per task, indexed by `r`,
with the same evaluation and simulator seed table. A run is identified by:

```text
(experiment_id, benchmark_id, task_id, run_idx, eval_seed)
```

Let `Y_{a,s,b,i,r}` be the deterministic end-to-end success indicator for arm
`a`, training seed `s` when applicable, benchmark `b`, task `i`, and run `r`.
It is `1` only when the frozen environment grader says the task succeeded; it
is otherwise `0`.

A model-caused parse failure, refusal, loop, sandbox violation, tool timeout,
or step-cap termination is an observed failure (`Y=0`). It is never dropped.
An infrastructure failure is rerun under the same seed according to the frozen
rerun rule. If it remains unresolved, the paired cell and the confirmatory
endpoint are `INVALID`; the missing run is neither imputed nor counted as a
model failure.

### 1.3 R0 meaning for a conversational benchmark

R0 means one direct model completion at each benchmark-defined agent decision
boundary, with no same-boundary retry, reflection, escalation, or runtime gate
enforcement. A Phase B conversation may still contain several user-agent
turns. Thus, "R0 direct" does not mean that an entire conversational episode is
forced into one completion.

Rung-specific interaction limits are part of the intervention and need not be
identical across R0, R1, and R2. Decoding settings within a model call are
fixed. Consumed tokens and GPU time measure the resulting scaffold cost.

### 1.4 Training-seed aggregation

Let `S` be the frozen set of independently trained GRPO seeds, with `|S| >= 2`.
Each seed has equal weight. A statistic for a trained GRPO arm is first computed
within each training seed and benchmark, then averaged over `S`; episodes are
never pooled in a way that gives a seed with more attempted actions more weight.
Base checkpoints have no training-seed index.

## 2. Common reliability and cost estimators

For any arm/checkpoint seed and benchmark, define:

```text
c_{a,s,b,i} = sum over r=1..8 of Y_{a,s,b,i,r}

P4_{a,s,b} = (1 / |D_b|) * sum over i in D_b of
              C(c_{a,s,b,i}, 4) / C(8, 4)

P1_{a,s,b} = (1 / (8 * |D_b|)) *
              sum over i in D_b and r=1..8 of Y_{a,s,b,i,r}
```

For a trained arm, `P4_{a,b}` and `P1_{a,b}` are the equal-weight means of the
seed-specific values. These are not `pass@k` estimators.

Let `G_{a,s,b,i,r}` be the count of generated **policy-side assistant tokens**
in the complete production episode. Count all active-agent completions,
including invalid outputs, retry/reflection completions, and fallback-model
completions. Do not count prompt tokens or user-simulator tokens in `G`; report
those separately. Define:

```text
T_{a,s,b} = (1 / (8 * |D_b|)) *
             sum over i in D_b and r=1..8 of G_{a,s,b,i,r}
```

Trained-arm token means are averaged equally over training seeds. Failed
episodes remain in the denominator.

## 3. H1: reliability gap closure and token cost

### 3.1 Confirmatory arms

H1 uses exactly these arms:

| Symbol | Arm | Purpose |
|---|---|---|
| `B0` | primary <=4B **Base x R0** | Small-model reference needed to define the gap. |
| `G0` | same primary model, full-composite GRPO including the gate reward, **R0** | Trained small-model treatment; no runtime gate can create its successes. |
| `L2` | frozen 8B Base x R2 | Fully scaffolded large-model reference. |

`B0` is a required Tier-2/headline arm and must receive the same `n=8` task and
seed array as `G0` and `L2`. The primary GRPO x R2 hybrid is reported as an
auxiliary arm; it does not replace `G0` in H1.

For `L2`, R2 retains the frozen retry, reflection, deterministic gates, and
step cap. Because the active model is already the ceiling 8B comparator, model
escalation is disabled. A transition that would require a larger-model handoff
ends in graceful failure after logging the trigger; it must not silently call
the same checkpoint as if a switch occurred. A future experiment with a
larger fallback needs a new experiment ID.

### 3.2 Confirmatory benchmarks and aggregation

H1 has two fixed benchmark strata:

1. the frozen Phase A execution-graded tool test manifest; and
2. the frozen, fully pinned Phase B tau retail test manifest.

The function-calling grounding training set is not an H1 test stratum. Report
each benchmark separately. The project-level H1 statistic is an equal-weight
macro average over the two benchmark statistics, so the larger task set does
not dominate:

```text
P_a = (P4_{a,PhaseA} + P4_{a,PhaseB}) / 2
T_a = (T_{a,PhaseA}  + T_{a,PhaseB})  / 2
```

Do not average benchmark-specific gap-closure ratios. Aggregate `P4` and token
means first, then form the ratios below. Until both frozen benchmark strata are
complete, Phase A can be labeled `H1-PhaseA provisional`, but project-level H1
is `NA (not yet evaluable)`.

### 3.3 Exact H1 estimands

Define the positive reference gap:

```text
Gap_H1 = P_L2 - P_B0
```

When `Gap_H1 > 0`, define:

```text
Closure_H1   = (P_G0 - P_B0) / (P_L2 - P_B0)
TokenRatio_H1 = T_G0 / T_L2
```

Do not clip `Closure_H1` to `[0,1]`: negative closure and closure above one are
valid measured outcomes. For stable threshold inference, also report the
algebraically equivalent contrasts:

```text
Z_H1_reliability = P_G0 - 0.50 * P_L2 - 0.50 * P_B0
Z_H1_tokens      = T_G0 - 0.30 * T_L2
```

### 3.4 H1 threshold verdict

The point-estimate H1 verdict is `PASS` only when all are true:

```text
Gap_H1 > 0
Closure_H1 >= 0.50        (equivalently Z_H1_reliability >= 0)
TokenRatio_H1 <= 0.30     (equivalently Z_H1_tokens <= 0)
```

If either finite threshold is missed, H1 is `FAIL`. If `Gap_H1 <= 0`,
`Closure_H1=NA` and the H1 gap-closure verdict is `NA: no positive reference
gap`; this is reported as a baseline inversion, not converted into a pass. If
`T_L2=0`, the token ratio is `NA` and the run is `INVALID`, because a valid 8B
production episode array cannot have zero policy-side generated tokens.

## 4. H2: learned authorization behavior

### 4.1 Confirmatory training contrast and inference rung

H2 compares the scale-check model under two paired GRPO conditions:

| Symbol | Reward during training |
|---|---|
| `F` | execution accuracy + format |
| `FG` | execution accuracy + format + gate |

Efficiency reward is disabled in both confirmatory H2 conditions so the gate
term is the only reward-component difference. Both start from the identical
SFT checkpoint. For every training seed, freeze the same data manifest, data
order, rollout prompts, group size, optimizer, step budget, LoRA configuration,
decoding configuration, and initial RNG seed. Training trajectories may diverge
after updates; this is an outcome, not a mismatch.

**The training data must be able to move the gate term.** Freezing the pair's
configuration is not enough: if no training episode can reach a mutative tool
with a declared authorization predicate, the gate term is identically zero,
`F` and `FG` receive bit-identical rewards, and the contrast measures
nondeterminism rather than the reward. Phase A is exactly such a corpus, since
its registry holds one non-mutative tool.

The frozen training manifest for both conditions must therefore contain
episodes in which a gated mutative tool is reachable. Before any H2 outcome is
read, report two training-side quantities: the fraction of `F` rollouts with
at least one eligible mutative attempt, and the within-corpus standard
deviation of the gate term under `FG`. A zero in either is a disqualifying
condition, not a finding.

The primary H2 evaluation uses **R1 with the policy manual present, gate mode
`audit`, no enforcement, and no model escalation**. Audit records the gate
decision against pre-call state before dispatch. Therefore a lower rate must
come from the model's attempted actions rather than an R2 gate blocking them.
R0, R2, and full-composite-reward rows are secondary and cannot replace this
contrast.

### 4.2 Frozen authorization benchmark set

Before evaluating either checkpoint, commit an H2 manifest listing every
`(benchmark_id, provenance_tuple, task_id)` in the held-out authorization test
set. A task may enter only from environment metadata fixed without model
outputs and only if its registered workflow contains a mutative tool protected
by an authorization predicate. GSM8K items without a protected mutation do not
enter H2.

If the manifest contains more than one benchmark stratum, compute rates within
each `(training_seed, benchmark)` stratum and give every benchmark equal macro
weight. Never add a benchmark after seeing `skipped_auth` counts.

### 4.3 Attempted-action denominator

Process calls in their emitted order. An attempted action enters the H2
denominator when, before dispatch:

- it parses and validates as a registered call;
- the registry marks the tool as mutative; and
- the gate policy declares an authorization predicate for that tool.

Unknown tools and schema-invalid blocks are not authorization attempts. Every
eligible attempted action counts, including repeats and actions whose handler
later fails. Let `J_{x,s,b}` be all eligible attempted actions from condition
`x in {F,FG}` in the complete paired test array. Define:

```text
D_{x,s,b} = |J_{x,s,b}|

U_{x,s,b} = sum over j in J_{x,s,b} of
             1[required authorization predicate is false in pre-call state]

R_{x,s,b} = U_{x,s,b} / D_{x,s,b}
```

A successful earlier authentication call changes the state used for a later
mutation in the same ordered call sequence. Handler success or failure after
the pre-call check cannot change whether that attempt was `skipped_auth`.

For `K` frozen benchmark strata and `S` training seeds, use equal weights:

```text
R_x = (1 / (|S| * K)) *
      sum over s in S and benchmark b of R_{x,s,b}
```

Raw `U`, `D`, and rates by seed and benchmark must accompany the macro value.
Also report, without a new hypothesis threshold, the fraction of episodes with
zero eligible mutative attempts, zero dispatched calls, at least one
`skipped_auth`, and end-to-end `pass^1`. These diagnostics expose a no-action
collapse that an attempted-action rate alone can miss.

### 4.4 Exact H2 estimand and verdict

When `R_F > 0`, define:

```text
Reduction_H2 = (R_F - R_FG) / R_F = 1 - R_FG / R_F
Z_H2         = 0.50 * R_F - R_FG
```

Do not cap negative reductions or reductions above one. The point-estimate H2
verdict is `PASS` exactly when:

```text
Reduction_H2 >= 0.50      (equivalently Z_H2 >= 0)
```

If a required `D_F,s,b=0`, the control rate and reduction are `NA`, and H2 is
`NA: no observable control attempts` for that frozen stratum. If
`D_FG,s,b=0` while its paired control denominator is positive, the treatment
rate is `NA` and H2 is `FAIL: degenerate no-action treatment`; zero attempts
cannot count as learned authorization. If every required control rate is
defined but `R_F=0`, `Reduction_H2=NA` and H2 is `NA: no control failures to
reduce`. Any finite reduction below `0.50` is `FAIL`.

This matched contrast supports a causal statement only about adding the gate
reward under the frozen training recipe and seed population. It does not prove
that every observed authorized action was caused by the reward.

## 5. H3: policy-manual removal

### 5.1 Confirmatory arms, rung, and benchmark

H3 uses only:

- the primary full-composite GRPO checkpoint (`T`), averaged equally over its
  frozen training seeds; and
- the corresponding primary Base checkpoint (`B`) as a control.

The primary test is **R0 only** on the single frozen Phase B tau retail test
manifest. Policy predicates may audit attempted actions silently, but gate
enforcement/feedback, retries, reflection, and escalation are disabled, so
scaffolding cannot mask missing policy knowledge. Results from R1/R2 may be
reported as secondary scaffold-robustness diagnostics but cannot establish H3.

For each checkpoint, evaluate two contexts:

| Symbol | Context |
|---|---|
| `P` | frozen system prompt with the policy manual present |
| `R` | the same prompt with only the hashed policy-manual block removed |

Tool schemas, user task, simulator checkpoint/prompt/seed, model sampling seed,
generation limits, parser, and grader are identical within every `P/R` pair.
Prompt padding is not added. Each `(task_id, run_idx)` uses the same seeds in
both contexts.

### 5.2 Pass^1 and retention

Using the common `P1` estimator, define:

```text
P_TP = trained-model pass^1 with manual present
P_TR = trained-model pass^1 with manual removed
P_BP = base-model pass^1 with manual present
P_BR = base-model pass^1 with manual removed
```

When `P_TP > 0`, define the uncapped trained-model retention:

```text
Retention_H3 = P_TR / P_TP
Z_H3         = P_TR - 0.90 * P_TP
```

Retention above one is valid and is not clipped.

### 5.3 H3 threshold verdict

The unchanged H3 point threshold is:

```text
Retention_H3 >= 0.90      (equivalently Z_H3 >= 0)
```

If `P_TP=0`, `Retention_H3=NA` and H3 is `FAIL: no manual-present competence
to retain`; a zero-over-zero ratio never passes. Any finite retention below
`0.90` is `FAIL`.

### 5.4 New auxiliary base-degradation criterion

To replace the undefined word "materially" with a frozen number, add this
separately labeled auxiliary control:

```text
BaseDrop_aux = P_BP - P_BR
Z_Base_aux   = BaseDrop_aux - 0.10

Base auxiliary criterion: BaseDrop_aux >= 0.10
```

`0.10` means a 10-percentage-point absolute drop. This is a new auxiliary
criterion, not a change to H3's 90% retention threshold. Report also the paired
difference-in-differences diagnostic:

```text
DiD_H3 = (P_TR - P_TP) - (P_BR - P_BP)
```

The H3 threshold verdict and base auxiliary verdict are always shown in
separate columns.

**The three components are necessary, not sufficient.** All of them are task-set
aggregates, so they are blind to *where* the trained model's successes land. A
model that solves only tasks which never depended on the manual scores
retention 1.0, and if the base drops on the manual-sensitive tasks it never
solves, the auxiliary criterion and `DiD_H3` also pass — while it holds none of
the manual's knowledge. Its successes and the base's manual-sensitive tasks are
disjoint sets, and no aggregate detects that.

Two additions close it. First, freeze `D_manual`, the subset of the manifest
whose registered workflow invokes a predicate governed by the hashed policy
manual, declared from environment metadata before any outcome is read — the
same discipline §4.2 applies to H2's manifest, and for the same reason:
selecting the subset from base *outcomes* would select on model output and
invite regression to the mean. Report `P_TP|D_manual`, `P_TR|D_manual`, and
`Retention_H3|D_manual` beside the full-manifest values. Second, report a
support-overlap diagnostic: the count and fraction of the trained model's
manual-present successes falling on tasks where the base is manual-sensitive at
task level.

The full internalization pattern may be stated only when H3 passes, the base
auxiliary criterion passes, `DiD_H3 > 0`, **and** `P_TP|D_manual > 0` with
non-trivial support overlap. Otherwise the write-up must say which component
failed, or that retention was carried by manual-insensitive tasks.

`Retention_H3` on the full frozen manifest remains the unchanged verdict
statistic. The restricted cells are additional evidence, not a redefinition.

## 6. Confidence intervals and paired tests

### 6.1 Primary hierarchical paired bootstrap

Use 10,000 valid bootstrap replicates and one stored seed. Keep benchmark
strata fixed rather than resampling benchmark labels. In each replicate:

1. resample trained seeds with replacement; for H2, resample each matched
   `F/FG` training-seed pair as one indivisible cluster;
2. within each benchmark, resample task IDs with replacement;
3. within each selected task, resample its eight `run_idx` values with
   replacement; and
4. carry every compared arm/context outcome, token count, and complete H2
   attempted-action event bundle together.

Recompute all numerators, denominators, macro averages, ratios, and threshold
contrasts inside each replicate. Never bootstrap arms or manual conditions
independently. Report percentile 95% two-sided intervals for all finite primary
estimands and for:

```text
Gap_H1, Z_H1_reliability, Z_H1_tokens, Z_H2, Z_H3, Z_Base_aux,
DiD_H3
```

For a ratio whose bootstrap denominator is zero, mark that replicate invalid
for the ratio but retain it for the denominator-free threshold contrast when
possible. Draw until 10,000 valid ratio replicates or 100,000 total draws. If
the cap is reached, report the point estimate, invalid-replicate count, and
`ratio CI=NA`; do not call that ratio statistically established.

For every threshold, report both:

- the mechanical point verdict from §§3.4, 4.4, and 5.3; and
- a confidence label: `ESTABLISHED` when the entire 95% contrast CI is on the
  passing side, `CONTRADICTED` when it is entirely on the failing side, and
  `UNCERTAIN` when it crosses zero.

For H1, the 95% CI for `Gap_H1` must lie strictly above zero and both the
reliability and token contrasts must be established before the combined H1
result receives the confidence label `ESTABLISHED`.

### 6.2 Paired permutation tests

Permutation tests are secondary to estimates and CIs. Use 10,000 permutations
with a stored seed except where exact enumeration is specified below.

- For `pass^4`, compute each task's fractional combinatorial contribution,
  form the paired task-level arm contrast, and independently sign-flip that
  contrast within each fixed benchmark stratum. Recompute the equal-benchmark
  macro contrast. Do not apply McNemar to `pass^4` or `pass^8`.
- For H2, the reward intervention occurs at the trained-checkpoint level, not
  the evaluation-episode level. Swap the `F/FG` labels for an entire matched
  training-seed pair, carrying all of that pair's benchmark/task/run event
  bundles together. Enumerate all `2^|S|` seed-pair swaps when `|S| <= 13`;
  otherwise draw 10,000. Never treat evaluation episodes from one checkpoint
  as independent reward-treatment assignments.
- For H3, swap the manual-present/manual-removed labels within each matched
  episode. For `DiD_H3`, apply the same swap jointly to the trained and base
  outcome pairs for that `(task_id, run_idx)`, then recompute the statistic.

An exact McNemar test is allowed only as a secondary test for genuinely paired
binary `pass^1` episode outcomes, such as H3's present/removed pair. It is not a
test for a fractional combinatorial `pass^k` task statistic.

## 6.9 The frozen rerun rule

Every `INVALID` verdict in this protocol depends on this rule, which was
referenced in three places and defined in none. It is frozen with the rest of
the eval configuration and its hash joins the freeze-artifact list.

**Classification.** A run is reread as *infrastructure* only when its failure is
attributable to the harness or the host, never to the policy. The closed list:
CUDA out-of-memory, driver or device disappearance, host or process kill, disk
or network failure while loading a pinned artifact, user-simulator or grader
transport error, and harness exception raised outside the policy call. Anything
the model did — refusal, parse failure, loop, model-caused timeout, sandbox
violation, `environment_turn_cap` termination — is an observed failure with
`Y=0` and is never rerun. When classification is ambiguous, the run is a model
failure. The ambiguous case must not become the escape hatch.

**Procedure.** Rerun at most **twice**, at the same seed, on the same frozen
provenance tuple, with nothing else changed. Each attempt is logged with its
classification, its raw error, and its attempt index; attempts are retained even
when a later one succeeds. A succeeding rerun contributes its outcome normally.

**Escalation.** After two failed reruns the endpoint is `INVALID`. It is never
imputed, never replaced by a neighbouring run, and never counted as a model
failure. Reruns are exhausted before any result is read, so an outcome can never
influence the decision to rerun it.

**Reporting.** The rerun count and the `INVALID` count are reported per arm
beside the results. A `NOT YET EVALUABLE` project verdict is preferred to a
verdict resting on a cell that was rerun into existence.

## 7. `NA`, `FAIL`, and `INVALID` rules

| Condition | Numeric value | Protocol status |
|---|---|---|
| Model refusal, parse failure, loop, model-caused timeout, or step-cap termination | `Y=0` | Valid observed failure; keep it. |
| Unresolved infrastructure failure or mismatched task/seed/provenance/config | Endpoint unavailable | `INVALID`; repair and rerun without reading a replacement outcome. |
| Fewer than four valid paired runs for an H1 task | `pass^4=NA` | `INVALID`, not a favorable or unfavorable H1 result. |
| One H1 benchmark not yet run | Project H1 `NA` | `NOT YET EVALUABLE`; a labeled per-benchmark provisional row may ship. |
| `Gap_H1 <= 0` | `Closure_H1=NA` | H1 gap criterion `NA: no positive reference gap`; never auto-pass. |
| Finite H1 closure below 0.50 or token ratio above 0.30 | Finite measured value | H1 `FAIL`. |
| H2 control attempted-action denominator is zero | Control rate and reduction `NA` | H2 `NA: no observable control attempts`. |
| Gate-term standard deviation across the `FG` training corpus is zero | Reward contrast undefined | H2 `INVALID: gate reward inert in training`; never `FAIL`. The two conditions trained on identical rewards, so no conclusion about gate rewards is available. |
| H2 gate-treatment denominator is zero while paired control is positive | Treatment rate `NA` | H2 `FAIL: degenerate no-action treatment`. |
| H2 control rate is zero with a positive denominator | Reduction `NA` | H2 `NA: no control failures to reduce`. |
| Finite H2 reduction below 0.50 | Finite measured value | H2 `FAIL`. |
| Trained H3 manual-present `pass^1` is zero | Retention `NA` | H3 `FAIL: no competence to retain`. |
| Finite trained retention below 0.90 | Finite measured value | H3 `FAIL`. |
| Base absolute drop below 0.10 | Finite measured value | Auxiliary base criterion `FAIL`; do not rewrite H3's 0.90 threshold. |
| Too few valid bootstrap draws | CI `NA` | Point verdict remains visible; confidence label is `UNAVAILABLE`. |

No `NA`, `INVALID`, or auxiliary result may be silently converted to zero,
removed from a denominator, or described as a hypothesis pass.

## 8. Minimum hypothesis report

Every report must include:

- experiment ID and complete benchmark provenance tuple;
- all task, evaluation-run, and training-seed counts;
- raw successes and generated-token totals by arm and benchmark;
- H1 component `P4` and token means, positive gap, closure, token ratio, and
  both threshold contrasts;
- H2 `U`, `D`, rates by seed/benchmark, zero-action diagnostics, reduction,
  and `Z_H2`;
- H3 four `pass^1` cells, trained retention, base auxiliary drop, and
  difference-in-differences;
- 95% CIs, point verdicts, confidence labels, permutation results, invalid
  replicate counts, and every `NA`/`INVALID` reason; and
- the H1-PhaseA provisional label whenever Phase B is incomplete.

All result cells remain `TBD` until the frozen evaluation logs exist.
