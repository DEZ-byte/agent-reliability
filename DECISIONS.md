# DECISIONS.md — dated decision log

Append-only. Every architectural or experimental decision gets a dated entry
with the reason. This log is itself a portfolio artifact.

---

## 2026-08-17 — Project reset: v1 blueprints superseded by BLUEPRINT_v2.md

**Context.** Two near-duplicate v1 blueprints (in `Downloads/`) were reviewed
by a 7-agent multi-lens critique (71 findings) plus an external review. Both
reviews agreed on four fatal classes of problems: fabricated results presented
as measured, an infeasible 108-arm compute plan, trivially hackable
substring-based rewards, and a single-turn training stack specced for
multi-turn work. BLUEPRINT_v2.md is now the single source of truth.

### D-001 — Goal and audience
Portfolio project for post-training / ML engineer roles. Report leads with
training-method depth: reward design, ablations, curves, honest statistics.

### D-002 — Compute and budget
Local RTX 4060 8 GB (dev/tests only), ~199 Colab units (SFT/DPO + mid evals),
RunPod spot capped at $30/month (GRPO + Phase B eval). API spend: $0.

### D-003 — Scope: 23 arms, not 108
Full 4-regime × 3-rung grid on ONE primary ≤4B model; {Base, GRPO} × {R0, R2}
spot-checks on a 1.5B and on Llama-3.2-3B; Llama-3.1-8B never trained, used
only as the scaffolded comparator at R0/R1/R2.

### D-004 — Model registry (4 checkpoints, 2 families)
Qwen primary (2.5-3B vs 3-4B decided at M0 by smoke test — see D-013),
Qwen small, Llama-3.2-3B, Llama-3.1-8B. Dropped: Ministral-3B (the v1
checkpoint ID does not exist as open weights), Gemma-2 (fp16 soft-capping
trap), all other tiers (compute).

### D-005 — Benchmark order: single-turn first
Phase A: GSM8K wrapped in our own calculator/REPL tool env + an open
function-calling dataset. Phase B: tau2-bench retail via the upstream package
(pinned; likely tau2-bench-verified) with an adapter — never reimplemented.
Reason: stock TRL/Unsloth GRPO is single-turn; multi-turn RL is a stretch
goal (M6) with a 2-week kill criterion.

### D-006 — $0-API eval stack
User simulator and any diagnostic judge = local open models (Qwen 14B/7B via
vLLM). Headline grading = tau2's deterministic DB × COMMUNICATE basis; no LLM
judge in headline metrics. Cascade escalation target = local 8B, co-resident
in 4-bit on a 24 GB GPU.

### D-007 — Rewards are execution-backed only
All reward terms computed from the parsed, executed tool-call event log and
environment state; substring checks banned. One shared gate engine serves
runtime gates and training rewards. Gate violation zeroes accuracy. Explicit
−0.3 for zero tool calls on tool-required tasks (kills the do-nothing
optimum). Gaming-input unit tests required before any RL run.

### D-008 — Training pipeline order
Base → SFT → {DPO | GRPO}, both from the SFT checkpoint; π_ref = SFT via
adapter-disable. Assistant-token-only loss masking in SFT/DPO/GRPO
(labels = −100 elsewhere). Token-level GRPO loss with masks (v1's
sequence-level formula was wrong). DPO β≈0.1; GRPO β≈0–0.04. Rollout
T = 0.7–0.85. Zero-variance-group and zero-tool-call fractions logged as
health alarms. DPO kept (builder has experience; pairs come from rollouts
already sampled) — dropped last, not first.

### D-009 — Data plan
SFT: 1–3k teacher trajectories from a local open-weight teacher,
rejection-sampled through the deterministic grader; mixed with
xlam/Glaive-style open data (licenses recorded). DPO: 500–2k shared-prefix
pairs from SFT rollouts. Splits committed as JSON ID lists, template-level;
tau2 test split is eval-only. Contamination probe on the math set.

### D-010 — Eval protocol
One run array (n=8 Tier-2 / n=4 Tier-1); pass^k = mean C(c,k)/C(n,k) and
pass@k = mean [1 − C(n−c,k)/C(n,k)] computed from the same array. Paired
tests (McNemar/permutation) + hierarchical bootstrap CIs. Identical decoding
params, step caps, and pinned prompts across arms. Context-protocol table per
arm; internalization ablation (± policy manual) is hypothesis H3. Ablation
ladder isolates the gate reward (H2). ≥2 seeds on the headline GRPO arm.

### D-011 — Trajectory logging and demo
Every eval episode logged as JSONL (task_id, run_idx, prompt, raw_completion,
parsed_tool_calls, sandbox_trace, gate_events, ground_truth,
reward_breakdown). Results tables generated programmatically from logs.
Logs feed an HF Space / Streamlit trajectory viewer.

### D-012 — Reporting rules
No unmeasured number is ever bolded, quoted, or headlined. Hypotheses carry
TBD cells until logs exist. Negative results ship in the same format.
Fabricated v1 tables and the "5×" pre-registered conclusion are discarded;
thesis reframed as an open question with H1–H3.

### D-013 — Deferred to M0 (with owner and deadline)
- Qwen2.5 vs Qwen3 for primary + small models: decided by smoke test
  (Unsloth support, tool-template quality, VRAM fit, tokens/s); result and
  measurements recorded here.
- Exact open function-calling dataset (xlam vs Glaive) after license check.
- Repo license (MIT vs Apache-2.0) after the dataset/model license table
  exists.

### D-014 — Cut from scope
Kubernetes manifest + EKS/GKE diagram; "hardened/secure" sandbox claims
(relabeled best-effort resource sandbox); MCP inside the RL loop; the
one-shot whole-repo generation prompt; "GSM8K-Tool" and "tau2 error_tags"
naming (both nonexistent); `[cite: ...]` pseudo-citations.

## 2026-08-17 — M0 reliability-kernel bootstrap

### D-015 — First implementation slice is GPU-independent
The first code slice is the deterministic path from normalized tool-call
output through parsing, schema validation, dispatch, event logging, gate
replay, reward computation, and evaluation metrics. Model loading, datasets,
training, and benchmark claims remain out of this slice. This tests the shared
runtime/reward semantics before any compute budget is spent.

### D-016 — Gate audit and enforcement are explicit modes
The same predicate engine has two callers. `audit` records an unauthorized
mutative attempt and allows a controlled training-environment dispatch so the
reward can observe it; `enforce` blocks that attempt before runtime dispatch.
Both modes evaluate the same pre-call state. Events preserve `dispatched` and
`succeeded` separately.

### D-017 — Reward edge cases are fixed for M0 tests
- Format is +0.2 only when every emitted tool block parses, validates against
  a registered schema, and at least one call dispatches. Any emitted block
  failing that conjunction scores −0.5; no emitted blocks score 0.
- The gate term is a binary −0.6 per episode, not a per-violation sum.
- Any dispatched mutative attempt whose required pre-call predicate is false
  is a violation even if the handler later fails. Failed authentication does
  not change state, so a following mutation remains unauthorized.
- On tool-required tasks, zero dispatched calls forces accuracy to 0 and adds
  the specified −0.3 efficiency term. Reward code never reads answer prose.

These interpretations close ambiguities in §7.0 without changing the stated
reward magnitudes. Revisit them only as a new dated decision, never silently.

### D-018 — Initial Python lock is intentionally narrow
The CPU reliability kernel supports Python 3.11–3.12 and pins Pydantic plus
its transitive dependencies. The training stack receives a separate lock only
after the M0 Qwen/Unsloth/TRL/tau2 compatibility smoke test; choosing the
current desktop Python by accident would prematurely constrain that stack.

### D-019 — Cross-platform sandbox policy
The sandbox remains a best-effort resource sandbox, not a security boundary.
POSIX workers apply address-space, CPU, and file-size rlimits. Every platform
uses a parent timeout with `process.kill()`; Windows additionally uses a
parent-side memory watchdog when available. Platform capability is recorded
in the result/violation rather than overstated.

### D-020 — Tool execution and reward evidence fail closed
Handlers execute against a deep JSON working copy and commit state only after
both the handler and its output/state validation succeed. A failed auth call
therefore cannot leak authorization state into a following mutation. Every
business-mutative tool must declare at least one gate; execution defaults to
`enforce`, and a missing engine is rejected before any call dispatches.

Each trace records a deterministic fingerprint of the predicate/tool policy
and a digest of all reward-consumed evidence. Reward replay rejects a different
policy or mutated evidence rather than silently scoring it.

### D-021 — Tool calls and gate configuration use strict JSON semantics
Normalized tool blocks reject duplicate keys, `NaN`/infinity, non-finite
numbers, and schema coercion. The version-1 `configs/gates.yaml` file uses JSON
syntax (a YAML 1.2 subset), allowing a strict standard-library loader while the
CPU kernel keeps only one runtime dependency. The file's tool-policy table is
cross-checked against registered mutative tools. `exists` means a path is
present; `not_null` is used when a present null value must still fail, including
the configured order-ID gate.

### D-022 — Sandbox IPC is bounded data, never worker-controlled pickle
Worker results cross the process boundary with `send_bytes`/`recv_bytes` as
strict JSON, capped at 4 MiB in the parent. Source, stdout/stderr, final-value
representations, and exception messages have independent limits. This avoids
unbounded parent allocation and removes pickle deserialization from the
less-trusted worker result path.

### D-023 — Trajectory artifacts require explicit schema versions
Every JSONL record must carry `schema_version=1`; missing and unsupported
versions fail validation. Existing in-memory records are revalidated from
their current Python payload before writing, so shallow post-construction
mutation cannot be silently coerced into a different JSON value.

## 2026-08-18 — M0 live compatibility findings

### D-024 — Re-evaluate the “TRL is single-turn” assumption
The current TRL v1.8 documentation exposes multi-turn tool environments through
`GRPOTrainer(environment_factory=...)`, requires `transformers>=5.2.0` for that
path, and lists both Qwen2.5 and Qwen3 among tested model families:
https://huggingface.co/docs/trl/v1.8.0/en/grpo_trainer and
https://huggingface.co/docs/trl/en/openenv.

This supersedes the implementation assumption in BLUEPRINT_v2.md §7.1 that
stock TRL is single-turn by design. It does not change the milestone order:
Phase A remains first because it is cheaper and independently shippable. Before
M6, compare current TRL OpenEnv against `verifiers` with one small deterministic
episode instead of pre-committing to either multi-turn backend.

### D-025 — Library support claims are smoke-test inputs, not measurements
TRL's official quickstart demonstrates GRPO with Qwen2.5, and Unsloth's official
repository links a Qwen3 advanced-GRPO notebook. These establish candidate
support paths only. They do not establish this project's VRAM fit, throughput,
tool-template quality, or training stability. The Qwen2.5-versus-Qwen3 decision
remains pending until both candidates produce comparable versioned smoke-result
artifacts on the same named hardware and pinned environment.

### D-026 — Keep inference and training chat templates distinct
Transformers can return assistant-token masks only when the active template
contains `{% generation %}` markers. Current TRL supplies separate patched
training templates for Qwen2.5 and Qwen3; its Qwen3 patch also preserves the
assistant prefix when a tool observation is appended:
https://huggingface.co/docs/transformers/main_classes/tokenizer and
https://huggingface.co/docs/trl/en/chat_templates.

The smoke test must therefore hash and inspect both templates. Native model
templates remain the inference baseline. TRL training templates are used for
assistant-only loss only after a fixture proves correct assistant masking and
multi-turn prefix preservation. A successful import alone is not sufficient.

### D-027 — A benchmark result is identified by its full provenance tuple
The M0 scan found two distinct tau2 code/data lines: Sierra's evolving official
repository and Amazon's separately corrected `tau2-bench-verified` fork. The
name “tau2” is therefore not enough to identify a comparable result.

Before Phase B, pin the repository URL, immutable commit, task manifest,
simulator checkpoint and prompt, reward basis, and dependency lock as one run
artifact. Results from different tuples must remain in separate tables unless
a deliberate bridge study reruns both variants. The choice of tuple remains
pending; no existing score is treated as reproduced.

### D-028 — License verification precedes, but does not substitute for, selection
At the verified revisions, `Qwen/Qwen2.5-3B-Instruct` uses the non-commercial
Qwen Research License, while `Qwen/Qwen3-4B` and both scale candidates use
Apache-2.0. The xLAM dataset states CC BY 4.0 and documents its generation and
verification pipeline, but its card also uses “research purposes only” wording.
Glaive states Apache-2.0 only in metadata and supplies neither a full license
file nor comparable provenance documentation.

All model and dataset selections remain pending. The smoke test decides model
fitness, while release-license acceptability is a separate hard constraint.
xLAM is the provisional format-grounding preference, but it is not ingested or
redistributed until its wording is resolved and its access terms are accepted.

### D-029 — Self-correction requires a matched, same-model diagnostic branch
`SELF_CORRECTION_SPEC.md` defines one earliest rollback-safe opportunity per
episode and freezes the failed prefix before any comparison. A neutral
same-model resample, a diagnostic-aware same-model repair, and an immediate 8B
handoff then branch from that evidence. Production R1/R2 traces alone support
descriptive attribution; causal language requires the matched branch bundle.

Local action repair and deterministic end-to-end recovery are separate
outcomes. Retry luck, gate prevention, and success after 8B escalation are
reported separately and are never credited as small-model self-correction.
These diagnostics do not replace production rollouts or alter H1–H3.

### D-030 — Smoke dependencies use a two-stage lock lifecycle
The kernel lock remains small and CPU-only. Before selection measurements,
dependency reconnaissance may iterate without producing a result; it then
freezes a provisional `requirements-smoke.lock` plus an immutable environment
manifest. Every Qwen candidate is measured in a clean recreation of that same
environment. The later training lock may change only as a new recorded
artifact and cannot retroactively change the smoke result.

### D-031 — Qwen selection compares two size-paired generation bundles
The alternatives are Qwen2.5 {3B primary, 1.5B scale} and Qwen3 {4B primary,
1.7B scale}. Both checkpoints in a bundle must pass the technical hard gates;
the primary quality metrics rank bundles and the scale metrics serve as the
next tie-breaker. This preserves the scale arm as a within-generation check.

Qwen3 is scored with `enable_thinking=false` for parity with the direct
Qwen2.5 tool-call condition and the generated-token objective. Thinking mode
may be a labeled diagnostic only. Before a technical leader becomes the
selected bundle, the intended release scope must declare whether
non-commercial upstream model terms are acceptable and filter ineligible
bundles. Imports alone do not establish training support: the final hard gate
is a bounded executed forward/backward or tiny GRPO step.

### D-032 — Rungs are state-machine treatments, not framework labels
`RUNG_PROTOCOL.md` defines an environment turn, agent turn, model decision,
and tool attempt separately. R0 permits one policy generation per natural
agent turn, including in a conversational episode, with no same-turn model
retry. M1 implements the framework-neutral Python R0/R1 core; M4 adds an R2
adapter and LangGraph orchestration only after golden event/state parity.

Exact call redispatch is allowed only for a versioned transient failure that
proves no commit and idempotency. Parse, schema, non-transient tool, and gate
failures require a fresh model decision when the rung budget permits. A small
parent may hand off once to the frozen 8B; the 8B arm is explicitly
`R2-no-escalation` and never escalates to itself.

### D-033 — H1–H3 use frozen operational estimands
`HYPOTHESIS_PROTOCOL.md` defines the confirmatory arms, denominators, zero
handling, benchmark aggregation, paired seed matrix, confidence intervals, and
verdict states before any model result exists. H1 adds primary Base×R0 to the
headline n=8 set, forms gap closure from Base×R0 → GRPO×R0 relative to 8B
Base×R2-no-escalation, and reports the generated-token ratio separately. H2
uses matched scale-model {accuracy+format} versus {accuracy+format+gate}
training runs and measures pre-enforcement attempted mutations under R1 audit
mode. H3 uses paired policy-manual contexts under R0 so runtime gates cannot
mask missing policy knowledge.

McNemar is limited to genuinely paired binary pass^1 outcomes. Fractional
`pass^k` contrasts use a hierarchical paired bootstrap and a task-level paired
permutation statistic. Undefined denominators remain `NA` or `INVALID` under
the protocol and can never become a favorable zero.

### D-034 — Confirmatory protocols supersede stale matrix and cap shorthand
The production grid remains the 23 arms in D-003, but the confirmatory H2
contrast adds two scale-model reward variants evaluated at R1 audit, for 25
total registered arms/configurations. These two runs use n=8 on the frozen
authorization manifest and cannot be replaced by a full-composite checkpoint.

The common runtime limit is 20 environment turns. Rungs intentionally have
different model-decision budgets, so D-010's phrase “identical step caps” is
superseded. Tier-1 arrays support only k in {1,4}; pass^8 requires n=8. M3 may
report only `H1-PhaseA provisional`; project-level H1 remains `NA` until the
Phase B stratum completes at M5. A budget stop cannot redefine these frozen
requirements. H3's 90% trained-model retention is the primary verdict; the
10-point base-model drop is a separately labeled auxiliary criterion.

### D-035 — Smoke and live multi-turn training use separate evidence lanes
The Phase-A/M0 smoke lane uses the Windows stack pinned by
`requirements-smoke.in` and `requirements-smoke.lock`: Unsloth 2026.8.18,
TRL 0.24.0, and Transformers 5.5.0. The current runner implements P0-P6. P5
tests multi-message serialization and assistant-token masking; it does not
execute a live multi-turn environment. P2's exact-one-valid-call condition
defines a measured success, not a compatibility hard gate.

P6 is a bounded rank-4 `q_proj`/`v_proj` LoRA microstep. It reuses the exact P5
mask through the TRL collator, obtains a same-model reference with the PEFT
adapter disabled, and performs one ephemeral SGD step without writing a
checkpoint or making a quality claim. Its implementation has mock-only test
coverage; no checkpoint has executed it. All four executions remain required,
and the machine-readable release gate remains pending, before model selection.

M6 `environment_factory` work uses a separate TRL 1.8 environment without
Unsloth. It must have its own requirements input, lock, manifest, and executed
compatibility evidence. Evidence cannot transfer between these lanes, and a
later M6 lock cannot rewrite the Phase-A/M0 smoke record.

### D-036 — Parse evidence is attributed to a block, and the format term reads only block-attributable failures
An external audit reproduced a reward defect: a fully valid, schema-valid,
dispatched tool call scored the -0.5 format penalty instead of +0.2 whenever
the surrounding prose happened to contain the literal string `</tool_call>`.
The parser correctly refuses to count a stray closing tag as an emitted block,
but `score_episode` tested `not trace.parse.issues`, so any issue at all broke
the conjunction.

BLUEPRINT_v2.md s7.0 and D-017 define the format term as a conjunction over
*emitted blocks*. `ParseIssue` therefore gains `attached_to_block` (default
`True`), the `unexpected_close_tag` issue sets it to `False`, and the reward
reads only block-attributable issues. An unclosed block remains attached and
still fails format, because the parser does count it as emitted. Text outside
the envelope is still preserved as evidence; it simply no longer decides the
format term. The reward magnitudes in s7.0 are unchanged.

Consequence: `ParseIssue` gained a field, so evidence digests computed before
this change will not match. No measured artifact exists yet, so nothing is
invalidated.

### D-037 — Unpaired surrogates are rejected at the parser boundary
A model can emit the JSON escape for a lone surrogate inside tool-call
arguments. `json` accepts it and yields a `str` that UTF-8 cannot encode, so it
crashed evidence hashing with an uncaught `UnicodeEncodeError` and could kill
rollout scoring mid-run.

Such a value is now rejected where it enters, as a reward-visible
`unpaired_surrogate` parse issue covering nested values, tool names, and object
keys. Evidence hashing additionally converts any remaining encode failure into
a described `ValueError` rather than propagating `UnicodeEncodeError`. Failing
closed as scored parse evidence is preferred to crashing a training run.

### D-038 — Result writing is atomic and pre-encoded
`write_trajectory_jsonl` promised that records are validated before the
destination is opened, but encoding happened during the write. One unencodable
record therefore truncated an existing results file.

Records are now validated *and* encoded to bytes before the destination is
touched, then written to a sibling temporary file, flushed, fsynced, and moved
into place with `os.replace`. A failed write removes the temporary file and
leaves the previous results untouched. Evaluation artifacts are the project's
primary evidence, so a partial or destroyed result file is treated as a
correctness defect, not an inconvenience.

### D-039 — Model selection is registry-backed and fail-closed
The smoke configuration no longer accepts a free-form `selection_allowed`
switch. It pins `configs/model_candidates.json` by SHA-256. Each registry
entry carries an independent `release_eligibility` and `release_decision`, and
the four smoke checkpoints also carry their size-paired `smoke_bundle`.

A resolved release gate requires all four smoke entries to use the same
recorded `D-###` decision, each bundle to be consistently eligible or
ineligible, and the configured eligible-bundle list to match the registry.
The decision section must include exact `Release scope:` and
`Release-eligible bundles:` markers. The current gate and all registry entries
remain pending; this decision changes evidence validation, not model or
license selection.

Technical readiness now requires successful P1-P5 probe statuses plus an
executed, passing P6 microstep for all four checkpoints before the two bundles
are compared. P2 strict tool-call rates remain ranking observations rather
than binary compatibility gates, but a failed deterministic-generation probe
cannot be treated as selection-ready.

### D-040 — Pinned TRL 0.24 does not supply the assumed Qwen template helper
D-026 correctly requires `{% generation %}` spans for assistant-token masks,
but its claim that current TRL supplies a public Qwen training-template helper
does not hold for the pinned Phase-A lane. `trl==0.24.0` has no
`trl.chat_template_utils` module. The runner now uses the exact native template
resolved from the tokenizer returned by Unsloth and fails P5 when that template
cannot produce a complete assistant mask.

P3 loads through `FastLanguageModel.from_pretrained`, not raw Transformers,
because the installed Unsloth LoRA path requires the loader-attached training
tokenizer. The runner verifies its immutable revision and attachment, reruns
P5 with that exact tokenizer, and uses it for generation and P6. TRL's public
`DataCollatorForLanguageModeling` remains the P6 label-construction check. No
checkpoint has executed this corrected path yet, so support remains pending
measured evidence.

### D-041 — Concurrent trajectory replacement is serialized in-process
Unique sibling temporary files prevent writers from corrupting one another,
but Windows can still deny two simultaneous `os.replace` calls targeting the
same result path. The writer therefore serializes only the final replacement
step through a bounded set of in-process path locks. Encoding, file writing,
flush, and fsync remain independent and concurrent.

There is no ordering guarantee: the last completed replacement wins, but the
destination is always one complete validated artifact. Cross-process writers
must use separate output paths or an external run-level lock.

### D-042 — Smoke revision and Qwen masking evidence use public, reproducible contracts
D-040's reliance on a loader object's private `_commit_hash` is superseded.
Immutable revision evidence comes from resolving a known repository file in
the exact local Hugging Face `snapshots/<commit>/...` directory. Exposed model
or tokenizer revision metadata is optional, but it must agree with the
requested commit when present. A missing private attribute is not itself a
compatibility failure.

For P5, the native Qwen template remains unchanged and remains the inference
template. The runner derives a separate project-owned, training-only template
by adding `{% generation %}` markers around the single unambiguous assistant
branch. Native and training-only rendering must produce byte-for-byte equal
text and exactly equal token IDs; the only permitted difference is assistant
mask attribution. Missing, duplicated, already instrumented, or ambiguous
branch structure fails closed.

The first negative Qwen3-1.7B compatibility artifact is retained rather than
rewritten. A corrected rerun remains pending. Neither this implementation
decision nor the negative artifact names a winning bundle or supports a
model-quality claim.

### D-043 — Actual CUDA parameters are primary placement evidence when no map is retained
The D-042 rerun proved the immutable revision and assistant-mask corrections,
but Unsloth returned an empty `hf_device_map` after placing every parameter on
`cuda:0`. The runner incorrectly treated the empty map as a conflicting map
and skipped generation.

An absent or empty device map is now recorded as unavailable evidence, not as
affirmative placement evidence and not as a contradiction. P3 passes placement
only when the model has at least one parameter, every actual parameter is on
the configured CUDA device, and no CPU, disk, or meta offload target is
observed. If a non-empty device map is present, every entry must also name that
same CUDA device. A malformed or conflicting map still fails closed.

The second raw Qwen3-1.7B artifact remains preserved. This correction does not
waive the independent P5 prefix-preservation gate, select a model, or establish
model quality.

### D-044 — Strict tool-call validity is measured separately from expected-tool selection
`MODEL_SMOKE_PROTOCOL.md` ranks eligible bundles first on strict tool-call
validity. The generation probe filled `strict_tool_output_by_case` from
`exactly_one_expected_dispatchable_call`, the same expression that already fed
`expected_tool_dispatchable_by_case`, so the reported strict metric and
`every_output_is_strict_and_schema_valid` were duplicates of a different
observation and ranking key 1 did not measure what its name states.

The accumulator now reads `registered_schema_valid_output`: every emitted block
parses strictly, every parsed call is registered and schema-valid, and at least
one call exists. Choosing the expected tool remains the separate, stricter
observation. The two now differ, for example when a model emits a valid call to
the wrong registered tool, or two valid calls where one was required.

No selection has been made, so no recorded decision changes. The two committed
Qwen3-1.7B artifacts predate this fix and their P4 probe was skipped, so neither
contains a strict-validity value; both remain valid historical records.

### D-045 — A skipped probe names its real upstream cause
The deterministic-generation probe recorded the fixed string "4-bit model did
not pass the no-offload hard gate" whenever it was skipped. In the committed
Qwen3-1.7B artifacts the actual cause was a revision-resolution failure, so a
permanent record attributed the skip to a hardware verdict that was never
reached. The probe now reports the upstream probe status and its error text,
matching the pattern P6 already used for its prerequisites.

## 2026-08-18 — M0 gate scoping

### D-046 — `prefix_preserved_after_tool_observation` is a Phase-A diagnostic and stays a multi-turn hard gate

Demoted gate: `P5:prefix_preserved_after_tool_observation`
Demoted on: 2026-08-18
Timing: post_hoc_after_measurement
Scope: `phase-a-windows-unsloth-trl024` / blueprint_7_1_stage_1_single_turn
Still a hard gate for: BLUEPRINT_v2 7.1 Stage 2 scripted 2-4-turn episodes, Stage 3 tau2 multi-turn, the M6 `environment_factory` lane, and any Rung 1/2 scaffold that appends a tool observation to an already-tokenized context.
Re-arm conditions: (1) rerun P5 with this check enforced as a hard gate on the M6 TRL 1.8 / no-Unsloth lane, with its own requirements input, lock, manifest and artifact; (2) diagnose the root cause from the recorded `first_prefix_divergence_index` and the surrounding decoded windows; (3) re-verify assistant-mask exactness on a trajectory with at least two appended tool observations.

**What is demoted.** One of the eleven P5 checks, `full_ids[:len(prefix_ids)] == prefix_ids`. It is
not deleted, renamed, or softened. It is still computed under every scope, still recorded in
`metrics.checks` under the same name with the same expression, and now additionally records the
first divergent token index, so the demoted check reports strictly more evidence than it did as a
hard gate.

**Why.** The check asserts a multi-turn serialization property: that appending a tool observation
leaves the earlier tokenization byte-identical. BLUEPRINT_v2 section 3.1 and section 7.1 Stage 1
specify single-turn verifiable tool tasks for the Phase-A/M0 lane, where one trajectory is tokenized
once and assistant-only loss is taken over a single assistant block. That objective never exercises
the demoted property. This is a scope argument, not a refutation of the measurement.

**The measured fact this sets aside.** Qwen3-4B and Qwen3-1.7B fail this and only this check, at
both sizes, while `project_template_render_matches_native`,
`project_template_token_ids_match_native` and `assistant_mask_exactly_matches_generation_spans` all
pass. The P1 native diagnostic shows the same divergence on Qwen3's own inference template: prefix
309 tokens against a 322-token full render. Qwen2.5-3B and Qwen2.5-1.5B pass all eleven checks and
pass P6. The cause of the Qwen3 divergence is not diagnosed anywhere in this repository.

**The motive, stated plainly.** The owner wants the public portfolio repository to be releasable
under a permissive licence. The technically eligible Qwen2.5 bundle contains
`Qwen/Qwen2.5-3B-Instruct` under the non-commercial Qwen Research License; the Qwen3 bundle is
Apache-2.0. This demotion was proposed after the four measurements were known and after it was known
which bundle it favours. It is therefore recorded as `post_hoc_after_measurement`. The scoping
argument would have been equally correct had it been noticed before any model ran. It was not, and
recording that ordering is not optional. An undisclosed motive is how the v1 blueprint failed.

**The validity precondition.** This demotion is sound only while no Phase-A Stage-1 rollout appends
a tool observation into an already-tokenized model context. If any Stage-1 arm or rung feeds an
observation back, the property is in scope for that arm and this demotion does not cover it.

**What this does not do.** It does not make Qwen3 selection-eligible: P6 has never executed on any
Qwen3 checkpoint, and the demotion only makes P6 reachable. It does not select a bundle, resolve the
release gate, or establish multi-turn compatibility. It does not weaken assistant-only loss masking,
which rests on checks that remain hard gates in every scope and are permanently outside the
demotable set. It does not alter any Qwen2.5 measurement, the reward path, the gate engine, the
evaluation harness, or the ranking keys. It does not transfer Phase-A evidence to M6.

**Mechanism.** The demotion is declared in `configs/model_smoke.json` under `lane.gate_demotions`,
so it rides `config_sha256`. The demotable set is a closed Literal in code and must equal the
multi-turn re-arm set. Scope is resolved once, by `_applied_gate_demotions`, whose first statement
returns an empty tuple when multi-turn or M6 is in scope. A run that relied on the demotion reports
probe status `passed_with_demoted_gates`, never plain `passed`, with a mandatory error naming the
check and this decision, plus `passed_under_preregistered_p5_rule: false`, a candidate-level
`demoted_gate_failures` record, and the candidate's name in the top-level
`candidates_with_demoted_gate_failures` beside `post_hoc_gate_demotion_present: true`. The runner
fails closed if this section loses any of the six marker lines above, and this section's SHA-256 is
pinned into each artifact.

**Historical records.** The artifacts committed before this decision recorded a genuine hard failure
under the stronger rule. They are retained unmodified, are not regenerated, and are not retroactively
reinterpreted as passes. Their `config_sha256` differs from every post-D-046 artifact, so the two
evidence regimes are distinguishable by hash alone.

**Obligation.** Any multi-turn or M6 use of a Qwen3 checkpoint must first satisfy the re-arm
conditions above, on the M6 lane, with its own lock, manifest and artifact.

### D-047 — D-026 is narrowed, not waived
D-026 required assistant-only loss to be used only after a fixture proves correct assistant masking
and multi-turn prefix preservation. D-046 narrows the prefix half of that requirement to multi-turn
scope, where it is reaffirmed in full. The masking half is unchanged and remains a hard gate under
every scope. D-043's statement that its correction does not waive the independent P5 prefix gate
also stands: the gate is re-scoped by a dated decision, not waived.

## 2026-08-18 — M0 bundle selection

### D-048 — Qwen3 {4B, 1.7B} is the selected bundle, under a declared release scope and a disclosed demotion

Release scope: `public-portfolio-permissive`
Release-eligible bundles: `qwen3`

**The release scope, declared.** This repository is a public hiring portfolio.
Released artifacts — code, adapters, generated data, and model cards — must be
redistributable under a permissive licence, and derivative fine-tunes must be
publishable without a separate commercial grant. That requirement is the input
the licence gate needed and had been missing since the registry was written.

**Measured technical evidence, all four checkpoints.** Every candidate has now
executed P6 on the frozen Phase-A lane:

| Candidate | P1-P5 | P6 | selection_eligible |
| :-- | :-- | :-- | :-- |
| `Qwen/Qwen2.5-3B-Instruct` | 11 of 11 P5 checks, clean | executed, passed | true |
| `Qwen/Qwen2.5-1.5B-Instruct` | 11 of 11 P5 checks, clean | executed, passed | true |
| `Qwen/Qwen3-4B` | 10 of 11; `prefix_preserved_after_tool_observation` false | executed, passed | true |
| `Qwen/Qwen3-1.7B` | 10 of 11; same single check false | executed, passed | true |

Both bundles are technically eligible. The technical ladders were **not**
identical: Qwen2.5 cleared eleven of eleven P5 checks, Qwen3 cleared ten and
relies on the D-046 demotion for the eleventh. Any sentence claiming both
bundles "passed P1-P6" is false without that qualifier.

**Why Qwen3 wins.** Not on technical merit, which favours Qwen2.5. On the
declared release scope. `Qwen/Qwen2.5-3B-Instruct` is published under the
non-commercial Qwen Research License, so selecting the Qwen2.5 bundle would make
the primary checkpoint — and, conservatively, every adapter derived from it — a
non-commercial artifact at the centre of a public portfolio. `Qwen/Qwen3-4B` and
`Qwen/Qwen3-1.7B` are Apache-2.0. Under `public-portfolio-permissive` the
Qwen2.5 bundle is ineligible for release regardless of its stronger technical
record, and the licence gate is applied before ranking, exactly as the selection
rule requires.

**What this selection carries.** It is made **with** the D-046 demotion of
`prefix_preserved_after_tool_observation`, which is recorded as
`post_hoc_after_measurement`. The correct description of this bundle is
"Phase-A single-turn eligible, with one of eleven P5 checks demoted (D-046)",
never "passed P1-P6". Every Qwen3 artifact records `passed_with_demoted_gates`,
`passed_under_preregistered_p5_rule: false`, and the candidate's name in
`candidates_with_demoted_gate_failures`.

**The obligation this creates.** Before any Stage 2 scripted multi-turn work,
any Stage 3 tau2 work, or any M6 `environment_factory` work uses a Qwen3
checkpoint, the demoted gate must be re-armed and re-verified under the M6 lane
per the D-046 re-arm conditions. The recorded
`first_prefix_divergence_index` of 277, identical at both sizes, is the starting
point for that diagnosis. If the re-armed gate fails, Stage 2 and Stage 3 must
either fall back to the non-commercially licensed Qwen2.5 bundle — reopening the
licensing problem this selection resolves — or ship with a documented multi-turn
limitation. That contingency is accepted knowingly.

**Not decided here.** The function-calling dataset, the repository licence, and
the Meta gated-model requests remain open. The Qwen2.5 measurements are retained
in full and are not deleted; the bundle is marked ineligible for release, not
wrong.

## 2026-08-18 — M0 hardening

### D-049 — One surrogate rule, enforced at every strict-JSON boundary
The parser rejected unpaired surrogates (D-037), but two other boundaries did
not. `_json_clone` in `src/env/tools.py` proved values with
`json.dumps(value, allow_nan=False)`, whose default `ensure_ascii=True` escapes
a lone surrogate so it round-trips and is accepted, contradicting the function's
own docstring claim to prove strict JSON. `_validate_json_value` in
`src/evaluation/trajectory.py` checked object keys only for being strings, and
returned early on any string value. A surrogate arriving through a tool return
value, through environment state, or as an object key therefore passed
validation and aborted the run later, during evidence hashing or result writing.

The rule now has one definition, `contains_surrogate` in `src/env/models.py`,
used by the parser, the tool boundary, and the trajectory validator. A value
carrying an unpaired surrogate fails where it enters: a tool output or state
mutation raises at dispatch and is recorded as a dispatched, failed event, which
is scored evidence rather than a crash; a trajectory payload is rejected before
its destination file is opened. Ordinary non-ASCII text is unaffected.

### D-050 — The declared package table asserts about the lock, it does not replace it
`_lock_environment_consistency` parsed `requirements-smoke.lock` into
`locked_versions` and then immediately overwrote every entry it cared about with
the hardcoded `EXPECTED_PACKAGES` table. For those fifteen lane-defining
packages the lock was therefore never compared against anything: a drift between
the lock and the table would have been invisible, and the probe would have
reported a matching environment while silently checking the table against
itself.

The expected set is now derived from the lock alone. Direct-URL pins, which
carry no `==` and cover the four CUDA wheels, are read from their wheel
filenames by `_LOCK_URL_PIN_RE`, so all 93 locked distributions participate. The
declared table is retained but demoted to an assertion: any disagreement with
the lock is reported as `declared_lock_drift` and fails the probe. Three tests
pin this, including one asserting that the locked distribution count exceeds the
declared table, so the lock can never quietly shrink to the table again.

### D-051 — The thinking-parity control is measured, and it did hold
D-031 scored Qwen3 with `enable_thinking=false` so the bundles could be compared
on generated tokens without charging hidden reasoning to one side. The harness
passed that kwarg but never checked it: a Jinja template silently ignores a
variable it does not read, so the decisive parity control was an assumption.

Measured on the pinned revisions, offline, by rendering the probe messages twice
with the flag negated:

| Candidate | Control honored | Scored render ends with |
| :-- | :-- | :-- |
| `Qwen/Qwen2.5-3B-Instruct` | no | `<|im_start|>assistant` |
| `Qwen/Qwen2.5-1.5B-Instruct` | no | `<|im_start|>assistant` |
| `Qwen/Qwen3-4B` | yes | `<think>` opened and closed empty |
| `Qwen/Qwen3-1.7B` | yes | `<think>` opened and closed empty |

The control reached the templates that have a thinking mode and changed their
output; it is inert on Qwen2.5, which has none. That is the intended behaviour,
so the D-048 comparison stands. "Not honored" is recorded, not gated: it is
correct for a template without the feature and would be a finding only for a
model that claims one.

The P1 probe now records `chat_template_kwargs` and
`chat_template_kwargs_honored` per candidate, so no future run has to assume it
again. A template that raises while being probed records the error as evidence
instead of failing the run.

### D-052 — Committed measurement records are frozen by hash
`results/smoke_environment.json` was guarded against silent edits; the
model-smoke artifacts, which carry the actual measurements a selection rests on,
were not. Nothing detected a result being edited, re-signed, or deleted.

`results/artifact_manifest.json` now records the SHA-256, byte length, recording
commit, `config_sha256`, and evidence regime of every committed
`model_smoke-*.json`. Four tests enforce it: every artifact matches its frozen
hash, no listed artifact has been deleted, the manifest agrees with each
artifact about whether it declares a gate demotion, and both evidence regimes
remain present so the pre-D-046 failures cannot quietly disappear once the
post-D-046 passes exist.

Verified by tampering: flipping the recorded P5 failure in
`model_smoke-qwen3-4b-1906997.json` to `passed` fails four tests. Adding a run
means adding a manifest entry; it never means changing one.

### D-053 — Eight document contradictions closed
An audit found eight places where two documents disagreed, or where a
load-bearing term was used but never defined. None changed a measurement; all
would have produced an unanswerable question later.

1. **Compute versus arms.** §9 estimated "~8-10 arms x ~114 tasks" while §4
   defines 23 production-grid arms plus 2 confirmatory configurations, and it
   used the full tau2 task set rather than the test split and ignored the
   >=2-seed rule. §9 now derives from §4 and says so, and states that any figure
   disagreeing with §4 is wrong by construction.
2. **H3 population.** §8.4 said "probe set only" for a set defined nowhere,
   while `HYPOTHESIS_PROTOCOL.md` used the frozen Phase B test manifest. There
   is now one population and one normative document.
3. **The simulator risk response.** "Drop to scripted user turns for Phase B"
   would silently swap a frozen provenance item that H1's Phase B stratum and
   all of H3 depend on. The response is now: stop, record a dated decision,
   relabel as a separate variant, and rerun the affected arms.
4. **Self-correction.** It is half the project title and appeared nowhere in the
   canonical blueprint. New §4.1 states what it is, that it is not a hypothesis
   or an arm, that `SELF_CORRECTION_SPEC.md` is normative, and that no H1-H3
   verdict may cite it.
5. **The fallback 7B simulator.** Named with no ID, revision, or licence row, in
   a table whose own rule is that a checkpoint appears only if its ID resolves.
   The fallback is removed; a substitute requires a registry entry and a dated
   decision.
6. **The H2 SFT checkpoint.** Both confirmatory configurations initialize from a
   scale-model SFT checkpoint that is in no arm, no milestone, and no budget.
   §4 now separates required training artifacts from reported arms.
7. **The 20-turn cap.** `SELF_CORRECTION_SPEC.md` froze a literal "step cap"
   that `RUNG_PROTOCOL.md` defines as a derived `environment_turn_cap`. It now
   cites the derived cap.
8. **The frozen rerun rule.** Every `INVALID` verdict depended on it and nothing
   defined it. `HYPOTHESIS_PROTOCOL.md` §6.9 now fixes the closed list of
   infrastructure causes, sends ambiguous cases to the model-failure side, caps
   reruns at two at the same seed, requires reruns to be exhausted before any
   outcome is read, and requires rerun and `INVALID` counts to be reported.

### D-054 — The repository is Apache-2.0, and that covers this repository only
The owner chose Apache-2.0 for the project's own source, configuration,
protocols, and documentation. `LICENSE` is the canonical text from
apache.org, byte-identical except for the appendix copyright line, which now
reads `Copyright 2026 Anirudh Raj Sharma`.

Apache-2.0 is the closer match to the selected bundle: D-048 chose Qwen3
{4B, 1.7B} under `public-portfolio-permissive`, and those weights are
Apache-2.0. Choosing the same licence for the repository keeps the code and the
model terms aligned and satisfies the attribution and notice duties in one
place.

A repository licence cannot relicense anything upstream, so `NOTICE` states the
boundary explicitly: model weights, adapters trained from them, generated
trajectories, and third-party datasets each keep their own terms, recorded per
artifact with verified revisions in `data/LICENSES.md`. It names the two traps
specifically — `Qwen/Qwen2.5-3B-Instruct` is non-commercial and is not covered
by the Apache-2.0 bundle decision, and Llama weights carry Meta's attribution
and derivative-naming obligations.

### D-055 — xLAM access is granted; the release-scope question is still open
The owner has accepted the `Salesforce/xlam-function-calling-60k` access gate,
so the dataset can now be fetched. That resolves access only.

The unresolved item from D-028 stands: the card declares CC BY 4.0 in metadata
while its ethical section says the release is "for research purposes only".
Under the `public-portfolio-permissive` scope declared in D-048, training a
published adapter on that data is exactly the case the two statements disagree
about. The dataset is therefore not ingested yet. Resolving it means one of:
obtaining clarification from the publisher; using xLAM but publishing neither
the data nor adapters derived from it; or generating format-grounding data from
an Apache-2.0 teacher instead. This is a licensing-risk judgement for the owner,
not a technical finding, and it is recorded as open rather than assumed away.

### D-056 — CI caught a lone surrogate inside the surrogate rule itself
The first GitHub Actions run failed all four jobs. The cause was in
`contains_surrogate`, the helper D-049 introduced: its docstring named the JSON
escape for a lone surrogate in a plain (non-raw) string, so Python turned those
six characters into a real U+D800 codepoint at compile time. The function
written to keep unpaired surrogates out of the project contained one.

The docstring is now a raw string. `tests/test_tools.py` adds a guard that
compiles every module under `src/` and `scripts/`, walks nested code objects,
and fails if any constant holds a surrogate. Nested traversal is required: a
function docstring is not a module-level constant, and a first version of the
guard missed the defect for exactly that reason. Test files are excluded on
purpose, because their fixtures must carry the hostile value.

Two things this establishes. The local suite passed on Python 3.12 with the
defect present, so a green local run was not evidence; only CI, on a clean
checkout, was. And the guard was verified by reintroducing the defect and
watching it fail, not by assuming it would.

### D-057 — Line-ending translation is disabled for every hashed file
The second CI run failed all four jobs again. The cause was not the code: it was
Git rewriting bytes on checkout.

This project's integrity model is content hashing. `results/artifact_manifest.json`
records a SHA-256 per measurement artifact, `results/smoke_environment.json`
records one per source file, and every artifact pins `config_sha256` and
`release_registry_sha256`. All of those digests were computed against a Windows
working tree with `core.autocrlf=true`, where the checked-out files carry CRLF
while the stored blobs carry LF. The same commit therefore produced different
bytes on Windows and Linux, and every recorded digest failed on any machine but
the one that wrote it. Reproduced in a clean clone: the artifact hashed to
`501bac8f…` on disk, and normalising CRLF to LF reproduced the recorded
`281af3ef…` exactly.

`.gitattributes` now sets `* text=auto eol=lf`, so source is LF in the working
tree on every platform, and marks `results/*.json`, `configs/*.json` and
`*.lock` as `-text`, so the files whose bytes are hashed are never translated by
any local setting. The evidence files keep the exact bytes they were recorded
with rather than being rewritten to match a convention, which is the right
trade for frozen measurement records (D-052).

The lesson is the one CI exists to teach: a green local suite proved only that
the digests matched the machine that produced them. Content-addressed files must
be exempt from any transformation the version-control system is allowed to
apply.

## 2026-08-20 — Function-calling dataset

### D-058 — xLAM is adopted for format grounding, with its licence conflict stated rather than resolved
The owner selected `Salesforce/xlam-function-calling-60k` and accepted that
artifacts derived from it will be published under the
`public-portfolio-permissive` scope of D-048. Glaive is not adopted.

**The conflict, unresolved.** The machine-readable licence field is
`cc-by-4.0`, confirmed live against the Hub API on 2026-08-20. The card's
ethical-considerations prose separately describes the release as being for
research purposes only (recorded in D-028; the card body is behind the access
gate and was not re-fetched for this entry). Those two statements disagree about
exactly this project's case: training a published adapter. This decision does
not resolve that disagreement. It records that the owner weighed it, chose the
declared licence field as controlling, and accepted the residual risk.

**Why it is defensible.** CC BY 4.0 is the formal licence identifier the
publisher set, it permits adaptation including commercially, and the
research-purposes sentence sits in an ethics narrative rather than in licence
terms. A reader can check both claims from this entry.

**Why it is still a risk.** A publisher's stated intent is not nothing. If
Salesforce clarifies that the narrower reading governs, adapters trained on this
data may need withdrawing or relicensing. That cost is accepted knowingly, and
this paragraph exists so the trade is visible rather than discovered later.

**Obligations this creates.** CC BY 4.0 requires, on every released artifact
derived from the data: attribution to Salesforce, a link to the licence, a
copyright and disclaimer notice, and an explicit statement that changes were
made. The access gate additionally requires citing APIGen. These go in the
release checklist in `data/LICENSES.md`, not in a footnote.

**Scope.** Format grounding only, mixed with teacher-generated trajectories per
BLUEPRINT_v2 section 5.2. The raw dataset is never redistributed from this
repository; only split and selection manifests referencing source IDs are
committed. Whether any grounding data is needed at all is a separate question,
answered at M1 by the measured format error rate: the four checkpoints already
emit strictly valid, correctly-selected tool calls on every probe case.

### D-059 — Meta gated access is granted and the pinned revisions still match
Verified on 2026-08-20 against the authenticated Hub API. Both
`meta-llama/Llama-3.2-3B-Instruct` and `meta-llama/Llama-3.1-8B-Instruct`
resolve, and each returns exactly the revision recorded in
`configs/model_candidates.json` (`0cb88a4f…` and `0e9e39f…`). Access was checked
by resolving the pinned commit, not by trusting the approval email, because an
approval says nothing about whether the revision this project depends on still
exists.

Weights are not downloaded. That is an M1 step: roughly 16 GB across the two
checkpoints, and it belongs in a watched run rather than a background one.

The Llama Community License obligations recorded in `data/LICENSES.md` apply
from here on. `meta-llama/Llama-3.2-3B-Instruct` is trained in this project, so
any released derivative must carry "Built with Llama" and a name beginning with
"Llama". `meta-llama/Llama-3.1-8B-Instruct` is inference-only (D-048's
comparator) and produces no derivative weights.

## 2026-08-20 — M1 Phase A environment

### D-060 — Phase A accuracy is the last executed calculator result
`src/env/phase_a.py` wraps GSM8K in this project's own calculator tool. The
grading rule is the reason the wrapper exists.

An episode's answer is the value returned by its last successful `calculator`
dispatch. Prose is never read, so a model that writes the correct number
without calling anything scores zero accuracy and takes the zero-tool-call
penalty. This is section 7.0's execution-backed accuracy applied to a
single-turn math task, and it is what makes a memorised GSM8K answer worthless
here.

Three supporting choices. Numbers are compared with an absolute tolerance of
1e-6 rather than by string, because a model may reach an integer answer through
division and land on a float. The gold answer is read from GSM8K's `####`
marker and a missing marker raises rather than defaulting, since an ungradeable
task must not silently become a wrong score. The calculator evaluates only
inside the existing sandbox, so imports, dunder access, and long-running code
are rejected at the boundary; a sandbox violation is recorded as a failed
dispatch, which is scored evidence, rather than an exception that ends the
episode.

The calculator declares no gates. It computes and returns a number without
touching state a gate would protect, and `ToolSpec` rejects gates on a
non-mutative tool.

### D-061 — Phase A task data is GSM8K at a pinned revision, split by manifest
`openai/gsm8k` config `main` at revision `740312add88f…`, MIT licensed, public,
verified through the Hub API on 2026-08-20. MIT sits comfortably inside the
`public-portfolio-permissive` scope, unlike the xLAM caveat in D-058.

`scripts/build_phase_a_splits.py` writes `configs/splits/phase_a_gsm8k.json`:
1,000 train tasks from the upstream train split, and 100 dev plus 150 test
tasks drawn disjointly from the upstream test split. Only IDs, source indices,
and content hashes are committed. The dataset is never redistributed here.

The build is deterministic: a pinned revision, seed 20260820, and a sort before
sampling. `--check` rebuilds and fails if one byte would differ, so the
committed manifest is verifiable rather than trusted.

GSM8K has no template field, so each item is its own template and the
template-level splitting required by BLUEPRINT section 5.4 reduces to
instance-level here. Exact-duplicate questions are dropped before sampling,
which is the only paraphrase twin detectable without a similarity model. Tests
assert that no task ID and no content hash appears in two splits, so a later
change cannot quietly let a training item into the evaluation set.

The split tests run offline against the committed manifest, because CI has no
network. Regenerating is a separate, deliberate step.

### D-062 — Execution-backed accuracy does not stop laundered recall, and that is now measured
D-060 stops a model scoring by writing a remembered answer in prose. It does not
stop the model recalling the answer and passing it to the tool as
`calculator("391")`. That call dispatches, succeeds, returns 391, and is graded
correct while performing no arithmetic.

This is a real gap in the pre-registered reward, found while building the
contamination probe BLUEPRINT section 5.4 requires. It matters here more than it
would elsewhere: GSM8K is public web text, the base checkpoints were trained on
public web text, and Phase A is the milestone that produces the first baseline
numbers.

Two measurements now exist. `expression_does_arithmetic` parses a calculator
expression and reports whether it computes anything; a bare literal, a
parenthesised literal, and unparseable text all count as no. `RecallProbe`
scores a no-tool attempt at a task, which is the direct question of whether the
model can state the answer without computing it. Prose is read in exactly one
place, `extract_final_number`, and its docstring says it must never score a
task; measuring recall is the one thing it is for.

**Recorded, not penalised.** `answered_without_arithmetic` is a diagnostic on
the trace. Turning it into a reward term would change a pre-registered reward
after seeing that it can be gamed, which needs its own decision and its own
disclosure, exactly as D-046 did. The honest order is: measure how often it
happens, then decide.

The obligation this creates: no Phase A accuracy number may be reported without
the recall rate and the no-arithmetic rate beside it. A baseline that looks
strong while both are high has not been shown to reason.

### D-063 — The no-tool probe measures two things, and the first is not memorisation
The first version of this probe asked each model to answer with no tool and
called the correct fraction a recall rate. Run on eight tasks, Qwen3-1.7B scored
0.625, and the retained completions show why that label was wrong: the model was
reasoning step by step in prose, ignoring an instruction not to show working.
It was solving, not remembering. Reporting that as memorisation would have
overstated contamination and understated capability, in the same document that
forbids exactly this kind of mislabelling.

The probe now runs two conditions against the same frozen test split:

`unconstrained` gives 256 new tokens and asks for the answer at the end. It
measures capability without a calculator, which is the honest floor a Phase A
baseline is compared against.

`token_starved` gives 12 new tokens and asks for the number alone. Multi-step
arithmetic does not fit in twelve tokens, so a correct answer here is evidence
the model recalled it. This is the contamination signal BLUEPRINT section 5.4
asks for.

Both are recorded per candidate as `no_tool_solve_rate` and
`immediate_answer_rate`, and the probe type is a closed literal on every record,
so a result cannot be stored without saying which question it answers. Raw
completions are retained under both conditions, which is what made the original
error visible.

Neither number is a task score. Together with the no-arithmetic rate from D-062
they set the context any Phase A accuracy figure must be read in.

## 2026-08-20 — Phase A contamination measured

### D-064 — GSM8K memorisation is at chance; unaided capability is high, which moves the risk
Measured over the frozen 150-task Phase A test split, greedy, seed 20260820,
artifact `results/contamination-qwen3-57b2bc2.json` at commit `57b2bc2`.

| Checkpoint | Unaided solve rate | 95% CI | Immediate answer rate | 95% CI | Shuffled-guess baseline |
| :-- | :-- | :-- | :-- | :-- | :-- |
| `Qwen/Qwen3-4B` | 0.707 (106/150) | [0.629, 0.774] | 0.040 (6/150) | [0.018, 0.085] | 0.015 |
| `Qwen/Qwen3-1.7B` | 0.640 (96/150) | [0.561, 0.712] | 0.013 (2/150) | [0.004, 0.047] | 0.012 |

**Contamination is not a problem here.** The token-starved rate is the recall
signal, and for the 1.7B it is 0.013 against a shuffled-guess baseline of 0.012,
which is chance. For the 4B it is 0.040 against 0.015, slightly above chance with
a confidence interval that reaches down near it. All six 4B hits are small round
numbers (3, 4, 7, 10, 300, 360), exactly where a guess coincides. Treat 0.040 as
an upper bound, not an estimate. GSM8K memorisation does not threaten Phase A.

**The unaided solve rate is the finding that matters.** Both checkpoints solve
most of this split by reasoning in prose with no calculator: 0.707 and 0.640.
Phase A therefore does not measure whether these models can do arithmetic. It
measures whether they will use a tool correctly and repeatably, which is what
this project is actually about, so the instrument is still the right one.

**It does relocate the risk, and sharply.** D-062 recorded that a model can
recall an answer and launder it through the tool as `calculator("391")`, scoring
correct while computing nothing. That was written as a hypothetical. These
numbers make it likely rather than hypothetical: a model that can reach the
answer unaided two times in three has every reason to compute mentally and pass
the result to the tool. `answered_without_arithmetic` therefore stops being a
curiosity and becomes a required reported quantity.

**Consequences, pre-committed now rather than after seeing baselines.** Every
Phase A accuracy figure is reported beside its no-arithmetic rate. If that rate
is material, the accuracy number describes tool-call compliance, not
tool-assisted problem solving, and must be described that way. Whether the
no-arithmetic case should carry a reward penalty is deliberately still open:
changing a pre-registered reward after learning it can be gamed needs its own
dated decision, and deciding it before the baseline exists would be guessing.

Llama checkpoints are not measured here. Their weights are approved but not
downloaded, and this artifact covers only the selected Qwen3 bundle.

## 2026-08-20 — Pre-training design review

### D-065 — H2 was unfalsifiable as scheduled, and five other design defects
An external review raised ten claims. I had each verified independently against
the files and the running code before changing anything. Twenty-three
sub-claims were confirmed, four partly, none refuted. The review was right, and
in two places the verification found the defect to be worse than reported.

**The critical one: H2 could not have fired.** The two H2 configurations and the
whole §7.4 ablation ladder were scheduled at M3, which trains on Phase A. The
Phase A registry holds one non-mutative tool, so `GateEngine.replay` discards
every event before a predicate is read and the gate term is identically 0.0 on
every Phase A episode. Verified by execution: a correct episode scores
`gate = 0.0` under an empty gate config and under one declaring an
`authenticated` predicate alike. With efficiency disabled in both conditions,
`F` and `FG` receive bit-identical rewards; under the frozen seeds and data
order of §4.1 they receive identical gradients. The contrast would have measured
GPU nondeterminism.

Worse than reported, in two ways. The verification found the whole four-rung
ladder inert for the same reason, not only the confirmatory pair. And §9 budgets
the H2 evaluation at M5 against gate-bearing tau2 data, so the design trained on
gate-free data at M3 and evaluated on gate-bearing data at M5 — a mismatch
across milestones, not within one. The `NA: no observable control attempts`
guard would not have caught it: that guard keys on the evaluation denominator,
which would be healthy, so H2 would have returned a finite near-zero reduction,
`FAIL`, and the pre-written sentence "gate rewards do not internalize the
constraint". A false conclusion about the project's own distinguishing design
choice.

Fixed by making the dependency explicit and schedulable. Stage 2 was the only
gate-bearing training data the blueprint named and belonged to no milestone; it
now has one (M3b) and a risk-register row. `HYPOTHESIS_PROTOCOL.md` §4.1 gains a
training-data requirement and two pre-outcome reported quantities, and §7 gains
`INVALID: gate reward inert in training`, which is never `FAIL`.

**The other five confirmed defects.**

H1 credited "GRPO" for a closure the design cannot attribute to GRPO alone,
since GRPO initialises from SFT. Reworded to credit the SFT-initialised
pipeline, with SFT×R0 available as a labelled auxiliary decomposition.

H3's three components are all task-set aggregates, so a model solving only
manual-insensitive tasks passes every one while holding none of the manual's
knowledge. The reviewer's fix — subset on tasks where the base shows an effect —
selects on model outcomes, which §4.2 forbids for H2's manifest and which would
invite regression to the mean. Instead `D_manual` is defined from environment
metadata before any outcome is read, restricted cells are reported beside the
full-manifest ones, and a support-overlap diagnostic is required before the
internalization pattern may be claimed.

§5.4 still said the execution-backed reward makes memorised answers score zero
"regardless". D-062 had already refuted that and D-064 had measured the unaided
solve rates that make the laundering path likely. The decisions were ahead of
the spec, which is the direction that misleads a reader of the canonical
document. §5.4 and the §7.0 accuracy row now both carry the caveat and the
reporting duty.

Stage 3 trains on full tau2 multi-turn, which needs user turns, while §6.2
claimed training never depends on the simulator and discharged that only for
Stages 1-2. Stage 3 is now named as the sole simulator-dependent training stage,
and the §11 swap rule extends to checkpoints it produces.

§4.1 scheduled the full C/R/E self-correction bundle on all seven Tier-2 arms.
The normative documents exclude four outright and reduce two to C/R-only,
because R1 has no escalation target. Exactly one arm supports the full bundle:
primary GRPO×R2.

**A correct piece of RL reasoning I had got wrong.** §7.3 listed the dense format
term as a mitigation for zero-variance groups. GRPO advantages are
group-relative, so a term identical across all G candidates shifts the mean,
leaves the standard deviation untouched, and cancels exactly. Verified
numerically: advantages computed with and without the format term are
bit-identical when it does not vary. The condition under which it contributes
nothing is zero within-group dispersion of that term — not "accuracy is still
0", which is what §7.3 said — and that is precisely the state mitigation 1 is
designed to produce. The mitigation self-extinguished in the regime it existed
for. Replaced with the real one: log per-component within-group standard
deviation, and resample or drop zero-variance groups rather than
backpropagating a zero advantage.

**Budget and staleness.** §9's Phase B estimate was written additively over a
subset, double-counting the seven Tier-2 arms by 28 runs per task; it now states
the correct decomposition. Five cost blocks the blueprint mandates elsewhere had
no line — the four ladder rungs, the scale-model SFT checkpoint, H3's
manual-removed re-runs, the self-correction branches, and Stage 2 itself — and
now do. Six smaller staleness items were corrected, including a `(or 7B)`
simulator fallback D-053 had already removed elsewhere, two wrong section
cross-references, a settled dataset choice still presented as open, a duplicated
PLAN entry both ticked and untitcked, and a 72B teacher example that no budgeted
card can hold.

**What this cost and what it bought.** No measurement changed; every fix is to a
pre-registration, which is the only time such fixes are free. Had the review
arrived after M3, the H2 result would have been a published false negative about
the project's central claim.

### D-066 — Closing the leftovers the D-065 sweep missed
A follow-up review found four real leftovers and five mechanical ones. All are
confirmed and fixed. Three sat in text D-065 had edited immediately beside,
which is the ordinary failure mode of a large sweep: the sentence being changed
gets attention and its neighbour does not.

**The arms table was broken by my own edit.** The H2 note was inserted between
the Scale H2 row and the Llama rows, splitting the §4 table so its last three
rows rendered as a headerless fragment. Moved below the table.

**§4 still asserted the double-count §9 had just corrected.** Tier 1 read "all
23 production-grid arms: n=4" and Tier 2 "adds pass^8", which is the additive
reading. Since §9 now declares §4 the source for arms, the false sentence had
moved into the section of record. Tier 1 is now the 16 arms that are not Tier 2,
and Tier 2 is n=8 **instead of** n=4, with all 23 arms still reporting pass^1
and pass^4.

**§9 sent the H2 pair to the wrong population.** It costed all three groups
"each over the frozen tau2-retail test split", but §4 and
`HYPOTHESIS_PROTOCOL.md` §4.2 put the confirmatory pair on the frozen
authorization manifest, which is different and smaller. Separated and costed on
its own.

**An undefined word inside a licence condition.** D-065 gated the
internalization claim on "non-trivial support overlap". §5.4 of the same
document exists to replace exactly that kind of word — it swaps "materially" for
0.10 — so this was a fresh instance of the defect the protocol already
prohibits. `Overlap_H3 >= 0.25` is now frozen with a stated numerator and
denominator, reported whatever the verdict.

**MODEL_SMOKE_PROTOCOL.md was never swept.** It still said no checkpoint had
executed P6 and that the release gate was pending with `eligible_bundles=[]`,
both false since D-048, and its "must pass P1-P6" wording carried no demotion
qualifier — which together produce the exact sentence D-048 forbids about the
selected bundle. Rewritten, including the qualifier.

**Also.** The H2 negative-outcome column still carried the false sentence
unconditionally; it is now conditional on the gate term having demonstrably
varied in training, with `INVALID` otherwise. The §11 simulator-swap rule now
also covers an SFT checkpoint trained on simulator-assisted trajectories, not
only checkpoints Stage 3 produces. M3 no longer claims the ablation ladder that
M3b owns. §7.4 now says why ladder rungs 2 and 3 are trained fresh rather than
reusing the confirmatory checkpoints: a ladder whose rungs differ in curriculum
as well as reward cannot attribute a difference to the reward. The `INVALID`
row no longer says the two conditions "trained on identical rewards", which is
exact only when the constant is zero. `data/README.md` no longer says no dataset
has been selected, which D-058 and D-061 falsified. Stray parenthesis in the
teacher bullet removed.

---

### D-067 — Evaluation decoding is frozen at temperature 0.7, one setting for every model
**Date:** 2026-08-20 · **Status:** accepted · **Supersedes:** nothing

`configs/eval.yaml` now pins decoding, run counts, seeds, the turn cap and the
reporting rules before the first measured run, as section 10.1 requires. Two
choices in it need justifying, because neither was pinned anywhere in the
existing documents and both are expensive to revisit.

**Sampling rather than greedy, at temperature 0.7 and top_p 0.95.** pass^k asks
whether a model succeeds on all k independent attempts. At temperature 0 the k
attempts are identical, pass^k equals pass^1 by construction, and the number
measures nothing this project is about. Sampling is therefore not a tuning
preference here; it is what makes the metric a metric. 0.7 with top_p 0.95 is
the ordinary default for this family of checkpoints.

**One setting for every model, not each publisher's recommendation.** Model
cards recommend different decoding per family. Adopting each family's own
recommendation would let a family's tuning enter a comparison that is supposed
to isolate training and scaffolding, and no later analysis could separate the
two. A shared setting disadvantages whichever family the setting suits least.
That cost is accepted, and it is stated rather than hidden, because the
alternative is a confound.

**What this costs.** Every artifact recorded under this configuration is
comparable only to other artifacts recorded under it. Changing any value in
`configs/eval.yaml` invalidates the comparison with everything already measured
and needs its own dated decision. The file's SHA-256 is written into every
result artifact, so a silent change shows up as a changed hash rather than as
an unexplained score movement.

---

### D-068 — First Phase A baseline: the failure is wrong arithmetic, not malformed calls
**Date:** 2026-08-20 · **Status:** accepted · **Supersedes:** nothing

The first measured run of the project. Both Qwen3 checkpoints, both rungs, the
frozen 150-task GSM8K test split, four runs per task: 2,400 episodes. Recorded
in `results/baseline-phase_a-3cc174f.json` and frozen by hash.

| model | rung | pass^1 [95% CI] | pass^4 [95% CI] | pass@4 | no-arith |
| --- | --- | --- | --- | --- | --- |
| Qwen3-4B | R0 | 0.598 [0.522, 0.673] | 0.547 [0.467, 0.627] | 0.647 | 0.000 |
| Qwen3-4B | R1 | 0.608 [0.532, 0.683] | 0.567 [0.487, 0.647] | 0.647 | 0.003 |
| Qwen3-1.7B | R0 | 0.303 [0.237, 0.373] | 0.247 [0.180, 0.320] | 0.353 | 0.015 |
| Qwen3-1.7B | R1 | 0.333 [0.262, 0.405] | 0.287 [0.213, 0.360] | 0.380 | 0.017 |

**The reliability gap the project exists to study is present and measurable.**
pass@4 minus pass^4 is 0.100 for the 4B and 0.107 for the 1.7B under R0. About
a tenth of all tasks are solved on some attempts and failed on others. Reporting
pass@4 alone would describe these models as 10 points better than they are when
every attempt has to hold.

**R1 is not distinguishable from R0 on the larger model, and barely so on the
smaller one.** A paired permutation test on per-task pass^4, 20,000 resamples,
same tasks and same seeds: 4B +0.020, p=0.25; 1.7B +0.040, p=0.031. The 1.7B
result is nominally significant and should still not be leaned on. Only 6 of
150 tasks disagree between the rungs, there is no correction for testing two
models, and a result resting on 6 discordant tasks is one relabelled task away
from crossing 0.05 in either direction. The honest summary is that one extra
model decision buys little here, and that any gain is small enough to need more
tasks to pin down.

**Why it buys little is the useful part, and it constrains what scaffolding can
do.** R1's second decision only fires when the first produced no action at all.
That happened in 22 of 600 episodes for the 4B and 62 of 600 for the 1.7B. When
it did fire it worked reasonably well — roughly 28% of retried episodes reached
a correct answer on both models. But it cannot fire on the dominant failure. Of
the 4B's R0 failures, 219 were a well-formed calculator call that computed the
wrong thing and 22 were a missing call; for the 1.7B, 356 against 62. Between
85% and 91% of failures are the model calling the tool correctly and getting the
arithmetic wrong.

This bounds the R-rungs from measurement rather than from argument. Feedback
scaffolding of this kind repairs the form of a call. The failures here are
overwhelmingly in the content of the call, where a retry that is told only that
nothing parsed has nothing to act on. It is a reason to expect the training arms
to have room the scaffolding arms do not, and it is a prediction that can now be
checked rather than assumed. It does not yet support H1, which needs the
trained arms measured.

**Laundering is near zero in the untrained checkpoints:** 0.000 to 0.017. The
D-062 risk is real but is not inflating these particular numbers. The rate must
keep being reported after training, when a reward for a passing call is exactly
the pressure that would raise it.

**One observation that is suggestive and not yet a result.** D-064 measured
unaided prose accuracy on the same 150 tasks at 0.707 for the 4B and 0.640 for
the 1.7B. With a calculator, pass^1 is 0.598 and 0.303. The tool interface
appears to cost accuracy, drastically so for the smaller model. This is not a
controlled contrast: the probe was greedy, the baseline samples at 0.7, and the
prompts differ. The gap for the 1.7B is far too large to be explained by
decoding alone, but the clean version — same decoding, same tasks, tool against
no tool — has not been run, so nothing is claimed from it yet.

---

### D-069 — Phase A cannot train self-correction on the failure that dominates it
**Date:** 2026-08-20 · **Status:** accepted · **Supersedes:** nothing
**Label:** `post_hoc_after_measurement`

M2's SFT set retains single-decision trajectories only. The `r1_recovery` shape
— a failed attempt, a structured observation, a successful retry — is dropped
from Phase A and moves to Stage 2 (M3b).

**The reason is structural, not a shortfall.** `run_episode` ends an episode as
soon as a calculator call executes successfully:

```python
answered = any(is_answering_event(event) for event in trace.tool_events)
if answered:
    terminal_reason = "answered"
    break
```

`is_answering_event` asks whether the call *ran*, never whether it was *right*.
So a model that computes `2 + 3` for a problem whose answer is 6 gets a
successful tool result, and the episode is over. R1's second decision fires only
when the first produced no successful call at all.

**That is correct behaviour and must not be changed.** The only way to give the
model a second decision after a wrong-but-successful call is to tell it the
answer is wrong, which is grader information. Section 8.2 requires cascade
triggers to be runtime-observable and keeps the grader invisible to the agent. A
Phase A environment that reveals wrongness at runtime would leak the grader into
every rollout and invalidate the arm.

**So the recoverable failures are exactly the observable ones**, and D-068
measured how few they are. Confirmed independently on the dev split, 400 R1
episodes per model:

| model | 2nd decision fired | recovered | distinct tasks yielding a recovery |
| --- | --- | --- | --- |
| Qwen3-4B | 17 of 400 (4.2%) | 1 (0.2%) | 1 of 100 |
| Qwen3-1.7B | 49 of 400 (12.2%) | 12 (3.0%) | 5 of 100 |

Scaled to the 1,000-task train split that is roughly 10 and 50 usable recovery
trajectories. The build plan this replaces assumed 15–25% of rows, or 150–250.
The gap is two orders of magnitude on the primary model, and it is not a
sampling artefact: it follows from the loop above.

**What this costs, stated plainly.** Self-correction is in this project's title
and Phase A cannot measure it. Phase A is single-turn with one non-mutative
tool, so almost nothing observable goes wrong; the observable band is 0.25–4.25%
of first decisions (D-070). Stage 2 is where tool errors, gate blocks and failed
preconditions are both frequent and legitimately visible, so that is where the
self-correction claims have to be made. Until Stage 2 exists this project has
measured tool reliability and not self-correction, and no summary of it should
say otherwise.

**Consequences.** `configs/train_config.yaml` pins `generation.rung: "R0"`.
D-046's demoted `prefix_preserved_after_tool_observation` gate stays untouched,
because a single-turn trajectory never appends a tool observation, so nothing
re-arms it and no multi-turn template swap is needed. The hybrid the build plan
floated — student first attempts corrected by a teacher — is not smuggled in
here; it is a candidate DPO-negative source for M3 and needs its own decision.

---

### D-070 — No format-grounding data is mixed in, because the schema failure rate is zero
**Date:** 2026-08-20 · **Status:** accepted · **Supersedes:** nothing
**Label:** `post_hoc_after_measurement`

`configs/train_config.yaml` pins `format_grounding.fraction: 0.0`.
`Salesforce/xlam-function-calling-60k` stays selected and licensed (D-058) but
contributes no rows to Phase A SFT.

Section 5.1 deferred this to a measured format error rate and allowed in
advance that the answer might be zero. It is. Measured on the **dev** split,
800 first decisions per model:

| failure class | Qwen3-4B | Qwen3-1.7B |
| --- | --- | --- |
| call validated against the tool schema and dispatched | 96.75% | 90.25% |
| **schema-invalid call** | **0.00%** | **0.00%** |
| unparseable block | 0.25% | 0.25% |
| no tool block emitted at all | 0.00% | 4.00% |
| well-formed call the sandbox rejected (words in the expression) | 3.00% | 5.50% |

xLAM teaches schema-conformant tool-call serialisation. Every emitted call
already validates. Mixing it in would spend data budget, dilute the retained
set, and take on the CC BY attribution duties in `data/LICENSES.md` to fix a
failure rate of zero.

**Measured on dev, not on test.** D-068's test-split numbers say the same thing,
but a training decision derived from the split the checkpoint is later scored on
is test-set selection under section 7.4, however innocuous the number looks.
Dev is otherwise unused and this is what it is for.

**What is not fixed by this.** The largest non-dispatch failure is not a format
failure at all: 3.0% and 5.5% of first decisions are well-formed, schema-valid
calls whose expression contains words — `(4 schools * (2 teams * 5 players))` —
which the sandbox rejects as invalid syntax. That is a habit SFT on clean
trajectories addresses directly, and no external dataset is needed for it.

---

### D-071 — The laundering filter's question-match rule is disabled, because GSM8K writes numbers as words
**Date:** 2026-08-20 · **Status:** accepted · **Supersedes:** nothing
**Label:** `post_hoc_after_measurement`

`configs/train_config.yaml` pins `retention.min_question_match_ratio: 0.0`,
which turns off the fourth rule in `training.retention.laundering_verdict`. The
first three rules stay on.

The rule was added because D-062's laundering check is defeated by one
character: `expression_does_arithmetic("391 + 0")` is true and the expression
computes nothing. Rules one to three close that — arithmetic required, at least
two operands, and the gold answer may not appear as a literal unless the
question contains it too. Rule four went further and required most of an
expression's literals to appear in the question.

**Measured on dev over 757 correct expressions**, rejections by threshold:

| `min_question_match_ratio` | rejected | of which by rule four |
| --- | --- | --- |
| 0.00 (rule off) | 5 (0.7%) | 0 |
| 0.25 | 21 (2.8%) | 16 |
| 0.50 | 75 (9.9%) | 70 |
| 1.00 | 382 (50.5%) | 377 |

Every rejection at 0.25 was inspected. All were genuine work. Both distinct
expressions came from the same task: *"Chatty prepared three dozen eggs for her
four children."* The correct answers `36 / 4` and `(3 * 12) / 4` contain no
literal that appears in the question, because the question spells its
quantities as words. GSM8K does this constantly, so the rule cannot tell an
invented number from a spelled-out one, and what it actually selects against is
multi-step arithmetic that introduces intermediate constants.

Rule four therefore rejects correct work and, on this data, caught nothing the
first three rules had not already caught. It stays implemented and configurable
so a later phase with digit-bearing prompts can switch it on, and it stays off
here.

**The measured laundering rate is low but not zero:** 5 of 757 correct
expressions, 0.66%, caught by rules one to three. That is the rate in untrained
checkpoints. It must be re-measured after training, when a reward for a passing
tool call is exactly the pressure that raises it.

---

### D-072 — The Phase A SFT teacher is Qwen3-4B teaching Qwen3-1.7B, not a 32B class model
**Date:** 2026-08-20 · **Status:** accepted · **Supersedes:** the teacher clause of D-009 for the Phase A scale-check arm only
**Label:** `post_hoc_after_measurement`

The M2 SFT arm trains **`Qwen/Qwen3-1.7B` only**, on trajectories generated by
**`Qwen/Qwen3-4B`**, both already pinned in `configs/model_candidates.json` and
both run locally. No GPU is rented and no 32B model is used.

**This deviates from a pre-registered commitment and the deviation is the
point of this record.** BLUEPRINT_v2 §5.2 says *"Teacher: strong open-weight
model — a 32B class in 4-bit fits a 24 GB card … Run locally on RunPod"*, and
D-009 says *"1–3k teacher trajectories from a local open-weight teacher"*. A 4B
is not a 32B class model. Two consequences follow and neither is hidden:

1. **The primary model gets no SFT arm from this.** Qwen3-4B cannot teach
   itself — measured, not argued: the pilot trained the 1.7B on its own
   retained outputs and the loss started at 0.045, because a model already
   assigns high probability to text it generated. Self-distillation has close
   to nothing to teach. So H1 for the primary model still needs the
   pre-registered teacher, and this arm does not answer it.
2. **The intervention is smaller than the one planned**, so a null result here
   is weaker evidence against post-training than a null result from the 32B
   teacher would be. Any report of this arm must say so rather than presenting
   it as the pre-registered comparison.

**Why it is still worth running.** The capability gap between the two
checkpoints is real and already measured on dev: pass^1 of 0.620 against 0.310,
and pass^4 of 0.590 against 0.270. The teacher solves roughly twice as many
tasks as the student, so the retained set contains correct expressions for
problems the student does not currently solve — which is exactly the property
§5.2 wanted from a larger teacher, at a smaller magnitude. It costs nothing,
runs unattended on the 8 GB card, and produces a genuine H1 data point for the
scale-check model rather than none.

**What stays open.** The 32B teacher path is not cancelled.
`Qwen/Qwen3-32B` at `9216db5781bf21249d130ec9da846c4624c16137` and
`Qwen/Qwen3-14B` at `40c069824f4251a91eefaf281ebe4c544efd3e18` both resolve as
Apache-2.0 and ungated. Neither is in the registry yet, because
`configs/model_smoke.json` pins the registry by SHA-256 as part of the resolved
D-048 release gate, and re-pinning a resolved gate for an unadopted model would
weaken the gate for nothing. They go in when one is adopted, in that decision.

**Naming.** The resulting checkpoint is `Qwen3-1.7B-sft-<confighash>` per §7.4.
Its model card and every artifact must name Qwen3-4B as the teacher, because a
reader who assumes §5.2's teacher would over-read the result.

**Everything else is unchanged.** Generation runs through the same harness,
same prompts, same grader, R0 only (D-069), the pinned laundering rules
(D-071), no format-grounding mix (D-070), one row per task, and checkpoint
selection on dev before test is touched exactly once.
