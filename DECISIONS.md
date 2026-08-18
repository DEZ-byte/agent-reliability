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
