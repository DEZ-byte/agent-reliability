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
