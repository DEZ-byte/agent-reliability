# Internalizing Agent Reliability — Blueprint v2 (canonical)

**Status: implementation started; no model experiment has run as of
2026-08-18.**
This file supersedes the two v1 blueprints in `Downloads/`
(`Unified_Agent_Reliability_RLVR_Master_Blueprint.md`,
`UNIFIED_AGENT_RELIABILITY_BLUEPRINT.md`). Those files contain fabricated
results tables, a nonexistent model checkpoint, and unimplementable specs.
Do not copy content from them into the repo. All decisions here are logged
with dates in [DECISIONS.md](DECISIONS.md).

---

## 1. Research question (open — not pre-answered)

> How much of the pass^k reliability gap between a small (≤4B) tool-calling
> model and a runtime-scaffolded 8B model can verifiable-reward post-training
> close — and at what token, latency, and GPU cost?

Both directions of the answer ship. "Post-training loses to scaffolding but
costs far less at inference" is a publishable, hireable result.

### 1.1 Falsifiable hypotheses (pre-registered)

| ID | Hypothesis | Negative outcome (also ships) |
| :-- | :-- | :-- |
| H1 | GRPO with gate rewards on the primary ≤4B model closes ≥50% of the pass^4 gap to the fully scaffolded 8B baseline, at ≤30% of the 8B arm's generated tokens per episode. | Gap closure <50% → reported as "scaffolding retains the reliability lead; post-training buys partial closure at X% of the cost." |
| H2 | Adding the gate-reward term reduces the `skipped_auth` failure rate by ≥50% relative to an otherwise-identical GRPO run without it (ablation ladder, §7.4). | No reduction → gate rewards do not internalize the constraint; format+accuracy explain the gains. |
| H3 | The GRPO-trained model retains ≥90% of its pass^1 when the domain policy manual is removed from the system prompt. Base-model degradation on the same probe set is a separately reported auxiliary criterion, not part of the H3 verdict. | Trained model still depends on the manual → policy knowledge was not internalized into weights; a weak base-model drop limits the interpretation but does not rewrite the H3 threshold. |

Decision rule: every hypothesis is reported with its measured outcome and CI.
None is dropped for being negative.
`HYPOTHESIS_PROTOCOL.md` is the normative definition of their arms,
denominators, aggregation, zero handling, and paired inference.

### 1.2 What is claimed as new vs. adopted
- **Adopted from prior work:** GRPO recipe and composite reward structure
  (DeepSeekMath-style), tau2-bench environment and grader, pass^k methodology.
- **Distinguishing design choice (novelty pending the M0 literature scan, §10):**
  measuring the training-vs-scaffolding trade-off on one axis system, and
  distilling a *runtime* deterministic gate engine into RLVR reward terms so
  the trained constraint and the runtime constraint are the same code.
- Do not use the phrase "core innovation" anywhere until the scan supports it.

---

## 2. Constraints (fixed inputs to every design decision)

| Constraint | Value |
| :-- | :-- |
| Purpose | Hiring portfolio — post-training / ML engineer roles |
| Local hardware | RTX 4060, 8 GB VRAM, Windows 11 (dev + tests only) |
| Colab | ~199 compute units total (L4 ≈ 4.8 units/h → ~40 L4-hours) |
| Cloud rental | RunPod/Vast spot, hard cap **$30/month** |
| API spend | **$0** — no paid API anywhere in training or eval |
| Timeline | 3+ months, part-time, no hard deadline |
| Builder experience | SFT/LoRA ✔, DPO/RL ✔, LangGraph ✘ (learning cost budgeted in M4) |

Consequences of $0 API:
- User simulator = local open model (§6.2). No frontier-API simulator.
- Grading = tau2's default deterministic basis (DB × COMMUNICATE). The NL
  LLM-judge assertion is **not** part of any headline metric (it is optional
  and diagnostic upstream too — do not misdescribe it as the retail default).
- Cascade escalation target = local 8B, not an API model (§8).
- SFT teacher = open-weight model run locally (§5.2). Avoids distillation-ToS
  questions and keeps the released dataset clean for a public repo.

---

## 3. Model registry (4 checkpoints, 2 families — verified IDs only)

Rule: a checkpoint may appear in this table only if its Hugging Face repo ID
resolves. Verify each link at M0. Accept the Meta gated licenses immediately
(response pre-committed in the risk register if approval stalls).

| Role | Checkpoint (decide at setup, §3.1) | Gated / license | Treatment |
| :-- | :-- | :-- | :-- |
| Primary small | `Qwen/Qwen2.5-3B-Instruct` **or** `Qwen/Qwen3-4B` | Public / Qwen2.5-3B uses the non-commercial Qwen Research License; Qwen3 uses Apache-2.0 | Full grid: SFT, DPO, GRPO × Rungs 0/1/2 |
| Scale check | `Qwen/Qwen2.5-1.5B-Instruct` **or** `Qwen/Qwen3-1.7B` | Apache-2.0 | Base + GRPO, Rungs 0 + 2 only |
| Cross-family check | `meta-llama/Llama-3.2-3B-Instruct` | **Gated**, Llama Community License (derivative names must start with "Llama") | Base + GRPO, Rungs 0 + 2 only |
| Scaffolded comparator | `meta-llama/Llama-3.1-8B-Instruct` | **Gated**, Llama Community License | **Never trained.** Base at Rungs 0/1/2 |
| User simulator (eval only) | `Qwen/Qwen2.5-14B-Instruct` | Apache-2.0 | Runs in vLLM beside the policy model. No fallback is named: this table's own rule is that a checkpoint appears only if its ID resolves, and any substitute must first be added to `configs/model_candidates.json` with a verified revision and licence row, then adopted by a dated decision (see the user-simulator risk row in §11) |

Dropped from v1 and why:
- `mistralai/Ministral-3B-Instruct` — **does not exist** as open weights (2024
  Ministral 3B was API-only). The open 3B that exists now is
  `mistralai/Ministral-3-3B-Instruct-2512` (Dec 2025, FP8-first, vision-capable)
  — usable in principle, but out of scope for the 4-model budget.
- `google/gemma-2-2b-it` — gated, and its logit soft-capping degrades in fp16
  (a trap on non-bf16 GPUs); superseded by Gemma 3 anyway.
- All remaining v1 tiers — compute (§9).

### 3.1 Setup-time decision: Qwen2.5 vs Qwen3
Select one size-paired generation bundle during M0 under
`MODEL_SMOKE_PROTOCOL.md`: Qwen2.5 {3B, 1.5B} or Qwen3 {4B, 1.7B}. Both sizes
must pass the frozen-stack tool-template, single-GPU NF4, training-mask, and
one-step training probes. Qwen3 is scored with thinking disabled; thinking is
diagnostic only. Apply the recorded release-license constraint before naming
the winner, then record the measurements and artifact paths in DECISIONS.md.

---

## 4. Experimental matrix (~25 arms, tiered — not 108)

Rungs: **R0** direct with no same-turn model retry; **R1** same-model
act/observe with bounded structured feedback; **R2** feedback + reflection +
gates and, for small parents only, one 8B handoff (§8 and
`RUNG_PROTOCOL.md`).

| Model | Regimes × Rungs | Arms |
| :-- | :-- | :-- |
| Primary (≤4B) | {Base, SFT, DPO, GRPO} × {R0, R1, R2} | 12 |
| Scale check (1.5–1.7B) | {Base, GRPO} × {R0, R2} | 4 |
| Scale H2 confirmatory | GRPO-{accuracy+format, accuracy+format+gate} × R1-audit | 2 |
| Llama-3.2-3B | {Base, GRPO} × {R0, R2} | 4 |
| Llama-3.1-8B | Base × {R0, R1, R2} | 3 |
| **Total** | | **25** |

**Training artifacts that are not arms.** §7.1 initializes GRPO from an SFT
checkpoint, so the two H2 confirmatory configurations both require a
**scale-model SFT checkpoint** that appears nowhere in this table and is never
evaluated as an arm. It is a prerequisite of M3, must be trained from the same
data and seed as the primary SFT run, and its GPU cost belongs in §9. The same
applies to any checkpoint an ablation branches from: a required artifact is not
the same thing as a reported arm, and neither may be silently borrowed from the
other.

Headline comparison: **primary Base×R0 → primary GRPO×R0 gap closure relative
to 8B Base×R2-no-escalation**; primary GRPO×R2 is the separate “hybrid” row.
Exact aggregation and token-ratio rules are in `HYPOTHESIS_PROTOCOL.md`.

Eval tiers (protocol in §7):
- Tier 1 — all 23 production-grid arms: n=4 runs, Phase A test set
  (N≈150) + tau2-retail test split. Reports pass^1, pass^4 (pass@k
  alongside).
- Tier 2 — 7 headline arms only (8B×R2-no-escalation, 8B×R1, primary
  Base×R0, primary Base×R1, primary SFT×R1, primary GRPO×R0, primary
  GRPO×R2): n=8 runs → adds pass^8. Base×R0 is required to define H1's gap.
- H2 confirmatory — the two additional scale-model reward variants run with
  n=8 under R1 audit on the frozen authorization manifest. They are not
  silently substituted with the full-composite production-grid checkpoint.

### 4.1 Self-correction, and where it sits
The project title names self-correction, and until now this document did not.
It is **not** a fifth hypothesis and not an extra arm. `SELF_CORRECTION_SPEC.md`
is normative: it defines one earliest rollback-safe correction opportunity per
episode, freezes the failed prefix, and compares a neutral same-model resample,
a diagnostic-aware same-model repair, and an immediate 8B handoff branching from
that same evidence (D-029).

Production R1/R2 traces support descriptive attribution only; causal language
requires the matched branch bundle. Local action repair and deterministic
end-to-end recovery are reported separately, and retry luck, gate prevention,
and success after 8B escalation are never credited as small-model
self-correction. The diagnostic branches run on the Tier-2 headline arms after
M4 and are budgeted in §9 as a separate line; they do not alter H1-H3, and no
H1-H3 verdict may cite them.

---

## 5. Data plan (was entirely missing in v1)

### 5.1 Benchmarks — honest names
- **Phase A (single-turn, first):** "GSM8K wrapped in our own calculator/REPL
  tool environment" (there is no benchmark called "GSM8K-Tool" — never use
  that name) + an open function-calling dataset for format grounding:
  `Salesforce/xlam-function-calling-60k` or Glaive-function-calling-v2
  (record the license of whichever is used in `data/README.md`).
- **Phase B (multi-turn):** tau2 retail through one pinned provenance tuple.
  Sierra's official repository and Amazon's separately corrected
  `tau2-bench-verified` fork are distinct choices, not upstream aliases.
  Before implementation, freeze the repository, commit, task manifest,
  simulator, reward basis, and dependency lock together (D-027). Never
  reimplement the environment, DB, or grader. Reproduce a reference result
  only on the exact same tuple; otherwise label the run as a new variant.

### 5.2 SFT data (target: 1–3k verified trajectories)
- Teacher: strong open-weight model (e.g., Qwen2.5-72B class in 4-bit, or the
  largest that fits the rented GPU) run locally on RunPod. No API teacher.
- Generation: teacher plays the agent on **train-split** tasks; every
  trajectory is graded by the deterministic grader; **only passing
  trajectories are kept** (rejection sampling — the grader doubles as the
  data filter).
- Serialization: each model family's native tool-call chat template.
- Mixed with the open function-calling set for schema-format grounding.

### 5.3 DPO pairs (target: 500–2k pairs)
- Sample 8–16 rollouts per train task from the **SFT checkpoint** at T≈0.8.
- Grade each with the deterministic grader.
- Pairs share an identical prompt prefix. For recovery-specific pairs:
  truncate both trajectories at the first tool-error observation and pair the
  continuation that recovers against the one that loops.

### 5.4 Splits and contamination
- Train / dev / test task IDs committed as JSON lists in `configs/splits/`.
- Split at the task-template level, so paraphrased twins never straddle splits.
- tau2-retail is small (~100–115 tasks): test split is eval-only, never
  trained on, never used for checkpoint selection.
- Contamination probe for the math set: base checkpoints attempt GSM8K test
  items with tools disabled; report the memorization rate. The
  execution-backed accuracy reward (§6.3) makes memorized text answers score
  0 regardless.

---

## 6. Environment, grading, and the $0 eval stack

### 6.1 Tool execution
- In-process Python tool registry (`src/env/tools.py`) for training rollouts
  and reward computation. **MCP is never inside the RL loop** — it may
  optionally expose the same registry at inference/demo time.
- Sandbox (`src/env/sandbox.py`) is labeled **"best-effort resource sandbox,
  not a trust boundary."** Requirements: restricted-builtins whitelist (no
  `__import__`/`eval`/`exec`/`open`/`compile`), POSIX rlimits (address space,
  CPU, file size) inside the worker, `process.kill()` on timeout, a defined
  `SandboxViolation` exception, stdout/stderr capture, and spawn-safe
  top-level worker functions (the dev box is Windows). Windows uses a parent
  private-bytes watchdog where available. Worker IPC is bounded strict JSON,
  never worker-controlled pickle (D-019, D-022). `tests/test_sandbox.py`
  proves the timeout and memory limits fire and the known escapes are blocked.

### 6.2 User simulator (Phase B eval only)
- Local Qwen2.5-14B-Instruct (or 7B) served by vLLM next to the policy model
  on the rented GPU. Pinned model ID, prompt, and temperature in `configs/`.
- Training never depends on the simulator: GRPO Stage 1–2 rollouts use
  single-turn tasks or scripted user turns (§7.1).

### 6.3 Grading
- Headline metric grader = tau2's deterministic basis: **DB-state check ×
  COMMUNICATE check**. No LLM judge in any headline number.
- Optional diagnostic: a pinned local judge for semantic failure tags only,
  validated against ~100 hand-labeled episodes with Cohen's kappa reported
  before any taxonomy percentage is cited.

---

## 7. Training pipeline

### 7.0 Reward functions v2 (execution-backed; the v1 substring rewards are banned)

One parser and one gate engine (`src/agent/gates.py`) serve **both** the
runtime scaffold and the reward computation, so the trained constraint and the
runtime constraint cannot diverge.

| Term | Source of truth | Value |
| :-- | :-- | :-- |
| Accuracy | Environment state only: sandbox execution result (Phase A) or final DB-state hash (Phase B). Text-only answers score 0. | +1.0 / 0.0 |
| Format | Every emitted tool block parses as strict JSON **and** validates against a registered tool's Pydantic schema **and** ≥1 call actually dispatched. | +0.2 when the conjunction holds / −0.5 when emitted blocks fail it / 0 when no block is emitted |
| Gate | Replay the tool-call event log; violation iff a mutative call was **dispatched** while a required pre-call predicate was false, even if its handler later failed. | Any violation **zeroes accuracy** and adds one binary −0.6 episode term |
| Efficiency | −0.05 × executed calls, capped at −0.3. | plus **−0.3 for zero tool calls** on tool-required tasks |

All four are summed (signed terms; no double-negation ambiguity).
On tool-required tasks, zero dispatched calls also forces accuracy to 0.
`tests/test_rewards.py` contains deliberate gaming inputs with exact expected
values: auth string in prose/comment, empty `<tool_call></tool_call>`,
out-of-order auth, failed-auth-then-modify, multiple `####` markers.

### 7.1 Order and curriculum
- Pipeline: **Base → SFT → {DPO | GRPO}**. GRPO and DPO both initialize from
  the SFT checkpoint. GRPO-from-base exists only as a labeled ablation.
- π_ref must be the frozen SFT policy used to initialize the trained arm.
  Disabling an SFT LoRA may expose the base model rather than the SFT policy,
  so the exact memory-safe implementation remains blocked on the executed
  SFT/LoRA smoke test and must be recorded before training.
- Curriculum:
  - **Stage 1:** single-turn verifiable tool tasks (Phase A). Use the frozen
    TRL/Unsloth stack after the one-step compatibility probe. G = 8–16.
  - **Stage 2:** short scripted 2–4-turn synthetic retail episodes with auth
    gates (no simulator; deterministic rewards).
  - **Stage 3 (stretch, M6):** full tau2 multi-turn through the backend chosen
    by a deterministic compatibility pilot between current TRL
    `environment_factory` support and `verifiers`. No backend is preselected.
    Kill criterion in §11.

### 7.2 Losses — exact formulations
- **Assistant-token-only loss masking everywhere** (SFT, DPO, GRPO):
  L(θ) = −Σ_{t∈T_assistant} log π_θ(x_t | x_<t), with labels[t] = −100 for all
  t outside assistant turns (tool observations, user/simulator turns). This is
  the direct defense against training the `fabricated_result` failure.
- **GRPO is token-level** (the v1 sequence-level ratio formula is wrong and
  numerically vacuous):
  L(θ) = −(1/G) Σ_i (1/|o_i|) Σ_t m_{i,t} [ min(r_{i,t} Â_i,
  clip(r_{i,t}, 1±ε) Â_i) − β·KL_t ], where r_{i,t} is the per-token ratio,
  m_{i,t} masks non-assistant tokens, Â_i = (R_i − mean)/(std + δ).
  (δ, not ε — v1 used one symbol for both.)
- β values: DPO β ≈ 0.1; GRPO β ≈ 0–0.04 with the per-token k3 estimator.

### 7.3 Zero-variance-group and collapse handling
1. SFT initialization guarantees non-zero early success probability.
2. Rollout temperature T = 0.7–0.85 so the G candidates do not collapse to
   identical strings.
3. The dense format term keeps gradient signal alive while accuracy is
   still 0.
4. Logged health alarms per step: fraction of zero-variance groups, fraction
   of zero-tool-call rollouts, mean KL, entropy.

### 7.4 Pre-committed training rules (no silent test-set selection)
- `configs/train_config.yaml` pins: max steps/epochs per method (SFT 2–3
  epochs; GRPO step budget per stage), checkpoint cadence (~every 25 steps,
  pushed to a private HF Hub repo), seeds, LoRA r=16 α=32 4-bit.
- **Checkpoint selection rule:** evaluate last + best-dev-pass^1 checkpoints
  on the **dev split only**; the dev winner is frozen and runs on test
  **exactly once**.
- Artifact naming: `<model>-<method>-<confighash>-step<N>`, model card embeds
  the training config and W&B run ID. Keep adapters + tokenizer configs;
  delete optimizer states after a run finishes.
- **Ablation ladder** (headline evidence for H2), on the scale-check model:
  accuracy-only → +format → +gate → full composite. Report the
  `skipped_auth` rate per rung. The confirmatory H2 contrast is the matched
  {accuracy+format} versus {accuracy+format+gate} pair with efficiency disabled
  in both, evaluated under R1 audit mode as specified in
  `HYPOTHESIS_PROTOCOL.md`. Headline GRPO arms run with ≥2 seeds; report the
  frozen paired CIs, not only mean ± range.

---

## 8. Inference scaffolding spec (v1 left this unimplementable)

### 8.1 Rungs
- **R0 Direct:** one generation per natural agent turn, tool schemas in the
  prompt, and no replacement generation at the same state. Conversational R0
  can have later generations after genuine environment/user observations.
- **R1 Act/observe:** the same model may make one structured-feedback decision
  after a correctable same-turn failure. The framework-neutral Python state
  machine is implemented in M1.
- **R2 Cascade + gates:** R1 plus one reflection decision, deterministic policy
  gates, and one-way 8B handoff for small parents. The 8B parent uses explicit
  `R2-no-escalation`. M4 may expose the same core through LangGraph only after
  golden event/state parity.

Every matched arm shares the **20 environment-turn** cap. Rung-specific model
decision budgets are treatments and are counted separately; graph hops,
parser passes, and gate blocks are not environment turns.

### 8.2 Half A — cascade triggers (all runtime-observable; the grader is never
consulted mid-episode)

| Signal | Transition |
| :-- | :-- |
| Confirmed transient, no-commit, idempotent tool failure | At most one exact redispatch under the common infrastructure policy |
| Parse/schema/unknown-tool or non-transient tool failure | Fresh same-model decision with structured feedback when the rung permits it |
| R2 feedback decision also fails | One pinned same-model reflection decision |
| R2 small parent exhausts same-model repair | One-way handoff to the frozen local 8B |
| R2 8B parent exhausts reflection | Graceful give-up; `escalation_target=none` |
| Gate block | No dispatch; follow the remaining feedback/reflect ladder rather than immediate escalation |
| Environment/decision cap hit | Graceful give-up with a specific termination label |
| N=3 identical consecutive calls | Failure signal without an additional identical dispatch |

- Escalation frequency is logged per arm — "escalations × cost" is a required
  column of the trade-off table.
- **Parity rule:** all scaffold knobs (retry counts, caps, reflect prompt) are
  tuned on the dev split only, with the same discipline as training
  hyperparameters, and frozen before test. An untuned scaffold would make the
  "training beats scaffolding" comparison unfair by construction.

### 8.3 Half B — deterministic gates
- Declarative predicates in `configs/gates.yaml`
  (`authenticated?`, `order_id_exists?`, `reversible?`), evaluated against
  environment state before any mutative tool executes.
- The same engine computes the training gate reward (§7.0).

### 8.4 Context protocol (the variable H3 measures — pinned per arm)

| Arm class | Tool schemas | Policy manual | Few-shot |
| :-- | :-- | :-- | :-- |
| All arms, default | Yes (identical) | Yes | Zero-shot (tau2 convention) |
| Internalization probe (H3) | Yes | **Removed** (trained + base arms, on the frozen Phase B tau2-retail test manifest — `HYPOTHESIS_PROTOCOL.md` is normative; there is no separate "probe set") | Zero-shot |

Every system prompt is pinned verbatim in `configs/prompts/`. Prompt tokens
and generated tokens are counted separately in all cost reporting.
The confirmatory H3 comparison is R0-only with paired present/removed contexts;
R1/R2 rows are secondary because gates or retries could mask prompt dependence.

---

## 9. Compute plan (honest budget)

| Resource | Use |
| :-- | :-- |
| RTX 4060 8 GB (Windows) | All development, unit tests, parsers/rewards/gates, smoke inference of the 1.5B in 4-bit. No training runs. |
| Colab (~199 units ≈ 40 L4-hours) | SFT and DPO runs, mid-size Phase A evals. |
| RunPod < $30/mo (~$90 over 3 months ≈ 120–250 spot 4090/L4 hours) | GRPO runs, Phase B eval with the co-resident user simulator, teacher-data generation. |

Estimates to validate at M1 (record actuals in DECISIONS.md):
- GRPO single-turn: 1.5B ≈ 4–8 h; 3–4B ≈ 8–15 h per run on a 4090.
- Phase B eval, derived from §4 rather than estimated separately: 23
  production-grid arms × the frozen tau2-retail **test split** (not the full
  ~114-task set) × n=4, plus n=8 on the 7 Tier-2 headline arms and on the 2 H2
  confirmatory configurations. Multiply the headline GRPO arm by its ≥2 seeds
  (§7.4). Any figure here that disagrees with §4 is wrong by construction; §4 is
  the source. Record the episode count and GPU-hours as measured actuals at M5.
- Every trainer is idempotent-resumable (checkpoint + RNG + dataloader cursor
  every ~25 steps to HF Hub); assume any session can die.
- The README reports actual spend ("all runs: $X on a single 4090") — this is
  itself a portfolio signal.
- Cost axes on Pareto plots: **tokens per episode** and **GPU-seconds per
  episode** (measured, hardware named). Dollar figures only with an explicit,
  stated conversion, labeled "estimated". No latency targets are published in
  advance — latency is measured on one fixed named GPU.

---

## 10. Evaluation protocol

### 10.1 Metrics — exact definitions (from ONE run array)
Run n=8 samples per task once (Tier 2 arms; n=4 for Tier 1), c_i = successes
on task i. Compute only requested k with `k ≤ n`: Tier 1 uses k ∈ {1,4};
Tier 2 uses k ∈ {1,4,8}.

- **pass^k** (reliability — all k succeed):
  pass^k = (1/|D|) Σ_i C(c_i, k) / C(n, k)
- **pass@k** (capability — at least one succeeds):
  pass@k = (1/|D|) Σ_i [ 1 − C(n−c_i, k) / C(n, k) ]
- pass^1 = pass@1 = mean success rate. ("Best-case single run" is a
  misdefinition — never use it.)

Temperature, `top_p`, prompts, environment-turn cap, and run seeds are pinned
in `configs/eval.yaml`. Rung-specific model-decision budgets are deliberate
treatments, not a shared step-cap setting, and are logged separately.

### 10.2 Statistics
- All arms evaluated on the identical task/run seed matrix. Use McNemar only
  for genuinely paired binary outcomes such as matched pass^1 trials. For
  fractional per-task `pass^k` contributions, use the hierarchical bootstrap
  plus a pre-registered paired permutation statistic.
- 95% CIs via hierarchical bootstrap (resample tasks, recompute the
  combinatorial estimator from each task's runs).
- Freeze a sensitivity/precision analysis from the actual task, seed, and
  paired-outcome design before interpreting small gaps. Report observed CI
  widths; do not use an unsupported blanket power threshold.

### 10.3 Trajectory logging (portfolio artifact)
Every eval episode is one JSONL record: `task_id`, `run_idx`, `prompt`,
`raw_completion`, `parsed_tool_calls`, `sandbox_trace`, `gate_events`,
`ground_truth`, `reward_breakdown`. The results table is **generated
programmatically** from these logs (one command per row), and the logs feed a
lightweight trajectory viewer (HF Space / Streamlit) so a reviewer can inspect
any row's reasoning and gate-block events in one click.

### 10.4 Failure taxonomy (honest version)
- Four deterministic tags computed from the event log: `tool_schema_error`,
  `tool_arg_error`, `skipped_auth`, `fabricated_result`.
- ≤2 semantic tags (`invented_policy`, `over_refused`) via the pinned local
  judge, only after the kappa validation in §6.3.
- Not called "extending tau2 error_tags" — tau2 has no such feature. Framed as
  "a custom taxonomy in the spirit of tau-bench's fault-assignment analysis."

### 10.5 Related work (M0 deliverable — 10–15 verified references)
Position against, at minimum: tau-bench / tau2-bench papers; DeepSeekMath
(GRPO); ToolRL; Nemotron-Research-Tool-N1; ReTool; ARTIST; Agent-R1 / RAGEN;
the `verifiers` ecosystem; DPO; xLAM/APIGen and ToolACE data work; BFCL.
Verify each citation exists before it enters the README. Reuse reward
magnitudes/hyperparameters from the closest published recipe where possible,
and cite it.

---

## 11. Milestones (every rung independently shippable) and risks

| Milestone | Contents | Exit criterion |
| :-- | :-- | :-- |
| **M0** (~1 wk) | Repo skeleton, kernel lock, provisional ML smoke lock, tool registry, sandbox + escape tests, literature scan, PLAN.md + DECISIONS.md in git, HF gated-license requests filed, Qwen 2.5-vs-3 smoke test | CI green; model choice recorded |
| **M1** (~2 wk) | Phase A env + rewards v2 + pass^k harness + framework-neutral R0/R1 core | **Real** baseline numbers, all 4 base models, R0/R1, with CIs — ships as a measurement study |
| **M2** (~1–2 wk) | SFT on primary + re-eval | "Does imitation help?" row filled |
| **M3** (~2–3 wk) | Single-turn GRPO + ablation ladder + ≥2 seeds + W&B curves; DPO arm alongside | Phase A GRPO rows and `H1-PhaseA provisional` filled; project-level H1 remains `NA` until M5 |
| **M4** (~2 wk) | LangGraph parity adapter + R2 cascade/gates on base models | First real Pareto plot |
| **M5** (~2–3 wk) | tau2-retail eval via the pinned-tuple adapter (`src/env/tau2_adapter.py`, budget 3–5 days) + local user sim | Phase B table filled; one same-tuple reference result reproduced when available |
| **M6** (stretch) | Multi-turn GRPO via the backend selected by the TRL-versus-`verifiers` pilot | Kill criterion below |
| **M7** (~1 wk) | Writeup, CIs, negative-results section, trajectory-viewer demo, optional video | Definition of done met |

Repo layout adds (v1 omissions): `tests/` + CI, `data/`, `scripts/`,
`results/`, `notebooks/` (Colab entrypoints), pinned `requirements.txt` +
lockfile, `.gitignore`, `LICENSE` (MIT/Apache-2.0 + dataset-license table),
`Makefile`. Cut from v1: `k8s-training-job.yaml`, the EKS/GKE diagram, the
"hardened/secure" sandbox claims, MCP in the training loop, the one-shot
repo-generation meta-prompt (build is milestone-by-milestone with real
commits and tests — the commit history is part of the portfolio).

### Risk register (pre-committed responses)

| Risk | Early indicator | Response |
| :-- | :-- | :-- |
| Multi-turn GRPO stalls | No reward slope within 2 calendar weeks of M6 start | Ship the single-turn headline; multi-turn → future work |
| GPU overspend | >$30 in any month | Stop new runs and queue the remainder. Keep project H1 `NA`/provisional until its frozen n=8 and both benchmark strata complete; any k=4/N=100 result is a separately versioned exploratory row. |
| Llama gating delayed | Not approved by end of M1 | Switch to Qwen-only registry |
| TRL/Unsloth breakage | Pinned env fails on upgrade | Stay on lockfile; upgrade only between milestones |
| tau2 adapter overruns | >5 days on the adapter | Ship Phase A only; tau2 → future work |
| User-sim too weak/slow | Episodes degenerate or eval > GPU budget | The simulator is a frozen provenance item (D-027). Swapping it invalidates every Phase B number already collected, including H1's Phase B stratum and all of H3. Response: stop, record a dated decision, relabel the affected results as a separate variant, and rerun the affected arms — never swap mid-campaign and never compare across the swap |

### Definition of done
README with measured numbers + CIs for ≥6 pre-registered arms, one-command
reproduction per table row, W&B links, failure-taxonomy breakdown with kappa,
the H1–H3 verdicts (positive or negative), and a "what didn't work" section.
Everything beyond this is stretch.

### Reporting rules (non-negotiable)
1. Never bold, quote, or headline a number that was not measured.
2. Hypothesis tables carry TBD cells until logs exist; every published number
   links to its run artifact.
3. Negative results are reported in the same table format as positive ones.
