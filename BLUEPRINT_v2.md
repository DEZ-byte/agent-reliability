# Internalizing Agent Reliability — Blueprint v2 (canonical)

**Status: hypotheses only — no experiments run as of 2026-08-17.**
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
| H3 | The GRPO-trained model retains ≥90% of its pass^1 when the domain policy manual is removed from the system prompt; the base model degrades materially on the same probe set. | Trained model still depends on the manual → policy knowledge was not internalized into weights. |

Decision rule: every hypothesis is reported with its measured outcome and CI.
None is dropped for being negative.

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
| Primary small | `Qwen/Qwen2.5-3B-Instruct` **or** `Qwen/Qwen3-4B` | No / Qwen license (2.5-3B is research-friendly but NOT Apache; Qwen3 is Apache-2.0 — check before release) | Full grid: SFT, DPO, GRPO × Rungs 0/1/2 |
| Scale check | `Qwen/Qwen2.5-1.5B-Instruct` **or** `Qwen/Qwen3-1.7B` | Apache-2.0 | Base + GRPO, Rungs 0 + 2 only |
| Cross-family check | `meta-llama/Llama-3.2-3B-Instruct` | **Gated**, Llama Community License (derivative names must start with "Llama") | Base + GRPO, Rungs 0 + 2 only |
| Scaffolded comparator | `meta-llama/Llama-3.1-8B-Instruct` | **Gated**, Llama Community License | **Never trained.** Base at Rungs 0/1/2 |
| User simulator (eval only) | `Qwen/Qwen2.5-14B-Instruct` (fallback 7B if VRAM-tight) | Apache-2.0 | Runs in vLLM beside the policy model |

Dropped from v1 and why:
- `mistralai/Ministral-3B-Instruct` — **does not exist** as open weights (2024
  Ministral 3B was API-only). The open 3B that exists now is
  `mistralai/Ministral-3-3B-Instruct-2512` (Dec 2025, FP8-first, vision-capable)
  — usable in principle, but out of scope for the 4-model budget.
- `google/gemma-2-2b-it` — gated, and its logit soft-capping degrades in fp16
  (a trap on non-bf16 GPUs); superseded by Gemma 3 anyway.
- All remaining v1 tiers — compute (§9).

### 3.1 Setup-time decision: Qwen2.5 vs Qwen3
Pick during M0 by running the same smoke test on both candidates:
Unsloth GRPO support, tool-call chat-template quality, 4-bit VRAM fit on a
4090/L4, and tokens/s. Record the choice and the measurements in DECISIONS.md.
If Qwen2.5 wins, the README must state the reason ("comparability with the
2025 GRPO literature"), because it is a 2024 model in a 2026 project.

---

## 4. Experimental matrix (~23 arms, tiered — not 108)

Rungs: **R0** direct single-turn; **R1** ReAct loop; **R2** cascade + gates (§8).

| Model | Regimes × Rungs | Arms |
| :-- | :-- | :-- |
| Primary (≤4B) | {Base, SFT, DPO, GRPO} × {R0, R1, R2} | 12 |
| Scale check (1.5–1.7B) | {Base, GRPO} × {R0, R2} | 4 |
| Llama-3.2-3B | {Base, GRPO} × {R0, R2} | 4 |
| Llama-3.1-8B | Base × {R0, R1, R2} | 3 |
| **Total** | | **23** |

Headline comparison: **primary GRPO×R0 (and ×R2 "hybrid") vs 8B Base×R2.**

Eval tiers (protocol in §7):
- Tier 1 — all 23 arms: n=4 runs, Phase A test set (N≈150) + tau2-retail test
  split. Reports pass^1, pass^4 (pass@k alongside).
- Tier 2 — 6 headline arms only (8B×R2, 8B×R1, primary Base×R1, primary
  SFT×R1, primary GRPO×R0, primary GRPO×R2): n=8 runs → adds pass^8.

---

## 5. Data plan (was entirely missing in v1)

### 5.1 Benchmarks — honest names
- **Phase A (single-turn, first):** "GSM8K wrapped in our own calculator/REPL
  tool environment" (there is no benchmark called "GSM8K-Tool" — never use
  that name) + an open function-calling dataset for format grounding:
  `Salesforce/xlam-function-calling-60k` or Glaive-function-calling-v2
  (record the license of whichever is used in `data/README.md`).
- **Phase B (multi-turn):** tau2-bench retail domain via the **upstream
  package as a pinned dependency** (evaluate `tau2-bench-verified` as the
  pinned choice — it fixes misannotated tasks). Never reimplement the env,
  DB, or grader: reimplementation makes every number incomparable to
  published results. Before running any project arm, reproduce one published
  small-model retail pass^1 to prove the harness is comparable.

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
  top-level worker functions (the dev box is Windows). `tests/test_sandbox.py`
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
| Format | Every emitted tool block parses as JSON **and** validates against a registered tool's Pydantic schema **and** ≥1 call actually dispatched. | +0.2 valid / −0.5 parse failure |
| Gate | Replay the executed tool-call event log; violation iff a mutative tool **executed** while the auth predicate on env state was false (order-aware, success-aware). | Violation **zeroes accuracy** and adds −0.6 |
| Efficiency | −0.05 × executed calls, capped at −0.3. | plus **−0.3 for zero tool calls** on tool-required tasks |

All four are summed (signed terms; no double-negation ambiguity).
`tests/test_rewards.py` contains deliberate gaming inputs with exact expected
values: auth string in prose/comment, empty `<tool_call></tool_call>`,
out-of-order auth, failed-auth-then-modify, multiple `####` markers.

### 7.1 Order and curriculum
- Pipeline: **Base → SFT → {DPO | GRPO}**. GRPO and DPO both initialize from
  the SFT checkpoint. GRPO-from-base exists only as a labeled ablation.
- π_ref = the SFT checkpoint realized by disabling the LoRA adapter
  (zero extra VRAM).
- Curriculum:
  - **Stage 1:** single-turn verifiable tool tasks (Phase A). Stock
    TRL/Unsloth GRPO works here (it is single-turn by design — v1's
    assumption that it handles multi-turn episodes was false). G = 8–16.
  - **Stage 2:** short scripted 2–4-turn synthetic retail episodes with auth
    gates (no simulator; deterministic rewards).
  - **Stage 3 (stretch, M6):** full tau2 multi-turn via the `verifiers`
    multi-turn RLVR library on a rented GPU. Kill criterion in §11.

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
  `skipped_auth` rate per rung. Headline GRPO arm runs with ≥2 seeds;
  report mean ± range.

---

## 8. Inference scaffolding spec (v1 left this unimplementable)

### 8.1 Rungs
- **R0 Direct:** one generation, tool schemas in prompt, no retries.
- **R1 ReAct:** thought/act/observe loop, step cap **20 env steps** —
  identical cap for every arm (cap changes pass^k and token counts by itself).
- **R2 Cascade + gates:** Half A escalation ladder + Half B deterministic
  gates, specified below. Built in LangGraph (builder is new to it — learning
  time budgeted inside M4).

### 8.2 Half A — cascade triggers (all runtime-observable; the grader is never
consulted mid-episode)

| Signal | Transition |
| :-- | :-- |
| 1st tool exception / schema-validation failure | Retry the same call |
| 2nd failure on the same step | Reflect: re-prompt same model with the error transcript (pinned reflect prompt in `configs/`) |
| 3rd failure, or a gate block | Escalate to the local 8B (both models co-resident in 4-bit on a 24 GB GPU) |
| Step cap hit | Graceful give-up |
| N=3 identical consecutive calls | Treated as a failure signal (loop detection) |

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
| Internalization probe (H3) | Yes | **Removed** (trained + base arms, probe set only) | Zero-shot |

Every system prompt is pinned verbatim in `configs/prompts/`. Prompt tokens
and generated tokens are counted separately in all cost reporting.

---

## 9. Compute plan (honest budget)

| Resource | Use |
| :-- | :-- |
| RTX 4060 8 GB (Windows) | All development, unit tests, parsers/rewards/gates, smoke inference of the 1.5B in 4-bit. No training runs. |
| Colab (~199 units ≈ 40 L4-hours) | SFT and DPO runs, mid-size Phase A evals. |
| RunPod < $30/mo (~$90 over 3 months ≈ 120–250 spot 4090/L4 hours) | GRPO runs, Phase B eval with the co-resident user simulator, teacher-data generation. |

Estimates to validate at M1 (record actuals in DECISIONS.md):
- GRPO single-turn: 1.5B ≈ 4–8 h; 3–4B ≈ 8–15 h per run on a 4090.
- Phase B eval: ~8–10 arms × ~114 tasks × n=4 (+n=8 on 6 arms) ≈ 6–8k episodes,
  batched ≈ 30–60 GPU-hours.
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
on task i. Compute for k ∈ {1,4,8}:

- **pass^k** (reliability — all k succeed):
  pass^k = (1/|D|) Σ_i C(c_i, k) / C(n, k)
- **pass@k** (capability — at least one succeeds):
  pass@k = (1/|D|) Σ_i [ 1 − C(n−c_i, k) / C(n, k) ]
- pass^1 = pass@1 = mean success rate. ("Best-case single run" is a
  misdefinition — never use it.)

Decoding: fixed temperature/top_p/step cap, identical across all arms, pinned
in `configs/eval.yaml`, per-run seeds recorded.

### 10.2 Statistics
- All arms evaluated on the identical task set → **paired** comparisons:
  McNemar / paired permutation on per-task outcomes.
- 95% CIs via hierarchical bootstrap (resample tasks, recompute the
  combinatorial estimator from each task's runs).
- N≈150 cannot resolve <5 pp gaps; the README says so and headline claims are
  restricted to gaps that survive the paired test.

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
| **M0** (~1 wk) | Repo skeleton, pinned env + lockfile, tool registry, sandbox + escape tests, literature scan, PLAN.md + DECISIONS.md in git, HF gated-license requests filed, Qwen 2.5-vs-3 smoke test | CI green; model choice recorded |
| **M1** (~2 wk) | Phase A env + rewards v2 + pass^k harness | **Real** baseline numbers, all 4 base models, R0/R1, with CIs — ships as a measurement study |
| **M2** (~1–2 wk) | SFT on primary + re-eval | "Does imitation help?" row filled |
| **M3** (~2–3 wk) | Single-turn GRPO + ablation ladder + ≥2 seeds + W&B curves; DPO arm alongside | **Headline experiment** rows filled |
| **M4** (~2 wk) | LangGraph R1/R2 + cascade + gates on base models | First real Pareto plot |
| **M5** (~2–3 wk) | tau2-retail eval via upstream adapter (`src/env/tau2_adapter.py`, budget 3–5 days) + local user sim | Phase B table filled; one published baseline reproduced |
| **M6** (stretch) | Multi-turn GRPO via `verifiers` | Kill criterion below |
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
| GPU overspend | >$30 in any month | Freeze eval at k=4, N=100; queue rest for next month |
| Llama gating delayed | Not approved by end of M1 | Switch to Qwen-only registry |
| TRL/Unsloth breakage | Pinned env fails on upgrade | Stay on lockfile; upgrade only between milestones |
| tau2 adapter overruns | >5 days on the adapter | Ship Phase A only; tau2 → future work |
| User-sim too weak/slow | Episodes degenerate or eval > GPU budget | Drop to scripted user turns for Phase B, note the limitation |

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
