# Related work: verifiable tool use and self-correction

**Verification date:** 2026-08-18
**Scope:** a targeted M0 scan, not a systematic review. Only original papers,
official technical reports, and official project repositories are used below.
Unless a venue is explicit in the linked primary source, a paper is described
only by its arXiv submission year. Reported findings belong to the cited
authors; this project has not reproduced them yet.

## Thematic synthesis

### Reliability needs interaction-level evaluation

Function-call correctness is necessary but not sufficient for reliable agents.
BFCL isolates important call-level and multi-turn function-calling abilities,
while $\tau$-bench evaluates a complete conversation against environment state
and domain rules. $\tau$-bench also introduced `pass^k`, the probability that
all of $k$ sampled attempts succeed, which makes consistency a first-class
target rather than hiding it inside mean accuracy. $\tau^2$-bench adds a
dual-control telecom setting in which both the agent and simulated user act on
a shared environment. These sources justify this project's combination of
strict tool-call diagnostics, execution-backed end-state grading, and
`pass^k`/`pass@k`; they do not establish that any particular training recipe
will improve those metrics.

### Verifiable rewards can train tool behavior, but reward design is the issue

DeepSeekMath introduced GRPO as a lower-memory relative-policy optimization
method for mathematical reasoning. ToolRL then studied fine-grained rewards
for tool selection and application. Nemotron-Research-Tool-N1 used a binary
rule-based reward over invocation format and functional correctness. ReTool
and ARTIST trained models with tools inside outcome-scored reasoning rollouts.
Together, these works support the feasibility of execution- or rule-backed
tool-use RL. They do not test this project's exact authorization predicate,
shared runtime/reward gate implementation, or customer-service reliability
comparison.

DPO is a useful offline comparator because it optimizes preferences without an
online reward-model/RL loop. It answers a different question from GRPO: whether
paired successful/recovering continuations are enough, rather than whether an
agent can learn directly from executable outcomes.

### Multi-turn agent RL is an infrastructure and stability problem

RAGEN identifies instability and shallow-strategy risks in multi-turn agent
RL, and argues for careful rollout shaping and sufficiently informative
rewards. Agent-R1 represents interaction as step-level transitions with modular
context and optimization interfaces. The `verifiers` project packages a task
dataset, harness/environment, and rubric for shared evaluation and RL use.
These sources support keeping full multi-turn RL as a separately gated stretch
milestone. [TRL v1.8 documentation](https://huggingface.co/docs/trl/v1.8.0/en/grpo_trainer)
separately describes multi-turn `environment_factory` support, so the
implementation question is now a measured compatibility choice rather than an
assumption that the trainer is single-turn-only.

### Verified synthetic data helps grounding, not policy proof

APIGen verifies generated examples through format checks, real function
execution, and semantic checks. xLAM unifies and augments heterogeneous agent
trajectories to train a family of action models. ToolACE combines agentic data
synthesis with rule- and model-based verification. These are strong precedents
for schema grounding and rejection-filtered trajectory generation. They do not
show that a model has internalized a domain policy, nor do their benchmark
results substitute for this project's matched H2/H3 ablations.

## Verified primary sources (15)

| # | Source | Year and source status | Contribution used here | Primary link |
|---:|---|---|---|---|
| 1 | *$\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains* | 2024, arXiv | Introduces simulated tool-agent-user conversations, end-state evaluation, domain-policy adherence, and reliability metric `pass^k`. | [arXiv:2406.12045](https://arxiv.org/abs/2406.12045) |
| 2 | *$\tau^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment* | 2025, arXiv | Introduces a dual-control telecom domain, compositional verifiable tasks, an environment-coupled user simulator, and coordination ablations. | [arXiv:2506.07982](https://arxiv.org/abs/2506.07982) |
| 3 | *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* | 2024, arXiv | Introduces GRPO, which estimates a relative group baseline without a separate value model; this is the optimization ancestor, not a tool-policy result. | [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) |
| 4 | *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* | 2023, arXiv | Derives an offline preference objective that avoids fitting an explicit reward model and running an online RL loop; anchors the DPO comparison arm. | [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) |
| 5 | *ToolRL: Reward is All Tool Learning Needs* | 2025, arXiv | Systematically studies reward type, scale, granularity, and dynamics for tool selection/application, then trains with GRPO. | [arXiv:2504.13958](https://arxiv.org/abs/2504.13958) |
| 6 | *Nemotron-Research-Tool-N1: Exploring Tool-Using Language Models with Reinforced Reasoning* | 2025, arXiv | Trains tool calling with a binary rule-based reward for format validity and functional correctness and compares SFT, RL, and SFT-then-RL. | [arXiv:2505.00024](https://arxiv.org/abs/2505.00024) |
| 7 | *ReTool: Reinforcement Learning for Strategic Tool Use in LLMs* | 2025, arXiv | Interleaves live code execution with reasoning and uses task outcomes to train when and how to invoke the tool. | [arXiv:2504.11536](https://arxiv.org/abs/2504.11536) |
| 8 | *Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning* (ARTIST) | 2025, arXiv technical report | Couples multi-turn reasoning, tool invocation, environment interaction, and outcome-based RL without step-level supervision. | [arXiv:2505.01441](https://arxiv.org/abs/2505.01441) |
| 9 | *RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning* | 2025, arXiv | Proposes StarPO/RAGEN and documents multi-turn stability, rollout-diversity, interaction-granularity, and reward-signal concerns. | [arXiv:2504.20073](https://arxiv.org/abs/2504.20073) |
| 10 | *Agent-R1: A Unified and Modular Framework for Agentic Reinforcement Learning* | 2025, arXiv technical report; revised 2026 | Defines step-level trajectory representation, flexible context management, and modular environment/optimization interfaces for agentic RL. | [arXiv:2511.14460](https://arxiv.org/abs/2511.14460) |
| 11 | *Verifiers: Environments for LLM Reinforcement Learning* | 2025-present, official software project | Defines reusable tasks/environments around a dataset, model harness, and reward rubric for both evaluation and RL; it is software, not a peer-reviewed paper. | [official repository](https://github.com/PrimeIntellect-ai/verifiers) |
| 12 | *APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets* | 2024, arXiv | Generates function-calling data and filters it with format, actual-execution, and semantic verification stages; releases the xLAM function-calling dataset. | [arXiv:2406.18518](https://arxiv.org/abs/2406.18518) |
| 13 | *xLAM: A Family of Large Action Models to Empower AI Agent Systems* | 2024, arXiv technical report | Unifies, augments, and synthesizes heterogeneous agent trajectories to train action-model checkpoints across several sizes. | [arXiv:2409.03215](https://arxiv.org/abs/2409.03215) |
| 14 | *ToolACE: Winning the Points of LLM Function Calling* | 2024, arXiv | Uses self-evolving, multi-agent synthesis and dual-layer rule/model verification to produce diverse tool-learning data. | [arXiv:2409.00920](https://arxiv.org/abs/2409.00920) |
| 15 | *A Function Calling Perspective on Scalable Large Language Model Agent Evaluation* (BFCL technical report) | 2025, official UC Berkeley technical report | Documents BFCL as a scalable function-calling evaluation spanning fundamental, live, multi-turn, and agentic settings; benchmark version must be pinned. | [UCB/EECS-2025-184](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-184.html) |

## Relevance to the pre-registered hypotheses

| Hypothesis | What the literature supports | What remains for this project to establish |
|---|---|---|
| **H1:** GRPO plus gate rewards on a $\leq$4B model closes at least 50% of the `pass^4` gap to a scaffolded 8B model at at most 30% of its generated tokens. | $\tau$-bench supplies the reliability construct; DeepSeekMath supplies GRPO; ToolRL, Tool-N1, ReTool, and ARTIST make outcome/rule-based tool RL plausible. BFCL and the synthetic-data papers offer call-format diagnostics and grounding sources. | None of the scanned sources establishes the exact small-trained-versus-large-scaffolded comparison, gap-closure threshold, or token-efficiency threshold. Only matched project runs can answer H1. |
| **H2:** adding the gate term halves `skipped_auth` versus otherwise identical GRPO. | ToolRL shows that reward decomposition matters; Tool-N1 shows that rule-based validity/correctness rewards can be enough to train tool calling; RAGEN warns that coarse rewards can induce shallow strategies. | No scanned source tests an authorization-state predicate replayed from dispatched mutative attempts with success recorded separately, or a gate-only causal ablation on `skipped_auth`. H2 requires a matched-seed protocol frozen before training. |
| **H3:** the trained model retains at least 90% of `pass^1` after removing the policy manual, while the base model degrades. | $\tau$-bench makes domain-policy context part of the agent problem. Tool-N1 and outcome-based RL papers suggest that strategies can arise without supervising intermediate reasoning. | None of the scanned sources demonstrates durable domain-policy knowledge under policy-manual removal. Tool-call accuracy or emergent reasoning is not evidence of policy internalization. H3 remains the clearest open empirical question in this scan. |

## Adopted choices versus potentially distinguishing choices

### Adopted or adapted from prior work

- `pass^k` as the reliability metric and complete environment-state grading
  are adopted from $\tau$-bench. `pass@k` is reported separately because it
  measures at-least-one success, not repeated reliability.
- The GRPO family is adopted from DeepSeekMath; the DPO arm is adopted as an
  offline preference-learning comparator. Tool use is an adaptation, not a
  reproduction of DeepSeekMath's math setting.
- Composite, verifiable tool rewards are adapted from the reward-design line
  represented by ToolRL and Tool-N1. ReTool and ARTIST motivate putting real
  tool feedback inside rollouts.
- The multi-turn stretch architecture is informed by RAGEN, Agent-R1, and
  `verifiers`; their systems and findings are not silently claimed as this
  project's implementation.
- Function-call grounding and rejection filtering are informed by APIGen,
  xLAM, and ToolACE. BFCL is suitable as an ancillary format/tool-selection
  diagnostic, not as the headline policy benchmark.

### Potentially distinguishing, pending a broader novelty search

- A controlled training-versus-runtime-scaffolding comparison that uses the
  same task set and decoding configuration, freezes each rung's opportunity
  caps, and reports `pass^k`, actual generated-token count, and GPU-seconds per
  episode. The rungs intentionally consume different budgets.
- One deterministic gate engine used in two roles: it blocks unsafe mutative
  actions at runtime and replays dispatched-attempt events to compute the gate
  reward during training, with handler success recorded separately.
- A causal gate-reward ablation centered on one operational failure,
  `skipped_auth`, rather than only aggregate function-call accuracy.
- A matched policy-manual-removal probe that separates prompt dependence from
  behavior retained after post-training.
- A pre-registered negative-result path and a small-model ($\leq$4B),
  constrained-compute focus.

These are **candidate distinguishing choices**, not established novel
contributions. This 15-source targeted scan is too small to prove novelty, so
the project must not call any of them a "core innovation" yet.

## Benchmark-version finding

The current implementation plan needs one explicit decision before Phase B:

- Sierra's [official `tau2-bench` repository](https://github.com/sierra-research/tau2-bench)
  now also carries later $\tau^3$-bench changes and task fixes.
- Amazon's [`tau2-bench-verified`](https://github.com/amazon-agi/tau2-bench-verified)
  is a separate corrected, human-verified fork with documented task, policy,
  database, and evaluation changes. It is **not** Sierra's upstream package.

Therefore, "upstream" and "`tau2-bench-verified`" cannot be used
interchangeably. The chosen repository, commit/tag, task set, simulator,
reward basis, and reproduced reference result must be pinned together. Scores
from Sierra's original/current tasks and Amazon's corrected tasks must not be
placed in one comparison table as if they used the same benchmark.

## Gaps and claims this project must not make

- Do not claim novelty from this scan. A broader search and experimental
  comparison are still required.
- Do not claim that tool-use RL already beats runtime scaffolding. None of the
  sources tests H1's matched comparison.
- Do not claim that format or functional correctness proves authorization
  policy adherence, or that a lower `skipped_auth` rate proves internalized
  policy without the H2/H3 controls.
- Do not generalize ReTool/ARTIST math results or Tool-N1/BFCL function-call
  results to $\tau$ retail policy reliability without running the target
  environment.
- Do not describe DPO as online RL, and do not compare its compute with GRPO
  without reporting rollout and data-generation costs.
- Do not call `verifiers` a paper or assume a current API supports the pinned
  training stack; pin and smoke-test a release.
- Do not call every $\tau^2$ domain dual-control. The 2025 paper introduces
  dual control specifically through its telecom setting; domain mechanics must
  be described separately.
- Do not say $\tau$-bench supplies `error_tags`. The project's failure tags are
  a custom taxonomy derived from its own event log.
- Do not confuse `pass^k` (all selected trials pass) with `pass@k` (at least
  one selected trial passes), or compare values computed with different trial
  counts and task sets without qualification.
- Do not present BFCL as a fixed dataset. It is a versioned, evolving
  benchmark, and published model rankings are time/version dependent.
- Do not treat synthetic-data paper results as license clearance. Dataset,
  model, and code licenses must be checked separately before redistribution.
- Do not report the cited authors' headline numbers as reproduced results.
  This repository still has no experimental measurements.

## Verification exceptions and naming notes

All requested source families were verified in primary sources. Two names need
care in later writing:

1. The current arXiv title for Agent-R1 is *Agent-R1: A Unified and Modular
   Framework for Agentic Reinforcement Learning*; older search text may show a
   different title.
2. `verifiers` is verified as an official software repository with its own
   citation guidance, not as a peer-reviewed publication.
