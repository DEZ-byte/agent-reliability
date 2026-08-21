# Decision records the configuration cites

The project's full decision log is append-only and kept outside this
repository. These four entries are extracted verbatim because
`configs/model_smoke.json` and `configs/model_candidates.json` name them,
and `scripts/smoke_models.py` verifies that each cited decision exists and
carries the exact markers the gate claims. Without them the licence and
gate-demotion checks cannot run.

The log is append-only, so these do not change once written.

---

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
