# Errata

Corrections to results this repository has already published. Each entry says
what was wrong, how we know, which version is authoritative, and what changed.
Nothing is rewritten in history. The original commits stay as they were, and
this file is where they are corrected.

| | What was wrong | Published in | Corrected in |
| :-- | :-- | :-- | :-- |
| **E1** | The manifest froze a hash for `utility-sft-30007ed.json` that matches no committed bytes | `7e33eb5` | `9bc4f14` changed the hash without saying so. This entry is the disclosure. |
| **E2** | The manifest briefly froze a re-run's copy of `utility-grpo-30007ed.json` | `9bc4f14` | `cb9ea8d` |
| **E3** | Four manifest entries named no recording commit | `7e33eb5`, `9bc4f14` | `9bc4f14`, `cb9ea8d`. The builder now refuses both E1 and E3 (`48869b1`) and is append-only (`fe42057`). |
| **E4** | The MMLU paired difference and the improved/regressed counts came from an incomplete file | `7e33eb5` | `5424bb2` |
| **E5** | "It did not forget anything" claimed more than the interval supports | `7e33eb5` | `5424bb2` |
| **E6** | ~~Truncation "a quarter as often" corrected to "less than half as often"~~ | `7e33eb5` | Withdrawn. Both rates were wrong (E7). |
| **E7** | The MMLU truncation rates counted batch padding as truncation | `7e33eb5` | Detector fixed in `72f7482`. Rates withdrawn until a re-run. |
| **E8** | Three GRPO results used the final step-400 adapter, not the dev-selected checkpoint | `8cef0b8`, `30007ed`, `9bc4f14` | Weight change re-measured (`ece15b0`). Phase B and MMLU marked, not yet re-run. |
| **E9** | Four artifacts name a source commit that did not hold the code that made them | `7e33eb5`, `9bc4f14` | Disclosed here. The weight-change script now refuses a dirty tree (`72f7482`). |
| **E10** | Four Phase B statements were false, and "worse than doing nothing" rested on an unstated rule | `30007ed` | This correction, with `phase_b-gates-d692d43.json` |
| **E11** | The headline left out the untrained 4B teacher, which beats every arm, and mixed rungs | `1249806` onward | This correction, with `sft-vs-teacher-*-d692d43.json` |
| **E12** | Ten phrases claimed more than their numbers | various | This correction |

E1 to E4 share one cause, explained under E1. Two evaluation processes were
writing the same result files at the same time, and the manifest and the
comparison were both built while one of them was still running.

---

## E1. A frozen hash that no commit holds

**What was wrong.** At `7e33eb5`, `results/artifact_manifest.json` recorded
this for `utility-sft-30007ed.json`:

| | SHA-256 | Bytes |
| :-- | :-- | --: |
| Manifest at `7e33eb5` | `7b830f874e23fbe92ca069729fc8be9547f7f74cd31c5fcdd61eb3aa90054482` | 6082 |
| File committed at `7e33eb5` | `591654473d824d26523301b9bcb84aa82d26e4e4ae124e8929aa441e14c2a7d2` | 6082 |

On any clean clone of that commit, `ArtifactImmutabilityTests` failed. So did
`python scripts/build_artifact_manifest.py --check`, with exit code 1. The
repository claimed that a changed artifact fails a test. That was true, and the
failing artifact was the one it had just published.

**Why.** Two MMLU evaluations were running at once and writing the same paths.
One was started with the fixed answer extractor. The other was left over from
an earlier launch that had not fully stopped. The committed artifacts show the
overlap. The runner evaluates one arm at a time, and each arm takes roughly
half an hour:

| Committed artifact | `created_at_utc` |
| :-- | :-- |
| `utility-base-30007ed.json` | 10:46:37 |
| `utility-sft-30007ed.json` | 11:22:19 |
| `utility-grpo-30007ed.json` | 11:24:45 |

A GRPO evaluation that starts 2 minutes 26 seconds after the SFT one cannot come
from the same sequential run.

The rest of the timeline comes from the working session's log and from file
times on the machine that ran the evaluations. It cannot be checked from the
repository:

| UTC | Event |
| :-- | :-- |
| 11:56:59 | The paired MMLU comparison runs. The SFT response file holds 391 of 400 rows (see E4). |
| ~11:58:08 | The manifest is rebuilt and hashes the SFT summary then on disk: `7b830f…` |
| 11:58:22 | The later SFT evaluation finishes and replaces that summary: `591654…` |
| 11:58:37 | The files are staged and committed as `7e33eb5`. |
| 12:08:27 | The later GRPO evaluation finishes and overwrites the GRPO summary and responses on disk (see E2). |

So the manifest indexed one copy and Git received another. Both are 6,082
bytes, so they most likely differ only in their timestamp. That cannot be
checked. The `7b830f…` bytes were never committed, and they no longer exist.

**Which bytes are authoritative.** The committed ones, `591654…`. They are the
only bytes Git has held under that name. The response file kept from the same
evaluation reproduces every figure in them exactly: accuracy, all five rates,
the mean answer length and all 57 per-subject scores.
[`scripts/compare_utility.py`](scripts/compare_utility.py) now checks that
before it uses a row, and `results/utility-comparison-48869b1.json` records the
check.

**How it was corrected, and what went wrong with that.** Commit `9bc4f14`
rebuilt the manifest to add a new artifact. That rebuild replaced `7b830f…` with
`591654…`, and the commit message did not mention it. The hash was repaired
silently, which is exactly what a manifest exists to prevent. This entry is the
disclosure that should have come with `9bc4f14`.

Since `fe42057` the builder reads the committed manifest first and refuses to
rewrite or drop any entry in it. The rebuild in `9bc4f14` would now stop with an
error instead of replacing the hash.

## E2. A re-run's timestamp, briefly frozen

The same rebuild in `9bc4f14` hashed `utility-grpo-30007ed.json` from the
working copy. The late GRPO evaluation had overwritten that copy at 12:08:27,
after `7e33eb5` was committed. Only `created_at_utc` differed. The manifest
recorded `5f29ac819b9228df6f950b8204612b60f53bca125e673a3a28ffb653aedc521e`
instead of the committed `1b88cf099ba522130737d4f933d7e56650e11a395264cdf4a06f5e169aeda834`.
`cb9ea8d` restored the committed bytes and rebuilt the manifest, and it said so.

One consequence remains. The GRPO response file kept locally was written by
that later evaluation, not by the one that wrote the committed GRPO summary.
Decoding is greedy, and the later file reproduces every committed GRPO summary
figure exactly, including all 57 subjects. Still, nothing can prove that the
two runs agreed on every individual question. The comparison artifact does not
flag this itself, so this entry is the caveat for the GRPO rows in
`utility-comparison-48869b1.json`. Those rows also come from the step-400
adapter (E8).

## E3. Entries with no recording commit

At `7e33eb5`, all three utility entries had `"recorded_in_commit": null`. The
manifest was built before those files were committed, so no commit held them
yet. `9bc4f14` filled the field in.

`weight-change-7e33eb5.json` had the same problem one commit later. At `9bc4f14`
its entry was also null, because that manifest was built before the file was
committed. `cb9ea8d` filled it in.

`scripts/build_artifact_manifest.py` now refuses to index an artifact unless a
full 40-character commit holds it, byte for byte. The E1 and E3 failure modes
now stop the build instead of going into the index. `tests/test_artifact_manifest.py`
covers both.

## E4. MMLU numbers from an incomplete file

**Published** (README section 5, FINDINGS D3), SFT against the untrained model:
paired difference +0.005, 95% interval −0.038 to +0.049, 38 questions improved
and 36 got worse.

**Why it cannot be right.** The committed summaries record 214/400 correct
untrained (0.535) and 217/400 after SFT (0.5425). That is three more questions
right. The mean paired difference must be +3/400 = +0.0075, and improved minus
regressed must be 3. But 38 − 36 = 2.

**Cause.** The comparison was typed into a terminal and run at 11:56:59. The
SFT response file was still being rewritten at that moment and held 391 rows, so
only 391 questions were paired (E1 timeline).

**Corrected**, from all 400 questions, in
[`results/utility-comparison-48869b1.json`](results/utility-comparison-48869b1.json):

| Comparison | Paired difference | 95% interval | Improved / got worse | Exact sign test |
| :-- | --: | :-- | :-- | --: |
| SFT vs untrained | **+0.0075** | −0.0350 to +0.0500 | **40 / 37** | p = 0.82 |
| GRPO vs untrained | +0.0025 | −0.0400 to +0.0450 | 40 / 39 | p = 1.00 |
| GRPO vs SFT | −0.0050 | −0.0225 to +0.0125 | 5 / 7 | p = 0.77 |

The GRPO rows compare the final step-400 adapter, not the selected GRPO
checkpoint (E8). The GRPO counts were right the first time, because its file was
complete. Its
difference was printed as +0.003, a rounding of +0.0025. Its upper bound moves
from +0.048 to +0.045 because the interval is now drawn from the project's pinned
seed, not the one-off seed the terminal script used.

## E5. "It did not forget anything"

The README table answered "It forgot things." with **No.** FINDINGS titled D3
"It did not forget anything" and called general knowledge "unchanged". The
interval does not support that. On this sample it allows a loss of up to 3.5
points. A null result inside an interval that wide shows no detectable change.
It does not show that nothing changed.

Both documents now say: **no detectable MMLU change on this 400-question
sample.**

The same retention paragraph said the model had been trained on "a thousand
calculator trajectories". The SFT set is 684 trajectories, at most one per task,
kept from 4,000 teacher attempts over the 1,000 training tasks
(`results/sft-dataset-54218c4.json`). That phrase was dropped in `5424bb2`
without a note. This is the note.

## E6. "A quarter as often" (withdrawn)

This entry first said the fine-tuned model runs out of token budget "less than
half as often" as the untrained one, because 16% / 40% = 0.4. Both rates were
wrong (E7), so the correction was wrong too. It is withdrawn, and no ratio is
claimed.

## E7. Truncation rates that measured batches, not rows

**What was wrong.** The MMLU truncation rates: 40% untrained, 16% after SFT,
20% after GRPO (README section 5 and Limits, FINDINGS D3, and E6 above).

**Why.** `scripts/run_utility_eval.py` flagged a row when its last new token was
not `tokenizer.eos_token_id`. But `generate()` pads every row in a batch to the
length of the longest one. Qwen3's pad token (`<|endoftext|>`, 151643) is not its
eos token (`<|im_end|>`, 151645). So a row that finished early ended in padding
and was flagged. Whenever one row in a batch of 8 used the full 320 tokens, all
8 rows were flagged.

**Evidence.** In all three response files the flags are all-or-none inside
every batch of 8. There are 0 mixed batches out of 50, for every arm. The flagged
counts are 160, 64 and 80, which is exactly 20, 8 and 10 whole batches. The
published rates are the share of batches with one long row, not the share of
rows that were cut off.

**What it does not affect.** Accuracy, the extraction-failure rates and the
readable-in-both comparison do not use this flag. The
`unreadable_within_budget_rate` did use it. It read 0.0 for every arm, because
every unreadable answer sat in a flagged batch, so it is withdrawn too. The
per-row `truncated` field in `utility-comparison-48869b1.json` copies the flag as
the evaluation wrote it, so that field is wrong as well.

**Status.** The detector is fixed in `72f7482`. `evaluation.utility.was_truncated`
flags a row only if it used the whole budget and emitted no stop token, and
tests cover a padded batch. The true rates need a re-run, and that has not been
done. Until then no truncation rate is published.

## E8. GRPO results from the final adapter, not the selected one

**What was wrong.** Three results labelled as the GRPO arm used the final
step-400 adapters:

| Artifact | Adapter used | Adapter that dev selection picked and test measured |
| :-- | :-- | :-- |
| `phase_b-grpo-8cef0b8.json` | run folder = step 400 (`83f7055c…`) | `checkpoint-200` (`024a1311…`) |
| `utility-grpo-30007ed.json` | run folder = step 400 (`83f7055c…`) | `checkpoint-200` (`024a1311…`) |
| `weight-change-7e33eb5.json` | step 400 at both rates (`83f7055c…`, `16aca300…`) | `checkpoint-200`, `checkpoint-300` (`39520ab1…`) |

The top of each GRPO run folder holds a copy of the last checkpoint. Its
adapter file is byte-identical to `checkpoint-400`. So an evaluation pointed at
the run folder evaluated step 400. Nothing caught this, because the Phase B and
MMLU artifacts record the adapter path, not a hash of its weights.

**Evidence.** `grpo-test-qwen3-1.7b-e7b8d74.json` records `checkpoint-200` with
weights `024a1311…`. `grpo-test-lr1e5-8182e7e.json` records `checkpoint-300`
with weights `39520ab1…`. The adapter hashes above come from the files on the
training machine.

**Corrected: weight change.** Re-measured on the selected checkpoints in
`weight-change-72f7482.json`, with the step-400 adapters beside them:

| Rate | Adapter | Adapter moved | Effective delta moved |
| :-- | :-- | --: | --: |
| 1e-6 | `checkpoint-200` (selected) | 0.408% | 1.38% |
| 1e-6 | step 400 | 0.449% | 1.55% |
| 1e-5 | `checkpoint-300` (selected) | 3.769% | 13.68% |
| 1e-5 | step 400 | 3.818% | 13.84% |

This settles why FINDINGS said 0.41% and the README said 0.45%. They described
different adapters. FINDINGS also paired 0.41% (checkpoint-200) with 3.82%
(step 400) in one table.

**Not yet corrected: Phase B and MMLU.** Re-running them on `checkpoint-200`
needs a GPU and has not been done. Until then, their GRPO columns in the README
and FINDINGS are marked as step-400 diagnostics. The SFT columns are not
affected: they use the dev-selected `checkpoint-86`.

## E9. Artifacts that name the wrong code

**What was wrong.** Four artifacts record a `source_commit` that did not contain
the code that produced them.

| Artifact | Names | Why that is wrong |
| :-- | :-- | :-- |
| `utility-{base,sft,grpo}-30007ed.json` | `30007ed` | At `30007ed`, `run_utility_eval.py` set `MAX_NEW_TOKENS = 128` and wrote no truncation field. The artifacts record `max_new_tokens: 320` and `truncated_rate`. Both first appear in `7e33eb5`, committed after the runs finished. |
| `weight-change-7e33eb5.json` | `7e33eb5` | Written 2026-08-30 16:57 UTC. `measure_weight_change.py` first appears in `9bc4f14`, committed 2026-08-31 16:54 UTC. |

**Why.** These scripts recorded `git rev-parse HEAD` without checking for
uncommitted changes. What ran was an uncommitted working tree. For the utility
runs, every recorded setting matches what `7e33eb5` committed, but that cannot
be proven byte for byte.

**Status.** The artifacts stay as written. `measure_weight_change.py` now
refuses a dirty tree (`72f7482`), and the artifact that replaces this one names
the commit that ran it. The evaluation runners still need the same guard. That
is planned with the GPU re-runs.

## E10. Phase B: what the model did, and what the reward assumed

Four statements about the transfer environment were false:

| Published | What the episodes show |
| :-- | :-- |
| It learned to operate "three tools it has never seen". | It operated two of the three offered tools, `authenticate` and `update_order`. |
| "It never once calls the lookup tool." | `get_order` was called in 3 of 600 SFT episodes and 2 of 600 GRPO episodes. |
| "It calls the verification tool, receives `authenticated: false`, and changes the record regardless." | Phase B is single-shot. The model writes every tool call before any tool runs, so it never receives a result. It emits the verification and the write together. |
| "The fine-tuned model is worse in this environment than the one that does nothing." | See below. |

**The reward verdict.** The fine-tuned model's mean reward, −0.394, is below the
untrained model's +0.286. That comparison counts two different violations the
same way. `results/phase_b-gates-d692d43.json` replays all 1,800 episodes,
reproduces every stored reward, and splits them:

| | After SFT | After GRPO (step 400) |
| :-- | --: | --: |
| Writes for an unverified customer | 267 | 262 |
| Verified writes that only skipped `get_order` | 285 | 289 |
| Mean reward, as scored | −0.394 | −0.387 |
| Mean reward, if skipping the lookup were not penalised | +0.366 | +0.384 |

The prompt tells the model to verify the customer. It never tells it to look the
order up first, yet the `order_id_exists` gate requires exactly that, and the
reward treats a verified write without a lookup like an unauthorised one. With
that penalty lifted, the fine-tuned model scores above the untrained one. So the
"worse than doing nothing" claim is withdrawn. The refusal failure stands: 267 of
296 writes on requests that should have been refused.

The rescoring is arithmetic on the same episodes, not a new run. A fair verdict
needs a re-run with a prompt that asks for the lookup.

## E11. The teacher was missing from the headline

**Omitted.** The untrained Qwen3-4B wrote the SFT trajectories and was measured
on the same test split (`baseline-phase_a-3cc174f.json`): `pass^1` 0.608 and
`pass^4` 0.567 at R1. That beats every arm in the headline table, which did not
show it. Paired against each SFT run at R1, the fine-tuned 1.7B is lower on
`pass^4` by 0.173, 0.153 and 0.107, and every 95% interval excludes zero
(`sft-vs-teacher-run{1,2,3}-d692d43.json`). The headline now says the fine-tuned
1.7B beats the 8B but does not reach its teacher.

**Mixed rungs.** The headline table showed the 8B at R1 and every other arm at
R0, while the paired intervals beneath it compared R1 with R1. Every arm is now
shown at R1, with a rung column. The largest shift is the untrained 1.7B, by up to
0.04.

**Cost.** "Costs about a third as much to run" is now "uses about 31% of the
parameter-weighted generated-token proxy per attempt". The proxy counts decoding
only, the column was per attempt rather than per task, and the parameter counts
come from model configs, not from an artifact.

## E12. Phrases that claimed more than their numbers

| Was | Now | Why |
| :-- | :-- | :-- |
| Format and tool reach were "most of the untrained gap" | 56 of the 149 removed failures, about 38% | 62 → 6 in those rows, 356 → 263 in the wrong-value row † |
| Wrong-value failures: "the third row barely moves" | Fell substantially, by 26%, but still 263 of 269 remaining failures | 356 → 263 † |
| "About +33 points" loosely, "+15" strictly | +27 to +33, and +15 to +21 | The old pair set the best `pass@4` run against the worst `pass^4` run |
| "~1 step in 4 carried no gradient: all 8 attempts scored alike" | 23% and 27% of steps, in which all 16 attempts (2 problems × 8) scored alike | The reward function never received the group size, so each step was logged as one group of 16: 400 groups in 400 steps |
| "Three of the four reward terms never varied" | Accuracy dominated; format and efficiency varied negligibly; the gate term was inert | Format (0.006) and efficiency (0.002) did vary, a little |
| "Nothing left to reach"; SFT "removed every failure a preference signal can fix" | Consistent with the null, not tested as a cause | No experiment tested it |
| "Reinforcement learning added nothing" | No detectable GRPO benefit at these learning rates and this budget | A null at one budget and one task |
| The dead-group fix "was not implemented" | It is built but has not been run | `--prompt-filter` in `scripts/train_grpo.py` |
| "More seeds would sharpen this: No." | Probably not; three runs make the between-run spread rough | The standard deviation came from three runs |
| MMLU "blind to losses under ~3.5 points" | Reliably detects only changes of about 6 points | 77 of 400 answers changed; that gives about 80% power at about 6 points |

Numbers that come only from local logs or configs are now marked † in the README
and FINDINGS: the failure counts in README section 2, the grader-gaming rates,
the untrained token ratio, the band-test p-value, the parameter counts and the
serving memory.
