# Errata

Corrections to results this repository has already published. Each entry says
what was wrong, how we know, which version is authoritative, and what changed.
Nothing is rewritten in history. The original commits stay as they were, and
this file is where they are corrected.

| | What was wrong | Published in | Corrected in |
| :-- | :-- | :-- | :-- |
| **E1** | The manifest froze a hash for `utility-sft-30007ed.json` that matches no committed bytes | `7e33eb5` | `9bc4f14` changed the hash without saying so. This entry is the disclosure. |
| **E2** | The manifest briefly froze a re-run's copy of `utility-grpo-30007ed.json` | `9bc4f14` | `cb9ea8d` |
| **E3** | All three utility entries named no recording commit | `7e33eb5` | `9bc4f14`. The builder now refuses both E1 and E3 (`48869b1`). |
| **E4** | The MMLU paired difference and the improved/regressed counts came from an incomplete file | `7e33eb5` | This correction |
| **E5** | "It did not forget anything" claimed more than the interval supports | `7e33eb5` | This correction |
| **E6** | Truncation was described as happening "a quarter as often" | `7e33eb5` | This correction |

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
two runs agreed on every individual question. The GRPO rows in
`utility-comparison-48869b1.json` carry that caveat.

## E3. Entries with no recording commit

At `7e33eb5`, all three utility entries had `"recorded_in_commit": null`. The
manifest was built before those files were committed, so no commit held them
yet. `9bc4f14` filled the field in.

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

The GRPO counts were right the first time, because its file was complete. Its
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

## E6. "A quarter as often"

Both documents said the fine-tuned model runs out of token budget "a quarter as
often" as the untrained one. The rates are 16% and 40%, so the ratio is 0.4. It
runs out less than half as often.
