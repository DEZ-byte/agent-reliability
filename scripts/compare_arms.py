"""Compare two arms on the same tasks: paired interval first, tests second.

HYPOTHESIS_PROTOCOL section 6.2 is explicit that "permutation tests are
secondary to estimates and CIs", and section 6.1 pins 10,000 replicates with a
stored seed. An earlier version of this script reported a p-value and no
interval at all, which inverts the protocol's own priority and leaves a reader
with a significance verdict and no idea how precise the estimate is.

Three things this reports that a naive comparison omits.

**An interval on the paired difference**, by task-level bootstrap. The tasks are
the sampling unit, so they are what gets resampled; resampling episodes would
treat four runs of one task as four independent observations and understate the
width badly.

**A p-value that cannot be zero.** A Monte Carlo permutation p-value is
`(hits + 1) / (resamples + 1)`, never `hits / resamples`. The uncorrected form
reports 0.0 when no resample beats the observed statistic, and 0.0 is not a
probability - it is the resample count showing through. The exact paired sign
test is reported alongside, because it does not depend on a resample count.

**Sampling degeneracy per arm.** pass^k only means anything if the k attempts
can differ. When an arm emits byte-identical completions for a task, that
task's pass^k collapses to pass^1 and the comparison is not matched on the
thing it claims to measure. The rate is reported per arm, and so is the
comparison restricted to tasks where both arms genuinely varied.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SCHEMA_VERSION: Final = 1

# HYPOTHESIS_PROTOCOL sections 6.1 and 6.2 both pin 10,000 with a stored seed.
BOOTSTRAP_REPLICATES: Final = 10000
PERMUTATION_RESAMPLES: Final = 10000
SEED: Final = 20260820
CONFIDENCE: Final = 0.95


class ComparisonError(RuntimeError):
    """The two arms cannot be compared as paired measurements."""


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() or "unknown"


def load_arm(
    path: Path, rung: str, candidate: str | None = None
) -> dict[str, dict[str, Any]]:
    """Per-task outcomes and completions for one rung and one candidate.

    The candidate filter is not optional in practice. A baseline episode log
    holds every model that was measured, all sharing the same task ids, so
    keying by task alone lets one model's rows silently overwrite another's and
    the comparison then reports whichever model happened to be written last. A
    file holding more than one candidate is refused rather than guessed at.
    """

    runs: dict[str, dict[int, bool]] = defaultdict(dict)
    texts: dict[str, dict[int, tuple[str, ...]]] = defaultdict(dict)
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            name = row.get("candidate")
            if name is not None:
                seen.add(name)
            if candidate is not None and name != candidate:
                continue
            if row["rung"] != rung:
                continue
            runs[row["task_id"]][row["run_index"]] = bool(row["correct"])
            texts[row["task_id"]][row["run_index"]] = tuple(row.get("completions") or ())

    if candidate is None and len(seen) > 1:
        raise ComparisonError(
            f"{path.name} holds {len(seen)} candidates ("
            + ", ".join(sorted(seen))
            + "); pass --candidate, or one model's rows would overwrite another's"
        )
    return {
        task: {
            "successes": [runs[task][index] for index in sorted(runs[task])],
            "distinct_completions": len(
                {texts[task][index] for index in sorted(texts[task])}
            ),
        }
        for task in runs
    }


def pass_power_k(successes: list[bool], k: int) -> float | None:
    """The unbiased chance that all k of k independent attempts succeed."""

    n = len(successes)
    if n < k:
        return None
    return math.comb(sum(successes), k) / math.comb(n, k)


def pass_at_k(successes: list[bool], k: int) -> float | None:
    """The unbiased chance that at least one of k attempts succeeds."""

    n = len(successes)
    failures = n - sum(successes)
    if n < k:
        return None
    if failures < k:
        return 1.0
    return 1.0 - math.comb(failures, k) / math.comb(n, k)


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile, symmetric about the median."""

    index = int(round(fraction * (len(values) - 1)))
    return values[min(max(index, 0), len(values) - 1)]


def bootstrap_interval(
    differences: list[float], *, rng: random.Random
) -> tuple[float, float]:
    """Task-level bootstrap of the mean paired difference.

    Tasks are the sampling unit. Resampling episodes instead would treat four
    runs of one task as four independent observations and report an interval
    far narrower than the evidence supports.
    """

    n = len(differences)
    means: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        total = 0.0
        for _ in range(n):
            total += differences[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    tail = (1.0 - CONFIDENCE) / 2.0
    return _percentile(means, tail), _percentile(means, 1.0 - tail)


def permutation_p(differences: list[float], *, rng: random.Random) -> float:
    """Two-sided sign-flip p-value, add-one corrected so it cannot be zero."""

    observed = abs(sum(differences) / len(differences))
    hits = 0
    for _ in range(PERMUTATION_RESAMPLES):
        total = sum(value if rng.random() < 0.5 else -value for value in differences)
        if abs(total / len(differences)) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (PERMUTATION_RESAMPLES + 1)


def sign_test_p(improved: int, regressed: int) -> float | None:
    """Exact two-sided paired sign test. No resample count to ride on."""

    n = improved + regressed
    if n == 0:
        return None
    extreme = max(improved, regressed)
    tail = sum(math.comb(n, i) for i in range(extreme, n + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def compare(
    baseline: dict[str, dict[str, Any]],
    treatment: dict[str, dict[str, Any]],
    *,
    k: int,
    rung: str,
    tasks: list[str] | None = None,
) -> dict[str, Any]:
    shared = tasks if tasks is not None else sorted(set(baseline) & set(treatment))
    if not shared:
        raise ComparisonError("the two arms share no task ids")

    differences: list[float] = []
    base_power: list[float] = []
    treat_power: list[float] = []
    base_at: list[float] = []
    treat_at: list[float] = []
    improved = regressed = 0

    for task in shared:
        left = pass_power_k(baseline[task]["successes"], k)
        right = pass_power_k(treatment[task]["successes"], k)
        if left is None or right is None:
            continue
        base_power.append(left)
        treat_power.append(right)
        base_at.append(pass_at_k(baseline[task]["successes"], k) or 0.0)
        treat_at.append(pass_at_k(treatment[task]["successes"], k) or 0.0)
        difference = right - left
        differences.append(difference)
        if difference > 0:
            improved += 1
        elif difference < 0:
            regressed += 1

    # A separate stream per comparison, so one comparison's draws cannot
    # correlate with another's through a shared position in one sequence.
    rng = random.Random(f"{SEED}:{rung}:{k}")
    low, high = bootstrap_interval(differences, rng=rng)

    base_mean = sum(base_power) / len(base_power)
    treat_mean = sum(treat_power) / len(treat_power)
    base_any = sum(base_at) / len(base_at)
    treat_any = sum(treat_at) / len(treat_at)

    return {
        "rung": rung,
        "k": k,
        "tasks_compared": len(base_power),
        "baseline_pass_power_k": base_mean,
        "treatment_pass_power_k": treat_mean,
        "difference": treat_mean - base_mean,
        "difference_ci95": [low, high],
        "difference_ci95_width": high - low,
        "baseline_pass_at_k": base_any,
        "treatment_pass_at_k": treat_any,
        # The band of tasks solved sometimes but not always. A treatment can
        # raise capability and widen this at the same time, and reporting only
        # pass^k hides that.
        "baseline_sometimes_band": base_any - base_mean,
        "treatment_sometimes_band": treat_any - treat_mean,
        "tasks_improved": improved,
        "tasks_regressed": regressed,
        "tasks_changed": improved + regressed,
        "p_permutation_two_sided": permutation_p(differences, rng=rng),
        "p_sign_test_exact": sign_test_p(improved, regressed),
        "permutation_resamples": PERMUTATION_RESAMPLES,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def degeneracy(arm: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """How often an arm produced the same completion every time.

    pass^k asks whether k independent attempts all succeed. Where the attempts
    are byte-identical the question is vacuous for that task, so the rate
    belongs beside every pass^k figure.
    """

    counts = [entry["distinct_completions"] for entry in arm.values()]
    identical = sum(1 for value in counts if value <= 1)
    return {
        "tasks": len(counts),
        "tasks_with_identical_completions": identical,
        "degenerate_rate": identical / len(counts) if counts else None,
        "mean_distinct_completions": sum(counts) / len(counts) if counts else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-episodes", required=True, type=Path)
    parser.add_argument("--treatment-episodes", required=True, type=Path)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--treatment-label", required=True)
    parser.add_argument("--baseline-candidate", default=None)
    parser.add_argument("--treatment-candidate", default=None)
    parser.add_argument("--rung", action="append", default=[])
    parser.add_argument("--k", action="append", type=int, default=[])
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    rungs = args.rung or ["R0", "R1"]
    ks = args.k or [1, 4]

    comparisons: list[dict[str, Any]] = []
    restricted: list[dict[str, Any]] = []
    degeneracies: dict[str, Any] = {}

    for rung in rungs:
        baseline = load_arm(args.baseline_episodes, rung, args.baseline_candidate)
        treatment = load_arm(args.treatment_episodes, rung, args.treatment_candidate)
        if not baseline or not treatment:
            continue
        degeneracies[rung] = {
            "baseline": degeneracy(baseline),
            "treatment": degeneracy(treatment),
        }
        varied = sorted(
            task
            for task in set(baseline) & set(treatment)
            if baseline[task]["distinct_completions"] > 1
            and treatment[task]["distinct_completions"] > 1
        )
        for k in ks:
            comparisons.append(compare(baseline, treatment, k=k, rung=rung))
            if k > 1 and varied:
                entry = compare(
                    baseline, treatment, k=k, rung=rung, tasks=varied
                )
                entry["restricted_to"] = "tasks where both arms varied"
                restricted.append(entry)

    if not comparisons:
        raise ComparisonError("no rung produced a comparison")

    alpha = 0.05 / len(comparisons)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "kind": "paired_arm_comparison",
        "method": (
            "Task-level bootstrap interval on the paired difference, which "
            "HYPOTHESIS_PROTOCOL section 6.2 makes primary, with the paired "
            "sign-flip permutation p-value and the exact paired sign test as "
            "secondary. Tasks are the sampling unit. Permutation p-values are "
            "add-one corrected so they cannot be reported as zero."
        ),
        "baseline_label": args.baseline_label,
        "treatment_label": args.treatment_label,
        "baseline_candidate": args.baseline_candidate,
        "treatment_candidate": args.treatment_candidate,
        "seed": SEED,
        "multiplicity": {
            "comparisons_reported": len(comparisons),
            "bonferroni_alpha": alpha,
            "note": (
                "No multiplicity policy was pre-registered. The Bonferroni "
                "threshold is reported so a reader can apply it; it is not "
                "used to filter what is shown."
            ),
        },
        "sampling_degeneracy": degeneracies,
        "comparisons": comparisons,
        "restricted_comparisons": restricted,
        "source_commit": _git_commit(),
        "platform": {"python": platform.python_version(), "system": platform.system()},
    }

    path = Path(args.summary)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(
        (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )
    os.replace(temporary, path)

    for entry in comparisons:
        print(
            "%s pass^%d: %.4f -> %.4f  %+.4f  CI95 [%+.4f, %+.4f]  "
            "perm p=%.5f  sign p=%.3g  changed=%d/%d"
            % (
                entry["rung"],
                entry["k"],
                entry["baseline_pass_power_k"],
                entry["treatment_pass_power_k"],
                entry["difference"],
                entry["difference_ci95"][0],
                entry["difference_ci95"][1],
                entry["p_permutation_two_sided"],
                entry["p_sign_test_exact"] if entry["p_sign_test_exact"] else float("nan"),
                entry["tasks_changed"],
                entry["tasks_compared"],
            )
        )
    for entry in restricted:
        print(
            "  restricted %s pass^%d: %.4f -> %.4f  %+.4f  (%d tasks varied in both)"
            % (
                entry["rung"],
                entry["k"],
                entry["baseline_pass_power_k"],
                entry["treatment_pass_power_k"],
                entry["difference"],
                entry["tasks_compared"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
