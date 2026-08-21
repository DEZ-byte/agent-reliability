"""Compare two arms on the same tasks with a paired test, and write the verdict.

BLUEPRINT_v2 section 10.2 requires paired inference. Two arms measured on the
same frozen split share every task, so treating them as independent samples
throws away the pairing and answers a weaker question than the one asked.

It also answers it misleadingly. Overlapping confidence intervals are not a
test: two arms can have heavily overlapping intervals and still differ
reliably on a paired comparison, and can have a significant-looking gap that a
paired test dissolves. Reading overlap as "no difference" is the error this
script exists to prevent, so it reports both, labelled.

The permutation test flips the sign of each task's difference. Under the null
that the arms are equivalent, a task's improvement is as likely to have gone
the other way, so the sign flip generates the reference distribution without
assuming normality across 100 or 150 tasks.

Discordant task counts are reported beside every p-value. A p-value resting on
six tasks that disagree is fragile whatever it says, and a reader who cannot
see that number cannot judge the claim.
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
RESAMPLES: Final = 20000
PERMUTATION_SEED: Final = 20260820


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


def load_runs(
    path: Path, rung: str, candidate: str | None = None
) -> dict[str, list[int]]:
    """Per-task success arrays for one rung and one candidate.

    The candidate filter is not optional in practice. A baseline episode log
    holds every model that was measured, all sharing the same task ids, so
    keying by task alone lets one model's rows silently overwrite another's and
    the comparison then reports whichever model happened to be written last.
    A file holding more than one candidate is refused rather than guessed at.
    """

    by_task: dict[str, dict[int, int]] = defaultdict(dict)
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
            by_task[row["task_id"]][row["run_index"]] = 1 if row["correct"] else 0

    if candidate is None and len(seen) > 1:
        raise ComparisonError(
            f"{path.name} holds {len(seen)} candidates ("
            + ", ".join(sorted(seen))
            + "); pass --candidate, or one model's rows would overwrite another's"
        )
    return {
        task: [runs[index] for index in sorted(runs)]
        for task, runs in by_task.items()
    }


def pass_power_k(runs: list[int], k: int) -> float | None:
    """The unbiased chance that all k of k independent attempts succeed."""

    n = len(runs)
    if n < k:
        return None
    return math.comb(sum(runs), k) / math.comb(n, k)


def paired_permutation(
    differences: list[float], *, resamples: int = RESAMPLES
) -> float:
    """Two-sided p-value from sign flips, which is what pairing licenses."""

    if not differences:
        raise ComparisonError("no paired tasks")
    observed = sum(differences) / len(differences)
    rng = random.Random(PERMUTATION_SEED)
    hits = 0
    for _ in range(resamples):
        total = sum(value if rng.random() < 0.5 else -value for value in differences)
        if abs(total / len(differences)) >= abs(observed) - 1e-12:
            hits += 1
    return hits / resamples


def compare(
    baseline: dict[str, list[int]],
    treatment: dict[str, list[int]],
    *,
    k: int,
) -> dict[str, Any]:
    shared = sorted(set(baseline) & set(treatment))
    if not shared:
        raise ComparisonError("the two arms share no task ids")

    differences: list[float] = []
    base_scores: list[float] = []
    treat_scores: list[float] = []
    discordant = 0
    improved = 0
    regressed = 0
    for task in shared:
        left = pass_power_k(baseline[task], k)
        right = pass_power_k(treatment[task], k)
        if left is None or right is None:
            continue
        base_scores.append(left)
        treat_scores.append(right)
        difference = right - left
        differences.append(difference)
        if difference > 0:
            discordant += 1
            improved += 1
        elif difference < 0:
            discordant += 1
            regressed += 1

    base_mean = sum(base_scores) / len(base_scores)
    treat_mean = sum(treat_scores) / len(treat_scores)
    return {
        "k": k,
        "tasks_compared": len(base_scores),
        "baseline": base_mean,
        "treatment": treat_mean,
        "difference": treat_mean - base_mean,
        "tasks_improved": improved,
        "tasks_regressed": regressed,
        "tasks_discordant": discordant,
        "p_value_two_sided": paired_permutation(differences),
        "resamples": RESAMPLES,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-episodes", required=True, type=Path)
    parser.add_argument("--treatment-episodes", required=True, type=Path)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--treatment-label", required=True)
    parser.add_argument("--rung", action="append", default=[])
    parser.add_argument(
        "--baseline-candidate",
        default=None,
        help="model id, required when the baseline log holds more than one",
    )
    parser.add_argument("--treatment-candidate", default=None)
    parser.add_argument("--k", action="append", type=int, default=[])
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    rungs = args.rung or ["R0", "R1"]
    ks = args.k or [1, 4]

    comparisons: list[dict[str, Any]] = []
    for rung in rungs:
        baseline = load_runs(
            args.baseline_episodes, rung, args.baseline_candidate
        )
        treatment = load_runs(
            args.treatment_episodes, rung, args.treatment_candidate
        )
        if not baseline or not treatment:
            continue
        for k in ks:
            entry = compare(baseline, treatment, k=k)
            entry["rung"] = rung
            comparisons.append(entry)

    if not comparisons:
        raise ComparisonError("no rung produced a comparison")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "kind": "paired_arm_comparison",
        "method": (
            "Paired permutation test on per-task pass^k, sign-flipping each "
            "task's difference. Section 10.2 requires paired inference; two "
            "arms on the same frozen split share every task, and overlapping "
            "confidence intervals are not a test."
        ),
        "baseline_label": args.baseline_label,
        "treatment_label": args.treatment_label,
        "baseline_candidate": args.baseline_candidate,
        "treatment_candidate": args.treatment_candidate,
        "permutation_seed": PERMUTATION_SEED,
        "comparisons": comparisons,
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
            "%s pass^%d: %.3f -> %.3f  (%+.3f)  p=%.4f  discordant=%d/%d"
            % (
                entry["rung"],
                entry["k"],
                entry["baseline"],
                entry["treatment"],
                entry["difference"],
                entry["p_value_two_sided"],
                entry["tasks_discordant"],
                entry["tasks_compared"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
