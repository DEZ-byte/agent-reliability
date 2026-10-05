"""Pair the MMLU arms question by question, and freeze the result.

The first write-up of the MMLU probe reported a paired difference of +0.005,
with 38 questions improved and 36 worse. Neither number fits the accuracies the
artifacts record: 214 of 400 before training and 217 after is a difference of
three questions, so improved minus regressed has to be three and the mean
difference has to be +0.0075. The comparison had been typed into a terminal and
run while the fine-tuned arm's response file was still being rewritten, with
391 of its 400 rows on disk.

So the comparison is now an artifact rather than terminal output, and it is
built only from rows that provably belong to the headline numbers. Before
comparing anything this refuses to run unless:

- every response file reproduces its committed summary exactly: accuracy, every
  rate, the mean answer length and every per-subject score;
- every arm answered the same frozen questions, with the same gold letters,
  prompts, decoding and split manifest;
- every summary is byte-identical to the copy committed in Git, and the working
  tree is clean, so `source_commit` names code that actually ran.

The rows are taken as graded when the evaluation ran. They are not re-scored
with whatever the answer extractor looks like today, because that would be a
different measurement wearing the old one's name.

The response files themselves stay uncommitted (they hold full completions and
fall under `results/**/*.jsonl`). The artifact carries every scored field per
question plus a SHA-256 of each completion, so anyone holding the response
files can bind them to this record.

The statistics are the ones `compare_arms.py` already uses: a question-level
bootstrap interval on the paired difference with 10,000 seeded replicates, the
paired sign-flip permutation p-value, and the exact sign test on discordant
pairs, which for right-or-wrong outcomes is McNemar's exact test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, NamedTuple

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from compare_arms import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    CONFIDENCE,
    PERMUTATION_RESAMPLES,
    SEED,
    bootstrap_interval,
    permutation_p,
    sign_test_p,
)
from evaluation.provenance import require_clean_worktree  # noqa: E402
from evaluation.utility import CHOICE_LABELS, UtilityScore, summarise  # noqa: E402
from run_utility_eval import SPLIT_MANIFEST_PATH, by_subject  # noqa: E402

SCHEMA_VERSION: Final = 1

# Fields that must be identical across arms for a paired comparison to mean
# anything. An arm scored on a different split, prompt or budget is a different
# experiment.
SHARED_SUMMARY_FIELDS: Final = (
    "benchmark",
    "model",
    "split_manifest_sha256",
    "decoding",
    "tools_offered",
    "prompt_sha256",
)

SCOPE_ALL: Final = "all_questions"
SCOPE_READABLE: Final = "readable_in_both"


class UtilityComparisonError(RuntimeError):
    """The arms cannot be compared as paired measurements of the same run."""


class Arm(NamedTuple):
    label: str
    summary: dict[str, Any]
    rows: list[dict[str, Any]]


def load_responses(path: Path) -> list[dict[str, Any]]:
    """One row per question, in the order the evaluation wrote them."""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise UtilityComparisonError(
                    f"{Path(path).name} line {number} is not valid JSON; a "
                    "partially written file cannot stand in for a finished run"
                ) from error
            if row["task_id"] in seen:
                raise UtilityComparisonError(
                    f"{Path(path).name} answers {row['task_id']} more than once"
                )
            seen.add(row["task_id"])
            rows.append(row)
    return rows


def check_rows_reproduce_summary(
    rows: list[dict[str, Any]], summary: dict[str, Any], *, label: str
) -> None:
    """Refuse rows that are not the rows behind this summary.

    Checked field by field against the stored grading, not by re-scoring, so a
    later change to the extractor cannot make old rows disagree with their own
    record.
    """

    for row in rows:
        if row["gold"] not in CHOICE_LABELS:
            raise UtilityComparisonError(f"{label}: {row['task_id']} has no valid gold")
        if row["extracted"] is not None and row["extracted"] not in CHOICE_LABELS:
            raise UtilityComparisonError(
                f"{label}: {row['task_id']} extracted {row['extracted']!r}"
            )
        if bool(row["correct"]) != (row["extracted"] == row["gold"]):
            raise UtilityComparisonError(
                f"{label}: {row['task_id']} is marked "
                f"{'correct' if row['correct'] else 'wrong'} against its own "
                "extracted and gold letters"
            )

    scores = [
        UtilityScore(
            correct=bool(row["correct"]),
            extracted=row["extracted"],
            emitted_tool_call=bool(row["emitted_tool_call"]),
            generated_chars=len(row["completion"]),
            truncated=bool(row["truncated"]),
        )
        for row in rows
    ]
    recomputed = summarise(scores)
    recorded = summary.get("summary")
    if recomputed != recorded:
        differing = sorted(
            key
            for key in set(recomputed) | set(recorded or {})
            if recomputed.get(key) != (recorded or {}).get(key)
        )
        raise UtilityComparisonError(
            f"{label}: the response rows do not reproduce the committed summary "
            f"(differs on {', '.join(differing)})"
        )
    if by_subject(rows) != summary.get("by_subject"):
        raise UtilityComparisonError(
            f"{label}: the response rows do not reproduce the per-subject scores"
        )


def check_arms_align(
    arms: list[Arm], split_manifest: dict[str, Any], *, split_manifest_sha256: str
) -> None:
    """Every arm answered the same frozen questions under the same conditions."""

    frozen = {entry["task_id"]: entry["subject"] for entry in split_manifest["questions"]}
    for arm in arms:
        answered = {row["task_id"]: row["subject"] for row in arm.rows}
        if set(answered) != set(frozen):
            raise UtilityComparisonError(
                f"{arm.label}: answered {len(answered)} questions, "
                f"{len(set(answered) ^ set(frozen))} differ from the frozen split"
            )
        if answered != frozen:
            raise UtilityComparisonError(
                f"{arm.label}: a question's subject differs from the frozen split"
            )
        if arm.summary.get("split_manifest_sha256") != split_manifest_sha256:
            raise UtilityComparisonError(
                f"{arm.label}: was scored against a different split manifest"
            )
        if arm.summary.get("executed") is not True:
            raise UtilityComparisonError(f"{arm.label}: the summary is a plan, not a run")

    reference = arms[0]
    gold = {row["task_id"]: row["gold"] for row in reference.rows}
    for arm in arms[1:]:
        for field in SHARED_SUMMARY_FIELDS:
            if arm.summary.get(field) != reference.summary.get(field):
                raise UtilityComparisonError(
                    f"{arm.label} and {reference.label} differ on {field}"
                )
        for row in arm.rows:
            if row["gold"] != gold[row["task_id"]]:
                raise UtilityComparisonError(
                    f"{arm.label} and {reference.label} disagree on the gold "
                    f"letter for {row['task_id']}"
                )


def compare_pair(
    baseline: Arm, treatment: Arm, *, scope: str = SCOPE_ALL
) -> dict[str, Any]:
    """Paired difference in accuracy, treatment minus baseline."""

    left = {row["task_id"]: row for row in baseline.rows}
    right = {row["task_id"]: row for row in treatment.rows}
    questions = sorted(left)
    if scope == SCOPE_READABLE:
        # Only questions where both arms named a letter. An unreadable answer is
        # scored wrong, and the arms were cut off by the token budget at very
        # different rates, so this separates format from knowledge.
        questions = [
            task
            for task in questions
            if left[task]["extracted"] is not None and right[task]["extracted"] is not None
        ]
    elif scope != SCOPE_ALL:
        raise UtilityComparisonError(f"unknown scope {scope!r}")
    if not questions:
        raise UtilityComparisonError(f"no questions to compare under {scope}")

    differences: list[float] = []
    improved = regressed = both_correct = both_wrong = 0
    for task in questions:
        before = bool(left[task]["correct"])
        after = bool(right[task]["correct"])
        differences.append(float(after) - float(before))
        if after and not before:
            improved += 1
        elif before and not after:
            regressed += 1
        elif before:
            both_correct += 1
        else:
            both_wrong += 1

    n = len(questions)
    baseline_correct = both_correct + regressed
    treatment_correct = both_correct + improved
    rng = random.Random(f"{SEED}:mmlu:{treatment.label}-vs-{baseline.label}:{scope}")
    low, high = bootstrap_interval(differences, rng=rng)
    return {
        "baseline": baseline.label,
        "treatment": treatment.label,
        "scope": scope,
        "questions": n,
        "baseline_correct": baseline_correct,
        "treatment_correct": treatment_correct,
        "baseline_accuracy": baseline_correct / n,
        "treatment_accuracy": treatment_correct / n,
        # Equal to (improved - regressed) / n by construction. The first
        # write-up broke exactly this identity.
        "difference": (treatment_correct - baseline_correct) / n,
        "difference_ci95": [low, high],
        "improved": improved,
        "regressed": regressed,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "p_sign_test_exact": sign_test_p(improved, regressed),
        "p_permutation_two_sided": permutation_p(differences, rng=rng),
    }


def question_rows(arms: list[Arm]) -> list[dict[str, Any]]:
    """Every scored field for every question, side by side across arms."""

    indexed = [{row["task_id"]: row for row in arm.rows} for arm in arms]
    table: list[dict[str, Any]] = []
    for task in sorted(indexed[0]):
        first = indexed[0][task]
        table.append(
            {
                "task_id": task,
                "subject": first["subject"],
                "gold": first["gold"],
                "arms": {
                    arm.label: {
                        "extracted": rows[task]["extracted"],
                        "correct": bool(rows[task]["correct"]),
                        "truncated": bool(rows[task]["truncated"]),
                        "emitted_tool_call": bool(rows[task]["emitted_tool_call"]),
                        "generated_chars": len(rows[task]["completion"]),
                        "completion_sha256": hashlib.sha256(
                            rows[task]["completion"].encode("utf-8")
                        ).hexdigest(),
                    }
                    for arm, rows in zip(arms, indexed)
                },
            }
        )
    return table


def _git(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True, check=False
    )


def _committed(path: Path) -> tuple[str, str]:
    """The file's SHA-256 and the commit holding exactly those bytes."""

    raw = path.read_bytes()
    relative = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    commit = _git("log", "-1", "--format=%H", "--", relative).stdout.decode().strip()
    held = _git("show", f"{commit}:{relative}") if commit else None
    if not commit or held is None or held.returncode != 0 or held.stdout != raw:
        raise UtilityComparisonError(
            f"{path.name} is not byte-identical to a committed copy"
        )
    return hashlib.sha256(raw).hexdigest(), commit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        nargs=3,
        action="append",
        required=True,
        metavar=("LABEL", "SUMMARY", "RESPONSES"),
        help="first arm is the reference for alignment checks",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        action="append",
        required=True,
        metavar=("TREATMENT", "BASELINE"),
    )
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    source_commit = require_clean_worktree(PROJECT_ROOT)

    split_raw = SPLIT_MANIFEST_PATH.read_bytes()
    split_sha256 = hashlib.sha256(split_raw).hexdigest()
    split_manifest = json.loads(split_raw.decode("utf-8"))

    arms: list[Arm] = []
    provenance: dict[str, Any] = {}
    for label, summary_path, responses_path in args.arm:
        summary_file = Path(summary_path)
        responses_file = Path(responses_path)
        summary_sha256, summary_commit = _committed(summary_file)
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        if summary.get("label") != label:
            raise UtilityComparisonError(
                f"{summary_file.name} is labelled {summary.get('label')!r}, not {label!r}"
            )
        rows = load_responses(responses_file)
        check_rows_reproduce_summary(rows, summary, label=label)
        arms.append(Arm(label, summary, rows))
        responses_raw = responses_file.read_bytes()
        provenance[label] = {
            "summary_artifact": summary_file.name,
            "summary_sha256": summary_sha256,
            "summary_recorded_in_commit": summary_commit,
            "model": summary["model"],
            "adapter": summary["adapter"],
            "summary_source_commit": summary["source_commit"],
            "responses_file": responses_file.name,
            "responses_sha256": hashlib.sha256(responses_raw).hexdigest(),
            "responses_rows": len(rows),
            "rows_reproduce_summary": True,
        }

    check_arms_align(arms, split_manifest, split_manifest_sha256=split_sha256)

    by_label = {arm.label: arm for arm in arms}
    comparisons: list[dict[str, Any]] = []
    for treatment, baseline in args.compare:
        if treatment not in by_label or baseline not in by_label:
            raise UtilityComparisonError(f"unknown arm in --compare {treatment} {baseline}")
        for scope in (SCOPE_ALL, SCOPE_READABLE):
            comparisons.append(
                compare_pair(by_label[baseline], by_label[treatment], scope=scope)
            )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "kind": "utility_paired_comparison",
        "benchmark": "mmlu",
        "split_manifest_sha256": split_sha256,
        "method": (
            "Paired per-question accuracy difference, treatment minus baseline, "
            "from the graded rows each evaluation wrote. Question-level bootstrap "
            "interval on the mean difference is primary; the paired sign-flip "
            "permutation p-value and the exact sign test on discordant pairs "
            "(McNemar's exact test) are secondary. Every response file was "
            "checked to reproduce its committed summary exactly before use."
        ),
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "permutation_resamples": PERMUTATION_RESAMPLES,
        "confidence": CONFIDENCE,
        "arms": provenance,
        "comparisons": comparisons,
        "rows": question_rows(arms),
        "source_commit": source_commit,
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
            "%s vs %s [%s] n=%d  %+.4f  CI95 [%+.4f, %+.4f]  +%d/-%d  sign p=%.3f"
            % (
                entry["treatment"],
                entry["baseline"],
                entry["scope"],
                entry["questions"],
                entry["difference"],
                entry["difference_ci95"][0],
                entry["difference_ci95"][1],
                entry["improved"],
                entry["regressed"],
                entry["p_sign_test_exact"] or 1.0,
            )
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UtilityComparisonError as error:
        print(f"refused: {error}", file=sys.stderr)
        raise SystemExit(1)
