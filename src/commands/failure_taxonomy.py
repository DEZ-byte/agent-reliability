"""Classify every failed GSM8K test attempt by how it failed.

The GSM8K artifacts report how often a model answers correctly. They do not
say why the other attempts fail, and until now those counts lived only in the
git-ignored episode logs. This writes them into one artifact.

Each failed episode falls into exactly one class:

- **No call, or an unparsed one.** The completion produced no tool event. This
  is split into replies with no `<tool_call>` tag at all, and tagged replies
  that do not parse into a valid call.
- **A call that failed to run.** A call was parsed, but none of its events was
  a calculator call that returned a number: a schema error, an unknown tool, or
  an expression the sandbox rejected.
- **A wrong value.** A calculator call ran and returned the wrong number.

At R1, an attempt that gives no answer gets one retry that sees the failure.
An R1 episode is classed by its last decision. The retries are also counted on
their own: how many fired, how many ended correct, how many ended with a wrong
number, and how many still gave no answer.

Only logged fields are read, plus a re-parse of each completion with the
evaluation's own parser and dialect rule. Before counting anything, this checks
three things. The re-parse must give back exactly the logged tool calls. The
stored rows must reproduce the committed summary's figures for each rung. And
each R1 first completion must equal the R0 completion for the same task and
run, which the shared seed guarantees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.dialects import OPEN_TAG, normalize_tool_dialect  # noqa: E402
from agent.parser import parse_tool_calls  # noqa: E402
from env.models import ToolEvent  # noqa: E402
from env.gsm8k import is_answering_event  # noqa: E402
from evaluation.metrics import compute_pass_metrics  # noqa: E402
from evaluation.provenance import (  # noqa: E402
    DirtyWorktreeError,
    recorded_path,
    require_clean_worktree,
)

SCHEMA_VERSION: Final = 1
RUNGS: Final = ("R0", "R1")
# The value of a result's tool_dialect for a model whose template asks for bare
# JSON. The evaluation normalized only those completions before parsing.
BARE_JSON: Final = "bare_json"

CORRECT: Final = "correct"
WRONG_VALUE: Final = "wrong_value"
CALL_FAILED: Final = "call_failed_to_run"
NO_CALL: Final = "no_call_or_unparsed"
NO_TAG: Final = "no_tag"
MALFORMED: Final = "malformed"
RETRY_FIELDS: Final = ("fired", "rescued", "answered_wrong", "still_no_answer")

# The per-rung figures of the committed summary that the stored rows must reproduce.
SUMMARY_FIELDS: Final = (
    "episodes",
    "tasks",
    "pass^1",
    "pass^4",
    "pass@4",
    "model_decisions",
    "no_arithmetic_episodes",
)

METHOD: Final = {
    CORRECT: (
        "The logged correct flag is true: a calculator call ran and returned the gold answer."
    ),
    WRONG_VALUE: (
        "The episode ended answered but not correct: a calculator call ran and "
        "returned the wrong number."
    ),
    CALL_FAILED: (
        "The last decision produced at least one tool event, but none was a "
        "calculator call that ran and returned a number."
    ),
    NO_CALL: "The last decision produced no tool event; this is no_tag plus malformed.",
    NO_TAG: (
        "No tool event, and the text the parser read has no <tool_call> opening tag. "
        "For a bare JSON model this includes untagged JSON that is not a valid call."
    ),
    MALFORMED: (
        "No tool event, and the text the parser read has a <tool_call> tag that "
        "does not parse into a valid call."
    ),
    "r1_final_outcome": (
        "An R1 episode is classed by its last decision, so a retry that ran is judged on the retry."
    ),
    "fired": "An R1 episode with two decisions: the first gave no answer and the retry ran.",
    "rescued": "A fired retry that ended correct.",
    "answered_wrong": "A fired retry that ended with a calculator value that is wrong.",
    "still_no_answer": "A fired retry that ended with no calculator value.",
    "checks": (
        "Each completion is re-parsed with agent.parser.parse_tool_calls, after "
        "agent.dialects.normalise_tool_dialect only when the summary records "
        "tool_dialect bare_json, and the re-parse must give back the logged calls "
        "exactly. The rows must reproduce the summary's figures for each rung, and "
        "every R1 first completion must equal the R0 completion for the same task and run."
    ),
}


class FailureAnalysisError(RuntimeError):
    """The stored episodes are not the ones behind the committed summary."""


# Refusals exit 1 with a one-line message instead of a traceback.
REFUSALS: Final = (FailureAnalysisError, DirtyWorktreeError)


def _where(row: dict[str, Any]) -> str:
    return f"{row['rung']} {row['task_id']} run {row['run_index']}"


def decisions(row: dict[str, Any], *, normalize: bool) -> list[tuple[str, list[ToolEvent]]]:
    """Each decision's parsed text, with the tool events it produced.

    The log keeps one flat list of events per episode. Every parsed call makes
    exactly one event, in order, so counting the calls in each completion splits
    the list. The re-parse must give back the logged calls exactly, or the split
    and every class built on it would be a guess.
    """

    events = [ToolEvent.model_validate_json(json.dumps(event)) for event in row["tool_events"]]
    split: list[tuple[str, list[ToolEvent]]] = []
    cursor = 0
    for completion in row["completions"]:
        text = normalize_tool_dialect(completion) if normalize else completion
        calls = parse_tool_calls(text).calls
        produced = events[cursor : cursor + len(calls)]
        if [event.call for event in produced] != calls:
            raise FailureAnalysisError(
                f"{_where(row)}: the re-parse does not give back the logged tool calls"
            )
        split.append((text, produced))
        cursor += len(calls)
    if cursor != len(events):
        raise FailureAnalysisError(
            f"{_where(row)}: {len(events)} logged events, but the re-parse finds {cursor} calls"
        )
    return split


def classify(row: dict[str, Any], *, normalize: bool) -> tuple[str, bool]:
    """The episode's class, judged on its last decision, and whether it answered."""

    text, events = decisions(row, normalize=normalize)[-1]
    answered = any(is_answering_event(event) for event in events)
    if answered != (row["terminal_reason"] == "answered") or (row["correct"] and not answered):
        raise FailureAnalysisError(
            f"{_where(row)}: the logged events do not match terminal reason "
            f"{row['terminal_reason']!r} and correct={row['correct']!r}"
        )
    if row["correct"]:
        return CORRECT, True
    if answered:
        return WRONG_VALUE, True
    if events:
        return CALL_FAILED, False
    if OPEN_TAG in text:
        return MALFORMED, False
    return NO_TAG, False


def check_rung(rows: list[dict[str, Any]], stored: dict[str, Any], rung: str) -> None:
    """Refuse rows that do not reproduce the summary's figures for one rung."""

    by_task: dict[str, dict[int, int]] = {}
    for row in rows:
        runs = by_task.setdefault(row["task_id"], {})
        if row["run_index"] in runs:
            raise FailureAnalysisError(f"{_where(row)} appears twice")
        runs[row["run_index"]] = 1 if row["correct"] else 0
    successes = [[runs[index] for index in sorted(runs)] for runs in by_task.values()]
    try:
        first, fourth = compute_pass_metrics(successes, 1), compute_pass_metrics(successes, 4)
    except ValueError as error:
        raise FailureAnalysisError(f"{rung}: {error}") from error

    recomputed = {
        "episodes": len(rows),
        "tasks": len(successes),
        "pass^1": first.pass_power_k,
        "pass^4": fourth.pass_power_k,
        "pass@4": fourth.pass_at_k,
        "model_decisions": sum(row["counters"]["policy_model_decision_count"] for row in rows),
        "no_arithmetic_episodes": sum(1 for row in rows if row["answered_without_arithmetic"]),
    }
    # The summary keeps the pass metrics under "metrics" and the counts beside it.
    metrics = stored.get("metrics", {})
    for field in SUMMARY_FIELDS:
        expected = metrics.get(field) if field.startswith("pass") else stored.get(field)
        if recomputed[field] != expected:
            raise FailureAnalysisError(
                f"{rung}: stored rows give {field}={recomputed[field]!r}, the summary {expected!r}"
            )


def count(rows: list[dict[str, Any]], *, normalize: bool, retry: bool) -> dict[str, Any]:
    classes: Counter[str] = Counter()
    retries: Counter[str] = Counter()
    for row in rows:
        name, answered = classify(row, normalize=normalize)
        classes[name] += 1
        if len(row["completions"]) > 1:
            retries["fired"] += 1
            if row["correct"]:
                retries["rescued"] += 1
            elif answered:
                retries["answered_wrong"] += 1
            else:
                retries["still_no_answer"] += 1

    failures = {
        NO_CALL: classes[NO_TAG] + classes[MALFORMED],
        CALL_FAILED: classes[CALL_FAILED],
        WRONG_VALUE: classes[WRONG_VALUE],
    }
    counted: dict[str, Any] = {
        "episodes": len(rows),
        "correct": classes[CORRECT],
        "failures": {**failures, "total": sum(failures.values())},
        "unparsed_detail": {NO_TAG: classes[NO_TAG], MALFORMED: classes[MALFORMED]},
    }
    if retry:
        counted["retry"] = {name: retries[name] for name in RETRY_FIELDS}
    return counted


def analyze_arm(
    rows: list[dict[str, Any]], summary: dict[str, Any], candidate: str
) -> dict[str, Any]:
    """Both rungs' failure classes for one candidate, after every check passes."""

    if summary.get("executed") is not True:
        raise FailureAnalysisError("the summary is not an executed run")
    results = [
        entry for entry in summary.get("results", []) if entry["candidate"]["id"] == candidate
    ]
    if len(results) != 1 or "rungs" not in results[0]:
        raise FailureAnalysisError(f"the summary has no measured result for {candidate}")
    result = results[0]
    normalize = result.get("tool_dialect") == BARE_JSON

    # A log can hold more than one model, all sharing task ids, so rows are
    # picked by the candidate they record.
    mine = [row for row in rows if row.get("candidate") == candidate]
    if not mine:
        raise FailureAnalysisError(f"the episode log holds no rows for {candidate}")
    by_rung = {rung: [row for row in mine if row["rung"] == rung] for rung in RUNGS}
    for rung in RUNGS:
        if rung not in result["rungs"]:
            raise FailureAnalysisError(f"the summary has no {rung} result for {candidate}")
        check_rung(by_rung[rung], result["rungs"][rung], rung)

    first = {(row["task_id"], row["run_index"]): row["completions"][0] for row in by_rung["R0"]}
    for row in by_rung["R1"]:
        if first.get((row["task_id"], row["run_index"])) != row["completions"][0]:
            raise FailureAnalysisError(
                f"{_where(row)}: the first completion differs from the R0 completion "
                "for the same task and run"
            )

    return {rung: count(by_rung[rung], normalize=normalize, retry=rung == "R1") for rung in RUNGS}


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        nargs=4,
        action="append",
        required=True,
        metavar=("LABEL", "SUMMARY", "EPISODES", "CANDIDATE"),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    source_commit = require_clean_worktree(PROJECT_ROOT)
    labels = [arm[0] for arm in args.arm]
    repeated = sorted({label for label in labels if labels.count(label) > 1})
    if repeated:
        raise FailureAnalysisError("each arm needs its own label; repeated: " + ", ".join(repeated))

    arms: dict[str, Any] = {}
    for label, summary_path, episodes_path, candidate in args.arm:
        summary_file, episodes_file = Path(summary_path), Path(episodes_path)
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        try:
            rungs = analyze_arm(read_rows(episodes_file), summary, candidate)
        except FailureAnalysisError as error:
            raise FailureAnalysisError(f"{label}: {error}") from error
        arms[label] = {
            "summary": recorded_path(summary_file, PROJECT_ROOT),
            "summary_sha256": hashlib.sha256(summary_file.read_bytes()).hexdigest(),
            "episodes": recorded_path(episodes_file, PROJECT_ROOT),
            "episodes_sha256": hashlib.sha256(episodes_file.read_bytes()).hexdigest(),
            "candidate": candidate,
            "rungs": rungs,
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "kind": "phase_a_failure_modes",
        "method": METHOD,
        "arms": arms,
        "source_commit": source_commit,
        "platform": {"python": platform.python_version(), "system": platform.system()},
    }
    path = Path(args.output)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes((json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    os.replace(temporary, path)
    for label, arm in arms.items():
        r0, r1 = arm["rungs"]["R0"], arm["rungs"]["R1"]
        print(
            f"{label}: R0 failures {r0['failures']}  unparsed {r0['unparsed_detail']}  "
            f"R1 failures {r1['failures']}  retry {r1['retry']}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except REFUSALS as error:
        print(f"refused: {error}", file=sys.stderr)
        raise SystemExit(1)
