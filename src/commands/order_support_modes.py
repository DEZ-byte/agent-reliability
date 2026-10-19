"""Pair each checkpoint's single-turn and multi-turn order-support runs task by task.

Both modes open with the same prompt on the same 150 test tasks, four sampled
runs per task, from the same weights and sampling settings. The intended
difference is the protocol: single-turn runs every tool call after the model
has written its whole reply, and multi-turn feeds each tool result back
before the next reply. A multi-turn episode stops when a reply makes no tool
call, after a reply whose `update_order` call was dispatched, or at the turn
cap the multi-turn summary records. The sampling path differs too:
multi-turn draws one sequence per `generate` call instead of four copies from
one call, so its random draws are not the single-turn ones, and on later
turns the chat template re-renders earlier replies without their empty think
block. The modes also ran at different source commits, so the replay step
below checks that the code grading multi-turn still gives every stored
single-turn grade. For base, SFT and GRPO alike, the per-task difference is
multi-turn minus single-turn under the whole closed-loop protocol, stopping
rule and sampling path included, not the effect of tool results alone.

For each checkpoint this command:

- checks that the two summaries name the same model, adapter, weights,
  split, prompt and sampling settings, that each episode log reproduces its
  summary, and that both logs cover every split task with runs 0 to 3;
- replays every stored single-turn reply as a one-turn episode through the
  multi-turn grader, which must reproduce the stored correctness, reward,
  violation flag, call count, write attempt and tools called, so the
  single-turn rows gain the same `clean_correct` flag and write class the
  multi-turn rows store;
- reports multi-turn minus single-turn for pass^1 and pass^4 on `correct` and
  pass^1 on `clean_correct`, with the bootstrap interval, permutation test and
  exact sign test from `compare_models`, called unchanged;
- reports per-mode behavior: completion on completable requests, refusal on
  refusal-required requests, writes by class, lookup rate, and for
  multi-turn the mean turns used, how many first replies already called
  `update_order`, and how many episodes that held off past the first reply
  then wrote after a failed `authenticate`.

Inputs are named relative to the results folder and recorded with their
SHA-256. The episode logs are not committed, so their hashes are the only
record of which bytes were read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.gates import GateEngine  # noqa: E402
from env.order_support_tasks import OrderSupportSplitError, load_split  # noqa: E402
from evaluation.multiturn import evaluate_episode, run_episode, summarize_multi_turn  # noqa: E402
from evaluation.order_support_modes import (  # noqa: E402
    ModeComparisonError,
    as_arm,
    check_same_setup,
    group_by_task,
    mode_metrics,
)
from evaluation.provenance import DirtyWorktreeError, require_clean_worktree  # noqa: E402
from commands import compare_models  # noqa: E402
from commands.evaluate_order_support import (  # noqa: E402
    GATES_PATH,
    MULTI_TURN,
    MULTI_TURN_KIND,
    RUNS_PER_TASK,
    SPLIT_MANIFEST_PATH,
    first_turn_messages,
    summarize,
)

SCHEMA_VERSION: Final = 1
KIND: Final = "order_support_mode_comparison"
DEFAULT_RESULTS_DIR: Final = PROJECT_ROOT / "results"

# label, single-turn summary, single-turn log, multi-turn summary, multi-turn log
DEFAULT_ARMS: Final = (
    (
        "base",
        "phase_b-base-10f9d97.json",
        "episodes-phase_b-base-10f9d97.jsonl",
        "order-support-multiturn-base-3cacc39.json",
        "episodes-order-support-multiturn-base-3cacc39.jsonl",
    ),
    (
        "sft",
        "phase_b-sft-aa59330.json",
        "episodes-phase_b-sft-aa59330.jsonl",
        "order-support-multiturn-sft-3cacc39.json",
        "episodes-order-support-multiturn-sft-3cacc39.jsonl",
    ),
    (
        "grpo",
        "phase_b-grpo-42e347c.json",
        "episodes-phase_b-grpo-42e347c.jsonl",
        "order-support-multiturn-grpo-3cacc39.json",
        "episodes-order-support-multiturn-grpo-3cacc39.jsonl",
    ),
)

# (metric, k): pass^1 and pass^4 on `correct`, and strict pass^1 on `clean_correct`.
COMPARISONS: Final = (("correct", 1), ("correct", RUNS_PER_TASK), ("clean_correct", 1))

# The summary fields each episode log must reproduce.
SUMMARY_FIELDS: Final = ("tasks", "episodes", "runs_per_task", "metrics", "by_intent", "diagnostics")
# Fields of a stored single-turn row that the replay must reproduce.
REPLAYED_FIELDS: Final = (
    "correct",
    "reward",
    "gate_violation",
    "executed_calls",
    "attempted_write",
    "tools_called",
)

METHOD: Final = (
    "Each checkpoint's single-turn and multi-turn runs are paired by task: same "
    "weights, split, prompt and sampling settings, every split task with runs 0 "
    "to 3 in both modes, and each episode log reproducing its summary. Differences "
    "are multi-turn minus single-turn, per task, using compare_models.compare "
    "unchanged: task-level bootstrap interval ({boot} replicates), add-one "
    "corrected sign-flip permutation p-value ({perm} resamples) and the exact "
    "paired sign test, with each comparison's random stream seeded from the "
    "string '{seed}:<stream>:<k>'. clean_correct is the reward's accuracy term: "
    "the order ends in the requested state, no gate failed, and at least one call "
    "ran. Single-turn "
    "logs do not store it, so every stored single-turn reply is replayed as a "
    "one-turn episode through evaluation.multiturn.run_episode and "
    "evaluate_episode, the code that sets it for multi-turn, and the replay must "
    "reproduce the stored correct, reward, gate_violation, executed_calls, "
    "attempted_write and tools_called exactly. Write classes and the lookup flag "
    "come from the same replay. Completion rate is the share of completable "
    "requests that end with the requested address. Refusal rate is the share of "
    "refusal-required requests where no update_order call ran. "
    "first_reply_update_order counts multi-turn episodes whose first reply, "
    "written before any tool result was shown, held an update_order call. "
    "later_write_after_failed_authenticate counts, per intent, the multi-turn "
    "episodes whose first reply held no update_order call (waited) and, of "
    "those, the ones where an update_order call ran in a later reply after an "
    "earlier authenticate call came back without authenticated: true."
)


# Refusals exit 1 with a one-line message instead of a traceback.
REFUSALS: Final = (
    ModeComparisonError,
    OrderSupportSplitError,
    compare_models.ComparisonError,
    DirtyWorktreeError,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve_input(results_dir: Path, name: str) -> tuple[Path, str]:
    """An input file inside the results folder, and its name relative to it."""

    if Path(name).is_absolute():
        raise ModeComparisonError(f"name inputs relative to the results folder, not {name!r}")
    path = (results_dir / name).resolve()
    if not path.is_relative_to(results_dir.resolve()):
        raise ModeComparisonError(f"{name} is outside the results folder")
    if not path.is_file():
        raise ModeComparisonError(f"{name} does not exist in the results folder")
    return path, path.relative_to(results_dir.resolve()).as_posix()


def load_tasks(summary: dict[str, Any]) -> list:
    """The split the summary ran on, after checking it is the same manifest."""

    if summary["split_manifest_sha256"] != _sha256(SPLIT_MANIFEST_PATH):
        raise ModeComparisonError("the summaries ran on a different split manifest")
    return load_split(SPLIT_MANIFEST_PATH, summary["split"])


def check_reproduces(rows: list[dict[str, Any]], tasks: list, summary: dict[str, Any], name: str) -> None:
    """The log must be the one behind its summary, field for field."""

    recomputed = summarize(rows, tasks)
    for field in SUMMARY_FIELDS:
        if recomputed[field] != summary.get(field):
            raise ModeComparisonError(f"{name} does not reproduce its summary's {field}")
    if summary.get("mode") == MULTI_TURN:
        stored = summary.get("multi_turn") or {}
        for field, value in summarize_multi_turn(rows, tasks, RUNS_PER_TASK).items():
            if stored.get(field) != value:
                raise ModeComparisonError(
                    f"{name} does not reproduce its summary's multi_turn.{field}"
                )


def replay_single_turn(
    row: dict[str, Any], task, *, engine: GateEngine, normalize: bool
) -> dict[str, Any]:
    """A stored single-turn row, plus what the multi-turn grader says about it."""

    messages = first_turn_messages(task)
    episode = run_episode(
        task,
        lambda _messages: row["completion"],
        system_prompt=messages[0]["content"],
        user_message=messages[1]["content"],
        gate_engine=engine,
        max_turns=1,
        normalize_dialect=normalize,
    )
    graded = evaluate_episode(episode, task, engine)
    for field in REPLAYED_FIELDS:
        if graded[field] != row[field]:
            raise ModeComparisonError(
                f"{row['task_id']} run {row['run_index']}: replay gives "
                f"{field}={graded[field]!r}, stored {row[field]!r}"
            )
    return {
        **row,
        "clean_correct": graded["clean_correct"],
        "write_class": graded["write_class"],
        "called_lookup": graded["called_lookup"],
        "failed_gates": graded["failed_gates"],
    }


def paired_comparison(
    single: dict[str, dict[str, Any]],
    multi: dict[str, dict[str, Any]],
    *,
    label: str,
    metric: str,
    k: int,
) -> dict[str, Any]:
    """compare_models.compare with single-turn as baseline, renamed for this artifact."""

    raw = compare_models.compare(single, multi, k=k, rung=f"order-support-modes:{label}:{metric}")
    entry: dict[str, Any] = {
        "metric": metric,
        "k": k,
        "difference_is": "multi_turn minus single_turn",
        "rng_stream": raw.pop("rung"),
    }
    for key, value in raw.items():
        if key == "k":
            continue
        entry[key.replace("baseline_", "single_turn_").replace("treatment_", "multi_turn_")] = value
    return entry


def analyze_model(
    label: str, single_summary: dict[str, Any], single_rows: list[dict[str, Any]],
    multi_summary: dict[str, Any], multi_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if single_summary.get("mode") == MULTI_TURN or single_summary.get("kind") == MULTI_TURN_KIND:
        raise ModeComparisonError(f"{label}: the single-turn summary is a multi-turn run")
    if multi_summary.get("mode") != MULTI_TURN or multi_summary.get("kind") != MULTI_TURN_KIND:
        raise ModeComparisonError(f"{label}: the multi-turn summary is not a multi-turn run")
    if single_summary.get("label") != label:
        raise ModeComparisonError(f"{label}: the summaries are labeled {single_summary.get('label')!r}")
    check_same_setup(single_summary, multi_summary)
    if single_summary["rollout"]["runs_per_task"] != RUNS_PER_TASK:
        raise ModeComparisonError(f"{label}: expected {RUNS_PER_TASK} runs per task")

    tasks = load_tasks(single_summary)
    by_id = {task.task_id: task for task in tasks}
    intents = {task.task_id: task.intent for task in tasks}
    # Grouping both logs against the same split and run count is the pairing
    # check: each mode must hold every split task with runs 0 to 3, once each.
    single_groups = group_by_task(single_rows, intents, RUNS_PER_TASK)
    multi_groups = group_by_task(multi_rows, intents, RUNS_PER_TASK)
    check_reproduces(single_rows, tasks, single_summary, f"{label} single-turn log")
    check_reproduces(multi_rows, tasks, multi_summary, f"{label} multi-turn log")

    engine = GateEngine.from_file(GATES_PATH)
    normalize = bool(single_summary["normalise_dialect"])
    single_groups = {
        task_id: [
            replay_single_turn(row, by_id[task_id], engine=engine, normalize=normalize)
            for row in runs
        ]
        for task_id, runs in single_groups.items()
    }

    comparisons = []
    for metric, k in COMPARISONS:
        entry = paired_comparison(
            as_arm(single_groups, metric, "completion"),
            as_arm(multi_groups, metric, "completions"),
            label=label,
            metric=metric,
            k=k,
        )
        if entry["tasks_compared"] != len(tasks):
            raise ModeComparisonError(f"{label}: only {entry['tasks_compared']} tasks were paired")
        comparisons.append(entry)

    single_flat = [row for runs in single_groups.values() for row in runs]
    multi_flat = [row for runs in multi_groups.values() for row in runs]
    return {
        "model": single_summary["model"],
        "adapter": single_summary["adapter"],
        "weights_sha256": single_summary["weights_sha256"],
        "tasks": len(tasks),
        "runs_per_task": RUNS_PER_TASK,
        "max_turns": multi_summary.get("max_turns"),
        "comparisons": comparisons,
        "single_turn": mode_metrics(single_flat, multi_turn=False),
        "multi_turn": mode_metrics(multi_flat, multi_turn=True),
        "sampling_degeneracy": {
            "single_turn": compare_models.degeneracy(as_arm(single_groups, "correct", "completion")),
            "multi_turn": compare_models.degeneracy(as_arm(multi_groups, "correct", "completions")),
        },
    }


def _describe(path: Path, name: str, summary: dict[str, Any] | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"file": name, "sha256": _sha256(path)}
    if summary is not None:
        entry["source_commit"] = summary.get("source_commit")
    return entry


def _table(models: dict[str, Any]) -> list[str]:
    lines = []
    for label, model in models.items():
        for entry in model["comparisons"]:
            low, high = entry["difference_ci95"]
            sign = entry["p_sign_test_exact"]
            lines.append(
                "%-5s %-13s pass^%d  single %.4f  multi %.4f  diff %+.4f  CI95 [%+.4f, %+.4f]  "
                "perm p=%.5f  sign p=%s  up/down %d/%d of %d"
                % (
                    label,
                    entry["metric"],
                    entry["k"],
                    entry["single_turn_pass_power_k"],
                    entry["multi_turn_pass_power_k"],
                    entry["difference"],
                    low,
                    high,
                    entry["p_permutation_two_sided"],
                    "n/a" if sign is None else "%.3g" % sign,
                    entry["tasks_improved"],
                    entry["tasks_regressed"],
                    entry["tasks_compared"],
                )
            )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--arm",
        nargs=5,
        action="append",
        default=None,
        metavar=("LABEL", "SINGLE_SUMMARY", "SINGLE_EPISODES", "MULTI_SUMMARY", "MULTI_EPISODES"),
        help="one checkpoint's four inputs, named relative to --results-dir "
        "(default: base, sft and grpo)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="the folder the inputs are read from (default: results/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="where to write the artifact (default: "
        "<results-dir>/order-support-modes-<commit7>.json)",
    )
    args = parser.parse_args(argv)
    arms = args.arm or [list(arm) for arm in DEFAULT_ARMS]
    labels = [arm[0] for arm in arms]
    if len(set(labels)) != len(labels):
        parser.error("each --arm needs its own label")

    source_commit = require_clean_worktree(PROJECT_ROOT)
    results_dir: Path = args.results_dir

    inputs: dict[str, Any] = {}
    models: dict[str, Any] = {}
    for label, single_name, single_log, multi_name, multi_log in arms:
        single_path, single_rel = resolve_input(results_dir, single_name)
        single_log_path, single_log_rel = resolve_input(results_dir, single_log)
        multi_path, multi_rel = resolve_input(results_dir, multi_name)
        multi_log_path, multi_log_rel = resolve_input(results_dir, multi_log)
        single_summary = json.loads(single_path.read_text(encoding="utf-8"))
        multi_summary = json.loads(multi_path.read_text(encoding="utf-8"))
        inputs[label] = {
            "single_turn": {
                "summary": _describe(single_path, single_rel, single_summary),
                "episodes": _describe(single_log_path, single_log_rel),
            },
            "multi_turn": {
                "summary": _describe(multi_path, multi_rel, multi_summary),
                "episodes": _describe(multi_log_path, multi_log_rel),
            },
        }
        models[label] = analyze_model(
            label,
            single_summary,
            _read_rows(single_log_path),
            multi_summary,
            _read_rows(multi_log_path),
        )

    reported = sum(len(model["comparisons"]) for model in models.values())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "kind": KIND,
        "method": METHOD.format(
            boot=compare_models.BOOTSTRAP_REPLICATES,
            perm=compare_models.PERMUTATION_RESAMPLES,
            seed=compare_models.SEED,
        ),
        "seed": compare_models.SEED,
        "bootstrap_replicates": compare_models.BOOTSTRAP_REPLICATES,
        "permutation_resamples": compare_models.PERMUTATION_RESAMPLES,
        "multiplicity": {
            "comparisons_reported": reported,
            "bonferroni_alpha": 0.05 / reported,
            "note": (
                "No multiplicity policy was pre-registered. The Bonferroni "
                "threshold is reported so a reader can apply it; it is not used "
                "to filter what is shown."
            ),
        },
        "inputs": inputs,
        "models": models,
        "source_commit": source_commit,
        "platform": {"python": platform.python_version(), "system": platform.system()},
    }

    path = args.output or results_dir / f"order-support-modes-{source_commit[:7]}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes((json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    os.replace(temporary, path)
    for line in _table(models):
        print(line)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except REFUSALS as error:
        print(f"refused: {error}", file=sys.stderr)
        raise SystemExit(1)
