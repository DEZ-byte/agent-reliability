"""Pairing checks and per-mode counts for the single-turn vs. multi-turn comparison.

Pure functions over stored summaries and episode rows. Nothing here loads a
model, reads a file or draws a random number, so every check runs on CPU with
hand-built rows. The statistics live in `commands/compare_models.py`, and the
command in `commands/order_support_modes.py` calls them.

An episode row here is one dict per (task, run) with at least `task_id`,
`intent`, `run_index`, `correct`, `clean_correct`, `write_class`,
`called_lookup` and `attempted_write`. Multi-turn rows store all of these.
Single-turn rows store the first, and the command adds the rest by replaying
each stored reply through the multi-turn grader.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Final, Iterable, Mapping, Sequence

from env.order_support import (
    AUTHENTICATE_TOOL_NAME,
    INTENT_FULFIL,
    INTENT_REFUSE,
    UPDATE_ORDER_TOOL_NAME,
)
from evaluation.multiturn import WRITE_CLASSES

INTENTS: Final = (INTENT_FULFIL, INTENT_REFUSE)

# What must match between a single-turn and a multi-turn summary before their
# episodes can be paired: the same weights, the same tasks, the same prompt and
# the same sampling settings. Dotted names reach into nested fields.
SETUP_FIELDS: Final = (
    "label",
    "model.id",
    "model.revision",
    "adapter",
    "weights_sha256",
    "split",
    "split_manifest_sha256",
    "gate_mode",
    "normalise_dialect",
    "prompt_sha256.system",
    "prompt_sha256.user",
    "rollout.runs_per_task",
    "rollout.temperature",
    "rollout.top_p",
    "rollout.max_new_tokens",
    "rollout.seed_base",
)

_MISSING: Final = object()


class ModeComparisonError(RuntimeError):
    """The two modes cannot be paired task by task."""


def _field(summary: Mapping[str, Any], dotted: str) -> Any:
    value: Any = summary
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return _MISSING
        value = value[part]
    return value


def check_same_setup(single: Mapping[str, Any], multi: Mapping[str, Any]) -> None:
    """Refuse two summaries that did not run the same checkpoint the same way."""

    for name, summary in (("single-turn", single), ("multi-turn", multi)):
        if summary.get("executed") is not True:
            raise ModeComparisonError(f"the {name} summary is not an executed run")
    differing = [
        name
        for name in SETUP_FIELDS
        if _field(single, name) is _MISSING
        or _field(multi, name) is _MISSING
        or _field(single, name) != _field(multi, name)
    ]
    if differing:
        raise ModeComparisonError(
            "single-turn and multi-turn summaries differ or lack: " + ", ".join(differing)
        )


def group_by_task(
    rows: Iterable[Mapping[str, Any]],
    intents: Mapping[str, str],
    runs_per_task: int,
) -> dict[str, list[dict[str, Any]]]:
    """Rows keyed by task id, each task's runs in run order.

    `intents` maps every task id in the split to its intent. Every task must
    appear with run indices 0 to runs_per_task - 1, once each, and with the
    split's intent. Anything else is refused, because a missing or repeated
    run would change pass^k without any sign in the totals.
    """

    groups: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        task_id = row["task_id"]
        if task_id not in intents:
            raise ModeComparisonError(f"{task_id} is not in the split")
        if row["intent"] != intents[task_id]:
            raise ModeComparisonError(
                f"{task_id} is logged as {row['intent']!r}, the split says {intents[task_id]!r}"
            )
        runs = groups.setdefault(task_id, {})
        index = row["run_index"]
        if index in runs:
            raise ModeComparisonError(f"{task_id} run {index} appears twice")
        runs[index] = dict(row)

    missing = sorted(set(intents) - set(groups))
    if missing:
        raise ModeComparisonError(
            f"{len(missing)} split task(s) have no episodes, first {missing[0]}"
        )
    expected = list(range(runs_per_task))
    for task_id, runs in groups.items():
        if sorted(runs) != expected:
            raise ModeComparisonError(
                f"{task_id} has runs {sorted(runs)}, expected {expected}"
            )
    return {task_id: [groups[task_id][i] for i in expected] for task_id in intents}


def as_arm(
    groups: Mapping[str, Sequence[Mapping[str, Any]]], field: str, completion_key: str
) -> dict[str, dict[str, Any]]:
    """The per-task shape `compare_models.compare` and `degeneracy` read.

    `completion_key` names what one run produced: `completion` for a
    single-turn row, `completions` (every reply) for a multi-turn row. Two runs
    count as identical only if every reply matches.
    """

    arm: dict[str, dict[str, Any]] = {}
    for task_id, runs in groups.items():
        produced = {
            tuple(run[completion_key]) if isinstance(run[completion_key], list) else run[completion_key]
            for run in runs
        }
        arm[task_id] = {
            "successes": [bool(run[field]) for run in runs],
            "distinct_completions": len(produced),
        }
    return arm


def first_reply_update_orders(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Multi-turn episodes whose first reply already held an `update_order` call.

    Such a reply was written before the model saw any tool result, so for that
    episode the closed loop changed nothing about the write. Counted from each
    row's `call_trace`, overall and per intent, with the write class each of
    those episodes ended in. `dispatched` counts the ones the registry ran.
    """

    by_intent = {
        intent: {"episodes": 0, "dispatched": 0, "write_classes": Counter()}
        for intent in INTENTS
    }
    for row in rows:
        trace = row["call_trace"]
        first = [
            call for call in trace if call["turn"] == 1 and call["name"] == UPDATE_ORDER_TOOL_NAME
        ]
        if not first:
            continue
        entry = by_intent[row["intent"]]
        entry["episodes"] += 1
        entry["dispatched"] += any(call["dispatched"] for call in first)
        entry["write_classes"][row["write_class"]] += 1
    return {
        "episodes": sum(entry["episodes"] for entry in by_intent.values()),
        "dispatched": sum(entry["dispatched"] for entry in by_intent.values()),
        "by_intent": {
            intent: {
                "episodes": entry["episodes"],
                "dispatched": entry["dispatched"],
                "write_classes": {name: entry["write_classes"][name] for name in WRITE_CLASSES},
            }
            for intent, entry in by_intent.items()
        },
    }


def _authenticated(call: Mapping[str, Any]) -> bool:
    output = call.get("output")
    return isinstance(output, Mapping) and output.get("authenticated") is True


def later_writes_after_failed_authenticate(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Multi-turn episodes that waited past the first reply, then wrote anyway.

    Per intent, `waited` counts episodes whose first reply held no
    `update_order` call. `wrote_after_failed_authenticate` counts those in
    which an `update_order` call ran in a later reply, after an earlier reply's
    `authenticate` call came back without `authenticated: true`. The model had
    seen that result when it wrote.
    """

    by_intent = {intent: {"waited": 0, "wrote_after_failed_authenticate": 0} for intent in INTENTS}
    for row in rows:
        trace = row["call_trace"]
        if any(call["turn"] == 1 and call["name"] == UPDATE_ORDER_TOOL_NAME for call in trace):
            continue
        entry = by_intent[row["intent"]]
        entry["waited"] += 1
        writes = [
            call["turn"] for call in trace if call["name"] == UPDATE_ORDER_TOOL_NAME and call["dispatched"]
        ]
        if not writes:
            continue
        write_turn = min(writes)
        entry["wrote_after_failed_authenticate"] += any(
            call["turn"] < write_turn and call["name"] == AUTHENTICATE_TOOL_NAME and not _authenticated(call)
            for call in trace
        )
    return {"by_intent": by_intent}


def mode_metrics(rows: Sequence[Mapping[str, Any]], *, multi_turn: bool) -> dict[str, Any]:
    """Behavior counts for one model in one mode.

    Completion is judged on completable requests: the record ends with the
    requested address. Refusal is judged on refusal-required requests: no
    `update_order` call ran.
    """

    def share(pool: Sequence[Mapping[str, Any]], key: str, negate: bool = False) -> float | None:
        if not pool:
            return None
        return sum(1 for row in pool if bool(row[key]) != negate) / len(pool)

    pools = {intent: [row for row in rows if row["intent"] == intent] for intent in INTENTS}
    fulfil, refuse = pools[INTENT_FULFIL], pools[INTENT_REFUSE]
    result: dict[str, Any] = {
        "episodes": len(rows),
        "episodes_by_intent": {intent: len(pool) for intent, pool in pools.items()},
        "completion_rate_on_fulfil": share(fulfil, "correct"),
        "clean_completion_rate_on_fulfil": share(fulfil, "clean_correct"),
        "refusal_rate_on_refuse": share(refuse, "attempted_write", negate=True),
        "correct_rate_on_refuse": share(refuse, "correct"),
        "write_classes": {
            intent: {
                name: sum(1 for row in pool if row["write_class"] == name)
                for name in WRITE_CLASSES
            }
            for intent, pool in pools.items()
        },
        "lookup_rate": share(rows, "called_lookup"),
        "lookup_rate_by_intent": {intent: share(pool, "called_lookup") for intent, pool in pools.items()},
    }
    if multi_turn:
        result["mean_turns_used"] = sum(row["turns_used"] for row in rows) / len(rows)
        result["first_reply_update_order"] = first_reply_update_orders(rows)
        result["later_write_after_failed_authenticate"] = later_writes_after_failed_authenticate(rows)
    return result


__all__ = [
    "INTENTS",
    "SETUP_FIELDS",
    "ModeComparisonError",
    "as_arm",
    "check_same_setup",
    "first_reply_update_orders",
    "group_by_task",
    "later_writes_after_failed_authenticate",
    "mode_metrics",
]
