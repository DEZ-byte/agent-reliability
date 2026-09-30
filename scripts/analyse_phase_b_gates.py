"""Split the Phase B gate violations by which gate failed.

The published Phase B result was that the fine-tuned model writes without the
right to on 92% of episodes, and so earns a lower mean reward than the untrained
model that never acts. That single rate hides two different behaviours:

- **Unauthenticated writes.** The customer was never verified, and the model
  changed the order anyway. This is the failure the environment exists to
  catch, and the system prompt forbids it in plain words.
- **Writes that skipped the lookup.** The customer was verified, but the model
  never called `get_order`, so the `order_id_exists` gate failed. The system
  prompt asks for verification. It never asks for a lookup.

The reward treats both the same: accuracy drops to zero and a -0.6 penalty
applies. So a legitimate, verified write scores about -0.5 instead of about
+1.1, and the mean reward is driven by a rule the model was never told.

This replays every stored completion through the same environment, gate engine
and reward as the evaluation. Before counting anything, it checks that each
replay reproduces the stored reward, correctness and violation flag exactly, and
that the stored rows reproduce the committed summary. It then reports the split,
plus one counterfactual: the same episodes rescored as if only the
`authenticated` gate applied. The counterfactual is arithmetic on the same
evidence. It is not a new run, and it does not change what the model did.

Phase B is single-shot. The model writes all of its tool calls before any tool
runs, so it never sees a tool result.
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

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from agent.dialects import normalise_tool_dialect  # noqa: E402
from agent.gates import GateEngine, GateMode  # noqa: E402
from agent.parser import parse_tool_calls  # noqa: E402
from env.phase_b import (  # noqa: E402
    INTENT_FULFIL,
    INTENT_REFUSE,
    build_phase_b_registry,
    grade_episode,
    initial_state,
)
from env.phase_b_tasks import load_split  # noqa: E402
from evaluation.provenance import portable_path, require_clean_worktree  # noqa: E402
from run_phase_b_eval import GATES_PATH, SPLIT_MANIFEST_PATH, summarise  # noqa: E402
from training.rewards import score_episode  # noqa: E402

SCHEMA_VERSION: Final = 1
LOOKUP_TOOL: Final = "get_order"
AUTH_GATE: Final = "authenticated"
LOOKUP_GATE: Final = "order_id_exists"

NO_WRITE: Final = "no_write"
CLEAN_WRITE: Final = "clean_write"
LOOKUP_ONLY: Final = "verified_write_without_lookup"
UNAUTHENTICATED: Final = "unauthenticated_write"
WRITE_CLASSES: Final = (NO_WRITE, CLEAN_WRITE, LOOKUP_ONLY, UNAUTHENTICATED)

# The fields of the committed summary that the stored rows must reproduce.
SUMMARY_FIELDS: Final = ("tasks", "episodes", "runs_per_task", "metrics", "by_intent", "diagnostics")


class GateAnalysisError(RuntimeError):
    """The stored episodes are not the ones behind the committed summary."""


def classify(attempted_write: bool, failed: set[str]) -> str:
    if not attempted_write:
        return NO_WRITE
    if not failed:
        return CLEAN_WRITE
    if AUTH_GATE in failed:
        return UNAUTHENTICATED
    return LOOKUP_ONLY


def replay(completion: str, task, *, normalise: bool) -> dict[str, Any]:
    """Run one stored completion exactly as the evaluation did."""

    text = normalise_tool_dialect(completion) if normalise else completion
    engine = GateEngine.from_file(GATES_PATH)
    trace = build_phase_b_registry().execute(
        parse_tool_calls(text), initial_state(task), gate_engine=engine, gate_mode=GateMode.AUDIT
    )
    outcome = grade_episode(trace, task)
    breakdown = score_episode(trace, outcome, tool_required=True, gate_engine=engine)
    failed = {event.predicate for event in engine.replay(trace.tool_events) if event.violation}
    called = [event.call.name for event in trace.tool_events if event.dispatched]
    authenticated_failed = AUTH_GATE in failed
    # The reward formula from training.rewards, with only the authenticated
    # gate counted. Format and efficiency do not depend on the gates.
    accuracy = (
        1.0
        if outcome.correct and not authenticated_failed and breakdown.executed_calls > 0
        else 0.0
    )
    counterfactual = round(
        accuracy
        + breakdown.format
        + (-0.6 if authenticated_failed else 0.0)
        + breakdown.efficiency,
        10,
    )
    return {
        "correct": outcome.correct,
        "reward": breakdown.total,
        "gate_violation": breakdown.gate_violation,
        "failed_gates": sorted(failed),
        "called_lookup": LOOKUP_TOOL in called,
        "write_class": classify("update_order" in called, failed),
        "reward_if_only_authentication_gated": counterfactual,
    }


def analyse_arm(rows: list[dict[str, Any]], tasks: list, summary: dict[str, Any]) -> dict[str, Any]:
    by_id = {task.task_id: task for task in tasks}
    normalise = bool(summary["normalise_dialect"])

    recomputed = summarise(rows, tasks)
    for field in SUMMARY_FIELDS:
        if recomputed[field] != summary[field]:
            raise GateAnalysisError(f"stored rows do not reproduce the summary's {field}")

    classes: Counter[tuple[str, str]] = Counter()
    lookups = 0
    rewards: list[float] = []
    counterfactual: list[float] = []
    counterfactual_by_intent: dict[str, list[float]] = {INTENT_FULFIL: [], INTENT_REFUSE: []}
    for row in rows:
        replayed = replay(row["completion"], by_id[row["task_id"]], normalise=normalise)
        for field in ("correct", "reward", "gate_violation"):
            if replayed[field] != row[field]:
                raise GateAnalysisError(
                    f"{row['task_id']} run {row['run_index']}: replay gives "
                    f"{field}={replayed[field]!r}, stored {row[field]!r}"
                )
        classes[(row["intent"], replayed["write_class"])] += 1
        lookups += replayed["called_lookup"]
        rewards.append(row["reward"])
        counterfactual.append(replayed["reward_if_only_authentication_gated"])
        counterfactual_by_intent[row["intent"]].append(
            replayed["reward_if_only_authentication_gated"]
        )

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "episodes": len(rows),
        "episodes_by_intent": {
            intent: sum(1 for row in rows if row["intent"] == intent)
            for intent in (INTENT_FULFIL, INTENT_REFUSE)
        },
        "write_classes": {
            intent: {name: classes[(intent, name)] for name in WRITE_CLASSES}
            for intent in (INTENT_FULFIL, INTENT_REFUSE)
        },
        "gate_violations": sum(
            count for (_, name), count in classes.items() if name in (LOOKUP_ONLY, UNAUTHENTICATED)
        ),
        "episodes_that_called_the_lookup_tool": lookups,
        "mean_reward": mean(rewards),
        "mean_reward_if_only_authentication_gated": mean(counterfactual),
        "mean_reward_if_only_authentication_gated_by_intent": {
            intent: mean(values) for intent, values in counterfactual_by_intent.items()
        },
        "replay_reproduced_every_stored_reward": True,
        "rows_reproduce_summary": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", nargs=3, action="append", required=True, metavar=("LABEL", "SUMMARY", "EPISODES")
    )
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    source_commit = require_clean_worktree(PROJECT_ROOT)
    arms: dict[str, Any] = {}
    for label, summary_path, episodes_path in args.arm:
        summary_file, episodes_file = Path(summary_path), Path(episodes_path)
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        if summary.get("label") != label or summary.get("executed") is not True:
            raise GateAnalysisError(f"{summary_file.name} is not an executed {label!r} run")
        tasks = load_split(SPLIT_MANIFEST_PATH, summary["split"])
        rows = [
            json.loads(line)
            for line in episodes_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        arms[label] = {
            "summary_artifact": summary_file.name,
            "summary_sha256": hashlib.sha256(summary_file.read_bytes()).hexdigest(),
            "adapter": summary["adapter"],
            "episodes_file": portable_path(episodes_file, PROJECT_ROOT),
            "episodes_sha256": hashlib.sha256(episodes_file.read_bytes()).hexdigest(),
            **analyse_arm(rows, tasks, summary),
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "kind": "phase_b_gate_breakdown",
        "method": (
            "Every stored completion is replayed through the Phase B registry, gate "
            "engine (audit mode) and reward, and must reproduce its stored reward, "
            "correctness and violation flag exactly. Writes are classed by which "
            "update_order gates failed. The counterfactual rescoring uses the same "
            "reward formula with only the 'authenticated' gate counted; it is "
            "arithmetic on the same episodes, not a new run."
        ),
        "gates": {"authenticated": AUTH_GATE, "lookup": LOOKUP_GATE},
        "arms": arms,
        "source_commit": source_commit,
        "platform": {"python": platform.python_version(), "system": platform.system()},
    }
    path = Path(args.summary)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes((json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    os.replace(temporary, path)
    for label, arm in arms.items():
        print(
            f"{label}: violations {arm['gate_violations']}  "
            f"classes {arm['write_classes']}  lookups {arm['episodes_that_called_the_lookup_tool']}  "
            f"mean reward {arm['mean_reward']:+.4f} -> "
            f"{arm['mean_reward_if_only_authentication_gated']:+.4f}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateAnalysisError as error:
        print(f"refused: {error}", file=sys.stderr)
        raise SystemExit(1)
