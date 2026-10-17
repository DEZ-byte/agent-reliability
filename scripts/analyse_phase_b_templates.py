"""Split the Phase B results by request template, and record the calls made.

The Phase B test split has three request templates:

- **address_change_verified** gives the right postcode. The model should
  authenticate and then change the address.
- **address_change_wrong_postcode** gives a postcode that is not on the
  account. The model must leave the order alone.
- **address_change_no_postcode** gives no postcode at all. The model must leave
  the order alone.

Phase B is single-shot. The model writes all of its tool calls before any tool
runs, so it never sees whether authentication worked. The first two templates
are worded differently, but nothing in either says whether its postcode is
right. A model that writes on both is wrong on one, a model that stops on both
is wrong on the other, and it has no evidence that would let it do better. Only
the no-postcode request shows, in the request itself, that the change must be
refused. It is the one template where a single reply can show judgement.

The per-intent rates in the summary merge the two refusal templates, and they
say nothing about which tools were called. This script counts each template on
its own, and records each episode's exact sequence of tool calls. It reads the
stored episode rows, and first checks that they reproduce the committed
summary. Given the gate breakdown, it also checks that they are the same bytes
that breakdown replayed.
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

from env.phase_b import (  # noqa: E402
    AUTHENTICATE_TOOL_NAME,
    INTENT_FULFIL,
    INTENT_REFUSE,
    UPDATE_ORDER_TOOL_NAME,
)
from env.phase_b_tasks import load_split  # noqa: E402
from evaluation.provenance import (  # noqa: E402
    DirtyWorktreeError,
    portable_path,
    require_clean_worktree,
)
from run_phase_b_eval import SPLIT_MANIFEST_PATH, summarise  # noqa: E402

SCHEMA_VERSION: Final = 1
SEPARATOR: Final = ">"
NO_CALLS: Final = "(none)"


class TemplateAnalysisError(RuntimeError):
    """The stored episodes are not the ones behind the committed artifacts."""


def checked_fields(summary: dict[str, Any]) -> dict[str, Any]:
    """The summary figures the stored rows must reproduce."""

    return {
        "episodes": summary["episodes"],
        "pass^1": summary["metrics"]["pass^1"],
        "fulfil_correct_rate": summary["by_intent"][INTENT_FULFIL]["correct_rate"],
        "refuse_correct_rate": summary["by_intent"][INTENT_REFUSE]["correct_rate"],
    }


def call_sequence(row: dict[str, Any]) -> str:
    return SEPARATOR.join(row["tools_called"]) or NO_CALLS


def count_template(rows: list[dict[str, Any]]) -> dict[str, Any]:
    intents = {row["intent"] for row in rows}
    if len(intents) != 1:
        raise TemplateAnalysisError(
            f"{rows[0]['template_id']} rows carry more than one intent: {sorted(intents)}"
        )
    sequences = Counter(call_sequence(row) for row in rows)
    return {
        "intent": intents.pop(),
        "episodes": len(rows),
        "correct": sum(1 for row in rows if row["correct"]),
        "wrote": sum(1 for row in rows if row["attempted_write"]),
        "wrote_without_authenticate": sum(
            1
            for row in rows
            if UPDATE_ORDER_TOOL_NAME in row["tools_called"]
            and AUTHENTICATE_TOOL_NAME not in row["tools_called"]
        ),
        "single_call": sum(1 for row in rows if len(row["tools_called"]) == 1),
        # Most common first, then by name, so the order never depends on the
        # order of the rows.
        "call_sequences": dict(sorted(sequences.items(), key=lambda item: (-item[1], item[0]))),
    }


def analyse_arm(rows: list[dict[str, Any]], tasks: list, summary: dict[str, Any]) -> dict[str, Any]:
    expected = checked_fields(summary)
    recomputed = checked_fields(summarise(rows, tasks))
    for field, value in expected.items():
        if recomputed[field] != value:
            raise TemplateAnalysisError(
                f"stored rows do not reproduce the summary's {field}: "
                f"rows give {recomputed[field]!r}, the summary records {value!r}"
            )

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["template_id"], []).append(row)
    return {
        "by_template": {template: count_template(groups[template]) for template in sorted(groups)},
        "totals": {
            "episodes": len(rows),
            "correct": sum(1 for row in rows if row["correct"]),
            "checked_against_summary": expected,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", nargs=3, action="append", required=True, metavar=("LABEL", "SUMMARY", "EPISODES")
    )
    parser.add_argument(
        "--gates", default=None, help="gate breakdown that must have replayed these same episodes"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_commit = require_clean_worktree(PROJECT_ROOT)
    labels = [label for label, _, _ in args.arm]
    if len(set(labels)) != len(labels):
        raise TemplateAnalysisError(f"each --arm label must be used once, got {labels}")

    gates_record = None
    replayed: dict[str, str] = {}
    if args.gates is not None:
        gates_file = Path(args.gates)
        gates_bytes = gates_file.read_bytes()
        gates = json.loads(gates_bytes.decode("utf-8"))
        replayed = {label: arm["episodes_sha256"] for label, arm in gates["arms"].items()}
        gates_record = {
            "artifact": portable_path(gates_file, PROJECT_ROOT),
            "sha256": hashlib.sha256(gates_bytes).hexdigest(),
        }

    arms: dict[str, Any] = {}
    for label, summary_path, episodes_path in args.arm:
        summary_file, episodes_file = Path(summary_path), Path(episodes_path)
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        if summary.get("label") != label or summary.get("executed") is not True:
            raise TemplateAnalysisError(f"{summary_file.name} is not an executed {label!r} run")
        # Hash and parse the same bytes, so the rows counted are the rows hashed.
        episodes_bytes = episodes_file.read_bytes()
        episodes_sha256 = hashlib.sha256(episodes_bytes).hexdigest()
        if gates_record is not None and replayed.get(label) != episodes_sha256:
            raise TemplateAnalysisError(
                f"{episodes_file.name} is not the {label!r} episodes file the gate breakdown replayed"
            )
        rows = [
            json.loads(line)
            for line in episodes_bytes.decode("utf-8").splitlines()
            if line.strip()
        ]
        tasks = load_split(SPLIT_MANIFEST_PATH, summary["split"])
        arms[label] = {
            "summary": portable_path(summary_file, PROJECT_ROOT),
            "summary_sha256": hashlib.sha256(summary_file.read_bytes()).hexdigest(),
            "episodes": portable_path(episodes_file, PROJECT_ROOT),
            "episodes_sha256": episodes_sha256,
            **analyse_arm(rows, tasks, summary),
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "kind": "phase_b_template_breakdown",
        "method": (
            "The stored episode rows of each arm are counted by request template. "
            "Before counting, the rows must reproduce the committed summary's episode "
            "count, pass^1 and per-intent correct rates exactly. When a gate breakdown "
            "is given, each episodes file must also hash to the one it replayed. "
            "'wrote' counts episodes that called update_order. "
            "'wrote_without_authenticate' counts those that called update_order and "
            "never called authenticate. A call sequence is the tools an episode "
            "dispatched, in order, joined by '>'; '(none)' means no tool ran. "
            "Phase B is single-shot, so the model never sees a tool result before "
            "it writes."
        ),
        "gates_breakdown": gates_record,
        "arms": arms,
        "source_commit": source_commit,
        "platform": {"python": platform.python_version(), "system": platform.system()},
    }
    path = Path(args.output)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes((json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    os.replace(temporary, path)
    for label, arm in arms.items():
        for template_id, counts in arm["by_template"].items():
            print(
                f"{label} {template_id}: correct {counts['correct']}/{counts['episodes']}  "
                f"wrote {counts['wrote']} ({counts['wrote_without_authenticate']} without "
                f"authenticate)  sequences {counts['call_sequences']}"
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (TemplateAnalysisError, DirtyWorktreeError) as error:
        print(f"refused: {error}", file=sys.stderr)
        raise SystemExit(1)
