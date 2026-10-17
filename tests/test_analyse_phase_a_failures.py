"""Every failed Phase A attempt must land in exactly one class, from logged fields.

The README states why Phase A attempts fail and how often the R1 retry rescues
one. These tests pin the classes, the retry counts, the bare JSON dialect rule
and every refusal. The episodes are written by the real rung loop with a
stand-in calculator, so there is no model, no sandbox and no network.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import analyse_phase_a_failures as failures  # noqa: E402
from agent.gates import GateEngine  # noqa: E402
from env.phase_a import CALCULATOR_TOOL_NAME, CalculatorArgs, PhaseATask  # noqa: E402
from env.tools import ToolRegistry, ToolSpec  # noqa: E402
from evaluation.metrics import compute_pass_metrics  # noqa: E402
from evaluation.provenance import DirtyWorktreeError  # noqa: E402
from evaluation.rungs import make_stub_policy, run_episode  # noqa: E402

MODEL = "Qwen/Qwen3-1.7B"
OTHER = "Qwen/Qwen3-4B"
HEAD = "c0ffee" * 6 + "c0ff"

# Stands in for the sandbox. An expression missing here raises, so its call
# runs and fails, as a sandbox error would.
VALUES = {"6 * 7": 42.0, "6 + 7": 13.0}


def call(expression: str) -> str:
    payload = json.dumps({"name": "calculator", "arguments": {"expression": expression}})
    return "<tool_call>\n" + payload + "\n</tool_call>"


GOOD = call("6 * 7")
WRONG = call("6 + 7")
FAILS = call("6 / 0")
PROSE = "The answer is 42."
UNPARSED = '<tool_call>\n{"name": "calculator", "arguments": \n</tool_call>'
BARE = json.dumps({"name": "calculator", "parameters": {"expression": "6 * 7"}})

# Three tasks of four runs. Each entry is the first reply, then the retry R1
# asks for when the first reply gives no answer. R0 sees only the first.
PLAN = [
    [[GOOD], [GOOD], [WRONG], [PROSE, GOOD]],
    [[GOOD], [FAILS, WRONG], [UNPARSED, PROSE], [PROSE, FAILS]],
    [[WRONG], [WRONG], [GOOD], [UNPARSED, UNPARSED]],
]
BARE_PLAN = [[[BARE], [BARE], [PROSE, BARE], [PROSE, PROSE]]]


def task(index: int) -> PhaseATask:
    return PhaseATask(
        task_id=f"gsm8k:test:{index}",
        template_id=f"gsm8k:test:{index}",
        question="Six boxes hold seven pens each. How many pens?",
        gold_answer=42.0,
        source="gsm8k",
    )


def registry() -> ToolRegistry:
    tools = ToolRegistry()
    tools.register(
        ToolSpec(
            name=CALCULATOR_TOOL_NAME,
            args_model=CalculatorArgs,
            handler=lambda args, state: VALUES[args.expression],
        )
    )
    return tools


def episode(
    rung: str,
    replies: list[str],
    *,
    task_index: int = 0,
    run_index: int = 0,
    candidate: str = MODEL,
    normalise: bool = False,
) -> dict:
    """One log row, exactly as run_phase_a_baseline.py writes it."""

    result = run_episode(
        task=task(task_index),
        registry=registry(),
        gate_engine=GateEngine.from_mapping({}),
        policy=make_stub_policy(replies if rung == "R1" else replies[:1]),
        rung=rung,  # type: ignore[arg-type]
        run_index=run_index,
        normalise_dialect=normalise,
    )
    return {"candidate": candidate, **result.to_json()}


def log(plan: list, *, candidate: str = MODEL, normalise: bool = False) -> list[dict]:
    return [
        episode(
            rung, replies, task_index=t, run_index=r, candidate=candidate, normalise=normalise
        )
        for rung in failures.RUNGS
        for t, runs in enumerate(plan)
        for r, replies in enumerate(runs)
    ]


def rung_summary(rows: list[dict]) -> dict:
    """The per-rung figures, computed the way run_phase_a_baseline.py does."""

    by_task: dict[str, list[int]] = {}
    for row in rows:
        by_task.setdefault(row["task_id"], []).append(1 if row["correct"] else 0)
    grid = list(by_task.values())
    first, fourth = compute_pass_metrics(grid, 1), compute_pass_metrics(grid, 4)
    return {
        "tasks": len(grid),
        "runs_per_task": 4,
        "episodes": len(rows),
        "model_decisions": sum(row["counters"]["policy_model_decision_count"] for row in rows),
        "metrics": {
            "pass^1": first.pass_power_k,
            "pass@1": first.pass_at_k,
            "pass^4": fourth.pass_power_k,
            "pass@4": fourth.pass_at_k,
        },
        "no_arithmetic_episodes": sum(1 for row in rows if row["answered_without_arithmetic"]),
    }


def summary(
    rows: list[dict], candidates: tuple[str, ...] = (MODEL,), dialect: str | None = None
) -> dict:
    results = []
    for candidate in candidates:
        result = {
            "candidate": {"id": candidate},
            "rungs": {
                rung: rung_summary(
                    [row for row in rows if row["candidate"] == candidate and row["rung"] == rung]
                )
                for rung in failures.RUNGS
            },
        }
        if dialect is not None:
            result["tool_dialect"] = dialect
        results.append(result)
    return {"executed": True, "results": results}


class ClassTests(unittest.TestCase):
    def _class(self, rung: str, replies: list[str]) -> str:
        return failures.classify(episode(rung, replies), normalise=False)[0]

    def test_a_call_that_returns_the_gold_answer_is_correct(self) -> None:
        row = episode("R0", [GOOD])
        self.assertEqual(failures.classify(row, normalise=False), ("correct", True))

    def test_a_call_that_returns_the_wrong_number_is_a_wrong_value(self) -> None:
        row = episode("R0", [WRONG])
        self.assertEqual(failures.classify(row, normalise=False), ("wrong_value", True))

    def test_a_call_that_runs_and_fails_is_a_call_that_failed_to_run(self) -> None:
        self.assertEqual(self._class("R0", [FAILS]), "call_failed_to_run")

    def test_a_schema_error_is_a_call_that_failed_to_run(self) -> None:
        payload = json.dumps({"name": "calculator", "arguments": {"expr": "6 * 7"}})
        wrong_argument = "<tool_call>\n" + payload + "\n</tool_call>"
        self.assertEqual(self._class("R0", [wrong_argument]), "call_failed_to_run")

    def test_prose_with_no_tag_is_no_tag(self) -> None:
        self.assertEqual(self._class("R0", [PROSE]), "no_tag")

    def test_a_tag_that_does_not_parse_is_malformed(self) -> None:
        self.assertEqual(self._class("R0", [UNPARSED]), "malformed")

    def test_an_r1_episode_is_classed_by_its_last_decision(self) -> None:
        self.assertEqual(self._class("R1", [UNPARSED, PROSE]), "no_tag")
        self.assertEqual(self._class("R1", [PROSE, FAILS]), "call_failed_to_run")
        self.assertEqual(self._class("R1", [FAILS, WRONG]), "wrong_value")
        self.assertEqual(self._class("R1", [PROSE, GOOD]), "correct")
        # The one case where judging the whole episode would differ: the first
        # call ran and failed, and the retry made no call at all.
        self.assertEqual(self._class("R1", [FAILS, PROSE]), "no_tag")

    def test_a_logged_call_the_re_parse_does_not_give_back_is_refused(self) -> None:
        """The same number of calls is not enough: they must be the same calls."""

        row = episode("R0", [WRONG])
        row["tool_events"][0]["call"]["arguments"]["expression"] = "6 - 7"
        with self.assertRaisesRegex(failures.FailureAnalysisError, "re-parse does not give back"):
            failures.classify(row, normalise=False)

    def test_events_that_contradict_the_terminal_reason_are_refused(self) -> None:
        row = episode("R0", [WRONG])
        row["terminal_reason"] = "no_action"
        with self.assertRaisesRegex(failures.FailureAnalysisError, "terminal reason"):
            failures.classify(row, normalise=False)


class ArmCountTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = log(PLAN)
        self.arm = failures.analyse_arm(self.rows, summary(self.rows), MODEL)

    def test_r0_failures_are_split_by_class(self) -> None:
        r0 = self.arm["R0"]
        self.assertEqual(r0["episodes"], 12)
        self.assertEqual(r0["correct"], 4)
        self.assertEqual(
            r0["failures"],
            {"no_call_or_unparsed": 4, "call_failed_to_run": 1, "wrong_value": 3, "total": 8},
        )
        self.assertEqual(r0["unparsed_detail"], {"no_tag": 2, "malformed": 2})
        self.assertNotIn("retry", r0)

    def test_r1_classes_follow_the_retry(self) -> None:
        r1 = self.arm["R1"]
        self.assertEqual(r1["correct"], 5)
        self.assertEqual(
            r1["failures"],
            {"no_call_or_unparsed": 2, "call_failed_to_run": 1, "wrong_value": 4, "total": 7},
        )
        self.assertEqual(r1["unparsed_detail"], {"no_tag": 1, "malformed": 1})

    def test_r1_retries_are_counted(self) -> None:
        self.assertEqual(
            self.arm["R1"]["retry"],
            {"fired": 5, "rescued": 1, "answered_wrong": 1, "still_no_answer": 3},
        )

    def test_rows_are_picked_by_candidate(self) -> None:
        """A baseline log holds two models that share every task id."""

        perfect = [[[GOOD]] * 4] * 3
        rows = self.rows + log(perfect, candidate=OTHER)
        both = summary(rows, (MODEL, OTHER))
        self.assertEqual(failures.analyse_arm(rows, both, MODEL), self.arm)
        other = failures.analyse_arm(rows, both, OTHER)
        self.assertEqual(other["R0"]["correct"], 12)
        self.assertEqual(other["R1"]["retry"]["fired"], 0)


class DialectTests(unittest.TestCase):
    """Bare JSON is normalised only when the summary says the model emits it."""

    def test_bare_json_is_a_call_for_a_bare_json_model(self) -> None:
        rows = log(BARE_PLAN, normalise=True)
        arm = failures.analyse_arm(rows, summary(rows, dialect="bare_json"), MODEL)
        self.assertEqual(arm["R0"]["correct"], 2)
        self.assertEqual(arm["R0"]["unparsed_detail"], {"no_tag": 2, "malformed": 0})
        self.assertEqual(arm["R1"]["correct"], 3)
        self.assertEqual(
            arm["R1"]["retry"],
            {"fired": 2, "rescued": 1, "answered_wrong": 0, "still_no_answer": 1},
        )

    def test_bare_json_is_no_tag_for_a_tagged_model(self) -> None:
        self.assertEqual(failures.classify(episode("R0", [BARE]), normalise=False)[0], "no_tag")

    def test_a_dialect_the_log_was_not_recorded_under_is_refused(self) -> None:
        rows = log(BARE_PLAN, normalise=True)
        with self.assertRaisesRegex(failures.FailureAnalysisError, "re-parse"):
            failures.analyse_arm(rows, summary(rows, dialect="canonical_tags"), MODEL)


class RefusalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = log(PLAN)
        self.summary = summary(self.rows)

    def test_rows_that_do_not_reproduce_each_summary_figure_are_refused(self) -> None:
        for rung in failures.RUNGS:
            for field in failures.SUMMARY_FIELDS:
                with self.subTest(rung=rung, field=field):
                    changed = copy.deepcopy(self.summary)
                    figures = changed["results"][0]["rungs"][rung]
                    if field.startswith("pass"):
                        figures["metrics"][field] += 0.25
                    else:
                        figures[field] += 1
                    pattern = f"{rung}: .*{re.escape(field)}"
                    with self.assertRaisesRegex(failures.FailureAnalysisError, pattern):
                        failures.analyse_arm(self.rows, changed, MODEL)

    def test_a_missing_rung_is_refused(self) -> None:
        del self.summary["results"][0]["rungs"]["R1"]
        with self.assertRaisesRegex(failures.FailureAnalysisError, "no R1 result"):
            failures.analyse_arm(self.rows, self.summary, MODEL)

    def test_a_repeated_run_is_refused(self) -> None:
        with self.assertRaisesRegex(failures.FailureAnalysisError, "appears twice"):
            failures.analyse_arm(self.rows + self.rows[:1], self.summary, MODEL)

    def test_an_r1_first_completion_that_differs_from_r0_is_refused(self) -> None:
        rows = [
            episode("R1", [WRONG], task_index=0, run_index=3)
            if (row["rung"], row["task_id"], row["run_index"]) == ("R1", "gsm8k:test:0", 3)
            else row
            for row in self.rows
        ]
        with self.assertRaisesRegex(failures.FailureAnalysisError, "differs from the R0"):
            failures.analyse_arm(rows, summary(rows), MODEL)

    def test_a_candidate_the_summary_does_not_hold_is_refused(self) -> None:
        with self.assertRaisesRegex(failures.FailureAnalysisError, "no measured result"):
            failures.analyse_arm(self.rows, self.summary, OTHER)

    def test_a_log_without_the_candidate_is_refused(self) -> None:
        both = summary(self.rows + log(PLAN, candidate=OTHER), (MODEL, OTHER))
        with self.assertRaisesRegex(failures.FailureAnalysisError, "no rows"):
            failures.analyse_arm(self.rows, both, OTHER)

    def test_a_planned_summary_is_refused(self) -> None:
        self.summary["executed"] = False
        with self.assertRaisesRegex(failures.FailureAnalysisError, "not an executed run"):
            failures.analyse_arm(self.rows, self.summary, MODEL)


class ArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.rows = log(PLAN)
        self.summary_path = self.root / "baseline-example.json"
        self.summary_path.write_text(json.dumps(summary(self.rows)), encoding="utf-8")
        self.episodes_path = self.root / "episodes-example.jsonl"
        self.episodes_path.write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8"
        )
        self.output = self.root / "failure-modes-example.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _main(self, labels: list[str], guard: mock.Mock) -> int:
        argv = ["analyse_phase_a_failures.py", "--output", str(self.output)]
        for label in labels:
            argv += ["--arm", label, str(self.summary_path), str(self.episodes_path), MODEL]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            failures, "require_clean_worktree", guard
        ), mock.patch("builtins.print"):
            return failures.main()

    def test_the_written_artifact_round_trips(self) -> None:
        self.assertEqual(self._main(["base", "again"], mock.Mock(return_value=HEAD)), 0)
        payload = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["kind"], "phase_a_failure_modes")
        self.assertEqual(payload["source_commit"], HEAD)
        self.assertEqual(set(payload["arms"]), {"base", "again"})
        for name in ("correct", "wrong_value", "call_failed_to_run", "no_call_or_unparsed",
                     "no_tag", "malformed"):
            self.assertIn(name, payload["method"])
        arm = payload["arms"]["base"]
        self.assertEqual(arm["rungs"], failures.analyse_arm(self.rows, summary(self.rows), MODEL))
        self.assertEqual(arm["candidate"], MODEL)
        # Both files sit outside the repository, so each is named by its file name.
        self.assertEqual(arm["summary"], "baseline-example.json")
        self.assertEqual(arm["episodes"], "episodes-example.jsonl")
        self.assertEqual(
            arm["summary_sha256"], hashlib.sha256(self.summary_path.read_bytes()).hexdigest()
        )
        self.assertEqual(
            arm["episodes_sha256"], hashlib.sha256(self.episodes_path.read_bytes()).hexdigest()
        )
        self.assertFalse(self.output.with_suffix(".tmp").exists())

    def test_a_dirty_tree_is_refused_and_nothing_is_written(self) -> None:
        guard = mock.Mock(side_effect=DirtyWorktreeError("dirty"))
        with self.assertRaises(DirtyWorktreeError):
            self._main(["base"], guard)
        guard.assert_called_once_with(failures.PROJECT_ROOT)
        self.assertFalse(self.output.exists())

    def test_repeated_labels_are_refused_and_nothing_is_written(self) -> None:
        with self.assertRaisesRegex(failures.FailureAnalysisError, "repeated: base"):
            self._main(["base", "base"], mock.Mock(return_value=HEAD))
        self.assertFalse(self.output.exists())

    def test_a_refused_arm_writes_nothing(self) -> None:
        changed = summary(self.rows)
        changed["results"][0]["rungs"]["R1"]["model_decisions"] += 1
        self.summary_path.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(failures.FailureAnalysisError, "base: R1"):
            self._main(["base"], mock.Mock(return_value=HEAD))
        self.assertFalse(self.output.exists())

    def test_a_path_inside_the_repository_is_recorded_relative_to_it(self) -> None:
        inside = PROJECT_ROOT / "results" / "baseline-example.json"
        self.assertEqual(failures.recorded_path(inside), "results/baseline-example.json")


if __name__ == "__main__":
    unittest.main()
