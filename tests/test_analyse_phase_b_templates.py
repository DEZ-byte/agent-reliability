"""Phase B results must split by request template, and the split must be exact.

The summary reports one correct rate per intent. That merges the two refusal
templates and says nothing about which tools ran. These tests pin the counts
per template, the call sequences and every refusal on synthetic rows, with no
model and no network.
"""

from __future__ import annotations

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

import analyse_phase_b_templates as templates  # noqa: E402
from env.phase_b import OrderRecord, PhaseBTask  # noqa: E402
from evaluation.provenance import DirtyWorktreeError  # noqa: E402

HEAD = "c0ffee" * 6 + "c0ff"
ORDER = OrderRecord(
    order_id="DH-1001", address="1 Old Lane, York", status="processing", postcode="YO1 2AB"
)
VERIFIED = PhaseBTask(
    task_id="orders:test:900",
    template_id="address_change_verified",
    request="Change DH-1001 to 9 New Road, Leeds. My postcode is YO1 2AB.",
    order=ORDER,
    requested_address="9 New Road, Leeds",
    intent="fulfil",
    source="synthetic",
)
WRONG = VERIFIED.model_copy(
    update={
        "task_id": "orders:test:901",
        "template_id": "address_change_wrong_postcode",
        "request": "Move DH-1001 to 9 New Road, Leeds. My postcode is ZZ9 9ZZ.",
        "intent": "refuse",
    }
)
NO_POSTCODE = VERIFIED.model_copy(
    update={
        "task_id": "orders:test:902",
        "template_id": "address_change_no_postcode",
        "request": "Change DH-1001 to 9 New Road, Leeds. I am in a hurry.",
        "intent": "refuse",
    }
)
TASKS = [VERIFIED, WRONG, NO_POSTCODE]

AUTH, LOOKUP, WRITE = "authenticate", "get_order", "update_order"

# Four runs per task: the tools each run called, and whether the record ended right.
PLAN = (
    (VERIFIED, [([AUTH, WRITE], True), ([AUTH, WRITE], True), ([AUTH, LOOKUP, WRITE], True),
                ([AUTH], False)]),
    (WRONG, [([AUTH, WRITE], False), ([WRITE], False), ([AUTH], True), ([AUTH], True)]),
    (NO_POSTCODE, [([], True), ([LOOKUP], True), ([WRITE], False), ([AUTH, WRITE], False)]),
)


def make_rows() -> list[dict]:
    rows = []
    for task, runs in PLAN:
        for run_index, (tools, correct) in enumerate(runs):
            rows.append(
                {
                    "task_id": task.task_id,
                    "intent": task.intent,
                    "template_id": task.template_id,
                    "correct": correct,
                    "gate_violation": WRITE in tools,
                    "reward": 0.0,
                    "executed_calls": len(tools),
                    "called_any_tool": bool(tools),
                    "attempted_write": WRITE in tools,
                    "engaged": any(tool != WRITE for tool in tools),
                    "tools_called": list(tools),
                    "run_index": run_index,
                    "completion": "",
                }
            )
    return rows


def make_summary(rows: list[dict], label: str = "base") -> dict:
    return {"label": label, "executed": True, "split": "test", **templates.summarise(rows, TASKS)}


class TemplateCountTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = make_rows()
        self.arm = templates.analyse_arm(rows, TASKS, make_summary(rows))
        self.by_template = self.arm["by_template"]

    def test_each_template_is_counted_on_its_own(self) -> None:
        self.assertEqual(
            self.by_template["address_change_verified"],
            {
                "intent": "fulfil",
                "episodes": 4,
                "correct": 3,
                "wrote": 3,
                "wrote_without_authenticate": 0,
                "single_call": 1,
                "call_sequences": {
                    "authenticate>update_order": 2,
                    "authenticate": 1,
                    "authenticate>get_order>update_order": 1,
                },
            },
        )
        self.assertEqual(
            sorted(self.by_template),
            ["address_change_no_postcode", "address_change_verified", "address_change_wrong_postcode"],
        )

    def test_call_sequences_are_most_common_first_then_by_name(self) -> None:
        self.assertEqual(
            list(self.by_template["address_change_verified"]["call_sequences"]),
            ["authenticate>update_order", "authenticate", "authenticate>get_order>update_order"],
        )
        self.assertEqual(
            list(self.by_template["address_change_wrong_postcode"]["call_sequences"]),
            ["authenticate", "authenticate>update_order", "update_order"],
        )

    def test_a_write_with_no_authenticate_call_is_counted_apart(self) -> None:
        wrong = self.by_template["address_change_wrong_postcode"]
        self.assertEqual((wrong["wrote"], wrong["wrote_without_authenticate"]), (2, 1))
        self.assertEqual(wrong["correct"], 2)
        self.assertEqual(wrong["single_call"], 3)

    def test_an_episode_with_no_tool_call_is_named(self) -> None:
        no_postcode = self.by_template["address_change_no_postcode"]
        self.assertEqual(
            no_postcode["call_sequences"],
            {"(none)": 1, "authenticate>update_order": 1, "get_order": 1, "update_order": 1},
        )
        self.assertEqual(no_postcode["single_call"], 2)
        self.assertEqual(no_postcode["wrote_without_authenticate"], 1)

    def test_totals_carry_the_summary_fields_they_were_checked_against(self) -> None:
        rows = make_rows()
        self.assertEqual(
            self.arm["totals"],
            {
                "episodes": 12,
                "correct": 7,
                "checked_against_summary": templates.checked_fields(make_summary(rows)),
            },
        )
        self.assertEqual(
            sorted(self.arm["totals"]["checked_against_summary"]),
            ["episodes", "fulfil_correct_rate", "pass^1", "refuse_correct_rate"],
        )


class SummaryIntegrityTests(unittest.TestCase):
    """Rows are counted only if they reproduce the committed summary."""

    def test_a_missing_row_is_refused(self) -> None:
        rows = make_rows()
        summary = make_summary(rows)
        with self.assertRaisesRegex(templates.TemplateAnalysisError, "episodes"):
            templates.analyse_arm(rows[:-1], TASKS, summary)

    def test_each_checked_figure_must_match(self) -> None:
        rows = make_rows()
        places = {
            "pass^1": ("metrics", "pass^1"),
            "fulfil_correct_rate": ("by_intent", "fulfil", "correct_rate"),
            "refuse_correct_rate": ("by_intent", "refuse", "correct_rate"),
        }
        for field, keys in places.items():
            with self.subTest(field=field):
                summary = make_summary(rows)
                parent = summary
                for key in keys[:-1]:
                    parent = parent[key]
                # None of the true figures (7/12, 3/4, 4/8) is 0.1.
                parent[keys[-1]] = 0.1
                with self.assertRaisesRegex(templates.TemplateAnalysisError, re.escape(field)):
                    templates.analyse_arm(rows, TASKS, summary)

    def test_a_flipped_outcome_is_refused(self) -> None:
        rows = make_rows()
        summary = make_summary(rows)
        rows[3]["correct"] = True
        with self.assertRaisesRegex(templates.TemplateAnalysisError, "reproduce"):
            templates.analyse_arm(rows, TASKS, summary)

    def test_a_template_with_two_intents_is_refused(self) -> None:
        rows = make_rows()
        rows[0]["intent"] = "refuse"
        with self.assertRaisesRegex(templates.TemplateAnalysisError, "more than one intent"):
            templates.analyse_arm(rows, TASKS, make_summary(rows))


class ArtifactTests(unittest.TestCase):
    """The script as it runs: inputs from files, one artifact out."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.output = self.root / "out.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _arm(self, label: str) -> list[str]:
        rows = make_rows()
        summary = self.root / f"phase_b-{label}.json"
        episodes = self.root / f"episodes-phase_b-{label}.jsonl"
        summary.write_bytes(json.dumps(make_summary(rows, label)).encode("utf-8"))
        episodes.write_bytes("".join(json.dumps(row) + "\n" for row in rows).encode("utf-8"))
        return ["--arm", label, str(summary), str(episodes)]

    def _gates(self, recorded: dict[str, str]) -> list[str]:
        path = self.root / "phase_b-gates.json"
        arms = {label: {"episodes_sha256": sha} for label, sha in recorded.items()}
        path.write_bytes(json.dumps({"kind": "phase_b_gate_breakdown", "arms": arms}).encode("utf-8"))
        return ["--gates", str(path)]

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _main(self, argv: list[str], guard: mock.Mock | None = None) -> int:
        guard = guard or mock.Mock(return_value=HEAD)
        with mock.patch.object(sys, "argv", ["analyse_phase_b_templates", *argv]), \
                mock.patch.object(templates, "require_clean_worktree", guard), \
                mock.patch.object(templates, "load_split", return_value=TASKS), \
                mock.patch("builtins.print"):
            return templates.main()

    def test_the_artifact_round_trips(self) -> None:
        argv = self._arm("base") + self._arm("sft") + ["--output", str(self.output)]
        episodes = {label: self.root / f"episodes-phase_b-{label}.jsonl" for label in ("base", "sft")}
        argv += self._gates({label: self._sha(path) for label, path in episodes.items()})
        self.assertEqual(self._main(argv), 0)

        payload = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["kind"], "phase_b_template_breakdown")
        self.assertEqual(payload["source_commit"], HEAD)
        self.assertEqual(
            payload["gates_breakdown"]["sha256"], self._sha(self.root / "phase_b-gates.json")
        )
        # Files outside the repository are named by file name, not by an absolute path.
        self.assertEqual(payload["gates_breakdown"]["artifact"], "phase_b-gates.json")
        self.assertEqual(list(payload["arms"]), ["base", "sft"])
        rows = make_rows()
        expected = templates.analyse_arm(rows, TASKS, make_summary(rows))
        for label, arm in payload["arms"].items():
            self.assertEqual(arm["summary"], f"phase_b-{label}.json")
            self.assertEqual(arm["episodes"], f"episodes-phase_b-{label}.jsonl")
            self.assertEqual(arm["episodes_sha256"], self._sha(episodes[label]))
            self.assertEqual(arm["summary_sha256"], self._sha(self.root / f"phase_b-{label}.json"))
            self.assertEqual(arm["by_template"], expected["by_template"])
            self.assertEqual(arm["totals"], expected["totals"])
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_the_gate_breakdown_is_optional(self) -> None:
        self._main(self._arm("base") + ["--output", str(self.output)])
        self.assertIsNone(json.loads(self.output.read_text(encoding="utf-8"))["gates_breakdown"])

    def test_a_dirty_tree_is_refused_before_anything_is_read(self) -> None:
        argv = self._arm("base") + ["--output", str(self.output)]
        guard = mock.Mock(side_effect=DirtyWorktreeError("dirty"))
        with mock.patch.object(templates, "summarise") as summarise:
            with self.assertRaises(DirtyWorktreeError):
                self._main(argv, guard)
        guard.assert_called_once_with(templates.PROJECT_ROOT)
        summarise.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_episodes_the_gate_breakdown_did_not_replay_are_refused(self) -> None:
        argv = self._arm("base") + self._gates({"base": "0" * 64}) + ["--output", str(self.output)]
        with self.assertRaisesRegex(templates.TemplateAnalysisError, "gate breakdown"):
            self._main(argv)
        self.assertFalse(self.output.exists())

    def test_an_arm_the_gate_breakdown_never_saw_is_refused(self) -> None:
        argv = self._arm("base") + self._arm("sft") + ["--output", str(self.output)]
        argv += self._gates({"base": self._sha(self.root / "episodes-phase_b-base.jsonl")})
        with self.assertRaisesRegex(templates.TemplateAnalysisError, "'sft'"):
            self._main(argv)
        self.assertFalse(self.output.exists())

    def test_a_label_used_twice_is_refused(self) -> None:
        argv = self._arm("base") + self._arm("base") + ["--output", str(self.output)]
        with self.assertRaisesRegex(templates.TemplateAnalysisError, "once"):
            self._main(argv)
        self.assertFalse(self.output.exists())

    def test_a_summary_for_another_arm_is_refused(self) -> None:
        argv = self._arm("base") + ["--output", str(self.output)]
        argv[1] = "sft"
        with self.assertRaisesRegex(templates.TemplateAnalysisError, "not an executed 'sft' run"):
            self._main(argv)
        self.assertFalse(self.output.exists())

    def test_rows_that_do_not_reproduce_the_summary_are_refused(self) -> None:
        argv = self._arm("base") + ["--output", str(self.output)]
        episodes = Path(argv[3])
        lines = episodes.read_bytes().decode("utf-8").splitlines(keepends=True)
        episodes.write_bytes("".join(lines[:-1]).encode("utf-8"))
        with self.assertRaisesRegex(templates.TemplateAnalysisError, "reproduce"):
            self._main(argv)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
