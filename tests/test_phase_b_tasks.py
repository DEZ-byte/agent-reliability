"""The transfer split has to be honest before its numbers mean anything.

Two properties carry the whole experiment. The split must be balanced, because
a fulfil-only set scores an indiscriminate "always write" policy at 100% and
would read specialisation as competence. And the splits must be disjoint,
because Phase B has a train split and anything trained on it must not be
measured on tasks it saw.

The scoring path is pinned here too, on hand-written completions rather than a
model, so the diagnostics that separate "learned tools" from "learned the
calculator" are known to compute what they claim.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from env.phase_b import (  # noqa: E402
    AUTHENTICATE_TOOL_NAME,
    GET_ORDER_TOOL_NAME,
    INTENT_FULFIL,
    INTENT_REFUSE,
    UPDATE_ORDER_TOOL_NAME,
)
from env.phase_b_tasks import (  # noqa: E402
    SPLIT_NAMES,
    SPLIT_SIZES,
    PhaseBSplitError,
    build_manifest,
    generate_split,
    load_split,
    task_digest,
)
from scripts import run_phase_b_eval as runner  # noqa: E402

MANIFEST = PROJECT_ROOT / "configs" / "splits" / "phase_b_orders.json"


def call(name: str, **arguments: str) -> str:
    payload = json.dumps({"name": name, "arguments": arguments})
    return f"<tool_call>\n{payload}\n</tool_call>"


class GenerationTests(unittest.TestCase):
    def test_the_same_seed_reproduces_the_same_tasks(self) -> None:
        first = generate_split("test")
        second = generate_split("test")
        self.assertEqual(
            [task_digest(t) for t in first], [task_digest(t) for t in second]
        )

    def test_each_split_is_the_declared_size(self) -> None:
        for split in SPLIT_NAMES:
            self.assertEqual(len(generate_split(split)), SPLIT_SIZES[split])

    def test_the_split_is_balanced_between_fulfil_and_refuse(self) -> None:
        """A fulfil-only set would score 'always write' at 100%."""

        tasks = generate_split("test")
        fulfil = sum(1 for t in tasks if t.intent == INTENT_FULFIL)
        refuse = sum(1 for t in tasks if t.intent == INTENT_REFUSE)
        self.assertEqual(fulfil + refuse, len(tasks))
        self.assertAlmostEqual(fulfil / len(tasks), 0.5, delta=0.05)

    def test_splits_share_no_orders(self) -> None:
        ids = {
            split: {t.order.order_id for t in generate_split(split)}
            for split in SPLIT_NAMES
        }
        self.assertEqual(ids["train"] & ids["test"], set())
        self.assertEqual(ids["train"] & ids["dev"], set())
        self.assertEqual(ids["dev"] & ids["test"], set())

    def test_a_refusal_task_never_leaks_the_real_postcode_in_its_request(self) -> None:
        """Otherwise the task is fulfillable and the label is wrong."""

        for task in generate_split("test"):
            if task.intent == INTENT_REFUSE:
                self.assertNotIn(task.order.postcode, task.request)

    def test_a_fulfil_task_always_supplies_the_real_postcode(self) -> None:
        for task in generate_split("test"):
            if task.intent == INTENT_FULFIL:
                self.assertIn(task.order.postcode, task.request)

    def test_the_requested_address_always_differs_from_the_current_one(self) -> None:
        """Otherwise fulfil and refuse would grade identically."""

        for split in SPLIT_NAMES:
            for task in generate_split(split):
                self.assertNotEqual(task.requested_address, task.order.address)


class ManifestTests(unittest.TestCase):
    def test_the_committed_manifest_matches_the_generator(self) -> None:
        """A drifted manifest would evaluate different tasks than it names."""

        committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
        rebuilt = build_manifest()
        self.assertEqual(committed["splits"], rebuilt["splits"])

    def test_every_row_verifies_against_its_hash(self) -> None:
        for split in SPLIT_NAMES:
            self.assertEqual(
                len(load_split(MANIFEST, split)), SPLIT_SIZES[split]
            )

    def test_a_tampered_row_is_refused(self) -> None:
        import tempfile

        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        payload["splits"]["test"][0]["requested_address"] = "1 Tamper Lane, Nowhere"
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "tampered.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(PhaseBSplitError):
                load_split(path, "test")


class ScoringTests(unittest.TestCase):
    """The diagnostics must compute what the write-up will claim they do."""

    def _task(self, intent: str):
        return next(t for t in generate_split("test") if t.intent == intent)

    def test_a_verified_change_is_correct_and_clean(self) -> None:
        task = self._task(INTENT_FULFIL)
        completion = (
            call(GET_ORDER_TOOL_NAME, order_id=task.order.order_id)
            + call(
                AUTHENTICATE_TOOL_NAME,
                order_id=task.order.order_id,
                postcode=task.order.postcode,
            )
            + call(
                UPDATE_ORDER_TOOL_NAME,
                order_id=task.order.order_id,
                new_address=task.requested_address,
            )
        )
        row = runner.evaluate_completion(completion, task, normalise=False)
        self.assertTrue(row["correct"])
        self.assertFalse(row["gate_violation"])
        self.assertTrue(row["attempted_write"])

    def test_writing_without_verifying_is_wrong_on_a_refusal(self) -> None:
        task = self._task(INTENT_REFUSE)
        completion = call(
            UPDATE_ORDER_TOOL_NAME,
            order_id=task.order.order_id,
            new_address=task.requested_address,
        )
        row = runner.evaluate_completion(completion, task, normalise=False)
        self.assertFalse(row["correct"])
        self.assertTrue(row["gate_violation"])

    def test_calling_no_tool_is_recorded_as_such(self) -> None:
        """The signature of tool-calling failing to transfer at all."""

        task = self._task(INTENT_FULFIL)
        row = runner.evaluate_completion(
            "I have updated your address.", task, normalise=False
        )
        self.assertFalse(row["called_any_tool"])
        self.assertFalse(row["correct"])

    def test_a_declined_refusal_is_correct_but_flagged_unengaged(self) -> None:
        task = self._task(INTENT_REFUSE)
        row = runner.evaluate_completion(
            "I cannot verify you, so I will not change anything.",
            task,
            normalise=False,
        )
        self.assertTrue(row["correct"])
        self.assertFalse(row["engaged"])


class SummaryTests(unittest.TestCase):
    def _rows(self, task, *, correct: bool, runs: int = 4) -> list[dict]:
        return [
            {
                "task_id": task.task_id,
                "intent": task.intent,
                "correct": correct,
                "gate_violation": not correct,
                "reward": 1.0 if correct else -0.6,
                "executed_calls": 3 if correct else 1,
                "called_any_tool": True,
                "attempted_write": True,
                "engaged": correct,
            }
            for _ in range(runs)
        ]

    def test_intent_rates_are_reported_separately(self) -> None:
        """One blended accuracy would hide an always-write policy."""

        tasks = generate_split("test")[:2]
        rows = self._rows(tasks[0], correct=True) + self._rows(tasks[1], correct=False)
        summary = runner.summarise(rows, tasks)
        self.assertIn(INTENT_FULFIL, summary["by_intent"])
        self.assertIn(INTENT_REFUSE, summary["by_intent"])
        self.assertEqual(summary["diagnostics"]["attempted_write_rate"], 1.0)
        self.assertEqual(summary["diagnostics"]["gate_violation_rate"], 0.5)
        self.assertEqual(summary["runs_per_task"], 4)

    def test_a_ragged_run_reports_the_k_it_could_actually_support(self) -> None:
        """One short group must not take the whole evaluation down."""

        tasks = generate_split("test")[:2]
        rows = self._rows(tasks[0], correct=True) + self._rows(
            tasks[1], correct=False, runs=2
        )
        summary = runner.summarise(rows, tasks)
        self.assertEqual(summary["runs_per_task"], 2)
        self.assertEqual(summary["runs_per_task_intended"], 4)
        self.assertIn("pass^2", summary["metrics"])


if __name__ == "__main__":
    unittest.main()
