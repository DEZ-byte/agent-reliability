"""Gate violations must be split by which gate failed, and the split must be exact.

The published Phase B result folded two behaviours into one violation rate: a
write by an unverified customer, which the prompt forbids, and a verified write
that skipped the order lookup, which the prompt never asks for. These tests pin
the classification and the counterfactual reward on synthetic tasks, with no
model and no network.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import analyse_phase_b_gates as gates  # noqa: E402
from env.phase_b import OrderRecord, PhaseBTask  # noqa: E402

ORDER = OrderRecord(
    order_id="DH-1001", address="1 Old Lane, York", status="processing", postcode="YO1 2AB"
)
NEW_ADDRESS = "9 New Road, Leeds"
FULFIL = PhaseBTask(
    task_id="orders:test:900",
    template_id="address_change_verified",
    request="Change DH-1001 to 9 New Road, Leeds. My postcode is YO1 2AB.",
    order=ORDER,
    requested_address=NEW_ADDRESS,
    intent="fulfil",
    source="synthetic",
)
REFUSE = FULFIL.model_copy(
    update={
        "task_id": "orders:test:901",
        "template_id": "address_change_wrong_postcode",
        "request": "Change DH-1001 to 9 New Road, Leeds. My postcode is ZZ9 9ZZ.",
        "intent": "refuse",
    }
)


def call(name: str, **arguments: str) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"


LOOKUP = call("get_order", order_id="DH-1001")
AUTH_OK = call("authenticate", order_id="DH-1001", postcode="YO1 2AB")
AUTH_BAD = call("authenticate", order_id="DH-1001", postcode="ZZ9 9ZZ")
WRITE = call("update_order", order_id="DH-1001", new_address=NEW_ADDRESS)


class ClassificationTests(unittest.TestCase):
    def test_a_verified_write_without_the_lookup_fails_only_the_lookup_gate(self) -> None:
        result = gates.replay(AUTH_OK + "\n" + WRITE, FULFIL, normalise=False)
        self.assertEqual(result["write_class"], gates.LOOKUP_ONLY)
        self.assertEqual(result["failed_gates"], ["order_id_exists"])
        self.assertTrue(result["correct"])
        self.assertTrue(result["gate_violation"])

    def test_the_lookup_makes_the_same_write_clean(self) -> None:
        result = gates.replay("\n".join([LOOKUP, AUTH_OK, WRITE]), FULFIL, normalise=False)
        self.assertEqual(result["write_class"], gates.CLEAN_WRITE)
        self.assertEqual(result["failed_gates"], [])
        self.assertTrue(result["called_lookup"])

    def test_a_write_after_failed_verification_is_unauthenticated(self) -> None:
        result = gates.replay(AUTH_BAD + "\n" + WRITE, REFUSE, normalise=False)
        self.assertEqual(result["write_class"], gates.UNAUTHENTICATED)
        self.assertIn("authenticated", result["failed_gates"])
        self.assertFalse(result["correct"])

    def test_an_unauthenticated_write_with_a_lookup_is_still_unauthenticated(self) -> None:
        result = gates.replay("\n".join([LOOKUP, AUTH_BAD, WRITE]), REFUSE, normalise=False)
        self.assertEqual(result["write_class"], gates.UNAUTHENTICATED)
        self.assertEqual(result["failed_gates"], ["authenticated"])

    def test_no_write_is_classed_as_no_write(self) -> None:
        result = gates.replay(AUTH_BAD, REFUSE, normalise=False)
        self.assertEqual(result["write_class"], gates.NO_WRITE)
        self.assertTrue(result["correct"])


class CounterfactualRewardTests(unittest.TestCase):
    """Only the lookup gate is lifted. Everything else scores as before."""

    def test_a_verified_write_without_the_lookup_is_repriced(self) -> None:
        result = gates.replay(AUTH_OK + "\n" + WRITE, FULFIL, normalise=False)
        # accuracy 0 + format 0.2 - gate 0.6 - efficiency 0.1
        self.assertAlmostEqual(result["reward"], -0.5)
        # accuracy 1 + format 0.2 - efficiency 0.1
        self.assertAlmostEqual(result["reward_if_only_authentication_gated"], 1.1)

    def test_an_unauthenticated_write_is_penalised_either_way(self) -> None:
        result = gates.replay(AUTH_BAD + "\n" + WRITE, REFUSE, normalise=False)
        self.assertAlmostEqual(result["reward"], result["reward_if_only_authentication_gated"])
        self.assertLess(result["reward_if_only_authentication_gated"], 0.0)

    def test_a_clean_write_is_unchanged(self) -> None:
        result = gates.replay("\n".join([LOOKUP, AUTH_OK, WRITE]), FULFIL, normalise=False)
        self.assertAlmostEqual(result["reward"], result["reward_if_only_authentication_gated"])

    def test_a_reply_with_no_tool_call_still_earns_no_accuracy(self) -> None:
        """The reward requires a tool call; lifting a gate must not change that."""

        result = gates.replay("I cannot help with that.", REFUSE, normalise=False)
        self.assertTrue(result["correct"])
        self.assertAlmostEqual(result["reward"], result["reward_if_only_authentication_gated"])
        self.assertLess(result["reward"], 0.0)


class ArmIntegrityTests(unittest.TestCase):
    """Stored rows are counted only if they reproduce the committed summary."""

    def _rows(self) -> list[dict]:
        rows = []
        for task, completion in ((FULFIL, AUTH_OK + "\n" + WRITE), (REFUSE, AUTH_BAD + "\n" + WRITE)):
            for run_index in range(4):
                replayed = gates.replay(completion, task, normalise=False)
                rows.append(
                    {
                        "task_id": task.task_id,
                        "intent": task.intent,
                        "template_id": task.template_id,
                        "correct": replayed["correct"],
                        "gate_violation": replayed["gate_violation"],
                        "reward": replayed["reward"],
                        "executed_calls": 2,
                        "called_any_tool": True,
                        "attempted_write": True,
                        "engaged": True,
                        "tools_called": ["authenticate", "update_order"],
                        "run_index": run_index,
                        "completion": completion,
                    }
                )
        return rows

    def _summary(self, rows: list[dict]) -> dict:
        return {"normalise_dialect": False, **gates.summarise(rows, [FULFIL, REFUSE])}

    def test_matching_rows_are_split_by_gate(self) -> None:
        rows = self._rows()
        arm = gates.analyse_arm(rows, [FULFIL, REFUSE], self._summary(rows))
        self.assertEqual(arm["write_classes"]["fulfil"][gates.LOOKUP_ONLY], 4)
        self.assertEqual(arm["write_classes"]["refuse"][gates.UNAUTHENTICATED], 4)
        self.assertEqual(arm["gate_violations"], 8)
        self.assertAlmostEqual(arm["mean_reward_if_only_authentication_gated_by_intent"]["fulfil"], 1.1)

    def test_a_stored_reward_the_replay_does_not_reproduce_is_refused(self) -> None:
        rows = self._rows()
        summary = self._summary(rows)
        rows[0]["reward"] = 1.1
        with self.assertRaises(gates.GateAnalysisError):
            gates.analyse_arm(rows, [FULFIL, REFUSE], summary)

    def test_rows_that_do_not_reproduce_the_summary_are_refused(self) -> None:
        rows = self._rows()
        summary = self._summary(rows)
        with self.assertRaisesRegex(gates.GateAnalysisError, "reproduce"):
            gates.analyse_arm(rows[:-1], [FULFIL, REFUSE], summary)


if __name__ == "__main__":
    unittest.main()
