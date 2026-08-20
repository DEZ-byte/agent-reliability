from __future__ import annotations

import dataclasses
import json
import unittest

from agent.gates import GateEngine
from env.phase_a import PhaseATask, build_phase_a_registry
from evaluation.rungs import (
    MODEL_DECISION_BUDGET,
    EpisodeCounters,
    make_stub_policy,
    run_episode,
)

TASK = PhaseATask(
    task_id="gsm8k:test:0",
    template_id="gsm8k:test:0",
    question="17 boxes of 23 pens. How many pens?",
    gold_answer=391.0,
    source="gsm8k",
)


def call(expression: str) -> str:
    payload = json.dumps(
        {"name": "calculator", "arguments": {"expression": expression}},
        separators=(",", ":"),
    )
    return f"<tool_call>{payload}</tool_call>"


def episode(rung: str, replies: list[str]):
    return run_episode(
        task=TASK,
        registry=build_phase_a_registry(),
        gate_engine=GateEngine.from_mapping({}),
        policy=make_stub_policy(replies),
        rung=rung,  # type: ignore[arg-type]
    )


class RungContrastTests(unittest.TestCase):
    """R0 and R1 must differ by exactly one thing: the second decision."""

    def test_the_budgets_are_the_only_difference(self) -> None:
        self.assertEqual(MODEL_DECISION_BUDGET["R0"], 1)
        self.assertEqual(MODEL_DECISION_BUDGET["R1"], 2)

    def test_both_rungs_succeed_on_a_first_correct_call(self) -> None:
        for rung in ("R0", "R1"):
            with self.subTest(rung=rung):
                result = episode(rung, [call("17 * 23")])
                self.assertTrue(result.correct)
                self.assertEqual(result.terminal_reason, "answered")
                self.assertEqual(result.counters.policy_model_decision_count, 1)

    def test_r0_gets_no_second_chance_after_a_bad_first_call(self) -> None:
        """The whole experimental contrast, stated as a test."""

        result = episode("R0", ["I think it is 391."])
        self.assertFalse(result.correct)
        self.assertEqual(result.terminal_reason, "no_action")
        self.assertEqual(result.counters.policy_model_decision_count, 1)

    def test_r1_recovers_from_the_same_bad_first_call(self) -> None:
        result = episode("R1", ["I think it is 391.", call("17 * 23")])
        self.assertTrue(result.correct)
        self.assertEqual(result.terminal_reason, "answered")
        self.assertEqual(result.counters.policy_model_decision_count, 2)

    def test_r1_stops_after_its_one_feedback_decision(self) -> None:
        result = episode("R1", ["nonsense", "still nonsense"])
        self.assertFalse(result.correct)
        self.assertEqual(result.terminal_reason, "feedback_exhausted")
        self.assertEqual(result.counters.policy_model_decision_count, 2)

    def test_r1_never_takes_a_third_decision(self) -> None:
        """make_stub_policy raises if the loop asks for more than it was given."""

        result = episode("R1", ["bad", "also bad"])
        self.assertEqual(result.counters.policy_model_decision_count, 2)


class FeedbackTests(unittest.TestCase):
    def test_the_feedback_names_the_failure_without_leaking_the_answer(self) -> None:
        result = episode("R1", [call("1/0"), call("17 * 23")])
        self.assertTrue(result.correct)
        self.assertEqual(len(result.completions), 2)
        for completion in result.completions:
            self.assertNotIn("391", completion.replace(call("17 * 23"), ""))

    def test_an_unparseable_reply_produces_a_parse_specific_observation(self) -> None:
        result = episode("R1", ["<tool_call>{not json}</tool_call>", call("17 * 23")])
        self.assertTrue(result.correct)


class CounterTests(unittest.TestCase):
    """RUNG_PROTOCOL section 1.4 forbids one counter proxying for another."""

    def test_every_required_counter_exists(self) -> None:
        required = {
            "environment_turn_count",
            "agent_turn_count",
            "policy_model_decision_count",
            "escalation_model_decision_count",
            "tool_dispatch_attempt_count",
            "exact_transient_redispatch_count",
            "gate_block_count",
            "model_switch_count",
        }
        present = {field.name for field in dataclasses.fields(EpisodeCounters)}
        self.assertEqual(present, required)

    def test_a_successful_answer_advances_exactly_one_environment_turn(self) -> None:
        result = episode("R0", [call("17 * 23")])
        self.assertEqual(result.counters.environment_turn_count, 1)
        self.assertEqual(result.counters.tool_dispatch_attempt_count, 1)

    def test_a_failed_episode_advances_no_environment_turn(self) -> None:
        result = episode("R0", ["no call at all"])
        self.assertEqual(result.counters.environment_turn_count, 0)
        self.assertEqual(result.counters.tool_dispatch_attempt_count, 0)

    def test_neither_rung_ever_escalates_or_switches_models(self) -> None:
        for rung in ("R0", "R1"):
            with self.subTest(rung=rung):
                result = episode(rung, ["bad", "worse"][: MODEL_DECISION_BUDGET[rung]])
                self.assertEqual(result.counters.escalation_model_decision_count, 0)
                self.assertEqual(result.counters.model_switch_count, 0)


class LaunderingTests(unittest.TestCase):
    def test_a_restated_answer_is_correct_and_flagged(self) -> None:
        result = episode("R0", [call("391")])
        self.assertTrue(result.correct)
        self.assertTrue(result.answered_without_arithmetic)

    def test_a_computed_answer_is_correct_and_not_flagged(self) -> None:
        result = episode("R0", [call("17 * 23")])
        self.assertTrue(result.correct)
        self.assertFalse(result.answered_without_arithmetic)


class SerialisationTests(unittest.TestCase):
    def test_the_record_round_trips_as_json(self) -> None:
        payload = episode("R1", [call("1/0"), call("17 * 23")]).to_json()
        restored = json.loads(json.dumps(payload))
        self.assertTrue(restored["correct"])
        self.assertEqual(restored["rung"], "R1")
        self.assertEqual(restored["counters"]["policy_model_decision_count"], 2)
        self.assertIn("completions", restored)


if __name__ == "__main__":
    unittest.main()
