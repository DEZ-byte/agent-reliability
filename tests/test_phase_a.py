from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from agent.gates import GateEngine, GateMode
from agent.parser import parse_tool_calls
from env.phase_a import (
    ANSWER_TOLERANCE,
    CalculatorArgs,
    PhaseATask,
    build_phase_a_registry,
    evaluate_expression,
    executed_answers,
    grade_episode,
    parse_gsm8k_answer,
)
from training.rewards import score_episode

TASK = PhaseATask(
    task_id="gsm8k:test:0",
    template_id="gsm8k:test:0",
    question="A shop sells 17 boxes with 23 pens each. How many pens?",
    gold_answer=391.0,
    source="gsm8k",
)


def block(expression: str) -> str:
    payload = json.dumps(
        {"name": "calculator", "arguments": {"expression": expression}},
        separators=(",", ":"),
    )
    return f"<tool_call>{payload}</tool_call>"


def run(raw: str):
    registry = build_phase_a_registry()
    engine = GateEngine.from_mapping({})
    trace = registry.execute(
        parse_tool_calls(raw), {}, gate_engine=engine, gate_mode=GateMode.ENFORCE
    )
    return trace, engine


class GoldAnswerParsingTests(unittest.TestCase):
    def test_reads_the_marker_including_thousands_separators(self) -> None:
        self.assertEqual(parse_gsm8k_answer("blah blah\n#### 391"), 391.0)
        self.assertEqual(parse_gsm8k_answer("#### 1,234"), 1234.0)
        self.assertEqual(parse_gsm8k_answer("#### -5"), -5.0)

    def test_a_missing_marker_raises_rather_than_guessing(self) -> None:
        with self.assertRaises(ValueError):
            parse_gsm8k_answer("the answer is 391")


class CalculatorTests(unittest.TestCase):
    def test_arithmetic_runs_in_the_sandbox(self) -> None:
        self.assertEqual(evaluate_expression("17 * 23"), 391.0)
        self.assertAlmostEqual(evaluate_expression("(48/2)+6"), 30.0)

    def test_non_numeric_and_unsafe_expressions_are_rejected(self) -> None:
        for expression in ("'hello'", "import os", "__import__"):
            with self.subTest(expression=expression):
                with self.assertRaises(Exception):
                    evaluate_expression(expression)

    def test_argument_schema_is_strict(self) -> None:
        with self.assertRaises(ValidationError):
            CalculatorArgs(expression="1+1", extra="no")
        with self.assertRaises(ValidationError):
            CalculatorArgs(expression="")


class GradingTests(unittest.TestCase):
    """The rule that makes this environment worth building."""

    def test_a_correct_executed_call_scores_correct(self) -> None:
        trace, _ = run(block("17 * 23"))
        self.assertEqual(executed_answers(trace), [391.0])
        self.assertTrue(grade_episode(trace, TASK).correct)

    def test_the_right_number_in_prose_scores_zero(self) -> None:
        trace, engine = run("The answer is 391. #### 391")

        self.assertEqual(executed_answers(trace), [])
        outcome = grade_episode(trace, TASK)
        self.assertFalse(outcome.correct)

        reward = score_episode(
            trace, outcome, tool_required=True, gate_engine=engine
        )
        self.assertEqual(reward.accuracy, 0.0)
        self.assertEqual(reward.efficiency, -0.3)

    def test_a_wrong_executed_call_scores_incorrect(self) -> None:
        trace, _ = run(block("17 + 23"))
        self.assertFalse(grade_episode(trace, TASK).correct)

    def test_the_last_executed_result_is_the_answer(self) -> None:
        trace, _ = run(block("17 + 23") + block("17 * 23"))
        self.assertEqual(executed_answers(trace), [40.0, 391.0])
        self.assertTrue(grade_episode(trace, TASK).correct)

    def test_a_failed_call_contributes_no_answer(self) -> None:
        trace, _ = run(block("1/0"))
        self.assertEqual(executed_answers(trace), [])
        self.assertFalse(grade_episode(trace, TASK).correct)

    def test_tolerance_accepts_float_paths_to_an_integer_answer(self) -> None:
        trace, _ = run(block("782 / 2"))
        self.assertTrue(grade_episode(trace, TASK).correct)
        self.assertLess(ANSWER_TOLERANCE, 1e-3)

    def test_a_correct_episode_earns_format_and_accuracy(self) -> None:
        trace, engine = run(block("17 * 23"))
        reward = score_episode(
            trace,
            grade_episode(trace, TASK),
            tool_required=True,
            gate_engine=engine,
        )
        self.assertEqual(reward.accuracy, 1.0)
        self.assertEqual(reward.format, 0.2)
        self.assertEqual(reward.efficiency, -0.05)
        self.assertEqual(reward.total, 1.15)


class TaskModelTests(unittest.TestCase):
    def test_task_is_frozen_and_strict(self) -> None:
        with self.assertRaises(ValidationError):
            PhaseATask(
                task_id="x",
                template_id="x",
                question="q",
                gold_answer=1.0,
                source="gsm8k",
                unexpected=True,
            )
        with self.assertRaises(ValidationError):
            TASK.task_id = "changed"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
