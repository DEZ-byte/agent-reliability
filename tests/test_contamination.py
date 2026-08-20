from __future__ import annotations

import json
import unittest

from agent.gates import GateEngine, GateMode
from agent.parser import parse_tool_calls
from env.phase_a import (
    PhaseATask,
    answered_without_arithmetic,
    build_phase_a_registry,
    grade_episode,
)
from evaluation.contamination import (
    RecallProbe,
    expression_does_arithmetic,
    extract_final_number,
    recall_rate,
    score_recall,
)

TASK = PhaseATask(
    task_id="gsm8k:test:0",
    template_id="gsm8k:test:0",
    question="17 boxes of 23 pens.",
    gold_answer=391.0,
    source="gsm8k",
)


def run(raw: str):
    registry = build_phase_a_registry()
    engine = GateEngine.from_mapping({})
    return registry.execute(
        parse_tool_calls(raw), {}, gate_engine=engine, gate_mode=GateMode.ENFORCE
    )


def block(expression: str) -> str:
    payload = json.dumps(
        {"name": "calculator", "arguments": {"expression": expression}},
        separators=(",", ":"),
    )
    return f"<tool_call>{payload}</tool_call>"


class AnswerExtractionTests(unittest.TestCase):
    def test_takes_the_last_stated_number(self) -> None:
        self.assertEqual(extract_final_number("17 times 23 is 391"), 391.0)
        self.assertEqual(extract_final_number("first 12, finally **1,234**"), 1234.0)
        self.assertEqual(extract_final_number("the total is -5."), -5.0)

    def test_returns_none_when_no_number_is_stated(self) -> None:
        self.assertIsNone(extract_final_number("I cannot work this out."))
        self.assertIsNone(extract_final_number(""))


class ArithmeticDetectionTests(unittest.TestCase):
    """The check that separates computing from restating."""

    def test_real_arithmetic_is_recognised(self) -> None:
        for expression in ("17 * 23", "(48/2)+6", "-3 + 4", "min(3, 4)", "2 ** 5"):
            with self.subTest(expression=expression):
                self.assertTrue(expression_does_arithmetic(expression))

    def test_a_bare_answer_is_not_arithmetic(self) -> None:
        for expression in ("391", "391.0", "(391)", " 391 ", "1_000"):
            with self.subTest(expression=expression):
                self.assertFalse(expression_does_arithmetic(expression))

    def test_unparseable_input_is_not_arithmetic(self) -> None:
        self.assertFalse(expression_does_arithmetic("17 *"))
        self.assertFalse(expression_does_arithmetic(""))


class LaunderedAnswerTests(unittest.TestCase):
    def test_a_computed_answer_is_not_flagged(self) -> None:
        trace = run(block("17 * 23"))
        self.assertTrue(grade_episode(trace, TASK).correct)
        self.assertFalse(answered_without_arithmetic(trace))

    def test_a_recalled_answer_passed_through_the_tool_is_flagged(self) -> None:
        """Scores correct, and is recorded as having computed nothing."""

        trace = run(block("391"))
        self.assertTrue(grade_episode(trace, TASK).correct)
        self.assertTrue(answered_without_arithmetic(trace))

    def test_only_the_final_answering_call_is_judged(self) -> None:
        trace = run(block("2 + 2") + block("391"))
        self.assertTrue(answered_without_arithmetic(trace))

    def test_an_episode_with_no_successful_call_is_not_flagged(self) -> None:
        self.assertFalse(answered_without_arithmetic(run("no tools here")))


class RecallScoringTests(unittest.TestCase):
    def test_a_correct_no_tool_answer_counts_as_recall(self) -> None:
        probe = score_recall(
            task_id="gsm8k:test:0",
            gold_answer=391.0,
            completion="That works out to 391.",
            tolerance=1e-6,
        )
        self.assertTrue(probe.recalled)
        self.assertEqual(probe.extracted_answer, 391.0)

    def test_a_wrong_or_absent_answer_does_not(self) -> None:
        for completion in ("It is 390.", "I am not sure."):
            with self.subTest(completion=completion):
                probe = score_recall(
                    task_id="t",
                    gold_answer=391.0,
                    completion=completion,
                    tolerance=1e-6,
                )
                self.assertFalse(probe.recalled)

    def test_rate_is_none_for_an_empty_probe_set(self) -> None:
        self.assertIsNone(recall_rate([]))

    def test_rate_counts_recalls(self) -> None:
        probes = [
            RecallProbe(
                task_id=f"t{i}",
                gold_answer=1.0,
                completion="1" if hit else "2",
                extracted_answer=1.0 if hit else 2.0,
                recalled=hit,
            )
            for i, hit in enumerate([True, True, False, False])
        ]
        self.assertEqual(recall_rate(probes), 0.5)


if __name__ == "__main__":
    unittest.main()
