"""The GRPO reward is the evaluator's reward, and its blind spot is measured.

Section 7.0 bans substring rewards, so every candidate is executed. These tests
pin the values that execution produces, because a policy-gradient method
optimises whatever the reward actually says rather than what it was meant to
say.

One of those values is uncomfortable and correct: a call that restates a
remembered answer scores exactly what genuine work scores. Section 7.0 chose to
measure that rather than penalise it. GRPO is the sharpest version of that
pressure, since laundering is the cheapest route to full accuracy, so the tests
below pin both the equal reward and the fact that it is detected.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.gates import GateEngine  # noqa: E402
from env.phase_a import build_phase_a_registry  # noqa: E402
from training.grpo_reward import (  # noqa: E402
    CompletionScore,
    group_health,
    score_completion,
)

QUESTION = "Ken had 2 boxes and twice as many bags. How many in total?"
GOLD = 6.0


def call(expression: str) -> str:
    return (
        '<tool_call>\n{"name": "calculator", "arguments": {"expression": "'
        + expression
        + '"}}\n</tool_call>'
    )


def score(text: str, gold: float = GOLD):
    return score_completion(
        text,
        gold_answer=gold,
        question=QUESTION,
        registry=build_phase_a_registry(),
        gate_engine=GateEngine.from_mapping({}),
    )


class RewardValueTests(unittest.TestCase):
    """Pinned so a change to the reward surface cannot pass unnoticed."""

    def test_correct_work_earns_accuracy_and_format(self) -> None:
        result = score(call("2 + 2*2"))
        self.assertTrue(result.correct)
        self.assertEqual(result.accuracy, 1.0)
        self.assertEqual(result.format, 0.2)
        self.assertAlmostEqual(result.total, 1.15)

    def test_a_well_formed_call_with_the_wrong_value_keeps_the_format_credit(self) -> None:
        """The dominant failure in this task: right shape, wrong arithmetic."""

        result = score(call("2+2"))
        self.assertFalse(result.correct)
        self.assertEqual(result.accuracy, 0.0)
        self.assertEqual(result.format, 0.2)

    def test_answering_in_prose_is_penalised_for_not_acting(self) -> None:
        result = score("The answer is 6.")
        self.assertEqual(result.executed_calls, 0)
        self.assertAlmostEqual(result.total, -0.3)

    def test_a_malformed_block_scores_below_emitting_nothing(self) -> None:
        """Otherwise a broken call would be a cheap alternative to silence."""

        malformed = score("<tool_call>{oops}</tool_call>")
        silent = score("The answer is 6.")
        self.assertLess(malformed.total, silent.total)


class LaunderingTests(unittest.TestCase):
    def test_a_laundered_call_earns_the_same_reward_as_genuine_work(self) -> None:
        """Section 7.0's deliberate choice, pinned because GRPO will find it.

        If this ever stops being true the reward has been changed, which is a
        decision that needs recording rather than a quiet improvement.
        """

        genuine = score(call("2 + 2*2"))
        laundered = score(call("6"))
        self.assertEqual(genuine.total, laundered.total)
        self.assertTrue(laundered.correct)

    def test_laundering_is_detected_even_though_it_is_not_penalised(self) -> None:
        self.assertFalse(score(call("2 + 2*2")).laundered)
        self.assertTrue(score(call("6")).laundered)

    def test_the_decorated_form_is_detected_too(self) -> None:
        """`6 + 0` computes nothing and parses as arithmetic."""

        self.assertTrue(score(call("6 + 0")).laundered)


class GroupHealthTests(unittest.TestCase):
    """Section 7.3: a group with no spread contributes no gradient at all."""

    def _fake(self, total: float, **kwargs) -> CompletionScore:
        base = dict(
            total=total,
            accuracy=0.0,
            format=0.2,
            gate=0.0,
            efficiency=-0.05,
            correct=False,
            executed_calls=1,
            laundered=False,
        )
        base.update(kwargs)
        return CompletionScore(**base)

    def test_identical_candidates_are_flagged_as_zero_variance(self) -> None:
        health = group_health([self._fake(1.15) for _ in range(8)])
        self.assertTrue(health["zero_variance"])
        self.assertEqual(health["std"]["total"], 0.0)

    def test_a_mixed_group_is_not_flagged(self) -> None:
        group = [self._fake(1.15), self._fake(0.15), self._fake(-0.3)]
        self.assertFalse(group_health(group)["zero_variance"])

    def test_a_component_constant_across_the_group_shows_zero_spread(self) -> None:
        """The point of reporting components separately.

        Format can be identical across every candidate while total still
        varies, and a constant component moves the group mean without moving
        the advantage, so it teaches nothing.
        """

        group = [self._fake(1.15), self._fake(0.15)]
        health = group_health(group)
        self.assertEqual(health["std"]["format"], 0.0)
        self.assertGreater(health["std"]["total"], 0.0)

    def test_laundered_and_correct_fractions_are_reported(self) -> None:
        group = [
            self._fake(1.15, correct=True, laundered=True),
            self._fake(1.15, correct=True),
            self._fake(0.15),
            self._fake(0.15),
        ]
        health = group_health(group)
        self.assertEqual(health["correct_fraction"], 0.5)
        self.assertEqual(health["laundered_fraction"], 0.25)


if __name__ == "__main__":
    unittest.main()
