"""The laundering filter must reject decoration and keep genuine work.

D-062 recorded that execution-backed grading does not stop a model solving the
problem in its head and handing the answer to the calculator. The existing
arithmetic check catches `calculator("391")`. It does not catch `391 + 0`,
which parses as arithmetic and computes nothing, so the training set would
fill with answer-first reconstructions.

Two failure directions matter equally here and both are tested. A filter that
misses decoration poisons the data. A filter that rejects real work shrinks the
dataset and biases it toward whatever it happens to accept, which is harder to
notice because the remaining rows all look fine.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.retention import (  # noqa: E402
    laundering_verdict,
    numbers_in_text,
    numeric_literals,
    rejection_counts,
)

QUESTION = (
    "Ken put 2 pounds of jelly beans in a box and twice as many pounds of "
    "brownies. How many pounds is the box?"
)
GOLD = 6.0


def judge(expression: str, *, question: str = QUESTION, gold: float = GOLD):
    return laundering_verdict(
        expression=expression, question=question, gold_answer=gold
    )


class LaunderingRejectionTests(unittest.TestCase):
    def test_a_bare_answer_is_rejected(self) -> None:
        self.assertEqual(judge("6").reason, "no_arithmetic")

    def test_a_parenthesised_answer_is_rejected(self) -> None:
        self.assertEqual(judge("(6)").reason, "no_arithmetic")

    def test_adding_zero_does_not_launder_the_answer_past_the_filter(self) -> None:
        """The decorated form. One character of effort defeats the old check."""

        self.assertEqual(judge("6 + 0").reason, "gold_answer_is_a_literal")

    def test_multiplying_by_one_does_not_launder_the_answer_either(self) -> None:
        self.assertEqual(judge("6 * 1").reason, "gold_answer_is_a_literal")

    def test_a_signed_answer_is_rejected_as_a_single_operand(self) -> None:
        self.assertEqual(judge("-6", gold=-6.0).reason, "single_operand")

    def test_an_expression_built_from_invented_numbers_is_rejected(self) -> None:
        """Nothing in the question mentions 97 or 41."""

        self.assertEqual(
            judge("97 - 41", gold=56.0).reason, "literals_absent_from_question"
        )


class GenuineWorkTests(unittest.TestCase):
    """The other direction: real computation must survive."""

    def test_a_repeated_operand_is_not_mistaken_for_a_restatement(self) -> None:
        """`2 + 2*2` has one distinct value and three operands.

        A distinct-value rule rejects this correct expression, which is why the
        rule counts occurrences.
        """

        verdict = judge("2 + 2*2")
        self.assertFalse(verdict.laundered)
        self.assertIsNone(verdict.reason)

    def test_the_answer_may_appear_when_the_question_also_contains_it(self) -> None:
        """Then the literal is evidence of nothing, so it cannot condemn."""

        question = "He had 6 apples and 2 pears. How many apples?"
        verdict = judge("6 * 2 / 2", question=question, gold=6.0)
        self.assertTrue(verdict.gold_appears_in_question)
        self.assertFalse(verdict.laundered)

    def test_thousands_separators_in_prose_still_match_plain_digits(self) -> None:
        """`1,000` in the question must match `1000` in the expression."""

        question = "She earns 1,000 dollars a month for 12 months. Total?"
        verdict = judge("1000 * 12", question=question, gold=12000.0)
        self.assertFalse(verdict.laundered)


class LiteralExtractionTests(unittest.TestCase):
    def test_literals_are_read_from_the_parsed_tree(self) -> None:
        """Order follows tree traversal and is not part of the contract."""

        self.assertEqual(sorted(numeric_literals("2 + 3*4")), [2.0, 3.0, 4.0])

    def test_a_negative_constant_is_counted_once_with_its_sign(self) -> None:
        """The positive twin the tree also carries must not be counted again."""

        self.assertEqual(sorted(numeric_literals("-5 + 2")), [-5.0, 2.0])

    def test_an_unparseable_expression_yields_no_literals(self) -> None:
        self.assertEqual(numeric_literals("4 schools * 2 teams"), ())

    def test_numbers_in_text_reads_decimals_and_separators(self) -> None:
        self.assertEqual(
            numbers_in_text("He paid 1,250 and 3.5 and 7."),
            (1250.0, 3.5, 7.0),
        )


class ReportingTests(unittest.TestCase):
    def test_rejection_counts_name_every_rule_that_fired(self) -> None:
        """Rejections are reported per rule so a filter change is visible."""

        counts = rejection_counts(
            [judge("6"), judge("6 + 0"), judge("2 + 2*2")]
        )
        self.assertEqual(counts["no_arithmetic"], 1)
        self.assertEqual(counts["gold_answer_is_a_literal"], 1)
        self.assertEqual(counts["retained"], 1)


if __name__ == "__main__":
    unittest.main()
