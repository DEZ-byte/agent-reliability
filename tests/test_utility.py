"""The utility probe must not confuse "answered badly" with "knows nothing".

This is the measurement that decides whether fine-tuning cost the model
anything, so its failure modes are the ones that matter. If answer extraction
is too strict, a model that says "The answer is (B)." scores as ignorant and
the probe reports damage that did not happen. If it is too loose, a stray
capital letter in a sentence becomes an answer and the probe reports knowledge
that is not there.

The second measurement is the interesting one. Nothing in this benchmark offers
a tool, so a tool call here is a habit that escaped its training context. That
is what specialisation looks like from the outside, and it is pinned separately
from accuracy because the two can move in opposite directions.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.utility import (  # noqa: E402
    emitted_tool_call,
    extract_choice,
    score_completion,
    summarise,
)


class ExtractionTests(unittest.TestCase):
    """Generous about phrasing, strict about guessing."""

    def test_a_bare_letter_is_read(self) -> None:
        self.assertEqual(extract_choice("B"), "B")

    def test_the_common_phrasings_are_read(self) -> None:
        for text in (
            "The answer is B.",
            "The answer is (B)",
            "Answer: B",
            "answer is **B**",
            "B) Paris",
            "B. Paris",
            "The correct option is B",
        ):
            with self.subTest(text=text):
                self.assertEqual(extract_choice(text), "B", text)

    def test_lowercase_is_read(self) -> None:
        self.assertEqual(extract_choice("the answer is b"), "B")

    def test_a_stated_answer_beats_a_letter_mentioned_while_reasoning(self) -> None:
        """Otherwise the probe scores the model's first thought, not its answer."""

        text = "A looks plausible at first, but on reflection the answer is C."
        self.assertEqual(extract_choice(text), "C")

    def test_prose_with_no_choice_extracts_nothing(self) -> None:
        """None, never a guess: unreadable is not the same as wrong."""

        self.assertIsNone(extract_choice("I am not sure about this question."))

    def test_an_empty_completion_extracts_nothing(self) -> None:
        self.assertIsNone(extract_choice(""))

    def test_a_letter_outside_the_range_is_not_accepted(self) -> None:
        self.assertIsNone(extract_choice("The answer is E."))


class ToolHabitTests(unittest.TestCase):
    """No tool is offered here, so any tool call is a habit out of context."""

    def test_a_qwen_style_call_is_detected(self) -> None:
        self.assertTrue(
            emitted_tool_call('<tool_call>\n{"name": "calculator"}\n</tool_call>')
        )

    def test_a_bare_json_call_is_detected(self) -> None:
        """Llama's dialect, and any half-written call, still shows the habit."""

        self.assertTrue(emitted_tool_call('{"name": "calculator", "arguments": {}}'))

    def test_an_unclosed_call_still_counts(self) -> None:
        self.assertTrue(emitted_tool_call("<tool_call>{oops"))

    def test_a_plain_answer_is_not_a_tool_call(self) -> None:
        self.assertFalse(emitted_tool_call("The answer is B."))

    def test_the_word_arguments_in_prose_is_not_a_tool_call(self) -> None:
        """Matching on a bare word would inflate the rate on any essay answer."""

        self.assertFalse(
            emitted_tool_call("Both arguments are valid, but the answer is B.")
        )


class ScoringTests(unittest.TestCase):
    def test_the_right_letter_scores_correct(self) -> None:
        self.assertTrue(score_completion("The answer is C.", gold_index=2).correct)

    def test_the_wrong_letter_scores_incorrect(self) -> None:
        self.assertFalse(score_completion("The answer is A.", gold_index=2).correct)

    def test_an_unreadable_answer_is_wrong_but_flagged(self) -> None:
        """Both facts are kept: it did not score, and it did not answer."""

        score = score_completion("Hmm, tricky.", gold_index=2)
        self.assertFalse(score.correct)
        self.assertTrue(score.extraction_failed)

    def test_a_gold_index_outside_the_choices_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            score_completion("B", gold_index=9)


class SummaryTests(unittest.TestCase):
    def _scores(self):
        return [
            score_completion("The answer is A.", gold_index=0),
            score_completion("The answer is B.", gold_index=0),
            score_completion("<tool_call>{...}</tool_call>", gold_index=0),
            score_completion("no idea", gold_index=0),
        ]

    def test_accuracy_counts_only_correct_letters(self) -> None:
        self.assertEqual(summarise(self._scores())["accuracy"], 0.25)

    def test_the_tool_habit_is_reported_separately_from_accuracy(self) -> None:
        """The case that matters: knowledge intact, answer format collapsed.

        Every answer below is correct *and* wrapped in a tool call. A probe
        that folded the two together would report a healthy model.
        """

        scores = [
            score_completion(
                '<tool_call>{"name": "x"}</tool_call> The answer is A.',
                gold_index=0,
            )
            for _ in range(4)
        ]
        summary = summarise(scores)
        self.assertEqual(summary["accuracy"], 1.0)
        self.assertEqual(summary["tool_call_rate"], 1.0)
        self.assertEqual(summarise(self._scores())["tool_call_rate"], 0.25)

    def test_extraction_failures_are_reported_so_they_cannot_hide(self) -> None:
        """A collapse in answer format would otherwise read as lost knowledge."""

        self.assertEqual(summarise(self._scores())["extraction_failure_rate"], 0.5)

    def test_an_empty_run_summarises_to_nothing(self) -> None:
        self.assertEqual(summarise([]), {"questions": 0})


if __name__ == "__main__":
    unittest.main()
