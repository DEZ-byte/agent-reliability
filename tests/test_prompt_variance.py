"""The prompt filter drops exactly the groups that cannot teach, and no others.

This filter decides what GRPO trains on, so its failure modes are asymmetric.
Dropping a live prompt loses signal quietly. Keeping a dead one wastes a step
and, worse, lets a run report that it filtered when it did not. Both are
pinned below, along with the boundary cases at solve rates of one and G-1,
where an off-by-one would silently discard the hardest and easiest prompts
that still carry a gradient.
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.prompt_variance import (  # noqa: E402
    DEAD_ALL_CORRECT,
    DEAD_ALL_WRONG,
    LIVE,
    SCHEMA_KIND,
    PromptVarianceError,
    classify_group,
    disagreements,
    live_task_ids,
    standard_deviation,
    summarise,
)


@dataclass(frozen=True, slots=True)
class Fake:
    """A scored candidate, structurally what the filter needs."""

    correct: bool
    total: float


def group(pattern: str, *, totals: list[float] | None = None) -> list[Fake]:
    """A group from a string like 'ccww': c is correct, w is not.

    Totals default to the reward those outcomes actually earn, 1.15 for
    correct work and 0.15 for a well-formed call with the wrong value, so a
    test that does not care about the total still gets a realistic one.
    """

    flags = [char == "c" for char in pattern]
    if totals is None:
        totals = [1.15 if flag else 0.15 for flag in flags]
    return [Fake(correct=flag, total=total) for flag, total in zip(flags, totals)]


class ClassificationTests(unittest.TestCase):
    def test_a_mixed_group_teaches(self) -> None:
        verdict = classify_group(group("ccww"), task_id="t")
        self.assertEqual(verdict.liveness, LIVE)
        self.assertTrue(verdict.teaches)
        self.assertEqual(verdict.correct, 2)
        self.assertEqual(verdict.solve_rate, 0.5)

    def test_all_correct_is_dead(self) -> None:
        verdict = classify_group(group("cccccccc"), task_id="t")
        self.assertEqual(verdict.liveness, DEAD_ALL_CORRECT)
        self.assertFalse(verdict.teaches)

    def test_all_wrong_is_dead(self) -> None:
        verdict = classify_group(group("wwwwwwww"), task_id="t")
        self.assertEqual(verdict.liveness, DEAD_ALL_WRONG)
        self.assertFalse(verdict.teaches)

    def test_one_correct_out_of_eight_still_teaches(self) -> None:
        """The boundary an off-by-one would eat: the hardest solvable prompts."""

        verdict = classify_group(group("cwwwwwww"), task_id="t")
        self.assertEqual(verdict.liveness, LIVE)
        self.assertEqual(verdict.solve_rate, 0.125)

    def test_seven_correct_out_of_eight_still_teaches(self) -> None:
        """The other boundary: nearly-solved prompts still carry a gradient."""

        verdict = classify_group(group("cccccccw"), task_id="t")
        self.assertEqual(verdict.liveness, LIVE)

    def test_an_empty_group_is_refused_rather_than_called_dead(self) -> None:
        with self.assertRaises(PromptVarianceError):
            classify_group([], task_id="t")

    def test_a_single_candidate_group_cannot_teach_and_says_so(self) -> None:
        """G=1 has no group-relative signal at all; it must not read as live."""

        self.assertEqual(classify_group(group("c"), task_id="t").liveness, DEAD_ALL_CORRECT)
        self.assertEqual(classify_group(group("w"), task_id="t").liveness, DEAD_ALL_WRONG)


class SpreadTests(unittest.TestCase):
    def test_identical_totals_have_no_spread(self) -> None:
        self.assertEqual(standard_deviation([1.15] * 8), 0.0)

    def test_fewer_than_two_values_have_no_spread(self) -> None:
        self.assertEqual(standard_deviation([1.15]), 0.0)
        self.assertEqual(standard_deviation([]), 0.0)

    def test_a_live_group_records_real_spread(self) -> None:
        self.assertGreater(classify_group(group("ccww"), task_id="t").total_std, 0.0)


class DisagreementTests(unittest.TestCase):
    """Where the accuracy criterion and the real gradient part company."""

    def test_all_correct_but_varying_efficiency_is_counted(self) -> None:
        """The honest case: this filter drops a group TRL would have used.

        Every candidate is right, so accuracy says dead, but the efficiency
        term moves with token count and leaves a whisper of spread. The
        gradient is real; it teaches brevity rather than correctness. Counting
        it keeps the trade visible instead of buried in the filter.
        """

        verdicts = [
            classify_group(
                group("cccc", totals=[1.15, 1.14, 1.16, 1.15]), task_id="t"
            )
        ]
        self.assertEqual(verdicts[0].liveness, DEAD_ALL_CORRECT)
        self.assertEqual(disagreements(verdicts)["dead_but_nonzero_std"], 1)

    def test_a_live_group_with_zero_spread_would_be_a_reward_bug(self) -> None:
        """Differing accuracy implies differing total, unless the reward broke."""

        verdicts = [classify_group(group("ccww"), task_id="t")]
        self.assertEqual(disagreements(verdicts)["live_but_zero_std"], 0)

    def test_a_clean_dead_group_is_not_counted_as_a_disagreement(self) -> None:
        verdicts = [classify_group(group("wwww"), task_id="t")]
        self.assertEqual(
            disagreements(verdicts),
            {"dead_but_nonzero_std": 0, "live_but_zero_std": 0},
        )


class SummaryTests(unittest.TestCase):
    def _verdicts(self):
        return [
            classify_group(group("ccww"), task_id="a"),
            classify_group(group("cccc"), task_id="b"),
            classify_group(group("wwww"), task_id="c"),
            classify_group(group("cwww"), task_id="d"),
        ]

    def test_counts_name_every_way_a_prompt_can_be_dead(self) -> None:
        summary = summarise(self._verdicts())
        self.assertEqual(summary["counts"][LIVE], 2)
        self.assertEqual(summary["counts"][DEAD_ALL_CORRECT], 1)
        self.assertEqual(summary["counts"][DEAD_ALL_WRONG], 1)

    def test_live_and_dead_fractions_account_for_everything(self) -> None:
        summary = summarise(self._verdicts())
        self.assertAlmostEqual(summary["live_fraction"] + summary["dead_fraction"], 1.0)

    def test_an_empty_probe_summarises_to_nothing_rather_than_dividing_by_zero(self) -> None:
        self.assertEqual(summarise([]), {"prompts": 0})


class FilterLoadingTests(unittest.TestCase):
    """A probe artifact that cannot be trusted must stop the run, not pass through."""

    def _payload(self, rows, *, executed=True, kind=SCHEMA_KIND):
        return {"kind": kind, "executed": executed, "prompts": rows}

    def test_only_live_prompts_are_kept(self) -> None:
        payload = self._payload(
            [
                {"task_id": "a", "liveness": LIVE},
                {"task_id": "b", "liveness": DEAD_ALL_CORRECT},
                {"task_id": "c", "liveness": DEAD_ALL_WRONG},
            ]
        )
        self.assertEqual(live_task_ids(payload), {"a"})

    def test_a_plan_only_artifact_is_refused(self) -> None:
        """Otherwise a killed probe silently trains on the whole split."""

        payload = self._payload([{"task_id": "a", "liveness": LIVE}], executed=False)
        with self.assertRaises(PromptVarianceError):
            live_task_ids(payload)

    def test_the_wrong_kind_of_artifact_is_refused(self) -> None:
        payload = self._payload(
            [{"task_id": "a", "liveness": LIVE}], kind="grpo_run"
        )
        with self.assertRaises(PromptVarianceError):
            live_task_ids(payload)

    def test_an_empty_probe_is_refused(self) -> None:
        with self.assertRaises(PromptVarianceError):
            live_task_ids(self._payload([]))

    def test_an_all_dead_probe_is_refused_rather_than_training_on_nothing(self) -> None:
        payload = self._payload([{"task_id": "a", "liveness": DEAD_ALL_WRONG}])
        with self.assertRaises(PromptVarianceError):
            live_task_ids(payload)


if __name__ == "__main__":
    unittest.main()
