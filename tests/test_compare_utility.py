"""The paired MMLU comparison must refuse rows that are not the headline's rows.

The published comparison was computed from a response file that another
process was still rewriting, and it reported an improved/regressed split that
the recorded accuracies could not produce. These tests pin the two guards that
would have caught it: rows must reproduce their committed summary exactly, and
the paired counts must add up to the accuracy difference.

Everything here is synthetic. Nothing touches a model, a dataset or the network.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import compare_utility as cu  # noqa: E402
from evaluation.utility import UtilityScore, summarise  # noqa: E402
from run_utility_eval import by_subject  # noqa: E402

SPLIT_SHA = "0" * 64
SUBJECTS = ("anatomy", "anatomy", "astronomy", "astronomy", "virology", "virology")
GOLD = ("A", "B", "C", "D", "A", "B")


def _rows(extracted: list[str | None], truncated: list[bool] | None = None) -> list[dict]:
    truncated = truncated or [False] * len(extracted)
    return [
        {
            "task_id": f"mmlu:test:{index}",
            "subject": SUBJECTS[index],
            "gold": GOLD[index],
            "extracted": letter,
            "correct": letter == GOLD[index],
            "emitted_tool_call": False,
            "truncated": truncated[index],
            "completion": f"The answer is {letter}." if letter else "Let me think",
        }
        for index, letter in enumerate(extracted)
    ]


def _summary(label: str, rows: list[dict]) -> dict:
    scores = [
        UtilityScore(
            correct=row["correct"],
            extracted=row["extracted"],
            emitted_tool_call=row["emitted_tool_call"],
            generated_chars=len(row["completion"]),
            truncated=row["truncated"],
        )
        for row in rows
    ]
    return {
        "kind": "utility_eval",
        "label": label,
        "benchmark": "mmlu",
        "model": {"id": "Qwen/Qwen3-1.7B", "revision": "a" * 40},
        "adapter": None,
        "split_manifest_sha256": SPLIT_SHA,
        "decoding": {"greedy": True, "max_new_tokens": 320},
        "tools_offered": False,
        "prompt_sha256": {"system": "s", "user": "u"},
        "executed": True,
        "summary": summarise(scores),
        "by_subject": by_subject(rows),
    }


def _split() -> dict:
    return {
        "questions": [
            {"task_id": f"mmlu:test:{index}", "subject": subject}
            for index, subject in enumerate(SUBJECTS)
        ]
    }


def _arm(label: str, extracted: list[str | None], **kwargs) -> cu.Arm:
    rows = _rows(extracted, **kwargs)
    return cu.Arm(label, _summary(label, rows), rows)


# base gets 0,1,2 right; sft gets 0,1,3,4 right: +2 improved (3,4), -1 regressed (2).
BASE = ["A", "B", "C", None, "B", "A"]
SFT = ["A", "B", "D", "D", "A", "C"]


class ReproductionTests(unittest.TestCase):
    """Rows are accepted only if they are exactly the rows behind the summary."""

    def test_matching_rows_are_accepted(self) -> None:
        arm = _arm("base", BASE)
        cu.check_rows_reproduce_summary(arm.rows, arm.summary, label="base")

    def test_a_summary_from_a_different_run_is_refused(self) -> None:
        arm = _arm("sft", SFT)
        summary = json.loads(json.dumps(arm.summary))
        summary["summary"]["accuracy"] = 0.5
        with self.assertRaisesRegex(cu.UtilityComparisonError, "accuracy"):
            cu.check_rows_reproduce_summary(arm.rows, summary, label="sft")

    def test_a_partial_file_is_refused(self) -> None:
        """391 of 400 rows is how the published numbers went wrong."""

        arm = _arm("sft", SFT)
        with self.assertRaises(cu.UtilityComparisonError):
            cu.check_rows_reproduce_summary(arm.rows[:-1], arm.summary, label="sft")

    def test_same_totals_with_a_different_subject_split_is_refused(self) -> None:
        arm = _arm("sft", SFT)
        summary = json.loads(json.dumps(arm.summary))
        summary["by_subject"]["anatomy"]["accuracy"] = 0.0
        with self.assertRaisesRegex(cu.UtilityComparisonError, "per-subject"):
            cu.check_rows_reproduce_summary(arm.rows, summary, label="sft")

    def test_a_row_graded_against_its_own_letters_is_refused(self) -> None:
        arm = _arm("base", BASE)
        arm.rows[3]["correct"] = True
        with self.assertRaisesRegex(cu.UtilityComparisonError, "marked correct"):
            cu.check_rows_reproduce_summary(arm.rows, arm.summary, label="base")

    def test_stored_grading_is_used_rather_than_re_scoring(self) -> None:
        """A later extractor change must not make old rows disagree with their record."""

        arm = _arm("base", BASE)
        arm.rows[0]["completion"] = "A rambling reply that names no letter at all"
        summary = _summary("base", arm.rows)
        cu.check_rows_reproduce_summary(arm.rows, summary, label="base")

    def test_a_truncated_partial_line_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "responses.jsonl"
            good = json.dumps(_rows(BASE)[0])
            path.write_text(good + "\n" + good[: len(good) // 2], encoding="utf-8")
            with self.assertRaisesRegex(cu.UtilityComparisonError, "not valid JSON"):
                cu.load_responses(path)

    def test_a_question_answered_twice_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "responses.jsonl"
            row = json.dumps(_rows(BASE)[0])
            path.write_text(row + "\n" + row + "\n", encoding="utf-8")
            with self.assertRaisesRegex(cu.UtilityComparisonError, "more than once"):
                cu.load_responses(path)


class AlignmentTests(unittest.TestCase):
    """Arms are paired only when they answered the same questions the same way."""

    def test_aligned_arms_pass(self) -> None:
        cu.check_arms_align(
            [_arm("base", BASE), _arm("sft", SFT)], _split(), split_manifest_sha256=SPLIT_SHA
        )

    def test_a_missing_question_is_refused(self) -> None:
        sft = _arm("sft", SFT)
        short = cu.Arm("sft", sft.summary, sft.rows[:-1])
        with self.assertRaisesRegex(cu.UtilityComparisonError, "frozen split"):
            cu.check_arms_align(
                [_arm("base", BASE), short], _split(), split_manifest_sha256=SPLIT_SHA
            )

    def test_a_different_decoding_budget_is_refused(self) -> None:
        sft = _arm("sft", SFT)
        sft.summary["decoding"] = {"greedy": True, "max_new_tokens": 128}
        with self.assertRaisesRegex(cu.UtilityComparisonError, "decoding"):
            cu.check_arms_align(
                [_arm("base", BASE), sft], _split(), split_manifest_sha256=SPLIT_SHA
            )

    def test_a_different_split_manifest_is_refused(self) -> None:
        with self.assertRaisesRegex(cu.UtilityComparisonError, "split manifest"):
            cu.check_arms_align(
                [_arm("base", BASE), _arm("sft", SFT)], _split(), split_manifest_sha256="1" * 64
            )

    def test_disagreeing_gold_letters_are_refused(self) -> None:
        sft = _arm("sft", SFT)
        sft.rows[0]["gold"] = "D"
        with self.assertRaisesRegex(cu.UtilityComparisonError, "gold"):
            cu.check_arms_align(
                [_arm("base", BASE), sft], _split(), split_manifest_sha256=SPLIT_SHA
            )


class PairedArithmeticTests(unittest.TestCase):
    """The counts must add up to the accuracy difference, every time."""

    def setUp(self) -> None:
        self.base = _arm("base", BASE)
        self.sft = _arm("sft", SFT)

    def test_counts_and_difference_agree(self) -> None:
        result = cu.compare_pair(self.base, self.sft)
        self.assertEqual(result["questions"], 6)
        self.assertEqual((result["baseline_correct"], result["treatment_correct"]), (3, 4))
        self.assertEqual((result["improved"], result["regressed"]), (2, 1))
        self.assertEqual((result["both_correct"], result["both_wrong"]), (2, 1))
        self.assertEqual(
            result["difference"], (result["improved"] - result["regressed"]) / 6
        )
        # Computed from counts, so it is exact; the difference of two rounded
        # accuracies is only equal to within float error.
        self.assertAlmostEqual(
            result["difference"],
            result["treatment_accuracy"] - result["baseline_accuracy"],
            places=12,
        )

    def test_readable_scope_drops_questions_either_arm_left_unreadable(self) -> None:
        result = cu.compare_pair(self.base, self.sft, scope=cu.SCOPE_READABLE)
        self.assertEqual(result["questions"], 5)
        self.assertEqual((result["improved"], result["regressed"]), (1, 1))

    def test_interval_is_seeded_and_brackets_the_estimate(self) -> None:
        first = cu.compare_pair(self.base, self.sft)
        second = cu.compare_pair(self.base, self.sft)
        self.assertEqual(first["difference_ci95"], second["difference_ci95"])
        low, high = first["difference_ci95"]
        self.assertLessEqual(low, first["difference"])
        self.assertGreaterEqual(high, first["difference"])

    def test_identical_arms_give_no_difference_and_no_evidence(self) -> None:
        result = cu.compare_pair(self.base, _arm("copy", BASE))
        self.assertEqual(result["difference"], 0.0)
        self.assertEqual(result["difference_ci95"], [0.0, 0.0])
        self.assertIsNone(result["p_sign_test_exact"])
        self.assertEqual(result["p_permutation_two_sided"], 1.0)

    def test_rows_carry_every_scored_field_for_every_arm(self) -> None:
        table = cu.question_rows([self.base, self.sft])
        self.assertEqual(len(table), 6)
        self.assertEqual(set(table[0]["arms"]), {"base", "sft"})
        self.assertEqual(
            set(table[0]["arms"]["sft"]),
            {
                "extracted",
                "correct",
                "truncated",
                "emitted_tool_call",
                "generated_chars",
                "completion_sha256",
            },
        )


class CommittedComparisonTests(unittest.TestCase):
    """The frozen comparison agrees with its own rows and the summaries it names."""

    RESULTS = PROJECT_ROOT / "results"

    def _artifacts(self) -> list[Path]:
        return sorted(self.RESULTS.glob("utility-comparison-*.json"))

    def test_a_comparison_is_committed(self) -> None:
        self.assertTrue(self._artifacts())

    def test_every_comparison_is_recomputable_from_its_rows(self) -> None:
        for path in self._artifacts():
            payload = json.loads(path.read_text(encoding="utf-8"))
            for entry in payload["comparisons"]:
                base, treat = entry["baseline"], entry["treatment"]
                with self.subTest(artifact=path.name, pair=f"{treat}-vs-{base}", scope=entry["scope"]):
                    chosen = [
                        row["arms"]
                        for row in payload["rows"]
                        if entry["scope"] == cu.SCOPE_ALL
                        or (
                            row["arms"][base]["extracted"] is not None
                            and row["arms"][treat]["extracted"] is not None
                        )
                    ]
                    before = [arms[base]["correct"] for arms in chosen]
                    after = [arms[treat]["correct"] for arms in chosen]
                    self.assertEqual(entry["questions"], len(chosen))
                    self.assertEqual(entry["baseline_correct"], sum(before))
                    self.assertEqual(entry["treatment_correct"], sum(after))
                    self.assertEqual(
                        entry["improved"], sum(a and not b for b, a in zip(before, after))
                    )
                    self.assertEqual(
                        entry["regressed"], sum(b and not a for b, a in zip(before, after))
                    )
                    self.assertEqual(
                        entry["difference"], (sum(after) - sum(before)) / len(chosen)
                    )
                    low, high = entry["difference_ci95"]
                    self.assertLessEqual(low, entry["difference"])
                    self.assertGreaterEqual(high, entry["difference"])

    def test_every_arm_matches_the_frozen_summary_it_names(self) -> None:
        manifest = json.loads(
            (self.RESULTS / "artifact_manifest.json").read_text(encoding="utf-8")
        )["artifacts"]
        for path in self._artifacts():
            payload = json.loads(path.read_text(encoding="utf-8"))
            for label, arm in payload["arms"].items():
                with self.subTest(artifact=path.name, arm=label):
                    self.assertEqual(
                        arm["summary_sha256"], manifest[arm["summary_artifact"]]["sha256"]
                    )
                    recorded = json.loads(
                        (self.RESULTS / arm["summary_artifact"]).read_text(encoding="utf-8")
                    )["summary"]
                    rows = [row["arms"][label] for row in payload["rows"]]
                    self.assertEqual(len(rows), recorded["questions"])
                    self.assertEqual(
                        sum(r["correct"] for r in rows) / len(rows), recorded["accuracy"]
                    )
                    self.assertEqual(
                        sum(r["truncated"] for r in rows) / len(rows),
                        recorded["truncated_rate"],
                    )
                    self.assertEqual(
                        sum(r["generated_chars"] for r in rows) / len(rows),
                        recorded["mean_generated_chars"],
                    )


if __name__ == "__main__":
    unittest.main()
