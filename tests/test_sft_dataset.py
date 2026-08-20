"""Selection must be deterministic, capped, and unable to admit a held-out task.

The leak these tests exist for is invisible once rows are shuffled together: a
dev or test task id in the training set invalidates every number the resulting
checkpoint produces, and nothing downstream would notice.

The cap matters for a quieter reason. Easy tasks pass on every sample and hard
tasks pass once, so an uncapped selection is weighted toward the problems the
model already solves - which is the opposite of what the retained set is for.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts import build_sft_dataset as builder  # noqa: E402

SPLIT_MANIFEST = PROJECT_ROOT / "configs" / "splits" / "phase_a_gsm8k.json"


def _split_ids(split: str) -> list[str]:
    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    return [entry["task_id"] for entry in manifest["splits"][split]]


def candidate(task_id: str, run_index: int, *, usable: bool = True) -> dict:
    return {
        "task_id": task_id,
        "run_index": run_index,
        "usable": usable,
        "messages": [{"role": "assistant", "content": "x"}],
        "messages_sha256": "0" * 64,
        "laundered": not usable,
        "laundering_reason": None if usable else "gold_answer_is_a_literal",
    }


class SelectionTests(unittest.TestCase):
    def test_unusable_candidates_are_never_selected(self) -> None:
        rows, stats = builder.select(
            [candidate("gsm8k:train:7", 0, usable=False)], per_task_cap=1
        )
        self.assertEqual(rows, [])
        self.assertEqual(stats["usable"], 0)

    def test_at_most_the_cap_is_kept_from_any_one_task(self) -> None:
        """Otherwise the easy tasks, which pass every time, dominate."""

        rows, stats = builder.select(
            [candidate("gsm8k:train:7", index) for index in range(4)],
            per_task_cap=1,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(stats["dropped_to_per_task_cap"], 3)
        self.assertEqual(stats["distinct_tasks"], 1)

    def test_a_higher_cap_keeps_more_rows_from_the_same_task(self) -> None:
        rows, _ = builder.select(
            [candidate("gsm8k:train:7", index) for index in range(4)],
            per_task_cap=2,
        )
        self.assertEqual(len(rows), 2)

    def test_selection_order_does_not_depend_on_input_order(self) -> None:
        """Two builds of the same candidates must be byte-identical."""

        shuffled = [
            candidate("gsm8k:train:17", 1),
            candidate("gsm8k:train:7", 3),
            candidate("gsm8k:train:7", 0),
        ]
        first, _ = builder.select(shuffled, per_task_cap=2)
        second, _ = builder.select(list(reversed(shuffled)), per_task_cap=2)
        self.assertEqual(
            [(row["task_id"], row["run_index"]) for row in first],
            [(row["task_id"], row["run_index"]) for row in second],
        )

    def test_the_lowest_run_index_is_the_one_kept(self) -> None:
        """A stable rule, and not one the model can influence by scoring well."""

        rows, _ = builder.select(
            [candidate("gsm8k:train:7", index) for index in (3, 1, 2)],
            per_task_cap=1,
        )
        self.assertEqual(rows[0]["run_index"], 1)


class SplitMembershipTests(unittest.TestCase):
    def test_train_ids_are_accepted(self) -> None:
        builder.check_split_membership(set(_split_ids("train")[:5]))

    def test_a_test_task_id_is_refused(self) -> None:
        """The leak that would invalidate every headline number."""

        with self.assertRaises(builder.DatasetError) as caught:
            builder.check_split_membership({_split_ids("test")[0]})
        self.assertIn("dev or test", str(caught.exception))

    def test_a_dev_task_id_is_refused(self) -> None:
        with self.assertRaises(builder.DatasetError):
            builder.check_split_membership({_split_ids("dev")[0]})

    def test_an_unknown_task_id_is_refused_separately(self) -> None:
        """Not a leak, but evidence the candidates came from another manifest."""

        with self.assertRaises(builder.DatasetError) as caught:
            builder.check_split_membership({"gsm8k:train:999999"})
        self.assertIn("not in the frozen train split", str(caught.exception))


class ManifestTests(unittest.TestCase):
    def test_the_manifest_records_only_what_identifies_a_row(self) -> None:
        """Task id, run index and hash. Enough to verify, too little to leak."""

        manifest = builder.build_manifest([candidate("gsm8k:train:7", 0)])
        self.assertEqual(
            set(manifest["rows"][0]), {"task_id", "run_index", "messages_sha256"}
        )


if __name__ == "__main__":
    unittest.main()
