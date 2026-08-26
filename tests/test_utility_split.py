"""The utility split must be frozen, held out, and shaped like MMLU.

Every arm is scored on these exact questions, so the split is the thing that
makes the arms comparable. Three properties matter and each has a way of
failing quietly.

If the sample is not stratified, it fills up with MMLU's largest subjects and
the score moves with the subject mix rather than with anything training did.
If the hashes are not checked, an upstream edit changes what "the same
questions" means without anything saying so. And if these questions ever
overlapped with something the project trains on, the number would measure
memorisation instead of retention.

The loader itself is not exercised here, because it downloads the dataset.
What is checked is the manifest that pins it.
"""

from __future__ import annotations

import collections
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.utility_split import (  # noqa: E402
    MMLU_MANIFEST_NAME,
    content_digest,
    load_manifest,
)

MANIFEST_PATH = PROJECT_ROOT / "configs" / "splits" / MMLU_MANIFEST_NAME


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = load_manifest(MANIFEST_PATH)

    def test_the_dataset_is_pinned_to_a_revision(self) -> None:
        """Without a revision, "the same questions" is not a claim."""

        dataset = self.manifest["dataset"]
        self.assertEqual(dataset["id"], "cais/mmlu")
        self.assertEqual(len(dataset["revision"]), 40)

    def test_every_question_carries_a_content_hash(self) -> None:
        for entry in self.manifest["questions"]:
            self.assertEqual(len(entry["content_sha256"]), 64, entry["task_id"])

    def test_question_ids_are_unique(self) -> None:
        ids = [entry["task_id"] for entry in self.manifest["questions"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_the_sample_is_spread_across_subjects(self) -> None:
        """A uniform draw would concentrate in MMLU's biggest subjects."""

        subjects = {entry["subject"] for entry in self.manifest["questions"]}
        self.assertGreaterEqual(len(subjects), 50)

    def test_no_single_subject_dominates(self) -> None:
        counts = collections.Counter(
            entry["subject"] for entry in self.manifest["questions"]
        )
        total = sum(counts.values())
        largest = counts.most_common(1)[0][1]
        self.assertLess(
            largest / total,
            0.2,
            "one subject carries too much of the sample; the score would track "
            "that subject rather than general ability",
        )

    def test_the_split_is_the_held_out_test_split(self) -> None:
        self.assertEqual(self.manifest["dataset"]["split"], "test")


class DigestTests(unittest.TestCase):
    """The hash has to notice a reordered option, not just a changed question."""

    def test_the_same_question_hashes_the_same(self) -> None:
        a = content_digest("What is 2+2?", ["3", "4", "5", "6"], 1)
        b = content_digest("What is 2+2?", ["3", "4", "5", "6"], 1)
        self.assertEqual(a, b)

    def test_reordered_choices_hash_differently(self) -> None:
        a = content_digest("What is 2+2?", ["3", "4", "5", "6"], 1)
        b = content_digest("What is 2+2?", ["4", "3", "5", "6"], 1)
        self.assertNotEqual(a, b)

    def test_a_changed_answer_index_hashes_differently(self) -> None:
        a = content_digest("What is 2+2?", ["3", "4", "5", "6"], 1)
        b = content_digest("What is 2+2?", ["3", "4", "5", "6"], 2)
        self.assertNotEqual(a, b)


class DisjointnessTests(unittest.TestCase):
    """These questions must not appear anywhere the project trains."""

    def test_no_utility_question_shares_a_hash_with_a_phase_a_task(self) -> None:
        """Different datasets, but the check is cheap and the claim is public."""

        phase_a = json.loads(
            (PROJECT_ROOT / "configs" / "splits" / "phase_a_gsm8k.json").read_text(
                encoding="utf-8"
            )
        )
        trained = {
            entry["content_sha256"]
            for split in phase_a["splits"].values()
            for entry in split
        }
        utility = {
            entry["content_sha256"]
            for entry in load_manifest(MANIFEST_PATH)["questions"]
        }
        self.assertEqual(trained & utility, set())


if __name__ == "__main__":
    unittest.main()
