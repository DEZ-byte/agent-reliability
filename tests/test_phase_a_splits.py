from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "configs" / "splits" / "phase_a_gsm8k.json"

TASK_ID = re.compile(r"^gsm8k:(train|test):[0-9]+$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PhaseASplitManifestTests(unittest.TestCase):
    """Offline checks. CI has no network, so this validates what is committed.

    Regenerating the manifest is a separate, deliberate step:
    `scripts/build_phase_a_splits.py --check` rebuilds from the pinned revision
    and fails if a single byte would differ.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.splits = cls.manifest["splits"]

    def test_dataset_is_pinned_to_an_immutable_revision(self) -> None:
        dataset = self.manifest["dataset"]
        self.assertEqual(dataset["id"], "openai/gsm8k")
        self.assertEqual(dataset["config"], "main")
        self.assertRegex(dataset["revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(dataset["license"], "mit")

    def test_expected_sizes(self) -> None:
        self.assertEqual(len(self.splits["train"]), 1000)
        self.assertEqual(len(self.splits["dev"]), 100)
        self.assertEqual(len(self.splits["test"]), 150)

    def test_task_ids_are_well_formed_and_unique_within_a_split(self) -> None:
        for name, rows in self.splits.items():
            ids = [row["task_id"] for row in rows]
            with self.subTest(split=name):
                self.assertEqual(len(ids), len(set(ids)))
                for task_id in ids:
                    self.assertRegex(task_id, TASK_ID)

    def test_no_task_appears_in_two_splits(self) -> None:
        """The property the whole manifest exists to guarantee."""

        seen: dict[str, str] = {}
        for name, rows in self.splits.items():
            for row in rows:
                previous = seen.get(row["task_id"])
                self.assertIsNone(
                    previous,
                    f"{row['task_id']} is in both {previous} and {name}",
                )
                seen[row["task_id"]] = name

    def test_no_content_hash_is_shared_across_splits(self) -> None:
        """Catches a duplicate question that carries two different IDs."""

        seen: dict[str, str] = {}
        for name, rows in self.splits.items():
            for row in rows:
                self.assertRegex(row["content_sha256"], SHA256)
                previous = seen.get(row["content_sha256"])
                self.assertIsNone(
                    previous,
                    f"identical question in {previous} and {name}",
                )
                seen[row["content_sha256"]] = name

    def test_train_comes_from_upstream_train_and_evaluation_from_upstream_test(
        self,
    ) -> None:
        for row in self.splits["train"]:
            self.assertTrue(row["task_id"].startswith("gsm8k:train:"))
        for name in ("dev", "test"):
            for row in self.splits[name]:
                self.assertTrue(row["task_id"].startswith("gsm8k:test:"))

    def test_policy_states_the_evaluation_only_rule(self) -> None:
        policy = self.manifest["policy"]
        self.assertTrue(policy["test_is_evaluation_only"])
        self.assertTrue(policy["dev_and_test_are_disjoint"])
        self.assertIsInstance(policy["seed"], int)
        self.assertIn("template", policy["template_granularity"])


if __name__ == "__main__":
    unittest.main()
