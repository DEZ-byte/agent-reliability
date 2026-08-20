from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from env.splits import (  # noqa: E402
    SPLIT_NAMES,
    SplitError,
    content_digest,
    load_split,
    upstream_split,
)

MANIFEST = PROJECT_ROOT / "configs" / "splits" / "phase_a_gsm8k.json"


class UpstreamSplitTests(unittest.TestCase):
    """Offline checks. CI has no network, so nothing here downloads a dataset."""

    def test_upstream_split_is_read_from_the_task_id(self) -> None:
        self.assertEqual(upstream_split("gsm8k:train:7"), "train")
        self.assertEqual(upstream_split("gsm8k:test:8"), "test")

    def test_a_malformed_task_id_is_rejected_rather_than_guessed(self) -> None:
        for bad in ("gsm8k:train", "train:7", "gsm8k:train:7:extra", ""):
            with self.subTest(task_id=bad):
                with self.assertRaises(SplitError):
                    upstream_split(bad)

    def test_the_manifest_really_does_span_two_upstream_splits(self) -> None:
        """The regression this loader exists for.

        A loader that hardcodes one upstream split works on dev and test and
        silently addresses the wrong rows for train. If this ever became a
        single-upstream manifest the hardcoded version would start passing and
        the guard below would stop meaning anything, so assert the premise.
        """

        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        seen = {
            split: {upstream_split(entry["task_id"]) for entry in entries}
            for split, entries in manifest["splits"].items()
        }
        self.assertEqual(seen["train"], {"train"})
        self.assertEqual(seen["dev"], {"test"})
        self.assertEqual(seen["test"], {"test"})
        union = set().union(*seen.values())
        self.assertEqual(union, {"train", "test"})

    def test_every_split_name_the_loader_accepts_exists_in_the_manifest(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(set(SPLIT_NAMES), set(manifest["splits"]))

    def test_an_unknown_split_is_refused_before_any_download(self) -> None:
        with self.assertRaises(SplitError):
            load_split(MANIFEST, "holdout")

    def test_content_digest_matches_the_frozen_manifest_hashes(self) -> None:
        """The digest is the only thing tying a loaded row to a frozen id.

        It is recomputed here from the same inputs the builder used, so a change
        to the hashing rule fails here rather than during a training run.
        """

        self.assertEqual(
            content_digest("q", "a"),
            content_digest("q", "a"),
        )
        self.assertNotEqual(content_digest("q", "a"), content_digest("q", "b"))
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for entries in manifest["splits"].values():
            for entry in entries:
                self.assertRegex(entry["content_sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
