"""The manifest may only freeze bytes that a commit actually holds.

At 7e33eb5 the three utility artifacts were indexed before they were committed,
so their entries named no recording commit, and one of them was rewritten by an
overlapping evaluation between indexing and committing, so the frozen hash
matched nothing in Git. ERRATA.md records both. These tests pin the guards.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import build_artifact_manifest as builder  # noqa: E402

COMMIT = "0123456789abcdef0123456789abcdef01234567"


class RecordingCommitTests(unittest.TestCase):
    def _artifact(self, directory: str) -> Path:
        path = Path(directory) / "utility-example.json"
        path.write_bytes(b'{"kind": "utility_eval"}\n')
        return path

    def test_every_listed_artifact_names_a_full_commit(self) -> None:
        manifest = json.loads(builder.MANIFEST_PATH.read_text(encoding="utf-8"))
        for name, entry in manifest["artifacts"].items():
            with self.subTest(artifact=name):
                self.assertRegex(
                    entry.get("recorded_in_commit") or "",
                    r"\A[0-9a-f]{40}\Z",
                    "a manifest entry names no commit that holds its bytes",
                )

    def test_an_uncommitted_artifact_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._artifact(tmp)
            with mock.patch.object(builder, "_recording_commit", return_value=None):
                with self.assertRaisesRegex(builder.ManifestError, "no recording commit"):
                    builder._entry(path)

    def test_an_abbreviated_commit_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._artifact(tmp)
            with mock.patch.object(builder, "_recording_commit", return_value="7e33eb5"):
                with self.assertRaises(builder.ManifestError):
                    builder._entry(path)

    def test_bytes_that_differ_from_the_committed_copy_are_refused(self) -> None:
        """What happened to utility-sft-30007ed.json: same length, different bytes."""

        with tempfile.TemporaryDirectory() as tmp:
            path = self._artifact(tmp)
            with mock.patch.object(builder, "_recording_commit", return_value=COMMIT), \
                    mock.patch.object(
                        builder, "_committed_bytes", return_value=b'{"kind": "utility_eva1"}\n'
                    ):
                with self.assertRaisesRegex(builder.ManifestError, "no commit holds"):
                    builder._entry(path)

    def test_a_committed_unmodified_artifact_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._artifact(tmp)
            with mock.patch.object(builder, "_recording_commit", return_value=COMMIT), \
                    mock.patch.object(builder, "_committed_bytes", return_value=path.read_bytes()):
                entry = builder._entry(path)
        self.assertEqual(entry["recorded_in_commit"], COMMIT)
        self.assertEqual(entry["kind"], "utility_eval")


if __name__ == "__main__":
    unittest.main()
