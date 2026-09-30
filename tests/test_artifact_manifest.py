"""The manifest may only freeze bytes that a commit actually holds.

At 7e33eb5 the three utility artifacts were indexed before they were committed,
so their entries named no recording commit, and one of them was rewritten by an
overlapping evaluation between indexing and committing, so the frozen hash
matched nothing in Git. A later rebuild then replaced that hash without notice.
ERRATA.md records all of it. These tests pin the guards, first with mocks and
then against a real throwaway Git repository.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
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


class RealGitTests(unittest.TestCase):
    """The same guards, exercised against a real repository rather than mocks."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.results = self.root / "results"
        self.results.mkdir()
        self.git("init", "-q")
        self.git("config", "core.autocrlf", "false")
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.invalid")
        for name, value in (
            ("PROJECT_ROOT", self.root),
            ("RESULTS_DIR", self.results),
            ("MANIFEST_PATH", self.results / "artifact_manifest.json"),
        ):
            patcher = mock.patch.object(builder, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=self.root, capture_output=True, text=True, check=True
        ).stdout.strip()

    def write(self, name: str, payload: dict) -> Path:
        path = self.results / name
        path.write_bytes((json.dumps(payload) + "\n").encode("utf-8"))
        return path

    def commit(self, message: str) -> str:
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")

    def freeze(self) -> None:
        manifest = builder.build(builder.load_previous())
        builder.MANIFEST_PATH.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        self.commit("chore: freeze")

    def check(self) -> int:
        argv = ["build_artifact_manifest.py", "--check"]
        quiet_out, quiet_err = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", argv), redirect_stdout(quiet_out):
            with redirect_stderr(quiet_err):
                return builder.main()

    def test_a_committed_artifact_records_the_commit_that_added_it(self) -> None:
        self.write("utility-a.json", {"kind": "utility_eval"})
        added = self.commit("feat: a")
        entry = builder.build()["artifacts"]["utility-a.json"]
        self.assertEqual(entry["recorded_in_commit"], added)

    def test_an_uncommitted_artifact_is_refused(self) -> None:
        self.write("utility-a.json", {"kind": "utility_eval"})
        with self.assertRaisesRegex(builder.ManifestError, "no recording commit"):
            builder.build()

    def test_a_working_copy_that_differs_from_its_commit_is_refused(self) -> None:
        self.write("utility-a.json", {"kind": "utility_eval", "created": 1})
        self.commit("feat: a")
        self.write("utility-a.json", {"kind": "utility_eval", "created": 2})
        with self.assertRaisesRegex(builder.ManifestError, "no commit holds"):
            builder.build()

    def test_a_committed_edit_to_a_recorded_artifact_is_refused(self) -> None:
        """The silent re-sign: edit, commit, rebuild. It must not pass."""

        self.write("utility-a.json", {"kind": "utility_eval", "accuracy": 0.50})
        self.commit("feat: a")
        self.freeze()
        self.write("utility-a.json", {"kind": "utility_eval", "accuracy": 0.60})
        self.commit("fix: a nicer number")
        with self.assertRaisesRegex(builder.ManifestError, "would change: utility-a.json"):
            builder.build(builder.load_previous())
        self.assertEqual(self.check(), 1)

    def test_a_deleted_recorded_artifact_is_refused(self) -> None:
        self.write("utility-a.json", {"kind": "utility_eval"})
        self.commit("feat: a")
        self.freeze()
        self.git("rm", "-q", "results/utility-a.json")
        self.commit("chore: drop a")
        with self.assertRaisesRegex(builder.ManifestError, "would disappear"):
            builder.build(builder.load_previous())
        self.assertEqual(self.check(), 1)

    def test_adding_an_artifact_keeps_every_existing_entry(self) -> None:
        self.write("utility-a.json", {"kind": "utility_eval"})
        self.commit("feat: a")
        self.freeze()
        before = builder.load_previous()
        self.write("utility-b.json", {"kind": "utility_eval"})
        self.commit("feat: b")
        after = builder.build(builder.load_previous())["artifacts"]
        self.assertEqual(after["utility-a.json"], before["utility-a.json"])
        self.assertIn("utility-b.json", after)

    def test_check_passes_on_a_frozen_untouched_tree(self) -> None:
        self.write("utility-a.json", {"kind": "utility_eval"})
        self.commit("feat: a")
        self.freeze()
        self.assertEqual(self.check(), 0)

    def test_a_rewritten_history_is_caught_by_check(self) -> None:
        """What a squash or rebase merge does to recorded commits."""

        (self.root / "README").write_text("start\n", encoding="utf-8")
        base = self.commit("chore: start")
        self.write("utility-a.json", {"kind": "utility_eval"})
        self.commit("feat: a")
        self.freeze()
        # Collapse "feat: a" and the freeze into one new commit, as a squash
        # merge does. The artifact now lives in a commit the manifest never saw.
        self.git("reset", "-q", "--soft", base)
        self.git("commit", "-q", "-m", "squashed")
        self.assertEqual(self.check(), 1)


if __name__ == "__main__":
    unittest.main()
