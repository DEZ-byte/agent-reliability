"""A script may only name a commit if it ran exactly that commit's code.

Four artifacts named a commit that did not yet contain the code that produced
them (ERRATA.md, E9). The guard runs against a real throwaway repository here,
because a mocked `git status` would prove nothing about the real one.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.provenance import (  # noqa: E402
    DirtyWorktreeError,
    portable_path,
    require_clean_worktree,
)


class CleanWorktreeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        for args in (
            ("init", "-q"),
            ("config", "core.autocrlf", "false"),
            ("config", "user.name", "Test"),
            ("config", "user.email", "test@example.invalid"),
        ):
            self.git(*args)
        (self.root / "code.py").write_text("x = 1\n", encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "start")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=self.root, capture_output=True, text=True, check=True
        ).stdout.strip()

    def test_a_clean_tree_returns_the_full_head_commit(self) -> None:
        self.assertEqual(require_clean_worktree(self.root), self.git("rev-parse", "HEAD"))

    def test_an_edited_file_is_refused(self) -> None:
        (self.root / "code.py").write_text("x = 2\n", encoding="utf-8")
        with self.assertRaises(DirtyWorktreeError):
            require_clean_worktree(self.root)

    def test_an_untracked_file_is_refused(self) -> None:
        """A stray file beside the code can change what runs."""

        (self.root / "helper.py").write_text("y = 1\n", encoding="utf-8")
        with self.assertRaises(DirtyWorktreeError):
            require_clean_worktree(self.root)

    def test_a_staged_but_uncommitted_change_is_refused(self) -> None:
        (self.root / "code.py").write_text("x = 3\n", encoding="utf-8")
        self.git("add", "code.py")
        with self.assertRaises(DirtyWorktreeError):
            require_clean_worktree(self.root)

    def test_outside_a_repository_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as bare:
            with self.assertRaises(DirtyWorktreeError):
                require_clean_worktree(Path(bare))


class PortablePathTests(unittest.TestCase):
    def test_a_path_inside_the_repository_becomes_relative(self) -> None:
        inside = PROJECT_ROOT / "checkpoints" / "run" / "checkpoint-200"
        self.assertEqual(
            portable_path(inside, PROJECT_ROOT), "checkpoints/run/checkpoint-200"
        )

    def test_a_path_outside_the_repository_is_left_alone(self) -> None:
        with tempfile.TemporaryDirectory() as elsewhere:
            self.assertEqual(portable_path(elsewhere, PROJECT_ROOT), elsewhere)


if __name__ == "__main__":
    unittest.main()
