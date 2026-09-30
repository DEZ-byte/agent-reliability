"""Provenance checks for scripts that write a measurement record.

Every artifact names the commit it came from in `source_commit`. That field is
only true if the code that ran is exactly the code in that commit, so a script
that writes a record must refuse to run on a dirty tree. Four artifacts broke
this before the rule was enforced here (ERRATA.md, E9): they named a commit that
did not yet contain the code that produced them.

Paths are recorded relative to the repository. An absolute path names one
person's machine and says nothing to anyone else (results/README.md lists the
older artifacts that carry them).
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class DirtyWorktreeError(RuntimeError):
    """The working tree differs from HEAD, so HEAD is not the code that ran."""


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )


def require_clean_worktree(root: Path) -> str:
    """The full HEAD commit, or an error if anything differs from it.

    Untracked files count. A stray script or config beside the code can change
    what runs just as an edit can.
    """

    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    if status.returncode != 0:
        raise DirtyWorktreeError("git status failed; cannot show the tree is clean")
    if status.stdout.strip():
        changed = len(status.stdout.strip().splitlines())
        raise DirtyWorktreeError(
            f"the working tree has {changed} uncommitted change(s); commit first "
            "so that source_commit names the code that actually ran"
        )
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    if len(head) != 40:
        raise DirtyWorktreeError("could not resolve HEAD to a full commit")
    return head


def portable_path(path: Path | str, root: Path) -> str:
    """A repository-relative POSIX path, or the path unchanged if outside it."""

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)


__all__ = ["DirtyWorktreeError", "portable_path", "require_clean_worktree"]
