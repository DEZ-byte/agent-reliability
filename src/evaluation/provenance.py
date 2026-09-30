"""Provenance checks for scripts that write a measurement record.

Every artifact names the commit it came from in `source_commit`. That field is
only true if the code that ran is exactly the code in that commit, so a script
that writes a record must refuse to run on a dirty tree. Four artifacts broke
this before the rule was enforced here (ERRATA.md, E9): they named a commit that
did not yet contain the code that produced them.

Paths are recorded relative to the repository. An absolute path names one
person's machine and says nothing to anyone else (results/README.md lists the
older artifacts that carry them).

A model is loaded at the revision configs/model_candidates.json pins, or not at
all. The default branch of a Hub repository can move under a published number.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


class DirtyWorktreeError(RuntimeError):
    """The working tree differs from HEAD, so HEAD is not the code that ran."""


class UnpinnedModelError(RuntimeError):
    """The model has no pinned revision, so it would load whatever is newest."""


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


def require_outside_worktree(path: Path | str, root: Path) -> None:
    """Refuse an output directory inside the repository.

    Each step of a chain refuses a dirty tree, and a new file counts as dirt.
    A chain that wrote inside the repository would block its own next step.
    """

    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return
    raise DirtyWorktreeError(
        f"{path} is inside the repository; the next step would refuse the "
        "files written there. Write run outputs outside the repository."
    )


def portable_path(path: Path | str, root: Path) -> str:
    """A repository-relative POSIX path, or the path unchanged if outside it."""

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)


def pinned_revision(model_id: str, registry_path: Path) -> str:
    """The revision the registry pins for `model_id`, or an error."""

    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    for entries in registry["roles"].values():
        for entry in entries:
            if entry["id"] == model_id and entry.get("revision"):
                return entry["revision"]
    raise UnpinnedModelError(
        f"{model_id} has no pinned revision in {Path(registry_path).name}; "
        "refusing to load whatever its default branch holds today"
    )


def pinned_load_kwargs(
    model_id: str, registry_path: Path, adapter: Path | str | None = None
) -> dict[str, Any]:
    """`FastLanguageModel.from_pretrained` arguments for the pinned base model.

    Without `use_exact_model_name`, Unsloth swaps the name for a pre-quantised
    mirror that does not have the pinned revision. Handed an adapter directory,
    it also drops `revision` for the base model. So the base is always loaded
    by id and the adapter is attached afterwards. The adapter's own tokenizer
    is kept, which is what loading the adapter directory used.
    """

    kwargs: dict[str, Any] = {
        "model_name": model_id,
        "revision": pinned_revision(model_id, registry_path),
        "use_exact_model_name": True,
    }
    if adapter is not None:
        config_path = Path(adapter) / "adapter_config.json"
        if not config_path.is_file():
            raise UnpinnedModelError(f"no adapter_config.json under {adapter}")
        base = json.loads(config_path.read_text(encoding="utf-8")).get(
            "base_model_name_or_path"
        )
        if base != model_id:
            raise UnpinnedModelError(
                f"{adapter} was trained on {base!r}, not on {model_id!r}"
            )
        kwargs["tokenizer_name"] = str(adapter)
    return kwargs


__all__ = [
    "DirtyWorktreeError",
    "UnpinnedModelError",
    "pinned_load_kwargs",
    "pinned_revision",
    "portable_path",
    "require_clean_worktree",
    "require_outside_worktree",
]
