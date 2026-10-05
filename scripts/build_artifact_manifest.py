"""Freeze every committed measurement record by hash.

Result artifacts are permanent records, not regenerable state. Editing an
unflattering one into a flattering one must fail a test, not pass silently.

This script writes that index. It covers every measurement family, so adding a
new kind of result means adding it to `ARTIFACT_GLOBS` rather than quietly
leaving it unprotected.

Adding a run means adding an entry. It never means changing one, and this
script enforces that: it reads the committed manifest first and refuses to
rewrite or drop any entry already in it. An earlier version rebuilt the index
from scratch, so a committed edit to a result would have been re-signed with a
new hash, and a deleted result silently dropped. That is how one hash was once
replaced without notice.

An artifact that a later run replaced can leave `results/`, but only on
purpose: `git rm` the file, then run this script with `--retire NAME`. Git
history keeps the bytes under the commit the entry recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
RESULTS_DIR: Final = PROJECT_ROOT / "results"
MANIFEST_PATH: Final = RESULTS_DIR / "artifact_manifest.json"

# Every family of committed measurement record. The manifest test walks the
# same list, so a family absent here is a family nothing protects.
ARTIFACT_GLOBS: Final = (
    "contamination-*.json",
    "baseline-*.json",
    "masking-*.json",
    "sft-*.json",
    # These families were produced after the list was first written and went
    # unprotected for a while. They hold the headline evidence, so a freeze
    # that missed them would protect the safe results and not the ones that
    # carry the claims.
    "grpo-*.json",
    "comparator-*.json",
    "h1-comparison-*.json",
    "phase_b-*.json",
    "utility-*.json",
    "weight-change-*.json",
)

SCHEMA_VERSION: Final = 1

FULL_COMMIT: Final = re.compile(r"[0-9a-f]{40}")

PURPOSE: Final = (
    "Frozen content hashes for every committed measurement artifact. These "
    "files are permanent records, not regenerable state. A test fails if any "
    "hash changes or any artifact is missing from this list, so a result "
    "cannot be edited, re-signed, or quietly dropped. Adding a new run means "
    "adding a new entry; it never means changing an existing one. An entry is "
    "removed only with --retire, after its file is deleted; git history keeps "
    "its bytes under the recorded commit."
)


class ManifestError(RuntimeError):
    """The manifest cannot be built without recording something false."""


def artifact_paths() -> list[Path]:
    found: list[Path] = []
    for pattern in ARTIFACT_GLOBS:
        found.extend(RESULTS_DIR.glob(pattern))
    return sorted(found, key=lambda path: path.name)


def _recording_commit(path: Path) -> str | None:
    relative = path.relative_to(PROJECT_ROOT).as_posix()
    completed = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", relative],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    commit = completed.stdout.strip()
    return commit or None


def _committed_bytes(commit: str, path: Path) -> bytes | None:
    relative = path.relative_to(PROJECT_ROOT).as_posix()
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
    )
    return completed.stdout if completed.returncode == 0 else None


def _entry(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    # The manifest may only freeze bytes that a commit actually holds. The
    # utility artifacts once broke both halves of that: they were indexed
    # before they were committed, so the entries named no commit, and one file
    # was still being rewritten by an overlapping evaluation, so the hash that
    # was frozen belonged to bytes that never reached Git.
    # Indexing only committed, unmodified files makes both impossible.
    commit = _recording_commit(path)
    if commit is None or not FULL_COMMIT.fullmatch(commit):
        raise ManifestError(
            f"{path.name} has no recording commit; commit the artifact first, "
            "then rebuild the manifest in a separate commit"
        )
    if _committed_bytes(commit, path) != raw:
        raise ManifestError(
            f"{path.name} differs from the copy committed in {commit[:7]}; "
            "the manifest would freeze bytes that no commit holds"
        )
    entry: dict[str, Any] = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "recorded_in_commit": commit,
        "kind": payload.get("kind"),
    }
    return entry


def check_append_only(
    previous: dict[str, Any], current: dict[str, Any]
) -> None:
    """Refuse a rebuild that would drop or rewrite a recorded entry."""

    missing = sorted(set(previous) - set(current))
    if missing:
        raise ManifestError(
            "recorded artifact(s) would disappear from the manifest: "
            + ", ".join(missing)
            + "; to remove a replaced record on purpose, use --retire"
        )
    changed = sorted(name for name in previous if previous[name] != current[name])
    if changed:
        raise ManifestError(
            "recorded entr(ies) would change: "
            + ", ".join(changed)
            + "; an existing entry is never rewritten"
        )


def retire(previous: dict[str, Any], names: list[str]) -> dict[str, Any]:
    """Drop the named entries, but only for files already gone from results/."""

    for name in names:
        if name not in previous:
            raise ManifestError(f"{name} is not in the manifest")
        if (RESULTS_DIR / name).exists():
            raise ManifestError(
                f"{name} is still in results/; git rm it before retiring its entry"
            )
    return {name: entry for name, entry in previous.items() if name not in names}


def load_previous() -> dict[str, Any] | None:
    if not MANIFEST_PATH.exists():
        return None
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8")).get("artifacts", {})


def build(previous: dict[str, Any] | None = None) -> dict[str, Any]:
    artifacts = {path.name: _entry(path) for path in artifact_paths()}
    if previous is not None:
        check_append_only(previous, artifacts)
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": PURPOSE,
        "artifacts": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="rebuild and fail if the committed manifest would change",
    )
    mode.add_argument(
        "--retire",
        nargs="+",
        metavar="NAME",
        default=[],
        help="drop these entries; each file must already be deleted from results/",
    )
    args = parser.parse_args()

    try:
        previous = load_previous()
        if args.retire:
            if previous is None:
                raise ManifestError("there is no manifest to retire entries from")
            previous = retire(previous, args.retire)
        manifest = build(previous)
    except ManifestError as error:
        print(str(error), file=sys.stderr)
        return 1
    payload = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not MANIFEST_PATH.exists():
            print("manifest missing", file=sys.stderr)
            return 1
        if MANIFEST_PATH.read_text(encoding="utf-8") != payload:
            print(
                "committed manifest differs from a fresh build; an artifact "
                "changed, appeared, or disappeared",
                file=sys.stderr,
            )
            return 1
        print("manifest reproduces exactly")
        return 0

    temporary = MANIFEST_PATH.with_suffix(".tmp")
    temporary.write_bytes(payload.encode("utf-8"))
    os.replace(temporary, MANIFEST_PATH)
    print(json.dumps({"artifacts": len(manifest["artifacts"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
