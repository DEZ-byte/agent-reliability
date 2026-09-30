"""Freeze every committed measurement record by hash.

D-052 made result artifacts immutable: they are permanent records, not
regenerable state, and editing an unflattering one into a flattering one must
fail a test rather than pass silently.

This script writes that index. It covers every measurement family, so adding a
new kind of result means adding it to `ARTIFACT_GLOBS` rather than quietly
leaving it unprotected.

Adding a run means adding an entry. It never means changing one, and this
script enforces that: it reads the committed manifest first and refuses to
rewrite or drop any entry already in it. An earlier version rebuilt the index
from scratch, so a committed edit to a result would have been re-signed with a
new hash, and a deleted result silently dropped. That is how one hash was
replaced without notice (ERRATA.md, E1).

If the committed manifest itself is wrong, correct the entry by hand and
explain why in ERRATA.md. That friction is deliberate.
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
    "model_smoke-*.json",
    "contamination-*.json",
    "baseline-*.json",
    "masking-*.json",
    "sft-*.json",
    # These four families were produced after the list was first written and
    # went unprotected for a while, which is worth naming because they are the
    # evidence behind the two headline claims: that the trained 1.7B beats the
    # scaffolded 8B, and that GRPO added nothing on top. A freeze that covers
    # the safe results and misses the load-bearing ones is not a freeze.
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
    "adding a new entry; it never means changing an existing one."
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
    # utility artifacts broke both halves of that (ERRATA.md, E1 and E2): they
    # were indexed before they were committed, so the entries named no commit,
    # and one file was still being rewritten by an overlapping evaluation, so
    # the hash that was frozen belonged to bytes that never reached Git.
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
        "kind": payload.get("kind", "model_smoke"),
    }
    # Model-smoke artifacts carry the two fields that tell the pre-D-046 and
    # post-D-046 evidence regimes apart. Other families have neither.
    if "config_sha256" in payload:
        entry["config_sha256"] = payload["config_sha256"]
    if "lane" in payload:
        entry["declares_gate_demotion"] = bool(payload["lane"].get("gate_demotions"))
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
            + "; a measurement record is never deleted"
        )
    changed = sorted(name for name in previous if previous[name] != current[name])
    if changed:
        raise ManifestError(
            "recorded entr(ies) would change: "
            + ", ".join(changed)
            + "; an existing entry is never rewritten. If the committed entry "
            "is wrong, correct it by hand and explain why in ERRATA.md"
        )


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
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild and fail if the committed manifest would change",
    )
    args = parser.parse_args()

    try:
        manifest = build(load_previous())
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
