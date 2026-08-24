"""Freeze every committed measurement record by hash.

D-052 made result artifacts immutable: they are permanent records, not
regenerable state, and editing an unflattering one into a flattering one must
fail a test rather than pass silently.

This script writes that index. It covers every measurement family, so adding a
new kind of result means adding it to `ARTIFACT_GLOBS` rather than quietly
leaving it unprotected.

Adding a run means adding an entry. It never means changing one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
)

SCHEMA_VERSION: Final = 1

PURPOSE: Final = (
    "Frozen content hashes for every committed measurement artifact. These "
    "files are permanent records, not regenerable state. A test fails if any "
    "hash changes or any artifact is missing from this list, so a result "
    "cannot be edited, re-signed, or quietly dropped. Adding a new run means "
    "adding a new entry; it never means changing an existing one."
)


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


def _entry(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    entry: dict[str, Any] = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "recorded_in_commit": _recording_commit(path),
        "kind": payload.get("kind", "model_smoke"),
    }
    # Model-smoke artifacts carry the two fields that tell the pre-D-046 and
    # post-D-046 evidence regimes apart. Other families have neither.
    if "config_sha256" in payload:
        entry["config_sha256"] = payload["config_sha256"]
    if "lane" in payload:
        entry["declares_gate_demotion"] = bool(payload["lane"].get("gate_demotions"))
    return entry


def build() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": PURPOSE,
        "artifacts": {path.name: _entry(path) for path in artifact_paths()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild and fail if the committed manifest would change",
    )
    args = parser.parse_args()

    payload = json.dumps(build(), indent=2, ensure_ascii=False) + "\n"

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
    print(json.dumps({"artifacts": len(build()["artifacts"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
