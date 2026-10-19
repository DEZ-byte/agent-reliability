"""Analyze recorded results on the CPU.

No model weights are loaded. backfill-tokens needs transformers and the cached
pinned tokenizers, and plots needs matplotlib.

Subcommands:
  compare-models           pair two models' episode logs: bootstrap interval, then tests
  compare-mmlu             pair the MMLU arms question by question
  failure-taxonomy         classify every failed GSM8K test attempt by how it failed
  order-support-gates      split the order-support gate violations by gate
  order-support-templates  split the order-support results by request template
  order-support-modes      pair single-turn and multi-turn order support task by task
  backfill-tokens          record token counts for existing episode logs
  plots                    draw the figures from the frozen results
  verify-manifest          check that results/ still matches its manifest

Example:
    python scripts/analyze.py failure-taxonomy --help
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from commands.cli import run  # noqa: E402

PROG = "analyze.py"


VERIFY_MANIFEST_HELP = """usage: analyze.py verify-manifest

Check that every result artifact matches its SHA-256 and byte length in
results/artifact_manifest.json, and that no file is missing from it. Takes no
options. To retire a replaced file, run
`python scripts/build_artifact_manifest.py --retire NAME` directly."""


def verify_manifest(argv: list[str]) -> int:
    """Run the manifest check in its own process and pass its exit code on."""

    if argv in (["-h"], ["--help"]):
        print(VERIFY_MANIFEST_HELP)
        return 0
    if argv:
        print(VERIFY_MANIFEST_HELP, file=sys.stderr)
        print(f"analyze.py verify-manifest: takes no options, got {' '.join(argv)}", file=sys.stderr)
        return 2
    script = PROJECT_ROOT / "scripts" / "build_artifact_manifest.py"
    return subprocess.run([sys.executable, str(script), "--check"], check=False).returncode


COMMANDS = {
    "compare-models": (
        "compare_models",
        "pair two models' episode logs: bootstrap interval, then tests",
    ),
    "compare-mmlu": ("compare_mmlu", "pair the MMLU arms question by question"),
    "failure-taxonomy": (
        "failure_taxonomy",
        "classify every failed GSM8K test attempt by how it failed",
    ),
    "order-support-gates": ("order_support_gates", "split the order-support gate violations by gate"),
    "order-support-templates": (
        "order_support_templates",
        "split the order-support results by request template",
    ),
    "order-support-modes": (
        "order_support_modes",
        "pair single-turn and multi-turn order support task by task",
    ),
    "backfill-tokens": ("backfill_tokens", "record token counts for existing episode logs"),
    "plots": ("plots", "draw the figures from the frozen results"),
    "verify-manifest": (verify_manifest, "check that results/ still matches its manifest"),
}


def main(argv: list[str] | None = None) -> int:
    return run(
        PROG,
        "Analyze recorded results on the CPU. No model weights are loaded; "
        "backfill-tokens needs the cached tokenizers and plots needs matplotlib.",
        COMMANDS,
        argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
