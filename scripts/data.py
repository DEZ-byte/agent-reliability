"""Build the frozen task splits and the SFT training data.

Subcommands:
  generate-teacher     roll out the teacher on the GSM8K train split (GPU)
  build-sft            select and freeze the SFT rows from graded candidates
  split-gsm8k          freeze the GSM8K dev and test splits
  split-order-support  freeze the order-support task splits
  split-mmlu           freeze the stratified MMLU subset

Example:
    python scripts/data.py split-gsm8k --help
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from commands.cli import run  # noqa: E402

PROG = "data.py"
COMMANDS = {
    "generate-teacher": ("generate_teacher", "roll out the teacher on the GSM8K train split (GPU)"),
    "build-sft": ("build_sft", "select and freeze the SFT rows from graded candidates"),
    "split-gsm8k": ("split_gsm8k", "freeze the GSM8K dev and test splits"),
    "split-order-support": ("split_order_support", "freeze the order-support task splits"),
    "split-mmlu": ("split_mmlu", "freeze the stratified MMLU subset"),
}


def main(argv: list[str] | None = None) -> int:
    return run(PROG, "Build the frozen task splits and the SFT training data.", COMMANDS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
