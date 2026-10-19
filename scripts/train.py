"""Train the SFT and GRPO adapters, and check what training did.

Subcommands:
  sft                train the SFT adapter on pre-masked rows (GPU)
  grpo               train the GRPO arm from the SFT checkpoint (GPU)
  select-checkpoint  score every saved checkpoint on dev and pick by the pinned rule (GPU)
  verify-masking     prove the training labels cover assistant tokens only
  weight-change      measure how far GRPO moved the adapter from its SFT start

Example:
    python scripts/train.py sft --help
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from commands.cli import run  # noqa: E402

PROG = "train.py"
COMMANDS = {
    "sft": ("train_sft", "train the SFT adapter on pre-masked rows (GPU)"),
    "grpo": ("train_grpo", "train the GRPO arm from the SFT checkpoint (GPU)"),
    "select-checkpoint": (
        "select_checkpoint",
        "score every saved checkpoint on dev and pick by the pinned rule (GPU)",
    ),
    "verify-masking": ("verify_masking", "prove the training labels cover assistant tokens only"),
    "weight-change": ("weight_change", "measure how far GRPO moved the adapter from its SFT start"),
}


def main(argv: list[str] | None = None) -> int:
    return run(PROG, "Train the SFT and GRPO adapters, and check what training did.", COMMANDS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
