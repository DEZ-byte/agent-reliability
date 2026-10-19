"""Run a model or adapter on a benchmark and record the result.

Subcommands:
  gsm8k          R0/R1 calculator episodes on GSM8K, with pass^k intervals (GPU)
  order-support  the order-support transfer test with three unseen tools (GPU)
  mmlu           MMLU with no tool offered, to measure what fine-tuning cost (GPU)
  contamination  the same GSM8K tasks with no calculator (GPU)

Example:
    python scripts/evaluate.py gsm8k --help
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from commands.cli import run  # noqa: E402

PROG = "evaluate.py"
COMMANDS = {
    "gsm8k": ("evaluate_gsm8k", "R0/R1 calculator episodes on GSM8K, with pass^k intervals (GPU)"),
    "order-support": (
        "evaluate_order_support",
        "the order-support transfer test with three unseen tools (GPU)",
    ),
    "mmlu": ("evaluate_mmlu", "MMLU with no tool offered, to measure what fine-tuning cost (GPU)"),
    "contamination": ("probe_contamination", "the same GSM8K tasks with no calculator (GPU)"),
}


def main(argv: list[str] | None = None) -> int:
    return run(PROG, "Run a model or adapter on a benchmark and record the result.", COMMANDS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
