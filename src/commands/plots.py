"""Draw the two README figures from the frozen results.

assets/reliability_cost_pareto.png: pass@4, pass^1 and pass^4 per model on
GSM8K tool use at R1, and pass^4 against the cost of one attempt.

assets/failure_and_transfer_breakdown.png: how the GSM8K attempts failed, the
order-support transfer rates, and the share of GRPO groups with no reward
variance.

Every number is read from results/ and from the token-counts artifact, which
is found by glob unless --token-counts names it. The figures are not results
themselves, so this command writes no provenance and runs on any tree. Needs
matplotlib.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.plotting import (  # noqa: E402
    PlotError,
    find_token_counts,
    reliability_cost_data,
    render_reliability_cost,
    render_transfer,
    transfer_data,
)

RESULTS_DIR: Final = PROJECT_ROOT / "results"
ASSETS_DIR: Final = PROJECT_ROOT / "assets"
RELIABILITY_COST: Final = "reliability_cost_pareto.png"
TRANSFER: Final = "failure_and_transfer_breakdown.png"

# Refusals exit 1 with a one-line message instead of a traceback.
REFUSALS: Final = (PlotError,)


def _write(render, data, path: Path) -> None:
    """Render to a temporary file, then move it into place."""

    temporary = path.with_name(path.stem + ".tmp.png")
    render(data, temporary)
    os.replace(temporary, path)


def _shown(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--token-counts",
        type=Path,
        default=None,
        help="token-counts artifact (default: the one results/token-counts-*.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ASSETS_DIR,
        help="folder for the two PNG files (default: assets/)",
    )
    args = parser.parse_args(argv)

    try:
        import matplotlib  # noqa: F401
    except ImportError as error:
        raise PlotError("matplotlib is not installed; install it to draw the figures") from error

    token_counts = args.token_counts or find_token_counts(RESULTS_DIR)
    reliability = reliability_cost_data(RESULTS_DIR, token_counts)
    transfer = transfer_data(RESULTS_DIR)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        (render_reliability_cost, reliability, args.output_dir / RELIABILITY_COST),
        (render_transfer, transfer, args.output_dir / TRANSFER),
    )
    for render, data, path in outputs:
        _write(render, data, path)
        print(f"wrote {_shown(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
