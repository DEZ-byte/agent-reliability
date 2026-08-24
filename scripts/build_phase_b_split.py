"""Freeze the Phase B task splits into configs/splits/phase_b_orders.json.

The tasks are generated rather than sampled from a dataset, so this writes them
out in full with a hash per row. Rerunning with the same seed reproduces the
file byte for byte; a diff means the generator changed, which is a decision
rather than a refresh.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from env.phase_b_tasks import GENERATOR_SEED, build_manifest  # noqa: E402

DEFAULT_OUTPUT: Final = PROJECT_ROOT / "configs" / "splits" / "phase_b_orders.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=GENERATOR_SEED)
    args = parser.parse_args()

    manifest = build_manifest(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "splits": {k: len(v) for k, v in manifest["splits"].items()},
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
