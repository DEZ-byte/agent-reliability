"""Freeze a stratified MMLU subset for the general-capability check.

Drawn once and reused by every arm, so the arms are compared on identical
questions. Rerunning with the same seed and the same dataset revision
reproduces the file exactly; a diff means one of those changed.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.utility_split import MMLU_MANIFEST_NAME, content_digest  # noqa: E402

DEFAULT_OUTPUT: Final = PROJECT_ROOT / "configs" / "splits" / MMLU_MANIFEST_NAME
DATASET_ID: Final = "cais/mmlu"
DATASET_CONFIG: Final = "all"
DATASET_SPLIT: Final = "test"
DATASET_REVISION: Final = "c30699e8356da336a370243923dbaf21066bb9fe"
SEED: Final = 20260826
QUESTIONS: Final = 400


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--questions", type=int, default=QUESTIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--allow-download", action="store_true", required=True)
    args = parser.parse_args()

    from datasets import load_dataset

    rows = load_dataset(
        DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT, revision=DATASET_REVISION
    )

    # Group by subject first, then take a proportional share of each, so the
    # sample keeps MMLU's shape instead of drifting toward its biggest subjects.
    by_subject: dict[str, list[int]] = collections.defaultdict(list)
    for index, subject in enumerate(rows["subject"]):
        by_subject[subject].append(index)

    rng = random.Random(args.seed)
    total = len(rows)
    chosen: list[int] = []
    for subject in sorted(by_subject):
        indices = by_subject[subject]
        share = max(1, round(args.questions * len(indices) / total))
        chosen.extend(rng.sample(indices, min(share, len(indices))))
    rng.shuffle(chosen)
    chosen = sorted(chosen[: args.questions])

    entries = []
    for index in chosen:
        row = rows[index]
        entries.append(
            {
                "task_id": f"mmlu:{DATASET_SPLIT}:{index}",
                "subject": row["subject"],
                "source_index": index,
                "content_sha256": content_digest(
                    row["question"], row["choices"], row["answer"]
                ),
            }
        )

    manifest = {
        "schema_version": 1,
        "dataset": {
            "id": DATASET_ID,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "revision": DATASET_REVISION,
            "license": "mit",
            "upstream_rows": total,
        },
        "policy": {
            "seed": args.seed,
            "questions": len(entries),
            "stratified_by": "subject",
            "note": (
                "Held out from everything this project trains on. No arm has "
                "seen these questions, and none of them involve a tool."
            ),
        },
        "questions": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "questions": len(entries),
                "subjects": len({e["subject"] for e in entries}),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
