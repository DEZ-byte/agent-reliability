"""Build frozen Phase A task splits from a pinned GSM8K revision.

Writes ID manifests, not data. BLUEPRINT_v2 section 5.4 requires the splits to
be committed as JSON lists so a later run cannot quietly evaluate on something
it trained on. The dataset itself is never redistributed from this repository.

Deterministic by construction: a fixed revision, a fixed seed, and a sort before
sampling. Running this twice produces byte-identical manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
SPLIT_DIR: Final = PROJECT_ROOT / "configs" / "splits"

DATASET_ID: Final = "openai/gsm8k"
DATASET_REVISION: Final = "740312add88f781978c0658806c59bc2815b9866"
DATASET_CONFIG: Final = "main"

SEED: Final = 20260820
TEST_SIZE: Final = 150
DEV_SIZE: Final = 100
TRAIN_SIZE: Final = 1000

SCHEMA_VERSION: Final = 1


def _load(split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows = load_dataset(
        DATASET_ID, DATASET_CONFIG, split=split, revision=DATASET_REVISION
    )
    return [{"question": r["question"], "answer": r["answer"]} for r in rows]


def _task_id(split: str, index: int) -> str:
    return f"gsm8k:{split}:{index}"


def _content_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sample(rows: list[dict[str, Any]], split: str, size: int, rng: random.Random):
    """Sample stable IDs, refusing to reuse a question that appears twice.

    GSM8K has no template field, so an item is its own template. Exact-duplicate
    questions are the only paraphrase twins detectable here, and dropping them
    keeps a duplicate from straddling two splits.
    """

    seen: set[str] = set()
    unique: list[int] = []
    for index, row in enumerate(rows):
        key = row["question"].strip()
        if key in seen:
            continue
        seen.add(key)
        unique.append(index)
    if size > len(unique):
        raise SystemExit(f"{split}: asked for {size} tasks, only {len(unique)} unique")
    chosen = sorted(rng.sample(unique, size))
    return [
        {
            "task_id": _task_id(split, index),
            "template_id": _task_id(split, index),
            "source_index": index,
            "content_sha256": _content_hash(rows[index]),
        }
        for index in chosen
    ]


def build() -> dict[str, Any]:
    train_rows = _load("train")
    test_rows = _load("test")

    rng = random.Random(SEED)
    # Dev and test both come from the held-out split and must not overlap.
    held_out = _sample(test_rows, "test", TEST_SIZE + DEV_SIZE, rng)
    test = held_out[:TEST_SIZE]
    dev = held_out[TEST_SIZE:]
    train = _sample(train_rows, "train", TRAIN_SIZE, rng)

    overlap = {t["task_id"] for t in test} & {d["task_id"] for d in dev}
    if overlap:
        raise SystemExit(f"dev and test overlap on {len(overlap)} tasks")

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "id": DATASET_ID,
            "config": DATASET_CONFIG,
            "revision": DATASET_REVISION,
            "license": "mit",
            "upstream_train_rows": len(train_rows),
            "upstream_test_rows": len(test_rows),
        },
        "policy": {
            "seed": SEED,
            "template_granularity": (
                "GSM8K has no template field, so each item is its own template. "
                "Exact-duplicate questions are removed before sampling so a "
                "duplicate cannot straddle two splits."
            ),
            "test_is_evaluation_only": True,
            "dev_and_test_are_disjoint": True,
            "train_source_split": "upstream train",
            "dev_and_test_source_split": "upstream test",
        },
        "splits": {"train": train, "dev": dev, "test": test},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(SPLIT_DIR / "phase_a_gsm8k.json"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild and fail if the committed manifest would change",
    )
    args = parser.parse_args()

    manifest = build()
    payload = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    path = Path(args.output)

    if args.check:
        if not path.exists():
            print("manifest missing", file=sys.stderr)
            return 1
        if path.read_text(encoding="utf-8") != payload:
            print("committed manifest differs from a fresh build", file=sys.stderr)
            return 1
        print("manifest reproduces exactly")
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(payload.encode("utf-8"))
    os.replace(temporary, path)
    counts = {name: len(rows) for name, rows in manifest["splits"].items()}
    print(json.dumps({"output": str(path), "counts": counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
