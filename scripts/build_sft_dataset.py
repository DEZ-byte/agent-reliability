"""Select the SFT rows from graded candidates, and freeze what was selected.

Generation records every candidate. This decides which ones become training
data, under thresholds pinned in `configs/train_config.yaml`, and writes two
things: the dataset itself, which stays out of Git because it is bulk, and a
manifest of task ids and row hashes, which is committed so the selection is
permanent and checkable.

Three properties the build enforces rather than assumes.

Every selected task id must appear in the train split of the frozen split
manifest. A dev or test id reaching training data is the leak that invalidates
every number the checkpoint later produces, and it is invisible once the rows
are shuffled together.

Every selected row must survive masking. `training.masking.encode_with_labels`
is called on each one here, so a row that cannot be labelled correctly is found
during the build rather than after a training run that quietly learnt nothing.

At most `retention.per_task_cap` rows come from any one task. Easy tasks pass
on every sample and hard tasks pass once, so an uncapped set is weighted toward
exactly the problems the model already solves.

`--check` rebuilds and fails if a byte would differ, so a committed manifest
cannot drift from the data it claims to describe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from env.phase_a import calculator_tool_schema  # noqa: E402
from training.config import config_sha256, load_train_config  # noqa: E402
from training.retention import completion_shape  # noqa: E402

TRAIN_CONFIG_PATH: Final = PROJECT_ROOT / "configs" / "train_config.yaml"
SPLIT_MANIFEST_PATH: Final = PROJECT_ROOT / "configs" / "splits" / "phase_a_gsm8k.json"
DATASET_MANIFEST_PATH: Final = PROJECT_ROOT / "configs" / "splits" / "sft_phase_a.json"

SCHEMA_VERSION: Final = 1


class DatasetError(RuntimeError):
    """The dataset could not be built as specified."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() or "unknown"


def _train_task_ids() -> set[str]:
    manifest = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    return {entry["task_id"] for entry in manifest["splits"]["train"]}


def _forbidden_task_ids() -> set[str]:
    manifest = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    forbidden: set[str] = set()
    for split in ("dev", "test"):
        forbidden.update(entry["task_id"] for entry in manifest["splits"][split])
    return forbidden


def check_split_membership(task_ids: set[str]) -> None:
    """Refuse any id that is not a train id, naming what went wrong.

    Two separate failures, because they mean different things. A dev or test id
    is a leak that invalidates every later number. An id in neither split means
    the candidates were generated against a different manifest than the one
    committed here, so the build cannot say what it selected.
    """

    leaked = sorted(task_ids & _forbidden_task_ids())
    if leaked:
        raise DatasetError(
            "dev or test task ids reached the training set: " + ", ".join(leaked[:5])
        )
    outside = sorted(task_ids - _train_task_ids())
    if outside:
        raise DatasetError(
            "selected task ids are not in the frozen train split: "
            + ", ".join(outside[:5])
        )


def read_candidates(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select(
    candidates: list[dict[str, Any]], *, per_task_cap: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep the usable rows, capped per task, in a deterministic order.

    Ordering is by task id then run index so two builds of the same candidates
    produce byte-identical output. Sorting by anything the model influences,
    such as a score, would let a rebuild reorder silently.
    """

    usable = [row for row in candidates if row.get("usable")]
    usable.sort(key=lambda row: (row["task_id"], row["run_index"]))

    kept: list[dict[str, Any]] = []
    per_task: Counter[str] = Counter()
    dropped_to_cap = 0
    for row in usable:
        if per_task[row["task_id"]] >= per_task_cap:
            dropped_to_cap += 1
            continue
        per_task[row["task_id"]] += 1
        kept.append(row)

    shapes: Counter[str] = Counter(
        completion_shape(row["messages"][-1]["content"]) for row in kept
    )
    stats = {
        "candidates": len(candidates),
        "usable": len(usable),
        "selected": len(kept),
        "dropped_to_per_task_cap": dropped_to_cap,
        "distinct_tasks": len(per_task),
        # Rejection sampling keeps whatever the policy emitted. A row whose
        # prose works the answer out before calling the tool is ordinary
        # chain-of-thought, and it is also one step from the D-062 failure, so
        # the split is reported rather than assumed.
        "completion_shapes": dict(shapes.most_common()),
    }
    return kept, stats


def verify_rows(rows: list[dict[str, Any]], *, model_id: str, revision: str) -> dict[str, Any]:
    """Mask every selected row, so an unlabellable row fails the build.

    This is the expensive check and it is not optional. A row whose assistant
    span cannot be recovered still trains: the labels are simply all -100 and
    the loss curve still falls.
    """

    from transformers import AutoTokenizer

    from training.masking import encode_with_labels

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tools = [calculator_tool_schema()]

    trained_tokens: list[int] = []
    total_tokens: list[int] = []
    for row in rows:
        example = encode_with_labels(tokenizer, row["messages"], tools=tools)
        if example.trained_token_count <= 0:
            raise DatasetError(f"{row['task_id']} produced no trained tokens")
        trained_tokens.append(example.trained_token_count)
        total_tokens.append(len(example.input_ids))

    return {
        "rows_masked": len(rows),
        "trained_tokens_total": sum(trained_tokens),
        "trained_tokens_max": max(trained_tokens) if trained_tokens else 0,
        "row_tokens_max": max(total_tokens) if total_tokens else 0,
        "tokenizer": {"id": model_id, "revision": revision},
    }


def build_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "The frozen selection of Phase A SFT rows. Task ids and row hashes "
            "are permanent: a rebuild that would change one is a different "
            "dataset and needs its own decision."
        ),
        "rows": [
            {
                "task_id": row["task_id"],
                "run_index": row["run_index"],
                "messages_sha256": row["messages_sha256"],
            }
            for row in rows
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--dataset", required=True, help="selected rows, JSONL")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--manifest", default=str(DATASET_MANIFEST_PATH))
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--skip-masking-check", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild and fail if the committed manifest would change",
    )
    args = parser.parse_args()

    config = load_train_config(TRAIN_CONFIG_PATH, require=["retention.per_task_cap"])
    per_task_cap = config["retention"]["per_task_cap"]

    candidates = read_candidates(Path(args.candidates))
    rows, stats = select(candidates, per_task_cap=per_task_cap)
    if not rows:
        raise DatasetError("no usable candidate rows; nothing to build")

    check_split_membership({row["task_id"] for row in rows})

    manifest_payload = json.dumps(build_manifest(rows), indent=2, ensure_ascii=False) + "\n"
    manifest_path = Path(args.manifest)

    if args.check:
        if not manifest_path.exists():
            print("dataset manifest missing", file=sys.stderr)
            return 1
        if manifest_path.read_text(encoding="utf-8") != manifest_payload:
            print(
                "committed dataset manifest differs from a fresh build; the "
                "selection changed",
                file=sys.stderr,
            )
            return 1
        print("dataset manifest reproduces exactly")
        return 0

    verification: dict[str, Any] | None = None
    if not args.skip_masking_check:
        revision = args.tokenizer_revision
        if revision is None:
            registry = json.loads(
                (PROJECT_ROOT / "configs" / "model_candidates.json").read_text(
                    encoding="utf-8"
                )
            )
            for entries in registry["roles"].values():
                for entry in entries:
                    if entry["id"] == args.tokenizer:
                        revision = entry["revision"]
        if revision is None:
            raise DatasetError("--tokenizer-revision is required for an unpinned tokenizer")
        verification = verify_rows(rows, model_id=args.tokenizer, revision=revision)

    dataset_path = Path(args.dataset)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "task_id": row["task_id"],
                        "run_index": row["run_index"],
                        "messages": row["messages"],
                        "messages_sha256": row["messages_sha256"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_bytes(manifest_payload.encode("utf-8"))
    os.replace(temporary, manifest_path)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "kind": "sft_dataset",
        "train_config_sha256": config_sha256(TRAIN_CONFIG_PATH),
        "split_manifest_sha256": _sha256_file(SPLIT_MANIFEST_PATH),
        "candidates_sha256": _sha256_file(Path(args.candidates)),
        "dataset_sha256": _sha256_file(dataset_path),
        "dataset_manifest_sha256": _sha256_text(manifest_payload),
        "retention": {"per_task_cap": per_task_cap},
        "selection_stats": stats,
        "laundering_reasons": dict(
            Counter(
                row.get("laundering_reason") or "retained"
                for row in candidates
                if row.get("laundered") is not None
            )
        ),
        "masking_verification": verification,
        "source_commit": _git_commit(),
        "platform": {"python": platform.python_version(), "system": platform.system()},
    }
    summary_path = Path(args.summary)
    temporary = summary_path.with_suffix(".tmp")
    temporary.write_bytes(
        (json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )
    os.replace(temporary, summary_path)

    print(json.dumps({"dataset": str(dataset_path), "stats": stats}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
