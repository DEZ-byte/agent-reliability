"""Load the frozen Phase A task splits, whichever upstream split they came from.

The split manifest draws from two different upstream GSM8K splits on purpose:
train tasks come from upstream `train`, while dev and test come from upstream
`test` (D-061). A loader that assumes one upstream split works for evaluation
and then refuses every training task, which is what the first baseline runner
did.

The upstream split is read from the task id rather than from the manifest's
prose policy field, because the id is machine-readable and the prose is not.
Every row is checked against the content hash frozen in the manifest, so a
loader that picked the wrong upstream split fails loudly instead of training on
silently mismatched questions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final

from env.phase_a import PhaseATask, parse_gsm8k_answer

SPLIT_NAMES: Final = ("train", "dev", "test")


class SplitError(RuntimeError):
    """A split could not be loaded exactly as it was frozen."""


def content_digest(question: str, answer: str) -> str:
    """The per-row hash frozen in the split manifest.

    Kept here rather than in each caller so the evaluation runner and the
    training data generator cannot drift into computing it differently.
    """

    payload = json.dumps(
        {"question": question, "answer": answer}, sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def upstream_split(task_id: str) -> str:
    """Which upstream dataset split a frozen task id refers to.

    Ids look like `gsm8k:train:7`. The middle field is the upstream split, not
    the project split, and the two differ for dev.
    """

    parts = task_id.split(":")
    if len(parts) != 3:
        raise SplitError(f"task id {task_id!r} is not source:split:index")
    return parts[1]


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def load_split(
    manifest_path: Path,
    split: str,
    *,
    limit: int | None = None,
) -> list[PhaseATask]:
    """Materialise one frozen split as tasks, verifying every row hash.

    Entries are grouped by upstream split so each upstream split is downloaded
    once even when a project split mixes them.
    """

    if split not in SPLIT_NAMES:
        raise SplitError(f"unknown split {split!r}")

    manifest = load_manifest(manifest_path)
    dataset = manifest["dataset"]
    entries = manifest["splits"][split]

    from datasets import load_dataset

    needed = sorted({upstream_split(entry["task_id"]) for entry in entries})
    rows_by_split = {
        name: load_dataset(
            dataset["id"], dataset["config"], split=name, revision=dataset["revision"]
        )
        for name in needed
    }

    tasks: list[PhaseATask] = []
    for entry in entries:
        row = rows_by_split[upstream_split(entry["task_id"])][entry["source_index"]]
        if content_digest(row["question"], row["answer"]) != entry["content_sha256"]:
            raise SplitError(
                f"{entry['task_id']} does not match its recorded content hash"
            )
        tasks.append(
            PhaseATask(
                task_id=entry["task_id"],
                template_id=entry["template_id"],
                question=row["question"],
                gold_answer=parse_gsm8k_answer(row["answer"]),
                source=dataset["id"],
            )
        )
    return tasks[:limit] if limit else tasks


__all__ = [
    "SPLIT_NAMES",
    "SplitError",
    "content_digest",
    "load_manifest",
    "load_split",
    "upstream_split",
]
