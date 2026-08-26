"""Load the frozen general-knowledge subset used to measure what training cost.

MMLU has 14,042 test questions and running all of them three times on an 8 GB
card is hours of GPU for a number that a stratified sample answers just as well.
The sample is drawn once, frozen with a content hash per question, and reused by
every arm, so the arms are compared on identical questions rather than on
independent draws that happen to differ in difficulty.

Stratified by subject rather than sampled uniformly. MMLU's subjects vary
enormously in size, and a uniform draw would fill the sample with the largest
ones. A 1.7B model's score would then move with the subject mix rather than with
anything training did.

The manifest stores indices and hashes rather than the questions themselves,
exactly as the Phase A split does, because the dataset is already public and
vendoring it would duplicate something upstream maintains.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final, NamedTuple

MMLU_MANIFEST_NAME: Final = "utility_mmlu.json"


class UtilitySplitError(RuntimeError):
    """The utility split could not be loaded exactly as it was frozen."""


class UtilityQuestion(NamedTuple):
    """One multiple-choice question, with its answer index."""

    task_id: str
    subject: str
    question: str
    choices: tuple[str, ...]
    gold_index: int


def content_digest(question: str, choices: list[str], answer: int) -> str:
    """The per-row hash frozen in the manifest.

    Covers the choices and the answer index as well as the question, since a
    question whose options were reordered upstream is a different question with
    the same text.
    """

    payload = json.dumps(
        {"question": question, "choices": list(choices), "answer": answer},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_questions(manifest_path: Path, *, limit: int | None = None):
    """Materialise the frozen subset, verifying every row hash."""

    manifest = load_manifest(manifest_path)
    dataset = manifest["dataset"]
    entries = manifest["questions"]

    from datasets import load_dataset

    rows = load_dataset(
        dataset["id"],
        dataset["config"],
        split=dataset["split"],
        revision=dataset["revision"],
    )

    questions: list[UtilityQuestion] = []
    for entry in entries:
        row = rows[entry["source_index"]]
        digest = content_digest(row["question"], row["choices"], row["answer"])
        if digest != entry["content_sha256"]:
            raise UtilitySplitError(
                f"{entry['task_id']} does not match its recorded content hash"
            )
        questions.append(
            UtilityQuestion(
                task_id=entry["task_id"],
                subject=entry["subject"],
                question=row["question"],
                choices=tuple(row["choices"]),
                gold_index=int(row["answer"]),
            )
        )
    return questions[:limit] if limit else questions


__all__ = [
    "MMLU_MANIFEST_NAME",
    "UtilityQuestion",
    "UtilitySplitError",
    "content_digest",
    "load_manifest",
    "load_questions",
]
