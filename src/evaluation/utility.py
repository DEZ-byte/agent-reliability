"""Score a model on a task with no tools in it, to see what training cost.

Every number this project has published so far was measured on the task the
model was trained for. That answers whether fine-tuning worked and says nothing
about what it broke, which is the first thing anyone sensible asks. A model that
gained 22 points on calculator use and lost general ability has not obviously
improved.

Two things are measured here and the second matters more.

The first is plain accuracy on multiple-choice questions, which is the ordinary
retention check: does the model still know things.

The second is whether the model reaches for a tool when there is no tool. The
training data was one tool-call per example, every example, so the pressure to
emit a tool call on anything that looks like a question is real and it is
exactly what specialisation would look like from the inside. Nothing in this
benchmark offers a tool, so a `<tool_call>` block here is a habit that has
escaped its context. That rate is reported alongside accuracy, because a model
can keep its knowledge and still become unusable outside its training task.

Answer extraction is deliberately generous. A model that answers "B" and one
that answers "The answer is (B)." know the same thing, and a strict parser would
score the second as ignorance. What is not generous is the fallback: when no
choice can be extracted the answer is scored wrong and counted separately, so an
extraction failure can never be mistaken for a knowledge failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final, Sequence

CHOICE_LABELS: Final = ("A", "B", "C", "D")

# Ordered most-specific first. "answer is B" must win over a stray "B" earlier
# in the reasoning, so the explicit forms are tried before the bare letter.
_PATTERNS: Final = (
    re.compile(r"\banswer\s*(?:is|:)\s*\(?\*{0,2}([A-D])\b", re.IGNORECASE),
    re.compile(r"^\s*\(?\*{0,2}([A-D])[\)\.\*:]", re.MULTILINE),
    re.compile(r"\boption\s+(?:is\s+)?\(?\*{0,2}([A-D])\b", re.IGNORECASE),
    re.compile(r"^\s*\*{0,2}([A-D])\s*$", re.MULTILINE),
)

_TOOL_CALL_MARKERS: Final = ("<tool_call>", "</tool_call>", '"arguments"', '"parameters"')


@dataclass(frozen=True, slots=True)
class UtilityScore:
    """One answered question, and how the answer was arrived at."""

    correct: bool
    extracted: str | None
    emitted_tool_call: bool
    generated_chars: int

    @property
    def extraction_failed(self) -> bool:
        return self.extracted is None


def extract_choice(completion: str) -> str | None:
    """The letter this completion settled on, or None if it named none.

    Tried in order of how explicit the phrasing is. Returning None rather than
    guessing keeps an unreadable answer separate from a wrong one, which is the
    difference between a model that has lost knowledge and one that has lost the
    ability to answer in the requested shape.
    """

    if not completion:
        return None
    for pattern in _PATTERNS:
        match = pattern.search(completion)
        if match:
            return match.group(1).upper()
    return None


def emitted_tool_call(completion: str) -> bool:
    """Whether the model reached for a tool that was never offered.

    Matched on markers rather than by parsing, because a half-written call still
    shows the habit. The point is not whether the call would have worked.
    """

    return any(marker in completion for marker in _TOOL_CALL_MARKERS)


def score_completion(completion: str, *, gold_index: int) -> UtilityScore:
    """Score one multiple-choice answer."""

    if not 0 <= gold_index < len(CHOICE_LABELS):
        raise ValueError(f"gold_index {gold_index} is outside the choice range")
    extracted = extract_choice(completion)
    return UtilityScore(
        correct=extracted == CHOICE_LABELS[gold_index],
        extracted=extracted,
        emitted_tool_call=emitted_tool_call(completion),
        generated_chars=len(completion),
    )


def summarise(scores: Sequence[UtilityScore]) -> dict[str, Any]:
    """What a whole benchmark run looked like."""

    if not scores:
        return {"questions": 0}
    total = len(scores)
    return {
        "questions": total,
        "accuracy": sum(1 for s in scores if s.correct) / total,
        # Reported next to accuracy on purpose. A model whose accuracy holds
        # while this climbs has not kept its general ability; it has kept its
        # knowledge and lost the ability to answer plainly.
        "tool_call_rate": sum(1 for s in scores if s.emitted_tool_call) / total,
        "extraction_failure_rate": sum(1 for s in scores if s.extraction_failed)
        / total,
        "mean_generated_chars": sum(s.generated_chars for s in scores) / total,
    }


__all__ = [
    "CHOICE_LABELS",
    "UtilityScore",
    "emitted_tool_call",
    "extract_choice",
    "score_completion",
    "summarise",
]
