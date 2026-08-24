"""Decide which training prompts can produce a gradient, and which cannot.

A GRPO advantage is group-relative. Every candidate in a group is scored, the
group mean is subtracted, and what remains is the learning signal. When all G
candidates score alike the differences are all zero, so the group contributes
exactly nothing however large its reward. The first GRPO run here measured
roughly a quarter of its steps in that state, which is a quarter of the budget
spent on arithmetic that cancels.

Worth being careful about what that does and does not explain. It is a real
waste and removing it buys back budget. It is not on its own the reason the run
came out null: the other three quarters did carry spread, and they reached the
same dev peak at 1e-6 and at 1e-5. Whether filtering moves the score is the
question this arm exists to answer, not a prediction it starts from.

Two published fixes exist and they differ in where they act.

DAPO (arXiv 2503.14476) resamples inside the training loop: it over-generates
prompts, discards every group whose candidates are all correct or all wrong,
and refills the batch until it is full of groups with spread. In DAPO's own
ablation this is the single largest contribution, worth 8 of the 20 points
their full recipe adds over naive GRPO.

GRESO (arXiv 2506.02177) makes the cheaper observation that deadness largely
persists, so the set can be measured once instead of rediscovered every step.
Their figure for how much it persists is easy to quote in the wrong direction,
and this project quoted it wrongly at first. GRESO reports that over 90% of the
prompts dead in the current epoch were also dead in an earlier one, which says
where today's dead prompts came from. The direction this design needs is the
opposite conditional, and they measure it separately: roughly 20% of previously
dead prompts become live again. Probing once therefore discards about one
prompt in five that would have carried gradient later.

This project takes the GRESO route, and the reason is a constraint rather than
a preference. `unsloth` rewrites `GRPOTrainer` on import, so overriding its
generation loop is fragile across versions, and over-generating candidates does
not fit an 8 GB card. Measuring once and filtering the dataset gets most of the
effect with none of that risk. It is weaker than DAPO's, and weaker than
GRESO's own method, which keeps sampling dead prompts occasionally for exactly
the revival reason above. Results built on this should say so rather than claim
DAPO.

The criterion has no tunable threshold. A group teaches something when at least
one candidate is correct and at least one is not. There is nothing to choose,
so there is nothing to choose badly after seeing which choice flatters a
number.

This module is the pure half: given what one group scored, say whether it can
teach and why not if it cannot. The GPU pass that produces those scores lives
in `scripts/probe_prompt_variance.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final, Iterable, Protocol, Sequence

LIVE: Final = "live"
DEAD_ALL_CORRECT: Final = "dead_all_correct"
DEAD_ALL_WRONG: Final = "dead_all_wrong"

SCHEMA_KIND: Final = "prompt_variance_probe"


class PromptVarianceError(RuntimeError):
    """A probe artifact could not be used as a filter."""


class Scored(Protocol):
    """The part of a scored candidate this module needs.

    Structural rather than imported: `CompletionScore` satisfies it, and so
    does a test double, without dragging the sandbox and gate engine into a
    module that only does arithmetic.
    """

    correct: bool
    total: float


@dataclass(frozen=True, slots=True)
class GroupVerdict:
    """One prompt's group, and whether training on it would do anything."""

    task_id: str
    group_size: int
    correct: int
    total_std: float
    liveness: str

    @property
    def solve_rate(self) -> float:
        return self.correct / self.group_size if self.group_size else 0.0

    @property
    def teaches(self) -> bool:
        return self.liveness == LIVE

    def as_row(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "group_size": self.group_size,
            "correct": self.correct,
            "solve_rate": self.solve_rate,
            "total_std": self.total_std,
            "liveness": self.liveness,
        }


def standard_deviation(values: Sequence[float]) -> float:
    """Sample standard deviation, zero for fewer than two values."""

    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def classify_group(scores: Sequence[Scored], *, task_id: str) -> GroupVerdict:
    """Whether this prompt's group carries a gradient, by DAPO's criterion.

    Liveness is decided on accuracy, not on the total reward, and the two can
    disagree. The total also carries an efficiency term that moves with the
    token count, so a group where all eight candidates are correct can still
    show a whisper of spread. That spread is real but it teaches brevity, not
    correctness, and DAPO's criterion is the one with published evidence
    behind it. `disagreements` below counts how often the two part company so
    the choice stays visible rather than assumed.
    """

    if not scores:
        raise PromptVarianceError(f"{task_id}: an empty group cannot be classified")
    correct = sum(1 for s in scores if s.correct)
    if correct == 0:
        liveness = DEAD_ALL_WRONG
    elif correct == len(scores):
        liveness = DEAD_ALL_CORRECT
    else:
        liveness = LIVE
    return GroupVerdict(
        task_id=task_id,
        group_size=len(scores),
        correct=correct,
        total_std=standard_deviation([s.total for s in scores]),
        liveness=liveness,
    )


def disagreements(verdicts: Iterable[GroupVerdict]) -> dict[str, int]:
    """Where the accuracy criterion and the actual gradient part company.

    `dead_but_nonzero_std` is a group this filter drops even though TRL would
    have produced some gradient from it. `live_but_zero_std` should be empty by
    construction, since differing accuracy implies differing total; if it is
    ever non-empty the reward has changed shape and this filter is no longer
    measuring what it claims.
    """

    dead_nonzero = 0
    live_zero = 0
    for verdict in verdicts:
        if verdict.teaches and verdict.total_std == 0.0:
            live_zero += 1
        elif not verdict.teaches and verdict.total_std > 0.0:
            dead_nonzero += 1
    return {"dead_but_nonzero_std": dead_nonzero, "live_but_zero_std": live_zero}


def summarise(verdicts: Sequence[GroupVerdict]) -> dict[str, Any]:
    """What the probe found, in the shape an artifact records."""

    if not verdicts:
        return {"prompts": 0}
    counts = {
        LIVE: sum(1 for v in verdicts if v.liveness == LIVE),
        DEAD_ALL_CORRECT: sum(1 for v in verdicts if v.liveness == DEAD_ALL_CORRECT),
        DEAD_ALL_WRONG: sum(1 for v in verdicts if v.liveness == DEAD_ALL_WRONG),
    }
    total = len(verdicts)
    return {
        "prompts": total,
        "counts": counts,
        "live_fraction": counts[LIVE] / total,
        "dead_fraction": (counts[DEAD_ALL_CORRECT] + counts[DEAD_ALL_WRONG]) / total,
        "mean_solve_rate": sum(v.solve_rate for v in verdicts) / total,
        "mean_total_std": sum(v.total_std for v in verdicts) / total,
        "disagreements": disagreements(verdicts),
    }


def live_task_ids(payload: dict[str, Any]) -> set[str]:
    """The keep-set from a probe artifact, refusing anything ambiguous.

    A probe that did not execute, or that recorded no prompts, is a planning
    stub rather than a measurement. Training against it would silently keep
    every prompt and report that it had filtered.
    """

    if payload.get("kind") != SCHEMA_KIND:
        raise PromptVarianceError(
            f"artifact kind is {payload.get('kind')!r}, expected {SCHEMA_KIND!r}"
        )
    if not payload.get("executed"):
        raise PromptVarianceError("probe artifact was not executed; it has no verdicts")
    rows = payload.get("prompts") or []
    if not rows:
        raise PromptVarianceError("probe artifact records no prompts")
    keep = {row["task_id"] for row in rows if row.get("liveness") == LIVE}
    if not keep:
        raise PromptVarianceError(
            "every probed prompt was dead; filtering would leave nothing to train on"
        )
    return keep


__all__ = [
    "DEAD_ALL_CORRECT",
    "DEAD_ALL_WRONG",
    "LIVE",
    "SCHEMA_KIND",
    "GroupVerdict",
    "PromptVarianceError",
    "classify_group",
    "disagreements",
    "live_task_ids",
    "standard_deviation",
    "summarise",
]
