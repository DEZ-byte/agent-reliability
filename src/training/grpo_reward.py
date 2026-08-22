"""Score a GRPO completion by executing it, never by reading it.

BLUEPRINT_v2 section 7.0 bans substring rewards. The reward here is the same
composite the evaluator grades with: accuracy from the sandbox result, a format
term over emitted blocks, a gate term replayed from the event log, and an
efficiency penalty. One parser, one gate engine and one reward function serve
both the runtime scaffold and training, so the constraint the model is trained
against cannot drift from the one it is measured against.

The practical consequence is that every candidate costs a real tool execution.
That is the price of an execution-backed reward and it is not negotiable: a
cheaper proxy is exactly the substring reward the blueprint forbids.

Section 7.3 warns about the failure mode this setup invites. GRPO advantages are
group-relative, so any reward component identical across all G candidates
contributes exactly nothing to the gradient. A group where every candidate
scores the same is not a weak signal, it is no signal, and back-propagating it
wastes a step. `group_health` reports the per-component spread so that can be
seen rather than assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final, Sequence

from agent.dialects import normalise_tool_dialect
from agent.gates import GateEngine, GateMode
from agent.parser import parse_tool_calls
from env.models import EnvironmentOutcome, OutcomeSource
from env.phase_a import answer_from_events, build_phase_a_registry
from env.tools import ToolRegistry
from training.retention import laundering_verdict
from training.rewards import score_episode

ANSWER_TOLERANCE: Final = 1e-6


@dataclass(frozen=True, slots=True)
class CompletionScore:
    """One candidate's reward and the evidence behind it."""

    total: float
    accuracy: float
    format: float
    gate: float
    efficiency: float
    correct: bool
    executed_calls: int
    laundered: bool


def score_completion(
    completion: str,
    *,
    gold_answer: float,
    question: str = "",
    registry: ToolRegistry,
    gate_engine: GateEngine,
    normalise_dialect: bool = False,
) -> CompletionScore:
    """Execute one candidate and score it on execution evidence alone."""

    text = normalise_tool_dialect(completion) if normalise_dialect else completion
    trace = registry.execute(
        parse_tool_calls(text),
        {},
        gate_engine=gate_engine,
        gate_mode=GateMode.ENFORCE,
    )
    answer = answer_from_events(trace.tool_events)
    correct = answer is not None and math.isclose(
        answer, gold_answer, rel_tol=0.0, abs_tol=ANSWER_TOLERANCE
    )
    breakdown = score_episode(
        trace,
        EnvironmentOutcome(correct=correct, source=OutcomeSource.SANDBOX_RESULT),
        tool_required=True,
        gate_engine=gate_engine,
    )
    # Section 7.0 is explicit that a call restating a remembered answer still
    # scores +1.0, measured rather than penalised. GRPO is the sharpest possible
    # version of that pressure: laundering is the cheapest route to full
    # accuracy, so the rate is tracked every step rather than checked at the end.
    laundered = False
    for event in reversed(trace.tool_events):
        expression = event.call.arguments.get("expression")
        if isinstance(expression, str):
            laundered = laundering_verdict(
                expression=expression,
                question=question,
                gold_answer=gold_answer,
                min_question_match_ratio=0.0,
            ).laundered
            break

    return CompletionScore(
        laundered=laundered,
        total=breakdown.total,
        accuracy=breakdown.accuracy,
        format=breakdown.format,
        gate=breakdown.gate,
        efficiency=breakdown.efficiency,
        correct=correct,
        executed_calls=breakdown.executed_calls,
    )


def group_health(scores: Sequence[CompletionScore]) -> dict[str, Any]:
    """Per-component spread inside one group, and whether it can teach anything.

    Section 7.3 point 3b: log the within-group standard deviation of each reward
    component separately. A component with zero spread is invisible to a
    group-relative advantage however large its value, so an arm can look
    rewarded while contributing no gradient at all.
    """

    def spread(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))

    components = {
        "total": [s.total for s in scores],
        "accuracy": [s.accuracy for s in scores],
        "format": [s.format for s in scores],
        "gate": [s.gate for s in scores],
        "efficiency": [s.efficiency for s in scores],
    }
    stds = {name: spread(values) for name, values in components.items()}
    return {
        "group_size": len(scores),
        "std": stds,
        "zero_variance": stds["total"] == 0.0,
        "correct_fraction": (
            sum(1 for s in scores if s.correct) / len(scores) if scores else 0.0
        ),
        "no_tool_call_fraction": (
            sum(1 for s in scores if s.executed_calls == 0) / len(scores)
            if scores
            else 0.0
        ),
        "laundered_fraction": (
            sum(1 for s in scores if s.laundered) / len(scores) if scores else 0.0
        ),
    }


def make_reward_function(*, normalise_dialect: bool = False, health_log: list | None = None):
    """Build the callable TRL invokes, closing over one registry and gate engine.

    TRL calls this as `f(prompts=..., completions=..., **columns)` where every
    dataset column arrives already expanded to one entry per generation, so the
    gold answer for each candidate travels alongside it.
    """

    registry = build_phase_a_registry()
    gate_engine = GateEngine.from_mapping({})

    def reward(completions, gold_answer, question=None, **kwargs):
        questions = question or [""] * len(completions)
        scores = [
            score_completion(
                text,
                gold_answer=gold,
                question=q,
                registry=registry,
                gate_engine=gate_engine,
                normalise_dialect=normalise_dialect,
            )
            for text, gold, q in zip(completions, gold_answer, questions)
        ]
        if health_log is not None:
            size = kwargs.get("num_generations") or len(scores)
            for start in range(0, len(scores), size):
                chunk = scores[start : start + size]
                if len(chunk) > 1:
                    health_log.append(group_health(chunk))
        return [s.total for s in scores]

    reward.__name__ = "execution_backed_composite"
    return reward


__all__ = [
    "ANSWER_TOLERANCE",
    "CompletionScore",
    "group_health",
    "make_reward_function",
    "score_completion",
]
