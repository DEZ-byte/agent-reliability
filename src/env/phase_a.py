"""Phase A: single-turn math tasks solved through a calculator tool.

BLUEPRINT_v2 section 5.1 names this honestly: GSM8K wrapped in this project's
own calculator environment. There is no benchmark called "GSM8K-Tool".

The point of the wrapper is the grading rule. A model that writes the right
number in prose scores nothing. Accuracy comes from a tool call that actually
executed in the sandbox and returned the right value, which is what section 7.0
means by execution-backed accuracy, and what makes a memorised answer worthless.
"""

from __future__ import annotations

import math
import re
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from env.models import EnvironmentOutcome, EpisodeTrace, OutcomeSource
from env.sandbox import SandboxViolation, run_code
from env.tools import ToolRegistry, ToolSpec, ToolState

CALCULATOR_TOOL_NAME: Final = "calculator"

# GSM8K answers are integers, but a model may reach one through division and
# land on a float. Compare numerically with a tolerance rather than by string.
ANSWER_TOLERANCE: Final = 1e-6

_EXPRESSION_MAX_CHARS: Final = 512
_SANDBOX_TIMEOUT_SECONDS: Final = 2.0

# GSM8K ships its gold answer after a "####" marker, with thousands separators.
_GSM8K_ANSWER_RE: Final = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)\s*$")


class CalculatorArgs(BaseModel):
    """Arguments accepted by the calculator tool."""

    model_config = ConfigDict(extra="forbid")

    expression: str = Field(min_length=1, max_length=_EXPRESSION_MAX_CHARS)


class PhaseATask(BaseModel):
    """One single-turn task with a verifiable numeric answer."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    task_id: str = Field(min_length=1)
    template_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    gold_answer: float
    source: str = Field(min_length=1)
    tool_required: bool = True


def parse_gsm8k_answer(answer_field: str) -> float:
    """Read the gold number out of a GSM8K answer field.

    Raises rather than guessing. A task whose answer cannot be read is a task
    that cannot be graded, and silently dropping the marker would turn a
    parsing bug into a wrong score.
    """

    match = _GSM8K_ANSWER_RE.search(answer_field.strip())
    if match is None:
        raise ValueError("GSM8K answer field has no '#### <number>' marker")
    return float(match.group(1).replace(",", ""))


def evaluate_expression(expression: str) -> float:
    """Evaluate one arithmetic expression inside the sandbox.

    The sandbox is the only path to a number. It rejects imports, dunder
    access, and long-running code, and it returns the final value as text.
    """

    result = run_code(
        expression,
        timeout_seconds=_SANDBOX_TIMEOUT_SECONDS,
    )
    if not result.succeeded:
        raise ValueError(
            f"{result.exception_type}: {result.exception_message}"
        )
    if result.value_repr is None:
        raise ValueError("expression produced no value")
    try:
        value = float(result.value_repr)
    except ValueError as exc:
        raise ValueError(
            f"expression produced a non-numeric value: {result.value_repr!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError("expression produced a non-finite value")
    return value


def _calculator_handler(args: BaseModel, state: ToolState) -> float:
    expression = args.expression  # type: ignore[attr-defined]
    try:
        value = evaluate_expression(expression)
    except SandboxViolation as exc:
        # A violation is the model's failure, recorded as a failed dispatch
        # rather than an exception that aborts the episode.
        raise ValueError(f"sandbox violation: {exc}") from exc
    state["last_calculator_value"] = value
    return value


def build_phase_a_registry() -> ToolRegistry:
    """Register the Phase A tool set.

    The calculator is read-only: it computes and returns a number without
    changing anything a policy gate would protect. It therefore declares no
    gates, which `ToolSpec` enforces.
    """

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name=CALCULATOR_TOOL_NAME,
            args_model=CalculatorArgs,
            handler=_calculator_handler,
        )
    )
    return registry


def executed_answers(trace: EpisodeTrace) -> list[float]:
    """Numbers this episode actually obtained from a successful tool call."""

    answers: list[float] = []
    for event in trace.tool_events:
        if not (event.dispatched and event.succeeded):
            continue
        if event.call.name != CALCULATOR_TOOL_NAME:
            continue
        if isinstance(event.output, (int, float)) and not isinstance(
            event.output, bool
        ):
            answers.append(float(event.output))
    return answers


def grade_episode(trace: EpisodeTrace, task: PhaseATask) -> EnvironmentOutcome:
    """Decide correctness from executed tool results only.

    The last successful calculator result is the episode's answer. Prose is
    never read. If the model called no tool, there is no answer to grade, and
    the outcome is incorrect regardless of what it wrote.
    """

    answers = executed_answers(trace)
    if not answers:
        return EnvironmentOutcome(
            correct=False, source=OutcomeSource.SANDBOX_RESULT
        )
    correct = math.isclose(
        answers[-1], task.gold_answer, rel_tol=0.0, abs_tol=ANSWER_TOLERANCE
    )
    return EnvironmentOutcome(
        correct=correct, source=OutcomeSource.SANDBOX_RESULT
    )
