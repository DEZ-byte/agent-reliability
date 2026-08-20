"""Framework-neutral R0 and R1 episode loops for Phase A.

`RUNG_PROTOCOL.md` section 4 defines these rungs, and section 1 defines the
counters every episode must record. This module implements both without
importing a model, a GPU, or an agent framework: the policy is a callable that
takes messages and returns text. That is what makes the rung semantics testable
on CPU, which matters because the whole study rests on the two rungs differing
by exactly one thing.

The difference is one model decision. R0 gets a single generation per agent
turn and no second chance when it fails. R1 gets one additional generation that
sees the structured failure. Nothing else changes: same checkpoint, same tools,
same environment-turn cap, same grader.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Final, Literal, Protocol

from agent.gates import GateEngine, GateMode
from agent.parser import parse_tool_calls
from env.models import EnvironmentOutcome, EpisodeTrace, OutcomeSource, ToolEvent
from env.phase_a import (
    CALCULATOR_TOOL_NAME,
    PhaseATask,
    answer_from_events,
    is_answering_event,
)
from env.tools import ToolRegistry

Rung = Literal["R0", "R1"]

REFERENCE_ENVIRONMENT_TURN_CAP: Final = 20

# RUNG_PROTOCOL section 3: model decisions permitted inside one agent turn.
MODEL_DECISION_BUDGET: Final[dict[str, int]] = {"R0": 1, "R1": 2}


class Policy(Protocol):
    """One model generation. Text in, text out."""

    def __call__(self, messages: list[dict[str, str]]) -> str: ...


@dataclass(slots=True)
class EpisodeCounters:
    """The counters RUNG_PROTOCOL section 1.4 requires on every episode.

    Kept as separate fields on purpose: the protocol forbids reporting one
    counter as another's proxy, and separate names make that hard to do by
    accident.
    """

    environment_turn_count: int = 0
    agent_turn_count: int = 0
    policy_model_decision_count: int = 0
    escalation_model_decision_count: int = 0
    tool_dispatch_attempt_count: int = 0
    exact_transient_redispatch_count: int = 0
    gate_block_count: int = 0
    model_switch_count: int = 0


@dataclass(slots=True)
class EpisodeResult:
    """Everything one episode produced, including why it stopped."""

    task_id: str
    rung: Rung
    run_index: int
    correct: bool
    terminal_reason: str
    answered_without_arithmetic: bool
    counters: EpisodeCounters
    completions: list[str] = field(default_factory=list)
    tool_events: list[ToolEvent] = field(default_factory=list)
    traces: list[EpisodeTrace] = field(default_factory=list)

    def to_json(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "rung": self.rung,
            "run_index": self.run_index,
            "correct": self.correct,
            "terminal_reason": self.terminal_reason,
            "answered_without_arithmetic": self.answered_without_arithmetic,
            "counters": asdict(self.counters),
            "completions": list(self.completions),
            "tool_events": [event.model_dump(mode="json") for event in self.tool_events],
        }


SYSTEM_PROMPT: Final = (
    "You solve arithmetic word problems using a calculator tool.\n"
    "Emit exactly one tool call, in this form and nothing else:\n"
    '<tool_call>{"name": "calculator", "arguments": {"expression": "<expr>"}}'
    "</tool_call>\n"
    "The expression must compute the answer. Do not put the answer in it "
    "directly, and do not write the answer in prose."
)

USER_PROMPT: Final = "Question: {question}"


def _failure_observation(trace: EpisodeTrace) -> str:
    """Describe, factually, why the last decision produced no answer.

    R1 gives this to its one extra decision. It states what happened and never
    hints at the answer, so the second decision is a genuine retry rather than
    a graded hint.
    """

    if trace.parse.issues:
        codes = ", ".join(sorted({issue.code for issue in trace.parse.issues}))
        return f"Your output could not be parsed as a tool call ({codes})."
    if not trace.parse.calls:
        return "You produced no tool call. A calculator call is required."
    for event in trace.tool_events:
        if not event.schema_valid:
            return (
                f"The call to '{event.call.name}' did not match the tool schema."
            )
        if event.dispatched and not event.succeeded:
            return (
                f"The calculator failed: {event.error_code or 'error'}: "
                f"{event.error_message or 'no detail'}"
            )
        if not event.dispatched:
            return f"The call to '{event.call.name}' was not dispatched."
    return "That attempt produced no usable result."


def _count(counters: EpisodeCounters, trace: EpisodeTrace) -> None:
    for event in trace.tool_events:
        if event.dispatched:
            counters.tool_dispatch_attempt_count += 1
    counters.gate_block_count += sum(
        1 for gate in trace.gate_events if gate.blocked
    )


def run_episode(
    *,
    task: PhaseATask,
    registry: ToolRegistry,
    gate_engine: GateEngine,
    policy: Policy,
    rung: Rung,
    run_index: int = 0,
    environment_turn_cap: int = REFERENCE_ENVIRONMENT_TURN_CAP,
) -> EpisodeResult:
    """Run one Phase A episode under R0 or R1.

    The episode ends as soon as a calculator call executes successfully, since
    D-060 makes that result the answer. It also ends when the rung's decision
    budget for the agent turn is spent, which for R0 is after a single failed
    generation. That asymmetry is the entire experimental contrast.
    """

    if rung not in MODEL_DECISION_BUDGET:
        raise ValueError(f"unknown rung {rung!r}")

    counters = EpisodeCounters()
    state: dict[str, object] = {}
    completions: list[str] = []
    traces: list[EpisodeTrace] = []
    events: list[ToolEvent] = []

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(question=task.question)},
    ]

    counters.agent_turn_count = 1
    budget = MODEL_DECISION_BUDGET[rung]
    terminal_reason = "decision_budget_exhausted"

    for decision in range(budget):
        if counters.environment_turn_count >= environment_turn_cap:
            terminal_reason = "environment_turn_cap"
            break

        completion = policy(messages)
        counters.policy_model_decision_count += 1
        completions.append(completion)

        trace = registry.execute(
            parse_tool_calls(completion),
            state,
            gate_engine=gate_engine,
            gate_mode=GateMode.ENFORCE,
        )
        traces.append(trace)
        events.extend(trace.tool_events)
        _count(counters, trace)

        answered = any(is_answering_event(event) for event in trace.tool_events)
        if answered:
            # An accepted action advances the environment and ends the turn.
            counters.environment_turn_count += 1
            terminal_reason = "answered"
            break

        if decision + 1 < budget:
            # R1's one feedback decision. It sees the failure, nothing else.
            messages = messages + [
                {"role": "assistant", "content": completion},
                {"role": "user", "content": _failure_observation(trace)},
            ]
        else:
            terminal_reason = "no_action" if rung == "R0" else "feedback_exhausted"

    answer = answer_from_events(events)
    correct = answer is not None and abs(answer - task.gold_answer) <= 1e-6
    outcome = EnvironmentOutcome(
        correct=correct, source=OutcomeSource.SANDBOX_RESULT
    )

    return EpisodeResult(
        task_id=task.task_id,
        rung=rung,
        run_index=run_index,
        correct=outcome.correct,
        terminal_reason=terminal_reason,
        answered_without_arithmetic=_laundered(events),
        counters=counters,
        completions=completions,
        tool_events=events,
        traces=traces,
    )


def _laundered(events: list[ToolEvent]) -> bool:
    """Whether the answering call restated a number instead of computing it."""

    from evaluation.contamination import expression_does_arithmetic

    for event in reversed(events):
        if not is_answering_event(event):
            continue
        expression = event.call.arguments.get("expression")
        if not isinstance(expression, str):
            return False
        return not expression_does_arithmetic(expression)
    return False


def make_stub_policy(replies: list[str]) -> Callable[[list[dict[str, str]]], str]:
    """A policy that returns scripted replies. For testing the loop itself."""

    remaining = list(replies)

    def policy(messages: list[dict[str, str]]) -> str:
        if not remaining:
            raise AssertionError("the loop asked for more decisions than the rung allows")
        return remaining.pop(0)

    return policy


__all__ = [
    "CALCULATOR_TOOL_NAME",
    "EpisodeCounters",
    "EpisodeResult",
    "MODEL_DECISION_BUDGET",
    "REFERENCE_ENVIRONMENT_TURN_CAP",
    "Rung",
    "SYSTEM_PROMPT",
    "USER_PROMPT",
    "make_stub_policy",
    "run_episode",
]
