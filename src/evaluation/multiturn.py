"""Closed-loop order support: the model sees each tool result before it acts again.

The single-turn evaluation asks for every tool call in one generation and runs
them afterward, so the model has to decide whether to write before it knows
whether `authenticate` succeeded. In this loop each reply's calls run, their
results go back to the model as `tool` messages, and the model writes its next
reply with them in view.

Everything else stays fixed so the two modes can be compared:

- Turn 1 is the single-turn prompt: same system prompt, same user message, same
  tool schemas. The policy is handed the whole message list each turn.
- All turns of an episode act on one persistent state, in `GateMode.AUDIT`, so a
  write that breaks the rule still lands and is still counted.
- Grading runs `grade_episode` and `score_episode` on one trace merged from all
  turns, so "correct" means what it means single-turn: the stored address.

An episode ends when a reply holds no tool call (a final answer or a refusal),
after the turn in which `update_order` was dispatched, or at the turn cap. A
reply whose tool blocks all fail to parse counts as holding no call; the parse
failure stays in the trace and the format term still charges for it. A
dispatched `update_order` ends the episode even if its handler failed, which
matches how the gate replay counts a write attempt.

The policy is a plain callable from messages to text, as in `rungs.py`, so the
loop runs and is tested on CPU without a model.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Final, Protocol, Sequence

from agent.dialects import normalize_tool_dialect
from agent.gates import GateEngine, GateMode
from agent.parser import parse_tool_calls
from env.models import (
    EpisodeTrace,
    GateEvent,
    ParseResult,
    ToolEvent,
    compute_evidence_digest,
)
from env.order_support import (
    GET_ORDER_TOOL_NAME,
    INTENT_FULFIL,
    INTENT_REFUSE,
    UPDATE_ORDER_TOOL_NAME,
    OrderSupportTask,
    build_order_support_registry,
    grade_episode,
    initial_state,
    touched_the_order,
)
from env.tools import ToolRegistry
from evaluation.metrics import compute_pass_metrics
from training.rewards import score_episode

DEFAULT_MAX_TURNS: Final = 4

NO_TOOL_CALL: Final = "no_tool_call"
WRITE_EXECUTED: Final = "write_executed"
MAX_TURNS: Final = "max_turns"
TERMINAL_REASONS: Final = (NO_TOOL_CALL, WRITE_EXECUTED, MAX_TURNS)

# The same classes, names and rule as the gate breakdown in
# commands/order_support_gates.py. A test checks that the two agree. They are
# not imported from there because that module imports the evaluation command,
# which imports this one.
AUTH_GATE: Final = "authenticated"
NO_WRITE: Final = "no_write"
CLEAN_WRITE: Final = "clean_write"
LOOKUP_ONLY: Final = "verified_write_without_lookup"
UNAUTHENTICATED: Final = "unauthenticated_write"
WRITE_CLASSES: Final = (NO_WRITE, CLEAN_WRITE, LOOKUP_ONLY, UNAUTHENTICATED)


class Policy(Protocol):
    """One model generation. The message list in, the reply text out."""

    def __call__(self, messages: list[dict[str, str]]) -> str: ...


def classify_write(attempted_write: bool, failed: set[str]) -> str:
    """Which kind of write, if any, an episode made."""

    if not attempted_write:
        return NO_WRITE
    if not failed:
        return CLEAN_WRITE
    if AUTH_GATE in failed:
        return UNAUTHENTICATED
    return LOOKUP_ONLY


def tool_message(event: ToolEvent) -> dict[str, str]:
    """The `tool` message the model sees for one call.

    A successful call returns its payload. Anything else returns only the error
    code, never the exception text: that text can quote internals the model
    has no use for, and it would make the transcript depend on library
    wording.
    """

    if event.dispatched and event.succeeded:
        payload: Any = event.output
    else:
        payload = {"error": event.error_code or "error"}
    return {
        "role": "tool",
        "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    }


def _wrote(trace: EpisodeTrace) -> bool:
    return any(
        event.dispatched and event.call.name == UPDATE_ORDER_TOOL_NAME
        for event in trace.tool_events
    )


def merge_traces(traces: Sequence[EpisodeTrace], gate_engine: GateEngine) -> EpisodeTrace:
    """One trace covering every turn, scored exactly like a single-turn trace.

    Block, call and event indices are offset so they stay unique across turns,
    and call ids gain a turn prefix. The final state is the last turn's, which
    is the state every turn wrote to.
    """

    if not traces:
        raise ValueError("an episode needs at least one turn")
    calls = []
    issues = []
    events: list[ToolEvent] = []
    gates: list[GateEvent] = []
    blocks = 0
    for turn, trace in enumerate(traces, start=1):
        call_offset = len(calls)
        renamed = {}
        for call in trace.parse.calls:
            new = call.model_copy(update={"call_id": f"turn-{turn}-{call.call_id}"})
            renamed[call.call_id] = new
            calls.append(new)
        for issue in trace.parse.issues:
            issues.append(issue.model_copy(update={"block_index": blocks + issue.block_index}))
        for event in trace.tool_events:
            events.append(
                event.model_copy(
                    update={
                        "index": call_offset + event.index,
                        "call": renamed.get(event.call.call_id, event.call),
                    }
                )
            )
        for gate in trace.gate_events:
            gates.append(gate.model_copy(update={"index": call_offset + gate.index}))
        blocks += trace.parse.emitted_blocks

    parse = ParseResult(emitted_blocks=blocks, calls=calls, issues=issues)
    final_state = dict(traces[-1].final_state)
    fingerprint = gate_engine.policy_fingerprint
    digest = compute_evidence_digest(
        parse=parse,
        tool_events=events,
        gate_events=gates,
        final_state=final_state,
        gate_policy_fingerprint=fingerprint,
    )
    return EpisodeTrace(
        parse=parse,
        tool_events=events,
        gate_events=gates,
        final_state=final_state,
        gate_policy_fingerprint=fingerprint,
        evidence_digest=digest,
    )


@dataclass(slots=True)
class MultiTurnEpisode:
    """What one closed-loop episode produced."""

    messages: list[dict[str, str]]
    completions: list[str]
    calls_per_turn: list[int]
    terminal_reason: str
    trace: EpisodeTrace
    turn_of_event: list[int] = field(default_factory=list)

    @property
    def turns_used(self) -> int:
        return len(self.completions)

    def call_trace(self) -> list[dict[str, Any]]:
        """Every call in order, with the turn it came from and what it returned."""

        return [
            {
                "turn": turn,
                "name": event.call.name,
                "arguments": event.call.arguments,
                "dispatched": event.dispatched,
                "succeeded": event.succeeded,
                "error_code": event.error_code,
                "output": event.output,
            }
            for turn, event in zip(self.turn_of_event, self.trace.tool_events)
        ]


def run_episode(
    task: OrderSupportTask,
    policy: Policy,
    *,
    system_prompt: str,
    user_message: str,
    gate_engine: GateEngine,
    registry: ToolRegistry | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    normalize_dialect: bool = False,
) -> MultiTurnEpisode:
    """Run one order-support episode, feeding tool results back each turn."""

    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")
    registry = registry or build_order_support_registry()
    state = initial_state(task)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    completions: list[str] = []
    calls_per_turn: list[int] = []
    traces: list[EpisodeTrace] = []
    turn_of_event: list[int] = []
    terminal_reason = MAX_TURNS

    for turn in range(1, max_turns + 1):
        # The policy gets a copy, so a policy that keeps or edits the list
        # cannot change the transcript this loop records.
        completion = policy([dict(message) for message in messages])
        completions.append(completion)
        text = normalize_tool_dialect(completion) if normalize_dialect else completion
        parsed = parse_tool_calls(text)
        trace = registry.execute(
            parsed, state, gate_engine=gate_engine, gate_mode=GateMode.AUDIT
        )
        traces.append(trace)
        calls_per_turn.append(len(parsed.calls))
        turn_of_event.extend(turn for _ in trace.tool_events)
        messages.append({"role": "assistant", "content": completion})

        if not parsed.calls:
            terminal_reason = NO_TOOL_CALL
            break
        messages.extend(tool_message(event) for event in trace.tool_events)
        if _wrote(trace):
            terminal_reason = WRITE_EXECUTED
            break

    return MultiTurnEpisode(
        messages=messages,
        completions=completions,
        calls_per_turn=calls_per_turn,
        terminal_reason=terminal_reason,
        trace=merge_traces(traces, gate_engine),
        turn_of_event=turn_of_event,
    )


def evaluate_episode(
    episode: MultiTurnEpisode, task: OrderSupportTask, gate_engine: GateEngine
) -> dict[str, Any]:
    """Grade one episode with the single-turn grader and reward.

    The first fields match the single-turn row, so the same summary code reads
    both. `clean_correct` is the reward's accuracy term: the record is right,
    no gate failed, and at least one call ran.
    """

    trace = episode.trace
    outcome = grade_episode(trace, task)
    breakdown = score_episode(trace, outcome, tool_required=True, gate_engine=gate_engine)
    failed = {event.predicate for event in gate_engine.replay(trace.tool_events) if event.violation}
    called = [event.call.name for event in trace.tool_events if event.dispatched]
    return {
        "task_id": task.task_id,
        "intent": task.intent,
        "template_id": task.template_id,
        "correct": outcome.correct,
        "gate_violation": breakdown.gate_violation,
        "reward": breakdown.total,
        "executed_calls": breakdown.executed_calls,
        "called_any_tool": bool(called),
        "attempted_write": UPDATE_ORDER_TOOL_NAME in called,
        "engaged": touched_the_order(trace.tool_events),
        "tools_called": called,
        "clean_correct": breakdown.accuracy == 1.0,
        "called_lookup": GET_ORDER_TOOL_NAME in called,
        "failed_gates": sorted(failed),
        "write_class": classify_write(UPDATE_ORDER_TOOL_NAME in called, failed),
        "turns_used": episode.turns_used,
        "calls_per_turn": list(episode.calls_per_turn),
        "terminal_reason": episode.terminal_reason,
        "call_trace": episode.call_trace(),
        "completions": list(episode.completions),
        # Everything after the prompt: replies and the tool messages they got.
        "transcript": episode.messages[2:],
    }


def summarize_multi_turn(
    rows: list[dict[str, Any]], tasks: Sequence[OrderSupportTask], runs_per_task: int
) -> dict[str, Any]:
    """The metrics a single-turn summary has no field for.

    `clean_metrics` is pass^k on `clean_correct`, so a write that skipped a gate
    does not count. Per intent: how often the record ended right, how often it
    ended right by the rules, how often the model wrote or left the record
    alone, and the write classes behind those numbers.
    """

    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row["task_id"], []).append(row)
    groups = [by_task[task.task_id] for task in tasks if task.task_id in by_task]
    if not groups or min(len(group) for group in groups) < runs_per_task:
        raise ValueError("every task needs at least runs_per_task episodes")
    clean = [[bool(row["clean_correct"]) for row in group[:runs_per_task]] for group in groups]
    pass1 = compute_pass_metrics(clean, 1)
    passk = compute_pass_metrics(clean, runs_per_task)

    def share(pool: list[dict[str, Any]], predicate) -> float:
        return sum(1 for row in pool if predicate(row)) / len(pool) if pool else 0.0

    by_intent: dict[str, Any] = {}
    for intent in (INTENT_FULFIL, INTENT_REFUSE):
        pool = [row for row in rows if row["intent"] == intent]
        classes = Counter(row["write_class"] for row in pool)
        by_intent[intent] = {
            "episodes": len(pool),
            "correct_rate": share(pool, lambda r: r["correct"]),
            "clean_correct_rate": share(pool, lambda r: r["clean_correct"]),
            "clean_write_rate": share(pool, lambda r: r["write_class"] == CLEAN_WRITE),
            "write_rate": share(pool, lambda r: r["attempted_write"]),
            "refusal_rate": share(pool, lambda r: not r["attempted_write"]),
            "write_classes": {name: classes[name] for name in WRITE_CLASSES},
        }

    reasons = Counter(row["terminal_reason"] for row in rows)
    turns = Counter(row["turns_used"] for row in rows)
    return {
        "clean_metrics": {
            "pass^1": pass1.pass_power_k,
            f"pass^{runs_per_task}": passk.pass_power_k,
            f"pass@{runs_per_task}": passk.pass_at_k,
        },
        "by_intent": by_intent,
        "terminal_reasons": {name: reasons[name] for name in TERMINAL_REASONS},
        "turns_used": {str(n): turns[n] for n in sorted(turns)},
        "mean_turns_used": sum(row["turns_used"] for row in rows) / len(rows),
        "lookup_rate": share(rows, lambda r: r["called_lookup"]),
    }


__all__ = [
    "CLEAN_WRITE",
    "DEFAULT_MAX_TURNS",
    "LOOKUP_ONLY",
    "MAX_TURNS",
    "NO_TOOL_CALL",
    "NO_WRITE",
    "TERMINAL_REASONS",
    "UNAUTHENTICATED",
    "WRITE_CLASSES",
    "WRITE_EXECUTED",
    "MultiTurnEpisode",
    "Policy",
    "classify_write",
    "evaluate_episode",
    "merge_traces",
    "run_episode",
    "summarize_multi_turn",
    "tool_message",
]
