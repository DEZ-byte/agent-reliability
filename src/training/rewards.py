"""Execution-backed composite reward computation."""

from __future__ import annotations

from agent.gates import GateEngine
from env.models import EnvironmentOutcome, EpisodeTrace, RewardBreakdown


def score_episode(
    trace: EpisodeTrace,
    outcome: EnvironmentOutcome,
    *,
    tool_required: bool,
    gate_engine: GateEngine,
) -> RewardBreakdown:
    """Score an episode using only normalized execution evidence.

    The API deliberately has no raw-completion argument. Accuracy arrives as
    an ``EnvironmentOutcome`` whose source is restricted to deterministic
    sandbox execution or database state.
    """

    trace.verify_evidence_digest()
    if trace.gate_policy_fingerprint is None:
        raise ValueError("episode trace has no gate policy fingerprint")
    if trace.gate_policy_fingerprint != gate_engine.policy_fingerprint:
        raise ValueError("gate policy fingerprint mismatch during reward replay")

    executed_calls = sum(event.dispatched for event in trace.tool_events)
    replayed_gates = gate_engine.replay(trace.tool_events)
    gate_violation = any(event.violation for event in replayed_gates)

    no_required_execution = tool_required and executed_calls == 0
    accuracy = 1.0 if outcome.correct and not gate_violation and not no_required_execution else 0.0

    parsed_every_block = (
        not trace.parse.issues
        and len(trace.parse.calls) == trace.parse.emitted_blocks
        and len(trace.tool_events) == len(trace.parse.calls)
    )
    schemas_valid = all(event.schema_valid for event in trace.tool_events)
    format_valid = (
        trace.parse.emitted_blocks > 0
        and parsed_every_block
        and schemas_valid
        and executed_calls >= 1
    )
    if format_valid:
        format_reward = 0.2
    elif trace.parse.emitted_blocks == 0 and not trace.parse.issues:
        format_reward = 0.0
    else:
        format_reward = -0.5

    gate_reward = -0.6 if gate_violation else 0.0
    if no_required_execution:
        efficiency = -0.3
    else:
        efficiency = -min(0.05 * executed_calls, 0.3)

    total = round(accuracy + format_reward + gate_reward + efficiency, 10)
    return RewardBreakdown(
        accuracy=accuracy,
        format=format_reward,
        gate=gate_reward,
        efficiency=efficiency,
        total=total,
        gate_violation=gate_violation,
        executed_calls=executed_calls,
        format_valid=format_valid,
    )
