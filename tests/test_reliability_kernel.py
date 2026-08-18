from __future__ import annotations

import json
import unittest
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from agent.gates import GateEngine, GateMode
from agent.parser import parse_tool_calls
from env.models import EnvironmentOutcome, OutcomeSource
from env.tools import ToolRegistry, ToolSpec
from training.rewards import score_episode


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AuthArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token: str


class UpdateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str


def build_kernel() -> tuple[ToolRegistry, GateEngine]:
    registry = ToolRegistry()

    def authenticate(args: BaseModel, state: dict) -> dict:
        if args.token != "secret":  # type: ignore[attr-defined]
            raise PermissionError("denied")
        state["session"]["authenticated"] = True
        return {"authenticated": True}

    def update_order(args: BaseModel, state: dict) -> dict:
        state["order"]["status"] = args.status  # type: ignore[attr-defined]
        return {"status": args.status}  # type: ignore[attr-defined]

    registry.register(ToolSpec("authenticate", AuthArgs, authenticate))
    registry.register(
        ToolSpec(
            "update_order",
            UpdateArgs,
            update_order,
            mutative=True,
            required_gates=("authenticated", "order_id_exists"),
        )
    )
    engine = GateEngine.from_file(PROJECT_ROOT / "configs" / "gates.yaml")
    return registry, engine


def update_block() -> str:
    return '<tool_call>{"name":"update_order","arguments":{"status":"cancelled"}}</tool_call>'


class ReliabilityKernelIntegrationTests(unittest.TestCase):
    def test_audit_and_reward_replay_report_the_same_failed_predicate(self) -> None:
        registry, engine = build_kernel()
        state = {
            "session": {"authenticated": False},
            "order": {"id": "A-1", "status": "open"},
        }

        trace = registry.execute(
            parse_tool_calls(update_block()),
            state,
            gate_engine=engine,
            gate_mode=GateMode.AUDIT,
        )
        replayed = engine.replay(trace.tool_events)
        reward = score_episode(
            trace,
            EnvironmentOutcome(correct=True, source=OutcomeSource.DB_STATE),
            tool_required=True,
            gate_engine=engine,
        )

        live_failures = {
            event.predicate for event in trace.gate_events if event.violation
        }
        replay_failures = {event.predicate for event in replayed if event.violation}
        self.assertEqual(live_failures, {"authenticated"})
        self.assertEqual(replay_failures, live_failures)
        self.assertTrue(trace.tool_events[0].dispatched)
        self.assertEqual(trace.final_state["order"]["status"], "cancelled")
        self.assertTrue(reward.gate_violation)
        self.assertEqual(reward.accuracy, 0.0)

    def test_enforce_blocks_and_reward_replay_does_not_invent_execution(self) -> None:
        registry, engine = build_kernel()
        state = {
            "session": {"authenticated": False},
            "order": {"id": "A-1", "status": "open"},
        }

        trace = registry.execute(
            parse_tool_calls(update_block()),
            state,
            gate_engine=engine,
        )
        reward = score_episode(
            trace,
            EnvironmentOutcome(correct=True, source=OutcomeSource.DB_STATE),
            tool_required=True,
            gate_engine=engine,
        )

        self.assertTrue(any(event.blocked for event in trace.gate_events))
        self.assertFalse(trace.tool_events[0].dispatched)
        self.assertEqual(trace.final_state["order"]["status"], "open")
        self.assertEqual(engine.replay(trace.tool_events), [])
        self.assertFalse(reward.gate_violation)
        self.assertEqual(reward.gate, 0.0)
        self.assertEqual(reward.total, -0.8)

    def test_authorize_then_update_is_order_aware_and_serializable(self) -> None:
        registry, engine = build_kernel()
        raw = (
            '<tool_call>{"name":"authenticate","arguments":{"token":"secret"}}</tool_call>'
            + update_block()
        )
        state = {
            "session": {"authenticated": False},
            "order": {"id": "A-1", "status": "open"},
        }

        trace = registry.execute(
            parse_tool_calls(raw),
            state,
            gate_engine=engine,
            gate_mode=GateMode.AUDIT,
        )
        reward = score_episode(
            trace,
            EnvironmentOutcome(correct=True, source=OutcomeSource.DB_STATE),
            tool_required=True,
            gate_engine=engine,
        )

        self.assertEqual(trace.tool_events[1].state_before["session"]["authenticated"], True)
        self.assertFalse(any(event.violation for event in engine.replay(trace.tool_events)))
        self.assertEqual(reward.total, 1.1)
        serialized = json.loads(trace.model_dump_json())
        self.assertEqual(serialized["final_state"]["order"]["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
