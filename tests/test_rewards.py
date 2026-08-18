from __future__ import annotations

import json
import unittest

from pydantic import BaseModel, ConfigDict, ValidationError

from agent.gates import GateEngine, GateMode
from agent.parser import parse_tool_calls
from env.models import EnvironmentOutcome, OutcomeSource
from env.tools import ToolRegistry, ToolSpec
from training.rewards import score_episode


class AuthArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token: str


class ModifyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quantity: int


class CalculatorArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: int


def block(name: str, arguments: dict) -> str:
    payload = json.dumps({"name": name, "arguments": arguments}, separators=(",", ":"))
    return f"<tool_call>{payload}</tool_call>"


def make_fixture() -> tuple[ToolRegistry, GateEngine]:
    registry = ToolRegistry()

    def authenticate(args: BaseModel, state: dict) -> dict:
        state["authenticated"] = True
        if args.token != "secret":  # type: ignore[attr-defined]
            raise PermissionError("invalid credentials")
        return {"authenticated": True}

    def modify(args: BaseModel, state: dict) -> dict:
        state["quantity"] = args.quantity  # type: ignore[attr-defined]
        return {"quantity": args.quantity}  # type: ignore[attr-defined]

    def calculator(args: BaseModel, state: dict) -> int:
        return args.value  # type: ignore[attr-defined,return-value]

    registry.register(ToolSpec("authenticate", AuthArgs, authenticate))
    registry.register(
        ToolSpec(
            "modify",
            ModifyArgs,
            modify,
            mutative=True,
            required_gates=("authenticated",),
        )
    )
    registry.register(ToolSpec("calculator", CalculatorArgs, calculator))
    engine = GateEngine.from_mapping(
        {"authenticated": {"op": "truthy", "path": "authenticated"}}
    )
    return registry, engine


def audit(raw: str, *, initial_state: dict | None = None):
    registry, engine = make_fixture()
    state = {"authenticated": False} if initial_state is None else initial_state
    trace = registry.execute(
        parse_tool_calls(raw),
        state,
        gate_engine=engine,
        gate_mode=GateMode.AUDIT,
    )
    return trace, engine


class RewardTests(unittest.TestCase):
    def test_auth_string_in_prose_gets_no_format_or_accuracy_credit(self) -> None:
        trace, engine = audit("Authenticated successfully in my reasoning. #### 42")

        reward = score_episode(
            trace,
            EnvironmentOutcome(correct=True, source=OutcomeSource.SANDBOX_RESULT),
            tool_required=True,
            gate_engine=engine,
        )

        self.assertEqual(reward.accuracy, 0.0)
        self.assertEqual(reward.format, 0.0)
        self.assertEqual(reward.efficiency, -0.3)
        self.assertEqual(reward.total, -0.3)

    def test_empty_tool_block_has_exact_negative_reward(self) -> None:
        trace, engine = audit("<tool_call></tool_call>")

        reward = score_episode(
            trace,
            EnvironmentOutcome(correct=True, source=OutcomeSource.SANDBOX_RESULT),
            tool_required=True,
            gate_engine=engine,
        )

        self.assertEqual(reward.accuracy, 0.0)
        self.assertEqual(reward.format, -0.5)
        self.assertEqual(reward.gate, 0.0)
        self.assertEqual(reward.efficiency, -0.3)
        self.assertEqual(reward.total, -0.8)

    def test_out_of_order_auth_zeroes_accuracy(self) -> None:
        raw = block("modify", {"quantity": 2}) + block(
            "authenticate", {"token": "secret"}
        )
        trace, engine = audit(raw)

        reward = score_episode(
            trace,
            EnvironmentOutcome(correct=True, source=OutcomeSource.DB_STATE),
            tool_required=True,
            gate_engine=engine,
        )

        self.assertEqual(reward.accuracy, 0.0)
        self.assertEqual(reward.format, 0.2)
        self.assertEqual(reward.gate, -0.6)
        self.assertAlmostEqual(reward.efficiency, -0.1)
        self.assertEqual(reward.total, -0.5)

    def test_failed_auth_then_modify_is_a_gate_violation(self) -> None:
        raw = block("authenticate", {"token": "wrong"}) + block(
            "modify", {"quantity": 2}
        )
        trace, engine = audit(raw)

        reward = score_episode(
            trace,
            EnvironmentOutcome(correct=True, source=OutcomeSource.DB_STATE),
            tool_required=True,
            gate_engine=engine,
        )

        self.assertFalse(trace.tool_events[0].succeeded)
        self.assertEqual(trace.tool_events[0].state_after["authenticated"], False)
        self.assertEqual(trace.tool_events[1].state_before["authenticated"], False)
        self.assertTrue(trace.tool_events[1].succeeded)
        self.assertTrue(reward.gate_violation)
        self.assertEqual(reward.total, -0.5)

    def test_reward_rejects_a_different_gate_policy(self) -> None:
        trace, engine = audit(block("modify", {"quantity": 2}))
        self.assertTrue(trace.gate_events[0].violation)
        different_engine = GateEngine.from_mapping(
            {"authenticated": {"op": "eq", "path": "authenticated", "value": False}}
        )

        with self.assertRaisesRegex(ValueError, "policy fingerprint mismatch"):
            score_episode(
                trace,
                EnvironmentOutcome(correct=True, source=OutcomeSource.DB_STATE),
                tool_required=True,
                gate_engine=different_engine,
            )

    def test_reward_rejects_nested_trace_mutation(self) -> None:
        raw = block("authenticate", {"token": "secret"}) + block(
            "modify", {"quantity": 2}
        )
        trace, engine = audit(raw)
        trace.tool_events[1].state_before["authenticated"] = False

        with self.assertRaisesRegex(ValueError, "evidence digest mismatch"):
            score_episode(
                trace,
                EnvironmentOutcome(correct=True, source=OutcomeSource.DB_STATE),
                tool_required=True,
                gate_engine=engine,
            )

    def test_environment_outcome_is_strict(self) -> None:
        with self.assertRaises(ValidationError):
            EnvironmentOutcome(
                correct="yes",  # type: ignore[arg-type]
                source=OutcomeSource.DB_STATE,
            )

    def test_multiple_answer_markers_cannot_override_wrong_execution(self) -> None:
        raw = "#### 42\n" + block("calculator", {"value": 7}) + "\n#### 42"
        trace, engine = audit(raw)

        reward = score_episode(
            trace,
            EnvironmentOutcome(correct=False, source=OutcomeSource.SANDBOX_RESULT),
            tool_required=True,
            gate_engine=engine,
        )

        self.assertEqual(reward.accuracy, 0.0)
        self.assertEqual(reward.format, 0.2)
        self.assertEqual(reward.efficiency, -0.05)
        self.assertEqual(reward.total, 0.15)

    def test_happy_path_has_exact_composite_reward(self) -> None:
        raw = block("authenticate", {"token": "secret"}) + block(
            "modify", {"quantity": 2}
        )
        trace, engine = audit(raw)

        reward = score_episode(
            trace,
            EnvironmentOutcome(correct=True, source=OutcomeSource.DB_STATE),
            tool_required=True,
            gate_engine=engine,
        )

        self.assertEqual(reward.accuracy, 1.0)
        self.assertEqual(reward.format, 0.2)
        self.assertEqual(reward.gate, 0.0)
        self.assertAlmostEqual(reward.efficiency, -0.1)
        self.assertEqual(reward.total, 1.1)

    def test_schema_failure_is_format_failure_and_never_dispatches(self) -> None:
        trace, engine = audit(block("modify", {}))

        reward = score_episode(
            trace,
            EnvironmentOutcome(correct=True, source=OutcomeSource.DB_STATE),
            tool_required=True,
            gate_engine=engine,
        )

        self.assertFalse(trace.tool_events[0].schema_valid)
        self.assertEqual(reward.format, -0.5)
        self.assertEqual(reward.executed_calls, 0)
        self.assertEqual(reward.total, -0.8)

    def test_gate_penalty_is_binary_with_multiple_violations(self) -> None:
        raw = block("modify", {"quantity": 2}) + block("modify", {"quantity": 3})
        trace, engine = audit(raw)

        reward = score_episode(
            trace,
            EnvironmentOutcome(correct=True, source=OutcomeSource.DB_STATE),
            tool_required=True,
            gate_engine=engine,
        )

        self.assertEqual(sum(event.violation for event in engine.replay(trace.tool_events)), 2)
        self.assertEqual(reward.gate, -0.6)
        self.assertEqual(reward.total, -0.5)

    def test_efficiency_penalty_caps_at_negative_point_three(self) -> None:
        raw = "".join(block("calculator", {"value": number}) for number in range(7))
        trace, engine = audit(raw)

        reward = score_episode(
            trace,
            EnvironmentOutcome(correct=False, source=OutcomeSource.SANDBOX_RESULT),
            tool_required=True,
            gate_engine=engine,
        )

        self.assertEqual(reward.executed_calls, 7)
        self.assertEqual(reward.efficiency, -0.3)
        self.assertEqual(reward.total, -0.1)


if __name__ == "__main__":
    unittest.main()
