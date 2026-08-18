from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from agent.gates import GateEngine, GateMode
from agent.parser import parse_tool_calls
from env.tools import ToolRegistry, ToolSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ModifyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quantity: int


def make_engine() -> GateEngine:
    return GateEngine.from_mapping(
        {
            "predicates": {
                "authenticated": {"op": "truthy", "path": "session.authenticated"},
                "order_exists": {"op": "exists", "path": "order.id"},
                "reversible": {"op": "eq", "path": "order.reversible", "value": True},
            }
        }
    )


class GateEngineTests(unittest.TestCase):
    def test_eq_truthy_and_exists_predicates(self) -> None:
        engine = make_engine()
        state = {
            "session": {"authenticated": True},
            "order": {"id": None, "reversible": True},
        }

        allowed, events = engine.check(
            ("authenticated", "order_exists", "reversible"),
            state,
            index=0,
            tool_name="modify",
            mode=GateMode.ENFORCE,
        )

        self.assertTrue(allowed)
        self.assertEqual([event.passed for event in events], [True, True, True])
        self.assertTrue(all(event.action == "allow" for event in events))

    def test_eq_does_not_treat_booleans_as_integers(self) -> None:
        expects_true = GateEngine.from_mapping(
            {"flag": {"op": "eq", "path": "value", "value": True}}
        )
        expects_one = GateEngine.from_mapping(
            {"one": {"op": "eq", "path": "value", "value": 1}}
        )

        _, true_events = expects_true.check(
            ("flag",),
            {"value": 1},
            index=0,
            tool_name="modify",
            mode=GateMode.AUDIT,
        )
        _, one_events = expects_one.check(
            ("one",),
            {"value": True},
            index=0,
            tool_name="modify",
            mode=GateMode.AUDIT,
        )

        self.assertFalse(true_events[0].passed)
        self.assertFalse(one_events[0].passed)

    def test_not_null_rejects_a_present_null_identifier(self) -> None:
        engine = GateEngine.from_mapping(
            {"order_id": {"op": "not_null", "path": "order.id"}}
        )

        _, null_events = engine.check(
            ("order_id",),
            {"order": {"id": None}},
            index=0,
            tool_name="modify",
            mode=GateMode.AUDIT,
        )

        self.assertFalse(null_events[0].passed)

    def test_audit_records_violation_but_enforce_blocks(self) -> None:
        engine = make_engine()
        state = {
            "session": {"authenticated": False},
            "order": {"id": "A-1", "reversible": True},
        }

        audit_allowed, audit_events = engine.check(
            ("authenticated",),
            state,
            index=2,
            tool_name="modify",
            mode=GateMode.AUDIT,
        )
        enforce_allowed, enforce_events = engine.check(
            ("authenticated",),
            state,
            index=2,
            tool_name="modify",
            mode=GateMode.ENFORCE,
        )

        self.assertTrue(audit_allowed)
        self.assertTrue(audit_events[0].violation)
        self.assertEqual(audit_events[0].action, "audit_violation")
        self.assertFalse(enforce_allowed)
        self.assertTrue(enforce_events[0].blocked)
        self.assertFalse(enforce_events[0].violation)
        self.assertEqual(enforce_events[0].action, "enforce_block")

    def test_replay_uses_pre_call_state_and_counts_failed_attempt(self) -> None:
        engine = make_engine()
        registry = ToolRegistry()

        def failed_modify(args: BaseModel, state: dict) -> None:
            raise RuntimeError("backend unavailable")

        registry.register(
            ToolSpec(
                "modify",
                ModifyArgs,
                failed_modify,
                mutative=True,
                required_gates=("authenticated",),
            )
        )
        parsed = parse_tool_calls(
            '<tool_call>{"name":"modify","arguments":{"quantity":2}}</tool_call>'
        )
        trace = registry.execute(
            parsed,
            {"session": {"authenticated": False}},
            gate_engine=engine,
            gate_mode=GateMode.AUDIT,
        )

        replayed = engine.replay(trace.tool_events)

        self.assertTrue(trace.tool_events[0].dispatched)
        self.assertFalse(trace.tool_events[0].succeeded)
        self.assertEqual(len(replayed), 1)
        self.assertTrue(replayed[0].violation)
        self.assertEqual(replayed[0].action, "replay_violation")

    def test_replay_ignores_calls_blocked_before_dispatch(self) -> None:
        engine = make_engine()
        registry = ToolRegistry()

        def modify(args: BaseModel, state: dict) -> None:
            state["modified"] = True

        registry.register(
            ToolSpec(
                "modify",
                ModifyArgs,
                modify,
                mutative=True,
                required_gates=("authenticated",),
            )
        )
        parsed = parse_tool_calls(
            '<tool_call>{"name":"modify","arguments":{"quantity":2}}</tool_call>'
        )
        trace = registry.execute(
            parsed,
            {"session": {"authenticated": False}},
            gate_engine=engine,
        )

        self.assertFalse(trace.tool_events[0].dispatched)
        self.assertEqual(trace.tool_events[0].error_code, "gate_blocked")
        self.assertEqual(engine.replay(trace.tool_events), [])

    def test_actual_config_file_loads_and_cross_checks_tool_policy(self) -> None:
        engine = GateEngine.from_file(PROJECT_ROOT / "configs" / "gates.yaml")
        registry = ToolRegistry()

        def update(args: BaseModel, state: dict) -> dict:
            state["quantity"] = args.quantity  # type: ignore[attr-defined]
            return {"quantity": args.quantity}  # type: ignore[attr-defined]

        registry.register(
            ToolSpec(
                "update_order",
                ModifyArgs,
                update,
                mutative=True,
                required_gates=("authenticated", "order_id_exists"),
            )
        )
        parsed = parse_tool_calls(
            '<tool_call>{"name":"update_order","arguments":{"quantity":2}}</tool_call>'
        )

        trace = registry.execute(
            parsed,
            {
                "session": {"authenticated": True},
                "order": {"id": "A-1"},
            },
            gate_engine=engine,
        )

        self.assertTrue(engine.has_tool_policies)
        self.assertEqual(
            engine.configured_requirements("update_order"),
            ("authenticated", "order_id_exists"),
        )
        self.assertEqual(len(engine.policy_fingerprint), 64)
        self.assertTrue(trace.tool_events[0].dispatched)

    def test_tool_policy_mismatch_and_missing_policy_fail_before_dispatch(self) -> None:
        engine = GateEngine.from_file(PROJECT_ROOT / "configs" / "gates.yaml")

        def update(args: BaseModel, state: dict) -> None:
            state["called"] = True

        state = {
            "session": {"authenticated": True},
            "order": {"id": "A-1"},
        }
        parsed_update = parse_tool_calls(
            '<tool_call>{"name":"update_order","arguments":{"quantity":2}}</tool_call>'
        )
        mismatched = ToolRegistry()
        mismatched.register(
            ToolSpec(
                "update_order",
                ModifyArgs,
                update,
                mutative=True,
                required_gates=("order_id_exists", "authenticated"),
            )
        )
        with self.assertRaisesRegex(ValueError, "do not match"):
            mismatched.execute(parsed_update, state, gate_engine=engine)
        self.assertNotIn("called", state)

        missing = ToolRegistry()
        missing.register(
            ToolSpec(
                "other_update",
                ModifyArgs,
                update,
                mutative=True,
                required_gates=("authenticated",),
            )
        )
        parsed_missing = parse_tool_calls(
            '<tool_call>{"name":"other_update","arguments":{"quantity":2}}</tool_call>'
        )
        with self.assertRaisesRegex(ValueError, "no configured gate policy"):
            missing.execute(parsed_missing, state, gate_engine=engine)
        self.assertNotIn("called", state)

    def test_policy_fingerprint_is_deterministic_and_covers_tool_policies(self) -> None:
        first = GateEngine.from_mapping(
            {
                "version": 1,
                "predicates": {
                    "a": {"op": "eq", "path": "a", "value": True},
                    "b": {"op": "exists", "path": "b"},
                },
                "tools": {"write": {"requires": ["a", "b"]}},
            }
        )
        reordered = GateEngine.from_mapping(
            {
                "tools": {"write": {"requires": ["a", "b"]}},
                "predicates": {
                    "b": {"path": "b", "op": "exists"},
                    "a": {"value": True, "path": "a", "op": "eq"},
                },
                "version": 1,
            }
        )
        changed = GateEngine.from_mapping(
            {
                "version": 1,
                "predicates": {
                    "a": {"op": "eq", "path": "a", "value": True},
                    "b": {"op": "exists", "path": "b"},
                },
                "tools": {"write": {"requires": ["b", "a"]}},
            }
        )

        self.assertEqual(first.policy_fingerprint, reordered.policy_fingerprint)
        self.assertNotEqual(first.policy_fingerprint, changed.policy_fingerprint)

    def test_invalid_config_and_unknown_predicates_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a value"):
            GateEngine.from_mapping({"broken": {"op": "eq", "path": "x"}})
        with self.assertRaisesRegex(ValueError, "unsupported"):
            GateEngine.from_mapping({"broken": {"op": "python", "path": "x"}})
        with self.assertRaisesRegex(ValueError, "unsupported gate config version"):
            GateEngine.from_mapping(
                {
                    "version": 2,
                    "predicates": {"ok": {"op": "truthy", "path": "ok"}},
                }
            )
        with self.assertRaisesRegex(ValueError, "require a config version"):
            GateEngine.from_mapping(
                {
                    "predicates": {"ok": {"op": "truthy", "path": "ok"}},
                    "tools": {"write": {"requires": ["ok"]}},
                }
            )
        with self.assertRaisesRegex(ValueError, "unknown predicates"):
            GateEngine.from_mapping(
                {
                    "version": 1,
                    "predicates": {"ok": {"op": "truthy", "path": "ok"}},
                    "tools": {"write": {"requires": ["missing"]}},
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unversioned-gates.yaml"
            path.write_text(
                '{"predicates":{"ok":{"op":"truthy","path":"ok"}}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must declare a version"):
                GateEngine.from_file(path)

        engine = make_engine()
        with self.assertRaisesRegex(KeyError, "unknown gate predicate"):
            engine.check(
                ("not_configured",),
                {},
                index=0,
                tool_name="modify",
                mode=GateMode.AUDIT,
            )


if __name__ == "__main__":
    unittest.main()
