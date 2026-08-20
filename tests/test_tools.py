from __future__ import annotations

import json
import pathlib
import unittest

from pydantic import BaseModel, ConfigDict

from agent.gates import GateEngine
from agent.parser import parse_tool_calls
from env.tools import ToolRegistry, ToolSpec


class SetArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: int


def allow_engine() -> GateEngine:
    return GateEngine.from_mapping(
        {"allowed": {"op": "eq", "path": "allowed", "value": True}}
    )


class ToolRegistryTests(unittest.TestCase):
    def test_dispatch_records_deep_state_snapshots_and_json_round_trip(self) -> None:
        registry = ToolRegistry()

        def set_value(args: BaseModel, state: dict) -> dict:
            state["nested"]["value"] = args.value  # type: ignore[attr-defined,index]
            return {"stored": args.value}  # type: ignore[attr-defined]

        registry.register(
            ToolSpec(
                "set_value",
                SetArgs,
                set_value,
                mutative=True,
                required_gates=("allowed",),
            )
        )
        parsed = parse_tool_calls(
            '<tool_call>{"name":"set_value","arguments":{"value":7}}</tool_call>'
        )
        state = {"allowed": True, "nested": {"value": 1}}

        trace = registry.execute(parsed, state, gate_engine=allow_engine())
        state["nested"]["value"] = 99

        event = trace.tool_events[0]
        self.assertTrue(event.schema_valid)
        self.assertTrue(event.dispatched)
        self.assertTrue(event.succeeded)
        self.assertEqual(event.state_before, {"allowed": True, "nested": {"value": 1}})
        self.assertEqual(event.state_after, {"allowed": True, "nested": {"value": 7}})
        self.assertEqual(trace.final_state, {"allowed": True, "nested": {"value": 7}})
        restored = json.loads(trace.model_dump_json())
        self.assertEqual(restored["tool_events"][0]["state_before"]["nested"]["value"], 1)

    def test_unknown_tool_and_schema_failure_are_never_dispatched(self) -> None:
        registry = ToolRegistry()
        calls = 0

        def handler(args: BaseModel, state: dict) -> None:
            nonlocal calls
            calls += 1

        registry.register(ToolSpec("set_value", SetArgs, handler))
        parsed = parse_tool_calls(
            '<tool_call>{"name":"missing","arguments":{}}</tool_call>'
            '<tool_call>{"name":"set_value","arguments":{}}</tool_call>'
        )

        trace = registry.execute(parsed, {})

        self.assertEqual(calls, 0)
        self.assertEqual(
            [event.error_code for event in trace.tool_events],
            ["unknown_tool", "schema_validation_error"],
        )
        self.assertTrue(all(not event.dispatched for event in trace.tool_events))

    def test_handler_exception_is_a_dispatched_failed_event(self) -> None:
        registry = ToolRegistry()

        def fail(args: BaseModel, state: dict) -> None:
            state["attempted"] = True
            raise RuntimeError("boom")

        registry.register(
            ToolSpec(
                "fail",
                SetArgs,
                fail,
                mutative=True,
                required_gates=("allowed",),
            )
        )
        parsed = parse_tool_calls(
            '<tool_call>{"name":"fail","arguments":{"value":1}}</tool_call>'
        )

        state = {"allowed": True}
        event = registry.execute(
            parsed,
            state,
            gate_engine=allow_engine(),
        ).tool_events[0]

        self.assertTrue(event.dispatched)
        self.assertFalse(event.succeeded)
        self.assertEqual(event.error_code, "tool_exception")
        self.assertIn("RuntimeError: boom", event.error_message or "")
        self.assertEqual(event.state_after, {"allowed": True})
        self.assertEqual(state, {"allowed": True})

    def test_invalid_output_rolls_back_working_state(self) -> None:
        registry = ToolRegistry()

        def invalid_output(args: BaseModel, state: dict) -> object:
            state["value"] = args.value  # type: ignore[attr-defined]
            return object()

        registry.register(
            ToolSpec(
                "invalid_output",
                SetArgs,
                invalid_output,
                mutative=True,
                required_gates=("allowed",),
            )
        )
        parsed = parse_tool_calls(
            '<tool_call>{"name":"invalid_output","arguments":{"value":9}}</tool_call>'
        )
        state = {"allowed": True, "value": 1}

        event = registry.execute(
            parsed,
            state,
            gate_engine=allow_engine(),
        ).tool_events[0]

        self.assertTrue(event.dispatched)
        self.assertFalse(event.succeeded)
        self.assertEqual(event.error_code, "invalid_tool_output")
        self.assertEqual(state, {"allowed": True, "value": 1})
        self.assertEqual(event.state_after, {"allowed": True, "value": 1})

    def test_argument_validation_is_strict(self) -> None:
        registry = ToolRegistry()
        dispatches = 0

        def handler(args: BaseModel, state: dict) -> int:
            nonlocal dispatches
            dispatches += 1
            return args.value  # type: ignore[attr-defined,return-value]

        registry.register(ToolSpec("read", SetArgs, handler))
        parsed = parse_tool_calls(
            '<tool_call>{"name":"read","arguments":{"value":"7"}}</tool_call>'
        )

        event = registry.execute(parsed, {}).tool_events[0]

        self.assertFalse(event.schema_valid)
        self.assertFalse(event.dispatched)
        self.assertEqual(dispatches, 0)

    def test_missing_gate_engine_fails_before_any_dispatch(self) -> None:
        registry = ToolRegistry()
        dispatches: list[str] = []

        def handler(args: BaseModel, state: dict) -> int:
            dispatches.append("called")
            return args.value  # type: ignore[attr-defined,return-value]

        registry.register(ToolSpec("read", SetArgs, handler))
        registry.register(
            ToolSpec(
                "write",
                SetArgs,
                handler,
                mutative=True,
                required_gates=("allowed",),
            )
        )
        parsed = parse_tool_calls(
            '<tool_call>{"name":"read","arguments":{"value":1}}</tool_call>'
            '<tool_call>{"name":"write","arguments":{"value":2}}</tool_call>'
        )

        with self.assertRaisesRegex(ValueError, "required before any mutative"):
            registry.execute(parsed, {"allowed": True})
        self.assertEqual(dispatches, [])

    def test_duplicate_registration_and_invalid_gate_metadata_fail_fast(self) -> None:
        registry = ToolRegistry()

        def handler(args: BaseModel, state: dict) -> None:
            return None

        spec = ToolSpec("set_value", SetArgs, handler)
        registry.register(spec)

        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(spec)
        with self.assertRaisesRegex(ValueError, "only mutative tools"):
            ToolSpec(
                "read_value",
                SetArgs,
                handler,
                mutative=False,
                required_gates=("authenticated",),
            )
        with self.assertRaisesRegex(ValueError, "must declare required gates"):
            ToolSpec("unsafe_write", SetArgs, handler, mutative=True)


class SurrogateBoundaryTests(unittest.TestCase):
    """A tool may not smuggle an unencodable string into the evidence log."""

    def test_tool_output_with_an_unpaired_surrogate_is_rejected(self) -> None:
        from env.tools import _json_clone

        with self.assertRaisesRegex(TypeError, "JSON-serializable"):
            _json_clone({"answer": chr(0xD800)})

    def test_object_key_with_an_unpaired_surrogate_is_rejected(self) -> None:
        from env.tools import _json_clone

        with self.assertRaisesRegex(TypeError, "JSON-serializable"):
            _json_clone({chr(0xD800): "value"})

    def test_ordinary_non_ascii_text_still_round_trips(self) -> None:
        from env.tools import _json_clone

        payload = {"note": "café — 日本語", "n": [1, 2.5, None, True]}
        self.assertEqual(_json_clone(payload), payload)


class SourceEncodabilityTests(unittest.TestCase):
    """No source file may itself contain what the surrogate rule forbids.

    A plain docstring turns a backslash-u escape into a real codepoint at
    compile time, so merely naming the escape in prose can embed a lone
    surrogate in the module. That breaks import on some interpreters and was
    caught only by CI. Raw strings avoid it; this test enforces it.
    """

    def test_no_source_file_contains_an_unpaired_surrogate(self) -> None:
        from env.models import contains_surrogate

        # Shipped code only. Test fixtures deliberately carry the hostile
        # value, which is the whole point of them.
        root = pathlib.Path(__file__).resolve().parents[1]
        checked = 0
        for folder in ("src", "scripts"):
            for path in sorted((root / folder).rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                checked += 1
                text = path.read_text(encoding="utf-8")
                with self.subTest(path=str(path.relative_to(root))):
                    self.assertFalse(
                        contains_surrogate(text),
                        "source text contains an unpaired surrogate",
                    )
                    # The compiled module must also survive encoding, which is
                    # where a docstring escape actually bites.
                    compiled = compile(text, str(path), "exec")
                    pending = [compiled]
                    while pending:
                        code = pending.pop()
                        for const in code.co_consts:
                            if isinstance(const, str):
                                self.assertFalse(
                                    contains_surrogate(const),
                                    "a compiled constant contains an unpaired "
                                    "surrogate (a docstring escape, most likely)",
                                )
                            elif hasattr(const, "co_consts"):
                                pending.append(const)
        self.assertGreater(checked, 10)


if __name__ == "__main__":
    unittest.main()
