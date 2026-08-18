"""Typed in-process tool registry with deterministic execution traces."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable, MutableMapping, TypeAlias, TypeVar, cast

from pydantic import BaseModel, JsonValue, ValidationError

from agent.gates import GateEngine, GateMode
from env.models import (
    EpisodeTrace,
    ParseResult,
    ToolCall,
    ToolEvent,
    compute_evidence_digest,
)


ToolState: TypeAlias = MutableMapping[str, JsonValue]
ToolHandler: TypeAlias = Callable[[BaseModel, ToolState], JsonValue]
_T = TypeVar("_T")


def _json_clone(value: _T) -> _T:
    """Deep-copy a value while proving it is strict JSON data."""

    def validate(item: object, path: str) -> None:
        if item is None or isinstance(item, (str, bool)) or type(item) is int:
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise TypeError(f"non-finite float at {path}")
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                validate(child, f"{path}[{index}]")
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise TypeError(f"non-string object key at {path}")
                validate(child, f"{path}.{key}")
            return
        raise TypeError(f"non-JSON value of type {type(item).__name__} at {path}")

    try:
        validate(value, "$")
        encoded = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("tool state and outputs must be JSON-serializable") from exc
    return cast(_T, json.loads(encoded))


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Registration metadata and implementation for one tool."""

    name: str
    args_model: type[BaseModel]
    handler: ToolHandler
    mutative: bool = False
    required_gates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool name must not be empty")
        if self.mutative and not self.required_gates:
            raise ValueError("business-mutative tools must declare required gates")
        if self.required_gates and not self.mutative:
            raise ValueError("only mutative tools may declare required gates")
        if any(not isinstance(name, str) or not name for name in self.required_gates):
            raise ValueError("required gate names must be non-empty strings")
        if len(set(self.required_gates)) != len(self.required_gates):
            raise ValueError("required gates must not contain duplicates")


class ToolRegistry:
    """Validate and dispatch normalized calls through registered handlers."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"tool '{spec.name}' is already registered")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    @staticmethod
    def _event_without_dispatch(
        *,
        index: int,
        call: ToolCall,
        state: ToolState,
        spec: ToolSpec | None,
        schema_valid: bool,
        error_code: str,
        error_message: str,
    ) -> ToolEvent:
        snapshot = _json_clone(dict(state))
        return ToolEvent(
            index=index,
            call=call,
            schema_valid=schema_valid,
            dispatched=False,
            succeeded=False,
            mutative=spec.mutative if spec is not None else False,
            required_gates=spec.required_gates if spec is not None else (),
            state_before=snapshot,
            state_after=_json_clone(snapshot),
            error_code=error_code,
            error_message=error_message,
        )

    def execute(
        self,
        parsed: ParseResult,
        state: ToolState,
        *,
        gate_engine: GateEngine | None = None,
        gate_mode: GateMode = GateMode.ENFORCE,
    ) -> EpisodeTrace:
        """Execute parsed calls in order and snapshot state around every call."""

        gate_mode = GateMode(gate_mode)
        tool_events: list[ToolEvent] = []
        gate_events = []

        called_specs = [
            spec
            for call in parsed.calls
            if (spec := self._specs.get(call.name)) is not None
        ]
        if any(spec.mutative for spec in called_specs) and gate_engine is None:
            raise ValueError("a GateEngine is required before any mutative tool dispatch")
        if gate_engine is not None and gate_engine.has_tool_policies:
            for spec in called_specs:
                gate_engine.validate_tool_policy(
                    tool_name=spec.name,
                    mutative=spec.mutative,
                    required_gates=spec.required_gates,
                )

        for index, call in enumerate(parsed.calls):
            spec = self._specs.get(call.name)
            if spec is None:
                tool_events.append(
                    self._event_without_dispatch(
                        index=index,
                        call=call,
                        state=state,
                        spec=None,
                        schema_valid=False,
                        error_code="unknown_tool",
                        error_message=f"tool '{call.name}' is not registered",
                    )
                )
                continue

            try:
                validated_args = spec.args_model.model_validate(call.arguments, strict=True)
            except ValidationError as exc:
                tool_events.append(
                    self._event_without_dispatch(
                        index=index,
                        call=call,
                        state=state,
                        spec=spec,
                        schema_valid=False,
                        error_code="schema_validation_error",
                        error_message=str(exc),
                    )
                )
                continue

            state_before = _json_clone(dict(state))
            if spec.required_gates:
                if gate_engine is None:
                    raise ValueError(
                        f"tool '{spec.name}' requires gates but no GateEngine was provided"
                    )
                allowed, call_gate_events = gate_engine.check(
                    spec.required_gates,
                    state_before,
                    index=index,
                    tool_name=spec.name,
                    mode=gate_mode,
                )
                gate_events.extend(call_gate_events)
                if not allowed:
                    tool_events.append(
                        self._event_without_dispatch(
                            index=index,
                            call=call,
                            state=state,
                            spec=spec,
                            schema_valid=True,
                            error_code="gate_blocked",
                            error_message="one or more required gate predicates failed",
                        )
                    )
                    continue

            working_state = _json_clone(state_before)
            try:
                raw_output = spec.handler(validated_args, working_state)
            except Exception as exc:  # Tool failures are data, not control flow.
                output = None
                succeeded = False
                error_code = "tool_exception"
                error_message = f"{type(exc).__name__}: {exc}"
            else:
                try:
                    output = _json_clone(raw_output)
                except TypeError as exc:
                    output = None
                    succeeded = False
                    error_code = "invalid_tool_output"
                    error_message = str(exc)
                else:
                    try:
                        committed_state = _json_clone(dict(working_state))
                    except TypeError as exc:
                        output = None
                        succeeded = False
                        error_code = "invalid_tool_state"
                        error_message = str(exc)
                    else:
                        state.clear()
                        state.update(committed_state)
                        succeeded = True
                        error_code = None
                        error_message = None

            state_after = _json_clone(dict(state))
            tool_events.append(
                ToolEvent(
                    index=index,
                    call=call,
                    schema_valid=True,
                    dispatched=True,
                    succeeded=succeeded,
                    mutative=spec.mutative,
                    required_gates=spec.required_gates,
                    state_before=state_before,
                    state_after=state_after,
                    output=output,
                    error_code=error_code,
                    error_message=error_message,
                )
            )

        final_state = _json_clone(dict(state))
        gate_policy_fingerprint = (
            gate_engine.policy_fingerprint if gate_engine is not None else None
        )
        evidence_digest = compute_evidence_digest(
            parse=parsed,
            tool_events=tool_events,
            gate_events=gate_events,
            final_state=final_state,
            gate_policy_fingerprint=gate_policy_fingerprint,
        )
        return EpisodeTrace(
            parse=parsed,
            tool_events=tool_events,
            gate_events=gate_events,
            final_state=final_state,
            gate_policy_fingerprint=gate_policy_fingerprint,
            evidence_digest=evidence_digest,
        )
