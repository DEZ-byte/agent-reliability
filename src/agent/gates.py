"""Declarative state predicates shared by runtime enforcement and rewards."""

from __future__ import annotations

import hashlib
import json
import math
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, JsonValue

from env.models import GateEvent, ToolEvent


class GateMode(str, Enum):
    """Whether failed predicates are observed or enforced before dispatch."""

    AUDIT = "audit"
    ENFORCE = "enforce"


class PredicateSpec(BaseModel):
    """A safe state lookup with no executable expression language."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    op: str
    path: str
    value: JsonValue | None = None


_MISSING = object()


class GateEngine:
    """Evaluate declarative predicates against pre-dispatch state snapshots."""

    CONFIG_VERSION = 1
    _SUPPORTED_OPERATORS = frozenset({"eq", "truthy", "exists", "not_null"})

    def __init__(
        self,
        predicates: Mapping[str, PredicateSpec],
        *,
        version: int | None = None,
        tool_policies: Mapping[str, tuple[str, ...]] | None = None,
        tool_policies_declared: bool = False,
    ) -> None:
        self._predicates = dict(predicates)
        self._version = version
        self._tool_policies = dict(tool_policies or {})
        self._tool_policies_declared = tool_policies_declared
        self._policy_fingerprint = self._compute_policy_fingerprint()

    def _compute_policy_fingerprint(self) -> str:
        payload = {
            "version": self._version,
            "predicates": {
                name: spec.model_dump(mode="json")
                for name, spec in sorted(self._predicates.items())
            },
            "tools": (
                {
                    name: {"requires": list(required)}
                    for name, required in sorted(self._tool_policies.items())
                }
                if self._tool_policies_declared
                else None
            ),
        }
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "GateEngine":
        """Build an engine from either a predicate mapping or its wrapper.

        Accepted forms are ``{"authenticated": {...}}`` and
        ``{"predicates": {"authenticated": {...}}}``.
        """

        wrapped = any(key in config for key in ("version", "predicates", "tools"))
        if wrapped:
            unexpected = set(config) - {"version", "predicates", "tools"}
            if unexpected:
                raise ValueError(f"unknown top-level gate config keys: {sorted(unexpected)}")
            raw_version = config.get("version")
            if raw_version is not None and type(raw_version) is not int:
                raise ValueError("gate config version must be an integer")
            if raw_version is not None and raw_version != cls.CONFIG_VERSION:
                raise ValueError(f"unsupported gate config version: {raw_version}")
            if "tools" in config and "version" not in config:
                raise ValueError("versioned tool policies require a config version")
            version: int | None = raw_version
            raw_predicates: Any = config.get("predicates")
        else:
            version = None
            raw_predicates = config
        if not isinstance(raw_predicates, Mapping):
            raise ValueError("gate config 'predicates' must be a mapping")

        predicates: dict[str, PredicateSpec] = {}
        for name, raw_spec in raw_predicates.items():
            if not isinstance(name, str) or not name:
                raise ValueError("predicate names must be non-empty strings")
            if not isinstance(raw_spec, Mapping):
                raise ValueError(f"predicate '{name}' must be a mapping")
            if raw_spec.get("op") == "eq" and "value" not in raw_spec:
                raise ValueError(f"eq predicate '{name}' requires a value")
            if raw_spec.get("op") != "eq" and "value" in raw_spec:
                raise ValueError(f"predicate '{name}' may only set value with the eq operator")
            spec = PredicateSpec.model_validate(raw_spec)
            if spec.op not in cls._SUPPORTED_OPERATORS:
                raise ValueError(f"unsupported gate operator '{spec.op}'")
            if not spec.path or any(not segment for segment in spec.path.split(".")):
                raise ValueError(f"predicate '{name}' has an invalid state path")
            predicates[name] = spec

        tool_policies_declared = wrapped and "tools" in config
        tool_policies: dict[str, tuple[str, ...]] = {}
        if tool_policies_declared:
            raw_tools = config["tools"]
            if not isinstance(raw_tools, Mapping):
                raise ValueError("gate config 'tools' must be a mapping")
            for tool_name, raw_policy in raw_tools.items():
                if not isinstance(tool_name, str) or not tool_name:
                    raise ValueError("tool policy names must be non-empty strings")
                if not isinstance(raw_policy, Mapping) or set(raw_policy) != {"requires"}:
                    raise ValueError(
                        f"tool policy '{tool_name}' must contain exactly a requires list"
                    )
                raw_required = raw_policy["requires"]
                if (
                    not isinstance(raw_required, Sequence)
                    or isinstance(raw_required, (str, bytes))
                    or not raw_required
                ):
                    raise ValueError(
                        f"tool policy '{tool_name}' requires a non-empty predicate list"
                    )
                required = tuple(raw_required)
                if any(not isinstance(name, str) or not name for name in required):
                    raise ValueError(
                        f"tool policy '{tool_name}' predicates must be non-empty strings"
                    )
                if len(set(required)) != len(required):
                    raise ValueError(
                        f"tool policy '{tool_name}' contains duplicate predicates"
                    )
                unknown = set(required) - set(predicates)
                if unknown:
                    raise ValueError(
                        f"tool policy '{tool_name}' references unknown predicates: {sorted(unknown)}"
                    )
                tool_policies[tool_name] = required

        return cls(
            predicates,
            version=version,
            tool_policies=tool_policies,
            tool_policies_declared=tool_policies_declared,
        )

    @classmethod
    def from_file(cls, path: str | PathLike[str]) -> "GateEngine":
        """Load a gate config written as strict JSON (a YAML-compatible subset)."""

        def reject_constant(value: str) -> None:
            raise ValueError(f"non-standard JSON constant '{value}' is not allowed")

        def finite_float(value: str) -> float:
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError(f"non-finite JSON number '{value}' is not allowed")
            return parsed

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON object key '{key}'")
                result[key] = value
            return result

        try:
            config = json.loads(
                Path(path).read_text(encoding="utf-8"),
                object_pairs_hook=unique_object,
                parse_constant=reject_constant,
                parse_float=finite_float,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid gate config file '{path}': {exc}") from exc
        if not isinstance(config, Mapping):
            raise ValueError("gate config document must be a JSON object")
        if "version" not in config:
            raise ValueError("gate config file must declare a version")
        return cls.from_mapping(config)

    @property
    def predicate_names(self) -> frozenset[str]:
        return frozenset(self._predicates)

    @property
    def policy_fingerprint(self) -> str:
        return self._policy_fingerprint

    @property
    def has_tool_policies(self) -> bool:
        return self._tool_policies_declared

    def configured_requirements(self, tool_name: str) -> tuple[str, ...] | None:
        return self._tool_policies.get(tool_name)

    def validate_tool_policy(
        self,
        *,
        tool_name: str,
        mutative: bool,
        required_gates: Sequence[str],
    ) -> None:
        """Cross-check called tool metadata against a declared policy table."""

        if not self._tool_policies_declared:
            return
        configured = self._tool_policies.get(tool_name)
        if not mutative:
            if configured is not None:
                raise ValueError(
                    f"tool '{tool_name}' has a gate policy but is not registered as mutative"
                )
            return
        if configured is None:
            raise ValueError(f"mutative tool '{tool_name}' has no configured gate policy")
        if tuple(required_gates) != configured:
            raise ValueError(
                f"tool '{tool_name}' gate requirements do not match configured policy"
            )

    @staticmethod
    def _resolve_path(state: Mapping[str, JsonValue], path: str) -> Any:
        current: Any = state
        for segment in path.split("."):
            if not isinstance(current, Mapping) or segment not in current:
                return _MISSING
            current = current[segment]
        return current

    def _evaluate(self, predicate: str, state: Mapping[str, JsonValue]) -> bool:
        try:
            spec = self._predicates[predicate]
        except KeyError as exc:
            raise KeyError(f"unknown gate predicate '{predicate}'") from exc

        actual = self._resolve_path(state, spec.path)
        if spec.op == "exists":
            return actual is not _MISSING
        if spec.op == "not_null":
            return actual is not _MISSING and actual is not None
        if spec.op == "truthy":
            return actual is not _MISSING and bool(actual)
        if spec.op == "eq":
            if actual is _MISSING:
                return False
            if isinstance(actual, bool) or isinstance(spec.value, bool):
                return type(actual) is type(spec.value) and actual == spec.value
            return actual == spec.value
        raise AssertionError(f"unreachable operator: {spec.op}")

    def _evaluate_required(
        self,
        required: Sequence[str],
        state: Mapping[str, JsonValue],
    ) -> list[tuple[str, bool]]:
        return [(predicate, self._evaluate(predicate, state)) for predicate in required]

    def check(
        self,
        required: Sequence[str],
        state: Mapping[str, JsonValue],
        *,
        index: int,
        tool_name: str,
        mode: GateMode,
    ) -> tuple[bool, list[GateEvent]]:
        """Check a pending call, returning whether it may be dispatched."""

        mode = GateMode(mode)
        evaluated = self._evaluate_required(required, state)
        allowed = all(passed for _, passed in evaluated)
        events: list[GateEvent] = []
        for predicate, passed in evaluated:
            if passed:
                action = "allow"
                blocked = False
                violation = False
            elif mode is GateMode.ENFORCE:
                action = "enforce_block"
                blocked = True
                violation = False
            else:
                action = "audit_violation"
                blocked = False
                violation = True
            events.append(
                GateEvent(
                    index=index,
                    tool_name=tool_name,
                    predicate=predicate,
                    passed=passed,
                    action=action,
                    blocked=blocked,
                    violation=violation,
                )
            )
        return allowed or mode is GateMode.AUDIT, events

    def replay(self, tool_events: Sequence[ToolEvent]) -> list[GateEvent]:
        """Replay executed mutative attempts using their pre-call snapshots.

        A dispatched attempt is checked even when its handler failed. Calls
        blocked before dispatch and non-mutative calls cannot be violations.
        """

        replayed: list[GateEvent] = []
        for event in tool_events:
            if not event.dispatched or not event.mutative:
                continue
            for predicate, passed in self._evaluate_required(
                event.required_gates,
                event.state_before,
            ):
                replayed.append(
                    GateEvent(
                        index=event.index,
                        tool_name=event.call.name,
                        predicate=predicate,
                        passed=passed,
                        action="allow" if passed else "replay_violation",
                        blocked=False,
                        violation=not passed,
                    )
                )
        return replayed
