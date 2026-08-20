"""Typed, JSON-serializable contracts for tool execution and rewards.

The models in this module intentionally do not retain a model's raw completion.
Reward computation can therefore depend only on normalized calls, environment
events, and an environment-backed outcome.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


def contains_surrogate(text: str) -> bool:
    r"""True when a string holds a surrogate codepoint.

    ``json`` accepts an escape such as ``\ud800`` and yields a ``str`` holding a
    lone surrogate. Python stores real codepoints, so any surrogate present here
    is unpaired and UTF-8 cannot encode it. Every strict-JSON boundary in this
    package rejects such a value, so it fails where it enters and becomes scored
    evidence instead of aborting a run later during hashing or result writing.
    """

    return any(0xD800 <= ord(char) <= 0xDFFF for char in text)


class _ContractModel(BaseModel):
    """Strict base class shared by all serialized reliability contracts."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ToolCall(_ContractModel):
    """A normalized request to invoke one registered tool."""

    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class ParseIssue(_ContractModel):
    """A deterministic parser failure observed while reading a completion.

    ``attached_to_block`` records whether the failure belongs to an emitted
    ``<tool_call>`` block. Stray text outside the envelope is preserved as
    evidence but must not make an otherwise valid block score as malformed;
    the format term is a conjunction over emitted blocks only.
    """

    block_index: int = Field(ge=0)
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    attached_to_block: bool = True


class ParseResult(_ContractModel):
    """Normalized calls and failures extracted from one completion."""

    emitted_blocks: int = Field(ge=0)
    calls: list[ToolCall] = Field(default_factory=list)
    issues: list[ParseIssue] = Field(default_factory=list)


class ToolEvent(_ContractModel):
    """The observable result of validating or dispatching one tool call."""

    index: int = Field(ge=0)
    call: ToolCall
    schema_valid: bool
    dispatched: bool
    succeeded: bool
    mutative: bool
    required_gates: tuple[str, ...] = ()
    state_before: dict[str, JsonValue] = Field(default_factory=dict)
    state_after: dict[str, JsonValue] = Field(default_factory=dict)
    output: JsonValue | None = None
    error_code: str | None = None
    error_message: str | None = None


class GateEvent(_ContractModel):
    """One predicate check performed for a mutative tool attempt."""

    index: int = Field(ge=0)
    tool_name: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    passed: bool
    action: Literal["allow", "audit_violation", "enforce_block", "replay_violation"]
    blocked: bool
    violation: bool


def compute_evidence_digest(
    *,
    parse: ParseResult,
    tool_events: list[ToolEvent],
    gate_events: list[GateEvent],
    final_state: dict[str, JsonValue],
    gate_policy_fingerprint: str | None,
) -> str:
    """Hash every piece of evidence consumed by reward computation."""

    payload = {
        "parse": parse.model_dump(mode="json"),
        "tool_events": [event.model_dump(mode="json") for event in tool_events],
        "gate_events": [event.model_dump(mode="json") for event in gate_events],
        "final_state": final_state,
        "gate_policy_fingerprint": gate_policy_fingerprint,
    }
    serialized = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    try:
        encoded = serialized.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "reward evidence contains text that UTF-8 cannot encode "
            "(most likely an unpaired surrogate): %s" % exc
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


class EpisodeTrace(_ContractModel):
    """All normalized evidence produced while executing one completion."""

    parse: ParseResult
    tool_events: list[ToolEvent] = Field(default_factory=list)
    gate_events: list[GateEvent] = Field(default_factory=list)
    final_state: dict[str, JsonValue] = Field(default_factory=dict)
    gate_policy_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    evidence_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    def calculated_evidence_digest(self) -> str:
        return compute_evidence_digest(
            parse=self.parse,
            tool_events=self.tool_events,
            gate_events=self.gate_events,
            final_state=self.final_state,
            gate_policy_fingerprint=self.gate_policy_fingerprint,
        )

    def verify_evidence_digest(self) -> None:
        """Reject nested mutations that bypass the frozen outer model."""

        if not hmac.compare_digest(self.evidence_digest, self.calculated_evidence_digest()):
            raise ValueError("episode trace evidence digest mismatch")

    @model_validator(mode="after")
    def _validate_evidence_digest(self) -> "EpisodeTrace":
        self.verify_evidence_digest()
        return self


class OutcomeSource(str, Enum):
    """Allowed deterministic sources for an accuracy outcome."""

    SANDBOX_RESULT = "sandbox_result"
    DB_STATE = "db_state"


class EnvironmentOutcome(_ContractModel):
    """Accuracy established by the environment, never by generated prose."""

    correct: bool
    source: OutcomeSource


class RewardBreakdown(_ContractModel):
    """Signed components of the execution-backed composite reward."""

    accuracy: float
    format: float
    gate: float
    efficiency: float
    total: float
    gate_violation: bool
    executed_calls: int = Field(ge=0)
    format_valid: bool
