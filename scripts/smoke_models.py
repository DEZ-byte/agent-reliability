"""Offline-first smoke-test planner and runner for Qwen model candidates.

The default command only validates configuration, inventories the local
environment, and writes the exact probe plan.  It does not import ML runtime
libraries or access model repositories.  Model/tokenizer access is gated by
the explicit combination of ``--run-load`` and ``--allow-download``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)


CONFIG_SCHEMA_VERSION: Final = 1
RESULT_SCHEMA_VERSION: Final = 1
DEFAULT_CONFIG: Final = Path(__file__).resolve().parents[1] / "configs" / "model_smoke.json"
DEFAULT_OUTPUT: Final = Path(__file__).resolve().parents[1] / "results" / "model_smoke.json"
PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
MAX_CONFIG_BYTES: Final = 256 * 1024
MAX_REGISTRY_BYTES: Final = 256 * 1024
MAX_RESULT_BYTES: Final = 8 * 1024 * 1024
MAX_CANDIDATES: Final = 4
MAX_MESSAGE_CHARS: Final = 4096
MAX_MESSAGES: Final = 16
MAX_TOOLS: Final = 16
MAX_TOOL_SCHEMA_CHARS: Final = 16 * 1024
MAX_GENERATION_CASES: Final = 8
MAX_GENERATION_CALLS: Final = 64
MAX_PERSISTED_PROMPT_CHARS: Final = 32 * 1024
MAX_TEMPLATE_TOKENS: Final = 8192
MAX_TEMPLATE_SOURCE_CHARS: Final = 256 * 1024
MAX_ERROR_CHARS: Final = 4096
MAX_P6_TOKENS: Final = 512
P6_LORA_RANK: Final = 4
P6_LEARNING_RATE: Final = 1e-3
P6_REFERENCE_ATOL: Final = 1e-4
P6_REFERENCE_RTOL: Final = 1e-4
GENERATION_START_MARKER: Final = "__CODEX_SMOKE_GENERATION_START_7D9C4A__"
GENERATION_END_MARKER: Final = "__CODEX_SMOKE_GENERATION_END_7D9C4A__"
OPTIONAL_LIBRARIES: Final = (
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "trl",
    "unsloth",
)
_MODEL_ID_RE: Final = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,95})/[A-Za-z0-9](?:[A-Za-z0-9._-]{0,95})$"
)
_CANDIDATE_NAME_RE: Final = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,63})$")
_SHA_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_PROTOCOL_PROBES: Final = tuple(f"P{index}" for index in range(7))


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class CandidateConfig(StrictModel):
    name: str
    bundle: Literal["qwen2.5", "qwen3"]
    role: Literal["primary_small", "scale_check"]
    model_id: str
    revision: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not _CANDIDATE_NAME_RE.fullmatch(value):
            raise ValueError("candidate name must be a lower-case slug")
        return value

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        if not _MODEL_ID_RE.fullmatch(value) or ".." in value:
            raise ValueError(
                "model_id must be a Hugging Face owner/repository ID, not a path"
            )
        return value

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        if not _SHA_RE.fullmatch(value):
            raise ValueError("revision must be an immutable 40-character lowercase commit SHA")
        return value


class ChatMessage(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class GeneratedToolFunction(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
    arguments: dict[str, JsonValue]


class GeneratedToolCall(StrictModel):
    type: Literal["function"]
    function: GeneratedToolFunction


class TrainingMessage(StrictModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(max_length=MAX_MESSAGE_CHARS)
    name: str | None = Field(
        default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,63}$"
    )
    tool_calls: list[GeneratedToolCall] | None = None

    @model_validator(mode="after")
    def validate_role_fields(self) -> "TrainingMessage":
        if self.role in {"system", "user"}:
            if not self.content or self.name is not None or self.tool_calls is not None:
                raise ValueError("system/user training messages only accept non-empty content")
        elif self.role == "assistant":
            if self.name is not None or not self.tool_calls:
                raise ValueError("assistant training message requires tool_calls and no name")
        elif self.role == "tool":
            if not self.content or not self.name or self.tool_calls is not None:
                raise ValueError("tool training message requires name and content")
        return self


class GenerationCase(StrictModel):
    name: str
    expected_tool: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
    messages: list[ChatMessage] = Field(min_length=1, max_length=MAX_MESSAGES)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not _CANDIDATE_NAME_RE.fullmatch(value):
            raise ValueError("generation case name must be a lower-case slug")
        return value

    @model_validator(mode="after")
    def includes_user_message(self) -> "GenerationCase":
        if not any(message.role == "user" for message in self.messages):
            raise ValueError("generation case must include a user message")
        return self


class FunctionParameters(StrictModel):
    type: Literal["object"]
    properties: dict[str, dict[str, JsonValue]] = Field(
        min_length=1, max_length=32
    )
    required: list[str] = Field(min_length=1, max_length=32)
    additionalProperties: Literal[False]

    @model_validator(mode="after")
    def required_properties_exist(self) -> "FunctionParameters":
        unknown = set(self.required).difference(self.properties)
        if unknown:
            raise ValueError(f"required keys missing from properties: {sorted(unknown)!r}")
        if len(self.required) != len(set(self.required)):
            raise ValueError("required property names must be unique")
        return self


class FunctionSpec(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
    description: str = Field(min_length=1, max_length=2048)
    parameters: FunctionParameters


class ToolSpec(StrictModel):
    type: Literal["function"]
    mutative: bool
    function: FunctionSpec


class ProbeConfig(StrictModel):
    messages: list[ChatMessage] = Field(
        min_length=1, max_length=MAX_MESSAGES
    )
    tools: list[ToolSpec] = Field(min_length=1, max_length=MAX_TOOLS)
    generation_cases: list[GenerationCase] = Field(
        min_length=2, max_length=MAX_GENERATION_CASES
    )
    training_trajectory: list[TrainingMessage] = Field(
        min_length=4, max_length=MAX_MESSAGES
    )
    mask_sentinels: dict[Literal["system", "user", "assistant", "tool"], str]
    chat_template_kwargs: dict[str, JsonValue] = Field(max_length=16)
    expected_template_markers: list[str] = Field(min_length=1, max_length=32)
    seed: int = Field(ge=0, le=2**32 - 1)
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(gt=0.0, le=1.0)
    max_new_tokens: int = Field(ge=1, le=256)
    warmup_runs: int = Field(ge=0, le=10)
    timed_runs: int = Field(ge=1, le=20)
    max_decoded_output_chars: int = Field(ge=128, le=16384)
    target_cuda_device_index: int = Field(ge=0, le=15)
    quantization: Literal["nf4"]
    double_quant: bool

    @model_validator(mode="after")
    def validate_probe(self) -> "ProbeConfig":
        if not any(message.role == "user" for message in self.messages):
            raise ValueError("probe messages must include a user message")
        if len(self.expected_template_markers) != len(set(self.expected_template_markers)):
            raise ValueError("expected template markers must be unique")
        if any(not marker.strip() for marker in self.expected_template_markers):
            raise ValueError("expected template markers must be non-blank")
        if any(len(marker) > 256 for marker in self.expected_template_markers):
            raise ValueError("expected template markers must be at most 256 characters")
        tool_names = [tool.function.name for tool in self.tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("tool names must be unique")
        case_names = [case.name for case in self.generation_cases]
        if len(case_names) != len(set(case_names)):
            raise ValueError("generation case names must be unique")
        unknown_tools = {
            case.expected_tool for case in self.generation_cases
        }.difference(tool_names)
        if unknown_tools:
            raise ValueError(f"generation cases reference unknown tools: {sorted(unknown_tools)!r}")
        expected_mutability = {False, True}
        covered_mutability = {
            next(
                tool.mutative
                for tool in self.tools
                if tool.function.name == case.expected_tool
            )
            for case in self.generation_cases
        }
        if not expected_mutability.issubset(covered_mutability):
            raise ValueError("generation cases must cover read-only and mutative tools")
        if set(self.mask_sentinels) != {"system", "user", "assistant", "tool"}:
            raise ValueError("mask_sentinels must define all four trajectory roles")
        if len(set(self.mask_sentinels.values())) != 4 or any(
            not value for value in self.mask_sentinels.values()
        ):
            raise ValueError("mask sentinels must be unique and non-empty")
        if any(len(value) > 128 for value in self.mask_sentinels.values()):
            raise ValueError("mask sentinels must be at most 128 characters")
        serialized_trajectory = json.dumps(
            [message.model_dump(mode="json") for message in self.training_trajectory],
            sort_keys=True,
        )
        if len(serialized_trajectory) > MAX_PERSISTED_PROMPT_CHARS:
            raise ValueError("training trajectory exceeds the prompt budget")
        missing_sentinels = [
            role for role, sentinel in self.mask_sentinels.items()
            if serialized_trajectory.count(sentinel) != 1
        ]
        if missing_sentinels:
            raise ValueError(
                "each mask sentinel must occur exactly once in training_trajectory: "
                f"{missing_sentinels!r}"
            )
        generation_calls = len(self.generation_cases) * (
            self.warmup_runs + self.timed_runs
        )
        if generation_calls > MAX_GENERATION_CALLS:
            raise ValueError(
                f"configured generation work exceeds {MAX_GENERATION_CALLS} calls"
            )
        total_content_chars = sum(len(message.content) for message in self.messages)
        total_content_chars += sum(
            len(message.content)
            for case in self.generation_cases
            for message in case.messages
        )
        total_content_chars += sum(
            len(message.content) for message in self.training_trajectory
        )
        if total_content_chars > MAX_PERSISTED_PROMPT_CHARS:
            raise ValueError("configured message content exceeds the prompt budget")
        for tool in self.tools:
            encoded_tool = json.dumps(
                tool.model_dump(mode="json"),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            if len(encoded_tool) > MAX_TOOL_SCHEMA_CHARS:
                raise ValueError(
                    f"tool schema '{tool.function.name}' exceeds the size limit"
                )
        if len(json.dumps(self.chat_template_kwargs, allow_nan=False)) > 4096:
            raise ValueError("chat_template_kwargs exceeds the size limit")
        return self


class SmokeLaneConfig(StrictModel):
    identity: Literal["phase-a-windows-unsloth-trl024"]
    lock_path: Literal["requirements-smoke.lock"]
    expected_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    m6_environment_factory_in_scope: Literal[False]
    probe_implementation: dict[
        Literal["P0", "P1", "P2", "P3", "P4", "P5", "P6"],
        Literal["implemented", "not_implemented"],
    ]

    @model_validator(mode="after")
    def validate_probe_implementation(self) -> "SmokeLaneConfig":
        expected = {probe_id: "implemented" for probe_id in _PROTOCOL_PROBES}
        if self.probe_implementation != expected:
            raise ValueError(
                "probe_implementation must mark P0-P6 implemented"
            )
        return self


class ReleaseSelectionGate(StrictModel):
    registry_path: Literal["configs/model_candidates.json"]
    expected_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["pending", "resolved"]
    intended_release_scope: str | None = Field(default=None, max_length=2048)
    decision_record: str | None = Field(
        default=None, pattern=r"^D-[0-9]{3}$"
    )
    eligible_bundles: list[Literal["qwen2.5", "qwen3"]] = Field(
        default_factory=list, max_length=2
    )

    @model_validator(mode="after")
    def state_is_consistent(self) -> "ReleaseSelectionGate":
        if self.status == "pending":
            if (
                self.intended_release_scope is not None
                or self.decision_record is not None
                or self.eligible_bundles
            ):
                raise ValueError(
                    "a pending release gate cannot define scope, decision, or eligible bundles"
                )
        elif (
            not self.intended_release_scope
            or not self.decision_record
            or not self.eligible_bundles
        ):
            raise ValueError(
                "a resolved release gate requires scope, a decision record, and eligible bundles"
            )
        if len(self.eligible_bundles) != len(set(self.eligible_bundles)):
            raise ValueError("eligible release bundles must be unique")
        return self

    @property
    def selection_allowed(self) -> bool:
        return self.status == "resolved" and bool(self.eligible_bundles)


class SmokeConfig(StrictModel):
    schema_version: Literal[1]
    lane: SmokeLaneConfig
    release_gate: ReleaseSelectionGate
    candidates: list[CandidateConfig] = Field(
        min_length=1, max_length=MAX_CANDIDATES
    )
    probe: ProbeConfig

    @model_validator(mode="after")
    def candidates_are_unique(self) -> "SmokeConfig":
        names = [candidate.name for candidate in self.candidates]
        model_ids = [candidate.model_id for candidate in self.candidates]
        revisions = [candidate.revision for candidate in self.candidates]
        if len(names) != len(set(names)):
            raise ValueError("candidate names must be unique")
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("candidate model IDs must be unique")
        if len(revisions) != len(set(revisions)):
            raise ValueError("candidate revisions must be unique")
        role_counts = {
            role: sum(candidate.role == role for candidate in self.candidates)
            for role in ("primary_small", "scale_check")
        }
        if role_counts != {"primary_small": 2, "scale_check": 2}:
            raise ValueError(
                "config must contain exactly two primary_small and two scale_check candidates"
            )
        bundle_roles = {
            bundle: {candidate.role for candidate in self.candidates if candidate.bundle == bundle}
            for bundle in ("qwen2.5", "qwen3")
        }
        if bundle_roles != {
            "qwen2.5": {"primary_small", "scale_check"},
            "qwen3": {"primary_small", "scale_check"},
        }:
            raise ValueError("each Qwen bundle must contain one primary and one scale candidate")
        return self


class CudaDevice(StrictModel):
    index: int = Field(ge=0)
    name: str
    driver_version: str
    memory_total_mib: int | None = Field(default=None, ge=0)
    compute_capability: str | None = None


class CudaFacts(StrictModel):
    status: Literal["available", "unavailable", "error"]
    cuda_visible_devices: str | None
    nvidia_smi_path: str | None
    devices: list[CudaDevice]
    error: str | None = None


class HardwareFacts(StrictModel):
    python_version: str
    python_implementation: str
    executable: str
    platform: str
    system: str
    release: str
    machine: str
    processor: str
    logical_cpu_count: int | None = Field(default=None, ge=1)
    cuda: CudaFacts


class ProbeResult(StrictModel):
    name: Literal[
        "tool_chat_template",
        "four_bit_load",
        "deterministic_generation",
        "training_stack_imports",
        "training_template_masking",
    ]
    status: Literal["planned", "passed", "failed", "unavailable", "skipped"]
    plan: dict[str, JsonValue]
    metrics: dict[str, JsonValue]
    error: str | None = Field(default=None, max_length=MAX_ERROR_CHARS)


class MinimalTrainingResult(StrictModel):
    probe_id: Literal["P6"] = "P6"
    name: Literal["minimal_training_execution"] = "minimal_training_execution"
    status: Literal["planned", "passed", "failed", "skipped"] = "planned"
    executed: bool = False
    passed: bool = False
    plan: dict[str, JsonValue] = Field(default_factory=dict)
    metrics: dict[str, JsonValue] = Field(default_factory=dict)
    error: str | None = Field(default=None, max_length=MAX_ERROR_CHARS)

    @model_validator(mode="after")
    def status_is_consistent(self) -> "MinimalTrainingResult":
        expected_executed = self.status in {"passed", "failed"}
        if self.executed != expected_executed:
            raise ValueError("P6 executed must be true exactly when its status passed or failed")
        if self.passed != (self.status == "passed"):
            raise ValueError("P6 passed must be true exactly when its status is passed")
        if self.status == "failed" and self.error is None:
            raise ValueError("a failed P6 result requires an error")
        if self.status == "passed" and self.error is not None:
            raise ValueError("a passed P6 result cannot include an error")
        if self.status == "skipped" and self.error is None:
            raise ValueError("a skipped P6 result requires a reason")
        return self


def _derived_environment_compatibility(
    probes: list[ProbeResult] | tuple[ProbeResult, ...],
) -> bool | None:
    statuses = {probe.name: probe.status for probe in probes}
    required = (
        "tool_chat_template",
        "four_bit_load",
        "deterministic_generation",
        "training_stack_imports",
        "training_template_masking",
    )
    if any(statuses.get(name) == "planned" for name in required):
        return None
    if any(name not in statuses for name in required):
        return None
    return all(statuses[name] == "passed" for name in required)


class CandidateResult(StrictModel):
    name: str
    bundle: Literal["qwen2.5", "qwen3"]
    role: Literal["primary_small", "scale_check"]
    model_id: str
    requested_revision: str
    resolved_revision: str | None
    probes: list[ProbeResult]
    p6: MinimalTrainingResult
    environment_compatible: bool | None
    selection_eligible: bool

    @model_validator(mode="after")
    def enforce_true_gates_and_p6(self) -> "CandidateResult":
        expected_compatibility = _derived_environment_compatibility(self.probes)
        if self.environment_compatible != expected_compatibility:
            raise ValueError(
                "environment_compatible must reflect every required P1-P5 probe"
            )
        if self.selection_eligible and not (
            self.environment_compatible is True
            and self.p6.executed
            and self.p6.status == "passed"
            and self.p6.passed
        ):
            raise ValueError(
                "selection eligibility requires the environment gates and executed/passed P6"
            )
        return self


def _candidate_result(
    *,
    name: str,
    bundle: Literal["qwen2.5", "qwen3"],
    role: Literal["primary_small", "scale_check"],
    model_id: str,
    requested_revision: str,
    resolved_revision: str | None,
    probes: list[ProbeResult],
    p6: MinimalTrainingResult | None = None,
) -> CandidateResult:
    p6_result = p6 or MinimalTrainingResult(plan=_minimal_training_plan())
    environment_compatible = _derived_environment_compatibility(probes)
    return CandidateResult(
        name=name,
        bundle=bundle,
        role=role,
        model_id=model_id,
        requested_revision=requested_revision,
        resolved_revision=resolved_revision,
        probes=probes,
        p6=p6_result,
        environment_compatible=environment_compatible,
        selection_eligible=(
            environment_compatible is True
            and p6_result.executed
            and p6_result.status == "passed"
            and p6_result.passed
        ),
    )


class RunOptions(StrictModel):
    dry_run: bool
    run_load: bool
    allow_download: bool
    selected_candidates: list[str]


class SmokeLaneResult(StrictModel):
    identity: Literal["phase-a-windows-unsloth-trl024"]
    lock_path: Literal["requirements-smoke.lock"]
    expected_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    actual_lock_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    lock_matches_expected: bool
    m6_environment_factory_in_scope: Literal[False]
    probe_implementation: dict[
        Literal["P0", "P1", "P2", "P3", "P4", "P5", "P6"],
        Literal["implemented", "not_implemented"],
    ]

    @model_validator(mode="after")
    def lock_identity_is_consistent(self) -> "SmokeLaneResult":
        matches = (
            self.actual_lock_sha256 is not None
            and self.actual_lock_sha256 == self.expected_lock_sha256
        )
        if self.lock_matches_expected != matches:
            raise ValueError("lock_matches_expected is inconsistent with the lock hashes")
        expected_probes = {probe_id: "implemented" for probe_id in _PROTOCOL_PROBES}
        if self.probe_implementation != expected_probes:
            raise ValueError("lane result has an invalid probe implementation contract")
        return self


class SourceIdentity(StrictModel):
    git_commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    smoke_script_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    smoke_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SmokeResult(StrictModel):
    schema_version: Literal[1]
    created_at_utc: str
    config_path: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lane: SmokeLaneResult
    release_gate: ReleaseSelectionGate
    release_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_identity: SourceIdentity
    command: list[str]
    options: RunOptions
    hardware: HardwareFacts
    library_versions: dict[str, str | None]
    candidates: list[CandidateResult]
    selection_eligible: bool

    @field_validator("created_at_utc")
    @classmethod
    def validate_utc_time(cls, value: str) -> str:
        if not value.endswith("Z"):
            raise ValueError("created_at_utc must use the UTC Z suffix")
        datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
        return value

    @model_validator(mode="after")
    def validate_identity_and_eligibility(self) -> "SmokeResult":
        if self.config_sha256 != self.source_identity.smoke_config_sha256:
            raise ValueError("config and source-identity hashes must match")
        if self.release_registry_sha256 != self.release_gate.expected_registry_sha256:
            raise ValueError("release registry hash must match the configured evidence hash")
        expected_selection = (
            self.release_gate.status == "resolved"
            and _has_selectable_bundle(self.candidates, self.release_gate.eligible_bundles)
        )
        if self.selection_eligible != expected_selection:
            raise ValueError(
                "top-level selection eligibility requires all four configured roles and candidates"
            )
        return self


def _all_four_candidates_eligible(
    candidates: list[CandidateResult] | tuple[CandidateResult, ...],
) -> bool:
    if len(candidates) != MAX_CANDIDATES:
        return False
    if len({candidate.name for candidate in candidates}) != MAX_CANDIDATES:
        return False
    role_counts = {
        role: sum(candidate.role == role for candidate in candidates)
        for role in ("primary_small", "scale_check")
    }
    return role_counts == {"primary_small": 2, "scale_check": 2} and all(
        candidate.selection_eligible for candidate in candidates
    )


def _has_selectable_bundle(
    candidates: list[CandidateResult] | tuple[CandidateResult, ...],
    eligible_bundles: list[Literal["qwen2.5", "qwen3"]]
    | tuple[Literal["qwen2.5", "qwen3"], ...],
) -> bool:
    if not _all_four_candidates_eligible(candidates):
        return False
    by_bundle = {
        bundle: [candidate for candidate in candidates if candidate.bundle == bundle]
        for bundle in ("qwen2.5", "qwen3")
    }
    return any(
        len(by_bundle[bundle]) == 2
        and {candidate.role for candidate in by_bundle[bundle]}
        == {"primary_small", "scale_check"}
        for bundle in eligible_bundles
    )


class SmokeConfigError(ValueError):
    """Raised when a smoke-test configuration cannot be decoded or validated."""


class SmokeResultError(ValueError):
    """Raised when a persisted result cannot be decoded or validated."""


class _StrictJSONError(ValueError):
    pass


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _StrictJSONError(f"duplicate JSON object key '{key}'")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise _StrictJSONError(f"non-standard JSON constant '{value}' is not allowed")


def _strict_json_loads(raw: str | bytes) -> object:
    return json.loads(
        raw,
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )


def load_config(path: str | os.PathLike[str]) -> SmokeConfig:
    config_path = Path(path)
    try:
        raw = config_path.read_bytes()
        if len(raw) > MAX_CONFIG_BYTES:
            raise _StrictJSONError(
                f"config exceeds the {MAX_CONFIG_BYTES}-byte size limit"
            )
        payload = _strict_json_loads(raw)
        return SmokeConfig.model_validate(payload)
    except (OSError, json.JSONDecodeError, _StrictJSONError, ValidationError) as exc:
        raise SmokeConfigError(f"invalid smoke config {config_path}: {exc}") from exc


def read_result(path: str | os.PathLike[str]) -> SmokeResult:
    result_path = Path(path)
    try:
        raw = result_path.read_bytes()
        if len(raw) > MAX_RESULT_BYTES:
            raise _StrictJSONError(
                f"result exceeds the {MAX_RESULT_BYTES}-byte size limit"
            )
        payload = _strict_json_loads(raw)
        return SmokeResult.model_validate(payload)
    except (OSError, json.JSONDecodeError, _StrictJSONError, ValidationError) as exc:
        raise SmokeResultError(f"invalid smoke result {result_path}: {exc}") from exc


def write_result_atomic(result: SmokeResult, path: str | os.PathLike[str]) -> None:
    """Validate and atomically replace a UTF-8 JSON result."""

    # Frozen models can still contain mutable JSON collections. Revalidation
    # prevents post-construction mutation from bypassing the result schema.
    validated = SmokeResult.model_validate(
        result.model_dump(mode="python", warnings=False)
    )
    try:
        encoded = json.dumps(
            validated.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise SmokeResultError(f"result is not strict JSON: {exc}") from exc
    if len(encoded.encode("utf-8")) > MAX_RESULT_BYTES:
        raise SmokeResultError(
            f"result exceeds the {MAX_RESULT_BYTES}-byte size limit"
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_lane_lock_path(lane: SmokeLaneConfig) -> Path:
    lock_path = (PROJECT_ROOT / lane.lock_path).resolve()
    try:
        lock_path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise SmokeConfigError("lane lock path must remain inside the project") from exc
    return lock_path


def _collect_lane_result(lane: SmokeLaneConfig) -> SmokeLaneResult:
    lock_path = _resolve_lane_lock_path(lane)
    try:
        actual_sha256 = _sha256_file(lock_path)
    except FileNotFoundError:
        actual_sha256 = None
    except OSError as exc:
        raise SmokeConfigError(f"cannot read lane lock {lock_path}: {exc}") from exc
    return SmokeLaneResult(
        identity=lane.identity,
        lock_path=lane.lock_path,
        expected_lock_sha256=lane.expected_lock_sha256,
        actual_lock_sha256=actual_sha256,
        lock_matches_expected=actual_sha256 == lane.expected_lock_sha256,
        m6_environment_factory_in_scope=lane.m6_environment_factory_in_scope,
        probe_implementation=dict(lane.probe_implementation),
    )


def _resolve_release_registry_path(gate: ReleaseSelectionGate) -> Path:
    registry_path = (PROJECT_ROOT / gate.registry_path).resolve()
    try:
        registry_path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise SmokeConfigError("release registry path must remain inside the project") from exc
    return registry_path


def _decision_section(decision_record: str) -> str:
    try:
        text = (PROJECT_ROOT / "DECISIONS.md").read_text(encoding="utf-8")
    except OSError as exc:
        raise SmokeConfigError("cannot read DECISIONS.md for the release gate") from exc
    match = re.search(
        rf"(?ms)^### {re.escape(decision_record)} —[^\r\n]*\r?\n"
        rf"(?P<body>.*?)(?=^### D-[0-9]{{3}} —|\Z)",
        text,
    )
    if match is None:
        raise SmokeConfigError(
            f"release decision {decision_record} is not a recorded DECISIONS.md section"
        )
    return match.group("body")


def _release_decision_markers(gate: ReleaseSelectionGate) -> tuple[str, str]:
    bundles = ", ".join(f"`{bundle}`" for bundle in gate.eligible_bundles)
    return (
        f"Release scope: `{gate.intended_release_scope}`",
        f"Release-eligible bundles: {bundles}",
    )


def _validate_release_registry(
    gate: ReleaseSelectionGate,
    candidates: list[CandidateConfig] | tuple[CandidateConfig, ...],
) -> str:
    """Validate the immutable registry and any resolved release decision."""

    registry_path = _resolve_release_registry_path(gate)
    try:
        raw = registry_path.read_bytes()
    except OSError as exc:
        raise SmokeConfigError(f"cannot read release registry {registry_path}: {exc}") from exc
    if len(raw) > MAX_REGISTRY_BYTES:
        raise SmokeConfigError("release registry exceeds its size limit")
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != gate.expected_registry_sha256:
        raise SmokeConfigError(
            "release registry SHA-256 does not match the configured evidence hash"
        )
    try:
        payload = _strict_json_loads(raw)
    except (json.JSONDecodeError, _StrictJSONError) as exc:
        raise SmokeConfigError(f"release registry is not strict JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise SmokeConfigError("release registry must be a schema-version 1 object")
    roles = payload.get("roles")
    if not isinstance(roles, dict):
        raise SmokeConfigError("release registry roles must be an object")

    entries: dict[str, tuple[str, dict[str, object]]] = {}
    for role, role_entries in roles.items():
        if not isinstance(role, str) or not isinstance(role_entries, list):
            raise SmokeConfigError("release registry roles contain invalid entries")
        for entry in role_entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
                raise SmokeConfigError("release registry contains an invalid candidate")
            model_id = entry["id"]
            if model_id in entries:
                raise SmokeConfigError(f"release registry repeats model ID {model_id}")
            entries[model_id] = (role, entry)

    states_by_bundle: dict[str, set[str]] = {"qwen2.5": set(), "qwen3": set()}
    for candidate in candidates:
        registered = entries.get(candidate.model_id)
        if registered is None:
            raise SmokeConfigError(
                f"smoke candidate {candidate.model_id} is absent from the release registry"
            )
        registry_role, entry = registered
        if (
            registry_role != candidate.role
            or entry.get("revision") != candidate.revision
            or entry.get("smoke_bundle") != candidate.bundle
        ):
            raise SmokeConfigError(
                f"release registry identity mismatch for {candidate.model_id}"
            )
        state = entry.get("release_eligibility")
        decision = entry.get("release_decision")
        if state not in {"pending", "eligible", "ineligible"}:
            raise SmokeConfigError(
                f"release registry has invalid eligibility for {candidate.model_id}"
            )
        if (state == "pending") != (decision is None):
            raise SmokeConfigError(
                f"release registry decision state is inconsistent for {candidate.model_id}"
            )
        if decision is not None and (
            not isinstance(decision, str) or re.fullmatch(r"D-[0-9]{3}", decision) is None
        ):
            raise SmokeConfigError(
                f"release registry decision is invalid for {candidate.model_id}"
            )
        states_by_bundle[candidate.bundle].add(state)
        if gate.status == "pending" and state != "pending":
            raise SmokeConfigError(
                "a pending release gate requires pending smoke-candidate registry entries"
            )
        if gate.status == "resolved" and decision != gate.decision_record:
            raise SmokeConfigError(
                f"release decision mismatch for {candidate.model_id}"
            )

    if gate.status == "resolved":
        if any(len(states) != 1 or "pending" in states for states in states_by_bundle.values()):
            raise SmokeConfigError(
                "a resolved release gate requires one resolved eligibility per complete bundle"
            )
        derived_bundles = [
            bundle
            for bundle in ("qwen2.5", "qwen3")
            if states_by_bundle[bundle] == {"eligible"}
        ]
        if derived_bundles != gate.eligible_bundles:
            raise SmokeConfigError(
                "release gate eligible bundles do not match the candidate registry"
            )
        section = _decision_section(gate.decision_record or "")
        missing_markers = [
            marker for marker in _release_decision_markers(gate) if marker not in section
        ]
        if missing_markers:
            raise SmokeConfigError(
                "release decision section does not contain the exact scope and bundle markers"
            )
    return actual_sha256


def _git_commit_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SmokeConfigError("cannot resolve the source Git commit SHA") from exc
    commit_sha = completed.stdout.strip().lower()
    if not _SHA_RE.fullmatch(commit_sha):
        raise SmokeConfigError("Git returned an invalid source commit SHA")
    return commit_sha


def _git_worktree_changes() -> list[str]:
    """Return all nonignored staged, unstaged, and untracked Git changes."""

    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SmokeConfigError("cannot verify that the Git worktree is clean") from exc
    return [line for line in completed.stdout.splitlines() if line]


def _require_clean_git_worktree() -> None:
    changes = _git_worktree_changes()
    if changes:
        raise SmokeConfigError(
            "measured smoke run requires a clean Git worktree; detected "
            f"{len(changes)} nonignored staged, unstaged, or untracked change(s)"
        )


def _collect_source_identity(config_sha256: str) -> SourceIdentity:
    if not _SHA256_RE.fullmatch(config_sha256):
        raise SmokeConfigError("config SHA-256 is invalid")
    return SourceIdentity(
        git_commit_sha=_git_commit_sha(),
        smoke_script_sha256=_sha256_file(Path(__file__).resolve()),
        smoke_config_sha256=config_sha256,
    )


def collect_library_versions() -> dict[str, str | None]:
    """Inspect installed distribution metadata without importing ML libraries."""

    versions: dict[str, str | None] = {}
    for distribution in OPTIONAL_LIBRARIES:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def collect_cuda_facts() -> CudaFacts:
    """Read CUDA device facts via nvidia-smi without importing torch."""

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return CudaFacts(
            status="unavailable",
            cuda_visible_devices=visible_devices,
            nvidia_smi_path=None,
            devices=[],
            error=None,
        )

    query = (
        "--query-gpu=index,name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    )
    try:
        completed = subprocess.run(
            [executable, *query],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        devices: list[CudaDevice] = []
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split(",", maxsplit=4)]
            if len(fields) != 5:
                raise ValueError(f"unexpected nvidia-smi row: {line!r}")
            memory = None if fields[3] in {"", "N/A"} else int(fields[3])
            capability = None if fields[4] in {"", "N/A"} else fields[4]
            devices.append(
                CudaDevice(
                    index=int(fields[0]),
                    name=fields[1],
                    driver_version=fields[2],
                    memory_total_mib=memory,
                    compute_capability=capability,
                )
            )
        return CudaFacts(
            status="available" if devices else "unavailable",
            cuda_visible_devices=visible_devices,
            nvidia_smi_path=executable,
            devices=devices,
            error=None,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return CudaFacts(
            status="error",
            cuda_visible_devices=visible_devices,
            nvidia_smi_path=executable,
            devices=[],
            error=_error_text(exc),
        )


def collect_hardware_facts() -> HardwareFacts:
    return HardwareFacts(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        executable=sys.executable,
        platform=platform.platform(),
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        processor=platform.processor(),
        logical_cpu_count=os.cpu_count(),
        cuda=collect_cuda_facts(),
    )


def _minimal_training_plan() -> dict[str, JsonValue]:
    return {
        "execution": "one assistant-only LoRA forward/backward and one ephemeral SGD step",
        "batch_size": 1,
        "maximum_sequence_tokens": MAX_P6_TOKENS,
        "reuse": "exact P5 input_ids and assistant_masks on the already-loaded NF4 model",
        "adapter": {
            "implementation": "unsloth.FastLanguageModel.get_peft_model",
            "rank": P6_LORA_RANK,
            "alpha": P6_LORA_RANK,
            "target_modules": ["q_proj", "v_proj"],
            "dropout": 0.0,
            "bias": "none",
            "gradient_checkpointing": "unsloth",
        },
        "masking": "trl.trainer.sft_trainer.DataCollatorForLanguageModeling assistant_masks",
        "reference_policy": "same PEFT model with adapters disabled; no second model",
        "optimizer": {
            "class": "torch.optim.SGD",
            "learning_rate": P6_LEARNING_RATE,
            "momentum": 0.0,
            "checkpoint": False,
        },
        "hard_checks": [
            "TRL labels exactly match the P5 assistant mask",
            "only LoRA parameters are trainable",
            "finite loss, reference log probabilities, gradients, and updates",
            "at least one nonzero LoRA gradient and parameter update",
            "adapter-disable context restores the policy adapter",
            "disabled-adapter reference is invariant across the optimizer step",
            "no optimizer or model checkpoint is written",
        ],
        "quality_claim": False,
    }


def probe_plans(probe: ProbeConfig) -> list[ProbeResult]:
    tool_names = [tool.function.name for tool in probe.tools]
    template_plan: dict[str, JsonValue] = {
        "artifact_access": "AutoTokenizer.from_pretrained at the configured immutable SHA",
        "render": "apply_chat_template(messages, tools, add_generation_prompt=True)",
        "checks": [
            "rendered prompt is non-empty",
            "two independent renders are byte-identical",
            "two independent tokenizations are identical",
            "every configured marker occurs in the rendered prompt",
            "the rendered prompt tokenizes to at least one token",
        ],
        "record": [
            "tokenizer metadata and native chat-template SHA-256",
            "native rendered prompt text and SHA-256",
            "native prefix preservation as a diagnostic, not a hard gate",
        ],
        "expected_markers": list(probe.expected_template_markers),
        "tool_names": tool_names,
    }
    load_plan: dict[str, JsonValue] = {
        "artifact_access": (
            "unsloth.FastLanguageModel.from_pretrained with the configured revision"
        ),
        "device_map": {
            "root": f"cuda:{probe.target_cuda_device_index}",
            "multi_gpu_allowed": False,
        },
        "load_in_4bit": True,
        "quantization_type": probe.quantization,
        "double_quant": probe.double_quant,
        "compute_dtype": "bfloat16 when supported, otherwise float16",
        "measurements": [
            "wall-clock load seconds",
            "model memory footprint bytes",
            "per-device peak CUDA allocated bytes",
            "per-device peak CUDA reserved bytes",
            "target GPU name and total memory",
            "torch CUDA runtime version",
            "actual parameter devices and dtypes",
        ],
        "hard_gate": (
            "every parameter and device-map entry must be on the configured single "
            "CUDA device, with verified effective NF4 4-bit state"
        ),
    }
    generation_plan: dict[str, JsonValue] = {
        "decode": "greedy (do_sample=False)",
        "seed": probe.seed,
        "temperature": probe.temperature,
        "top_p": probe.top_p,
        "sampling_parameters_active": False,
        "max_new_tokens": probe.max_new_tokens,
        "warmup_runs": probe.warmup_runs,
        "timed_runs": probe.timed_runs,
        "timing": "CUDA-synchronized perf_counter around model.generate",
        "compatibility_checks": [
            "all timed output token ID sequences are identical within each case",
            "at least one new token is produced across timed runs",
            "every timed generation completes with positive measured duration",
        ],
        "ranking_observations": [
            "exactly one parsed call is registered, schema-valid, and names the expected tool",
            "tool-call quality is recorded for ranking and does not gate environment compatibility",
        ],
        "generation_cases": [
            {
                "name": case.name,
                "expected_tool": case.expected_tool,
            }
            for case in probe.generation_cases
        ],
        "retained_output_limit_characters": probe.max_decoded_output_chars,
        "tool_scoring": [
            "strict JSON parse rate",
            "registered-schema-valid output rate",
            "dispatchable-call output rate (schema acceptance only)",
            "zero-tool-call rate",
        ],
        "execution_boundary": (
            "schema acceptance only; no tool handler or execution gate runs"
        ),
        "throughput": "sum(actual new tokens) / sum(measured generation seconds)",
    }
    import_plan: dict[str, JsonValue] = {
        "imports": [
            "trl.GRPOConfig",
            "trl.trainer.sft_trainer.DataCollatorForLanguageModeling",
            "unsloth.FastLanguageModel",
        ],
        "construct": "GRPOConfig only; no trainer and no training run",
        "preflight": "every planned import and configuration construction succeeds",
        "reference_policy_fixture": (
            "planned as a separate pre-compute unit fixture; this import smoke does not "
            "claim reference-policy correctness"
        ),
    }
    training_template_plan: dict[str, JsonValue] = {
        "template_source": "tokenizer.get_chat_template(tools=...) resolved native template",
        "record": [
            "training template SHA-256 separately from native template",
            "training rendered prompt SHA-256",
            "complete expected and returned assistant-token masks",
            "tokenized prefix before and after appending the tool observation",
        ],
        "hard_checks": [
            "assistant mask exactly equals all token spans emitted by template generation blocks",
            "pre-observation tokens are an exact prefix after the tool observation is appended",
        ],
        "note": "imports alone cannot pass this probe",
    }
    return [
        ProbeResult(
            name="tool_chat_template", status="planned", plan=template_plan, metrics={}
        ),
        ProbeResult(name="four_bit_load", status="planned", plan=load_plan, metrics={}),
        ProbeResult(
            name="deterministic_generation",
            status="planned",
            plan=generation_plan,
            metrics={},
        ),
        ProbeResult(
            name="training_stack_imports", status="planned", plan=import_plan, metrics={}
        ),
        ProbeResult(
            name="training_template_masking",
            status="planned",
            plan=training_template_plan,
            metrics={},
        ),
    ]


def build_result(
    config: SmokeConfig,
    *,
    config_path: Path,
    config_bytes: bytes,
    command: list[str],
    run_load: bool,
    allow_download: bool,
    selected_names: list[str],
) -> SmokeResult:
    if run_load != allow_download:
        raise SmokeConfigError(
            "run_load and allow_download must be enabled together for model access"
        )
    lane_result = _collect_lane_result(config.lane)
    release_registry_sha256 = _validate_release_registry(
        config.release_gate, config.candidates
    )
    if run_load and lane_result.actual_lock_sha256 is None:
        raise SmokeConfigError(
            f"measured smoke run requires the lane lock: {lane_result.lock_path}"
        )
    if run_load and not lane_result.lock_matches_expected:
        raise SmokeConfigError(
            "measured smoke run rejected because the lane lock SHA-256 does not "
            "match the configured expected value"
        )
    if run_load:
        _require_clean_git_worktree()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    source_identity = _collect_source_identity(config_sha256)
    selected = _select_candidates(config, selected_names)
    options = RunOptions(
        dry_run=not run_load,
        run_load=run_load,
        allow_download=allow_download,
        selected_candidates=[candidate.name for candidate in selected],
    )
    hardware = collect_hardware_facts()
    library_versions = collect_library_versions()
    candidates: list[CandidateResult] = []
    for candidate in selected:
        if run_load:
            candidate_result = _execute_candidate(candidate, config.probe)
        else:
            candidate_result = _candidate_result(
                name=candidate.name,
                bundle=candidate.bundle,
                role=candidate.role,
                model_id=candidate.model_id,
                requested_revision=candidate.revision,
                resolved_revision=None,
                probes=probe_plans(config.probe),
            )
        candidates.append(candidate_result)

    return SmokeResult(
        schema_version=RESULT_SCHEMA_VERSION,
        created_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        config_path=str(config_path.resolve()),
        config_sha256=config_sha256,
        lane=lane_result,
        release_gate=config.release_gate,
        release_registry_sha256=release_registry_sha256,
        source_identity=source_identity,
        command=command,
        options=options,
        hardware=hardware,
        library_versions=library_versions,
        candidates=candidates,
        selection_eligible=(
            config.release_gate.status == "resolved"
            and _has_selectable_bundle(
                candidates, config.release_gate.eligible_bundles
            )
        ),
    )


def _select_candidates(
    config: SmokeConfig, selected_names: list[str]
) -> list[CandidateConfig]:
    if not selected_names:
        return list(config.candidates)
    if len(selected_names) != len(set(selected_names)):
        raise SmokeConfigError("--candidate values must be unique")
    by_name = {candidate.name: candidate for candidate in config.candidates}
    unknown = sorted(set(selected_names).difference(by_name))
    if unknown:
        raise SmokeConfigError(f"unknown candidate name(s): {', '.join(unknown)}")
    return [by_name[name] for name in selected_names]


def _load_unsloth_four_bit_model(
    *,
    model_id: str,
    revision: str,
    quantization_config: Any,
    compute_dtype: Any,
    target_cuda_device_index: int,
    seed: int,
) -> tuple[Any, Any]:
    unsloth = importlib.import_module("unsloth")
    fast_language_model = getattr(unsloth, "FastLanguageModel")
    loader = getattr(fast_language_model, "from_pretrained")
    loaded = loader(
        model_name=model_id,
        revision=revision,
        max_seq_length=MAX_P6_TOKENS,
        dtype=compute_dtype,
        load_in_4bit=True,
        trust_remote_code=False,
        device_map={"": target_cuda_device_index},
        quantization_config=quantization_config,
        local_files_only=False,
        use_exact_model_name=True,
        fast_inference=False,
        random_state=seed,
        disable_log_stats=True,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise TypeError("Unsloth model loader did not return (model, tokenizer)")
    model, tokenizer = loaded
    if model is None or tokenizer is None:
        raise ValueError("Unsloth model loader returned an empty model or tokenizer")
    if _resolved_revision(tokenizer) != revision:
        raise ValueError(
            "Unsloth model loader tokenizer did not resolve the requested revision"
        )
    if getattr(model, "_saved_temp_tokenizer", None) is not tokenizer:
        raise ValueError(
            "Unsloth model loader did not attach its tokenizer for LoRA preparation"
        )
    return model, tokenizer


def _execute_candidate(candidate: CandidateConfig, probe: ProbeConfig) -> CandidateResult:
    plans = {planned.name: planned.plan for planned in probe_plans(probe)}
    import_result = _run_training_stack_import_probe(
        plans["training_stack_imports"]
    )
    try:
        transformers = importlib.import_module("transformers")
    except (ImportError, RuntimeError) as exc:
        error = f"transformers runtime unavailable: {_error_text(exc)}"
        return _candidate_result(
            name=candidate.name,
            bundle=candidate.bundle,
            role=candidate.role,
            model_id=candidate.model_id,
            requested_revision=candidate.revision,
            resolved_revision=None,
            probes=[
                ProbeResult(
                    name=name,
                    status="unavailable",
                    plan=plans[name],
                    metrics={},
                    error=error,
                )
                for name in (
                    "tool_chat_template",
                    "four_bit_load",
                    "deterministic_generation",
                )
            ]
            + [
                import_result,
                ProbeResult(
                    name="training_template_masking",
                    status="unavailable",
                    plan=plans["training_template_masking"],
                    metrics={},
                    error=error,
                ),
            ],
            p6=_skipped_minimal_training(error),
        )

    tokenizer: Any = None
    native_template: str | dict[str, str] | None = None
    resolved_revision: str | None = None
    tools = _tool_payloads(probe)
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            candidate.model_id,
            revision=candidate.revision,
            trust_remote_code=False,
            local_files_only=False,
        )
        resolved_revision = _resolved_revision(tokenizer)
        get_chat_template = getattr(tokenizer, "get_chat_template", None)
        native_template = (
            get_chat_template(tools=tools)
            if callable(get_chat_template)
            else getattr(tokenizer, "chat_template", None)
        )
        messages = [message.model_dump(mode="json") for message in probe.messages]
        rendered = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            **probe.chat_template_kwargs,
        )
        repeated = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            **probe.chat_template_kwargs,
        )
        if max(len(rendered), len(repeated)) > MAX_PERSISTED_PROMPT_CHARS:
            raise ValueError("native rendered prompt exceeds the character limit")
        token_ids = _flat_int_list(
            tokenizer(rendered, add_special_tokens=False)["input_ids"]
        )
        repeated_token_ids = _flat_int_list(
            tokenizer(repeated, add_special_tokens=False)["input_ids"]
        )
        if max(len(token_ids), len(repeated_token_ids)) > MAX_TEMPLATE_TOKENS:
            raise ValueError("native rendered prompt exceeds the token limit")
        checks = {
            "immutable_revision_resolved": resolved_revision == candidate.revision,
            "native_template_present": bool(native_template),
            "nonempty": bool(rendered),
            "repeat_render_identical": rendered == repeated,
            "repeat_tokenization_identical": token_ids == repeated_token_ids,
            "all_expected_markers_present": all(
                marker in rendered for marker in probe.expected_template_markers
            ),
            "nonzero_token_count": len(token_ids) > 0,
        }
        template_status = "passed" if all(checks.values()) else "failed"
        template_error = None if template_status == "passed" else "template quality check failed"
        template_result = ProbeResult(
            name="tool_chat_template",
            status=template_status,
            plan=plans["tool_chat_template"],
            metrics={
                "checks": checks,
                "tokenizer_class": type(tokenizer).__name__,
                "vocabulary_size": len(tokenizer),
                "special_tokens_map": _json_safe_metadata(
                    getattr(tokenizer, "special_tokens_map", {})
                ),
                "native_chat_template_sha256": _artifact_sha256(native_template),
                "native_rendered_prompt_sha256": _text_sha256(rendered),
                "native_rendered_prompt": rendered,
                "native_rendered_character_count": len(rendered),
                "native_rendered_token_count": len(token_ids),
                "native_prefix_preservation_diagnostic": _prefix_diagnostic(
                    tokenizer=tokenizer,
                    template=native_template,
                    probe=probe,
                    tools=tools,
                ),
            },
            error=template_error,
        )
    except Exception as exc:  # External tokenizer/runtime errors are result data.
        template_result = ProbeResult(
            name="tool_chat_template",
            status="failed",
            plan=plans["tool_chat_template"],
            metrics={},
            error=_error_text(exc),
        )

    if tokenizer is None:
        dependency_error = "tokenizer/template probe did not produce a generation prompt"
        return _candidate_result(
            name=candidate.name,
            bundle=candidate.bundle,
            role=candidate.role,
            model_id=candidate.model_id,
            requested_revision=candidate.revision,
            resolved_revision=resolved_revision,
            probes=[
                template_result,
                ProbeResult(
                    name="four_bit_load",
                    status="skipped",
                    plan=plans["four_bit_load"],
                    metrics={},
                    error=dependency_error,
                ),
                ProbeResult(
                    name="deterministic_generation",
                    status="skipped",
                    plan=plans["deterministic_generation"],
                    metrics={},
                    error=dependency_error,
                ),
                import_result,
                ProbeResult(
                    name="training_template_masking",
                    status="skipped",
                    plan=plans["training_template_masking"],
                    metrics={},
                    error=dependency_error,
                ),
            ],
            p6=_skipped_minimal_training(dependency_error),
        )

    training_template_result = _run_training_template_probe(
        tokenizer=tokenizer,
        native_template=native_template,
        probe=probe,
        plan=plans["training_template_masking"],
    )

    try:
        torch = importlib.import_module("torch")
        importlib.import_module("accelerate")
        importlib.import_module("bitsandbytes")
    except (ImportError, RuntimeError) as exc:
        runtime_error = f"4-bit runtime unavailable: {_error_text(exc)}"
        return _candidate_result(
            name=candidate.name,
            bundle=candidate.bundle,
            role=candidate.role,
            model_id=candidate.model_id,
            requested_revision=candidate.revision,
            resolved_revision=resolved_revision,
            probes=[
                template_result,
                ProbeResult(
                    name="four_bit_load",
                    status="unavailable",
                    plan=plans["four_bit_load"],
                    metrics={},
                    error=runtime_error,
                ),
                ProbeResult(
                    name="deterministic_generation",
                    status="skipped",
                    plan=plans["deterministic_generation"],
                    metrics={},
                    error="4-bit model did not load",
                ),
                import_result,
                training_template_result,
            ],
            p6=_skipped_minimal_training(runtime_error),
        )

    if not torch.cuda.is_available():
        return _candidate_result(
            name=candidate.name,
            bundle=candidate.bundle,
            role=candidate.role,
            model_id=candidate.model_id,
            requested_revision=candidate.revision,
            resolved_revision=resolved_revision,
            probes=[
                template_result,
                ProbeResult(
                    name="four_bit_load",
                    status="unavailable",
                    plan=plans["four_bit_load"],
                    metrics={},
                    error="torch reports that CUDA is unavailable",
                ),
                ProbeResult(
                    name="deterministic_generation",
                    status="skipped",
                    plan=plans["deterministic_generation"],
                    metrics={},
                    error="4-bit CUDA model did not load",
                ),
                import_result,
                training_template_result,
            ],
            p6=_skipped_minimal_training("torch reports that CUDA is unavailable"),
        )

    target_cuda_device_index = probe.target_cuda_device_index
    if target_cuda_device_index >= torch.cuda.device_count():
        error = (
            f"configured CUDA device {target_cuda_device_index} is unavailable; "
            f"torch reports {torch.cuda.device_count()} visible device(s)"
        )
        return _candidate_result(
            name=candidate.name,
            bundle=candidate.bundle,
            role=candidate.role,
            model_id=candidate.model_id,
            requested_revision=candidate.revision,
            resolved_revision=resolved_revision,
            probes=[
                template_result,
                ProbeResult(
                    name="four_bit_load",
                    status="unavailable",
                    plan=plans["four_bit_load"],
                    metrics={},
                    error=error,
                ),
                ProbeResult(
                    name="deterministic_generation",
                    status="skipped",
                    plan=plans["deterministic_generation"],
                    metrics={},
                    error=error,
                ),
                import_result,
                training_template_result,
            ],
            p6=_skipped_minimal_training(error),
        )

    model: Any = None
    load_result: ProbeResult
    try:
        with torch.cuda.device(target_cuda_device_index):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(target_cuda_device_index)
            bf16_supported = torch.cuda.is_bf16_supported()
        compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=probe.quantization,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=probe.double_quant,
        )
        started = time.perf_counter()
        model, tokenizer = _load_unsloth_four_bit_model(
            model_id=candidate.model_id,
            revision=candidate.revision,
            quantization_config=quantization_config,
            compute_dtype=compute_dtype,
            target_cuda_device_index=target_cuda_device_index,
            seed=probe.seed,
        )
        loaded_get_chat_template = getattr(tokenizer, "get_chat_template", None)
        loaded_native_template = (
            loaded_get_chat_template(tools=tools)
            if callable(loaded_get_chat_template)
            else getattr(tokenizer, "chat_template", None)
        )
        training_template_result = _run_training_template_probe(
            tokenizer=tokenizer,
            native_template=loaded_native_template,
            probe=probe,
            plan=plans["training_template_masking"],
        )
        _synchronize_cuda(torch, target_cuda_device_index)
        load_seconds = time.perf_counter() - started
        resolved_revision = _resolved_revision(model) or resolved_revision
        peak_allocated = torch.cuda.max_memory_allocated(target_cuda_device_index)
        peak_reserved = torch.cuda.max_memory_reserved(target_cuda_device_index)
        gpu_name = torch.cuda.get_device_name(target_cuda_device_index)
        gpu_total_memory = int(
            torch.cuda.get_device_properties(target_cuda_device_index).total_memory
        )
        footprint = model.get_memory_footprint() if hasattr(model, "get_memory_footprint") else None
        placement, placement_passed = _inspect_model_placement(
            model, target_cuda_device_index=target_cuda_device_index
        )
        quantization, quantization_passed = _inspect_effective_quantization(
            model,
            expected_quantization=probe.quantization,
            expected_double_quant=probe.double_quant,
            expected_compute_dtype=compute_dtype,
        )
        revision_matches = resolved_revision == candidate.revision
        load_passed = (
            placement_passed and quantization_passed and revision_matches
        )
        load_result = ProbeResult(
            name="four_bit_load",
            status="passed" if load_passed else "failed",
            plan=plans["four_bit_load"],
            metrics={
                "load_seconds": load_seconds,
                "model_loader": "unsloth.FastLanguageModel.from_pretrained",
                "unsloth_training_tokenizer_attached": True,
                "loaded_tokenizer_revision_matches_requested": True,
                "p5_ran_with_loaded_training_tokenizer": True,
                "resolved_revision_matches_requested": revision_matches,
                "model_memory_footprint_bytes": footprint,
                "target_cuda_device_index": target_cuda_device_index,
                "target_gpu_name": gpu_name,
                "target_gpu_total_memory_bytes": gpu_total_memory,
                "torch_cuda_runtime_version": getattr(
                    getattr(torch, "version", None), "cuda", None
                ),
                **placement,
                **quantization,
                "peak_cuda_allocated_bytes": peak_allocated,
                "peak_cuda_reserved_bytes": peak_reserved,
                "load_in_4bit": True,
                "quantization_type": probe.quantization,
                "double_quant": probe.double_quant,
                "compute_dtype": str(compute_dtype),
            },
            error=(
                None
                if load_passed
                else (
                    "4-bit load failed revision, single-device, or effective-NF4 hard gate"
                )
            ),
        )
    except Exception as exc:  # External model/runtime errors are result data.
        load_result = ProbeResult(
            name="four_bit_load",
            status="failed",
            plan=plans["four_bit_load"],
            metrics={},
            error=_error_text(exc),
        )

    if model is None or load_result.status != "passed":
        generation_result = ProbeResult(
            name="deterministic_generation",
            status="skipped",
            plan=plans["deterministic_generation"],
            metrics={},
            error="4-bit model did not pass the no-offload hard gate",
        )
    else:
        generation_result = _run_generation_probe(
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            probe=probe,
            plan=plans["deterministic_generation"],
        )

    candidate_probes = [
        template_result,
        load_result,
        generation_result,
        import_result,
        training_template_result,
    ]
    p6_prerequisite_error = _p6_prerequisite_error(candidate_probes)
    if p6_prerequisite_error is not None or model is None:
        p6_result = _skipped_minimal_training(
            p6_prerequisite_error or "P6 has no loaded model"
        )
    else:
        p6_result = _run_minimal_training_probe(
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            probe=probe,
            training_template_result=training_template_result,
        )

    return _candidate_result(
        name=candidate.name,
        bundle=candidate.bundle,
        role=candidate.role,
        model_id=candidate.model_id,
        requested_revision=candidate.revision,
        resolved_revision=resolved_revision,
        probes=candidate_probes,
        p6=p6_result,
    )


def _run_generation_probe(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    probe: ProbeConfig,
    plan: dict[str, JsonValue],
) -> ProbeResult:
    retained_outputs: list[dict[str, JsonValue]] = []
    prompt_token_counts: dict[str, JsonValue] = {}
    elapsed_seconds: list[float] = []
    token_counts: list[int] = []
    try:
        parser_module = importlib.import_module("agent.parser")
        model.eval()
        target_device = torch.device(f"cuda:{probe.target_cuda_device_index}")
        tools = _tool_payloads(probe)
        deterministic_by_case: dict[str, bool] = {}
        expected_dispatchable_by_case: dict[str, bool] = {}
        strict_tool_output_by_case: dict[str, bool] = {}

        for case in probe.generation_cases:
            messages = [message.model_dump(mode="json") for message in case.messages]
            rendered = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
                **probe.chat_template_kwargs,
            )
            if len(rendered) > MAX_PERSISTED_PROMPT_CHARS:
                raise ValueError(
                    f"generation prompt '{case.name}' exceeds the character limit"
                )
            encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
            encoded = {key: value.to(target_device) for key, value in encoded.items()}
            prompt_tokens = int(encoded["input_ids"].shape[-1])
            if prompt_tokens > MAX_TEMPLATE_TOKENS:
                raise ValueError(
                    f"generation prompt '{case.name}' exceeds the token limit"
                )
            prompt_token_counts[case.name] = prompt_tokens

            def generate_once() -> tuple[list[int], float]:
                torch.manual_seed(probe.seed)
                torch.cuda.manual_seed_all(probe.seed)
                _synchronize_cuda(torch, probe.target_cuda_device_index)
                started = time.perf_counter()
                with torch.inference_mode():
                    output = model.generate(
                        **encoded,
                        do_sample=False,
                        max_new_tokens=probe.max_new_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                _synchronize_cuda(torch, probe.target_cuda_device_index)
                elapsed = time.perf_counter() - started
                new_ids = output[0, prompt_tokens:].detach().cpu().tolist()
                return new_ids, elapsed

            for _ in range(probe.warmup_runs):
                generate_once()

            case_outputs: list[list[int]] = []
            case_dispatchable: list[bool] = []
            case_strict_outputs: list[bool] = []
            for run_index in range(probe.timed_runs):
                output_ids, elapsed = generate_once()
                decoded = tokenizer.decode(output_ids, skip_special_tokens=False)
                if len(decoded) > probe.max_decoded_output_chars:
                    raise ValueError(
                        f"decoded output for '{case.name}' exceeds the character limit"
                    )
                scored = _score_tool_output(
                    decoded,
                    expected_tool=case.expected_tool,
                    probe=probe,
                    parser_module=parser_module,
                )
                bounded = _bounded_text(decoded, probe.max_decoded_output_chars)
                retained_outputs.append(
                    {
                        "case": case.name,
                        "run_index": run_index,
                        "expected_tool": case.expected_tool,
                        "decoded_output": bounded["text"],
                        "decoded_output_truncated": bounded["truncated"],
                        "decoded_output_character_count": bounded["character_count"],
                        "decoded_output_sha256": _text_sha256(decoded),
                        "generated_token_count": len(output_ids),
                        "elapsed_seconds": elapsed,
                        "tool_score": scored,
                    }
                )
                case_outputs.append(output_ids)
                case_dispatchable.append(
                    bool(scored["exactly_one_expected_dispatchable_call"])
                )
                case_strict_outputs.append(
                    bool(scored["exactly_one_expected_dispatchable_call"])
                )
                elapsed_seconds.append(elapsed)
                token_counts.append(len(output_ids))
            deterministic_by_case[case.name] = all(
                output_ids == case_outputs[0] for output_ids in case_outputs[1:]
            )
            expected_dispatchable_by_case[case.name] = all(case_dispatchable)
            strict_tool_output_by_case[case.name] = all(case_strict_outputs)

        token_total = sum(token_counts)
        seconds_total = sum(elapsed_seconds)
        score_rows = [output["tool_score"] for output in retained_outputs]
        output_count = len(score_rows)

        def rate(key: str) -> float | None:
            if output_count == 0:
                return None
            return sum(bool(row[key]) for row in score_rows) / output_count  # type: ignore[index]

        compatibility_checks = {
            "timed_outputs_identical_within_each_case": all(
                deterministic_by_case.values()
            ),
            "nonzero_generated_tokens": token_total > 0,
            "positive_elapsed_time": seconds_total > 0,
        }
        quality_observations = {
            "every_output_is_strict_and_schema_valid": all(
                strict_tool_output_by_case.values()
            ),
            "every_output_has_exactly_one_expected_dispatchable_call": all(
                expected_dispatchable_by_case.values()
            ),
        }
        compatible = all(compatibility_checks.values())
        throughput = token_total / seconds_total if seconds_total > 0 else None
        ranking_metrics = {
            "strict_json_parse_rate": rate("strict_json_parse_success"),
            "registered_schema_valid_output_rate": rate(
                "registered_schema_valid_output"
            ),
            "dispatchable_call_output_rate": rate("has_dispatchable_call"),
            "zero_tool_call_rate": rate("zero_tool_call"),
        }
        return ProbeResult(
            name="deterministic_generation",
            status="passed" if compatible else "failed",
            plan=plan,
            metrics={
                "compatibility_checks": compatibility_checks,
                "quality_observations": quality_observations,
                "tool_call_quality_gates_environment_compatibility": False,
                "prompt_token_counts_by_case": prompt_token_counts,
                "deterministic_by_case": deterministic_by_case,
                "strict_tool_output_by_case": strict_tool_output_by_case,
                "expected_tool_dispatchable_by_case": expected_dispatchable_by_case,
                "generated_token_counts": token_counts,
                "elapsed_seconds_by_run": elapsed_seconds,
                "generated_tokens_total": token_total,
                "elapsed_seconds_total": seconds_total,
                "tokens_per_second": throughput,
                "decoding_options": {
                    "do_sample": False,
                    "temperature": probe.temperature,
                    "top_p": probe.top_p,
                    "seed": probe.seed,
                    "max_new_tokens": probe.max_new_tokens,
                },
                **ranking_metrics,
                "ranking_metrics": ranking_metrics,
                "retained_outputs": retained_outputs,
            },
            error=None if compatible else "generation compatibility check failed",
        )
    except Exception as exc:  # External model/runtime errors are result data.
        return ProbeResult(
            name="deterministic_generation",
            status="failed",
            plan=plan,
            metrics={
                "partial_prompt_token_counts_by_case": prompt_token_counts,
                "partial_generated_token_counts": token_counts,
                "partial_elapsed_seconds_by_run": elapsed_seconds,
                "partial_retained_outputs": retained_outputs,
            },
            error=_error_text(exc),
        )


def _run_training_stack_import_probe(plan: dict[str, JsonValue]) -> ProbeResult:
    imported: dict[str, JsonValue] = {}
    try:
        # Unsloth requires import before Transformers/TRL so its supported
        # patches are installed before those modules initialize.
        unsloth = importlib.import_module("unsloth")
        imported["unsloth"] = True
        trl = importlib.import_module("trl")
        sft_module = importlib.import_module("trl.trainer.sft_trainer")
        imported["trl"] = True
        grpo_config_class = getattr(trl, "GRPOConfig")
        getattr(sft_module, "DataCollatorForLanguageModeling")
        fast_language_model = getattr(unsloth, "FastLanguageModel")
        getattr(fast_language_model, "from_pretrained")
        getattr(fast_language_model, "get_peft_model")
        getattr(fast_language_model, "for_training")
        imported.update({
            "trl.GRPOConfig": True,
            "trl.trainer.sft_trainer.DataCollatorForLanguageModeling": True,
            "unsloth.FastLanguageModel": True,
            "unsloth.FastLanguageModel.from_pretrained": True,
            "unsloth.FastLanguageModel.get_peft_model": True,
            "unsloth.FastLanguageModel.for_training": True,
        })
        output_dir = str(Path(tempfile.gettempdir()) / "qwen-smoke-grpo-config")
        config = grpo_config_class(
            output_dir=output_dir,
            max_steps=1,
            report_to="none",
        )
        return ProbeResult(
            name="training_stack_imports",
            status="passed",
            plan=plan,
            metrics={
                "imports": imported,
                "constructed_config_class": (
                    f"{type(config).__module__}.{type(config).__qualname__}"
                ),
                "training_started": False,
            },
        )
    except ImportError as exc:
        return ProbeResult(
            name="training_stack_imports",
            status="unavailable",
            plan=plan,
            metrics={"imports_completed_before_error": imported},
            error=_error_text(exc),
        )
    except Exception as exc:
        return ProbeResult(
            name="training_stack_imports",
            status="failed",
            plan=plan,
            metrics={"imports_completed_before_error": imported},
            error=_error_text(exc),
        )


def _run_training_template_probe(
    *,
    tokenizer: Any,
    native_template: str | dict[str, str] | None,
    probe: ProbeConfig,
    plan: dict[str, JsonValue],
) -> ProbeResult:
    try:
        training_template = native_template
        if not isinstance(training_template, str) or not training_template:
            raise ValueError("tokenizer did not resolve one usable string training template")
        if len(training_template) > MAX_TEMPLATE_SOURCE_CHARS:
            raise ValueError("training template source exceeds the character limit")

        trajectory = [
            message.model_dump(mode="json", exclude_none=True)
            for message in probe.training_trajectory
        ]
        prefix_messages = trajectory[:-1]
        tools = _tool_payloads(probe)
        prefix_ids = _flat_int_list(
            tokenizer.apply_chat_template(
                prefix_messages,
                tools=tools,
                chat_template=training_template,
                add_generation_prompt=False,
                tokenize=True,
                **probe.chat_template_kwargs,
            )
        )
        if len(prefix_ids) > MAX_TEMPLATE_TOKENS:
            raise ValueError("training prefix exceeds the token limit")
        rendered = tokenizer.apply_chat_template(
            trajectory,
            tools=tools,
            chat_template=training_template,
            add_generation_prompt=False,
            tokenize=False,
            **probe.chat_template_kwargs,
        )
        if len(rendered) > MAX_PERSISTED_PROMPT_CHARS:
            raise ValueError("training rendered prompt exceeds the character limit")
        masked = tokenizer.apply_chat_template(
            trajectory,
            tools=tools,
            chat_template=training_template,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            **probe.chat_template_kwargs,
        )
        full_ids = _flat_int_list(masked["input_ids"])
        raw_mask = masked.get("assistant_masks", masked.get("assistant_tokens_mask"))
        if raw_mask is None:
            raise KeyError("assistant mask key is absent from tokenizer output")
        assistant_mask = _flat_int_list(raw_mask)
        if len(full_ids) > MAX_TEMPLATE_TOKENS:
            raise ValueError("training rendered prompt exceeds the token limit")
        if len(assistant_mask) > MAX_TEMPLATE_TOKENS:
            raise ValueError("assistant mask exceeds the token limit")
        tokenized_render = tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        render_ids = _flat_int_list(tokenized_render["input_ids"])
        if len(render_ids) > MAX_TEMPLATE_TOKENS:
            raise ValueError("independent training tokenization exceeds the token limit")
        offsets = [
            tuple(int(value) for value in pair)
            for pair in tokenized_render["offset_mapping"]
        ]
        if len(offsets) > MAX_TEMPLATE_TOKENS:
            raise ValueError("training offset mapping exceeds the token limit")
        expected_mask, generation_spans = _expected_assistant_mask(
            tokenizer=tokenizer,
            training_template=training_template,
            trajectory=trajectory,
            tools=tools,
            template_kwargs=probe.chat_template_kwargs,
            rendered=rendered,
            offsets=offsets,
        )
        checks = {
            "assistant_mask_length_matches_tokens": len(assistant_mask) == len(full_ids),
            "assistant_mask_is_binary": all(
                value in {0, 1} for value in assistant_mask
            ),
            "assistant_mask_nonempty": any(value == 1 for value in assistant_mask),
            "assistant_mask_has_unmasked_tokens": any(value == 0 for value in assistant_mask),
            "expected_mask_length_matches_tokens": len(expected_mask) == len(full_ids),
            "assistant_mask_exactly_matches_generation_spans": (
                assistant_mask == expected_mask
            ),
            "render_tokenization_matches_template": render_ids == full_ids,
            "prefix_nonempty": bool(prefix_ids),
            "prefix_preserved_after_tool_observation": (
                full_ids[: len(prefix_ids)] == prefix_ids
            ),
        }
        passed = all(checks.values())
        return ProbeResult(
            name="training_template_masking",
            status="passed" if passed else "failed",
            plan=plan,
            metrics={
                "checks": checks,
                "training_template_source": "resolved_native_chat_template",
                "native_chat_template_sha256": _artifact_sha256(native_template),
                "training_chat_template_sha256": _artifact_sha256(training_template),
                "training_rendered_prompt_sha256": _text_sha256(rendered),
                "training_rendered_prompt": rendered,
                "training_rendered_token_count": len(full_ids),
                "training_input_ids": full_ids,
                "assistant_token_mask": assistant_mask,
                "expected_assistant_token_mask": expected_mask,
                "assistant_mask_one_count": sum(value == 1 for value in assistant_mask),
                "assistant_mask_zero_count": sum(value == 0 for value in assistant_mask),
                "generation_character_spans": [list(span) for span in generation_spans],
                "pre_observation_prefix_token_count": len(prefix_ids),
            },
            error=None if passed else "training template mask or prefix hard gate failed",
        )
    except ImportError as exc:
        return ProbeResult(
            name="training_template_masking",
            status="unavailable",
            plan=plan,
            metrics={},
            error=_error_text(exc),
        )
    except Exception as exc:
        return ProbeResult(
            name="training_template_masking",
            status="failed",
            plan=plan,
            metrics={},
            error=_error_text(exc),
        )


def _p6_prerequisite_error(probes: list[ProbeResult]) -> str | None:
    statuses = {probe.name: probe.status for probe in probes}
    required = {
        "four_bit_load": "P3 four-bit load",
        "training_stack_imports": "training-stack imports",
        "training_template_masking": "P5 training mask",
    }
    failures = [
        f"{label}={statuses.get(name, 'missing')}"
        for name, label in required.items()
        if statuses.get(name) != "passed"
    ]
    if failures:
        return "P6 prerequisites did not pass: " + ", ".join(failures)
    return None


def _skipped_minimal_training(reason: str) -> MinimalTrainingResult:
    return MinimalTrainingResult(
        status="skipped",
        executed=False,
        passed=False,
        plan=_minimal_training_plan(),
        metrics={},
        error=_error_text(ValueError(reason)),
    )


def _assistant_only_labels(
    input_ids: list[int], assistant_mask: list[int]
) -> list[int]:
    if not input_ids or len(input_ids) > MAX_P6_TOKENS:
        raise ValueError(
            f"P6 requires 1-{MAX_P6_TOKENS} input tokens; received {len(input_ids)}"
        )
    if len(input_ids) != len(assistant_mask):
        raise ValueError("P6 input IDs and assistant mask lengths differ")
    if any(token_id < 0 for token_id in input_ids):
        raise ValueError("P6 input IDs must be non-negative")
    if any(value not in {0, 1} for value in assistant_mask):
        raise ValueError("P6 assistant mask must be binary")
    if not any(assistant_mask) or all(assistant_mask):
        raise ValueError("P6 assistant mask must contain supervised and ignored tokens")
    labels = [
        token_id if mask_value == 1 else -100
        for token_id, mask_value in zip(input_ids, assistant_mask)
    ]
    if not any(label != -100 for label in labels[1:]):
        raise ValueError("P6 has no causally shifted supervised token")
    return labels


def _p5_training_batch(
    training_template_result: ProbeResult,
) -> tuple[list[int], list[int], list[int]]:
    if training_template_result.status != "passed":
        raise ValueError("P6 requires a passed P5 training-template result")
    raw_input_ids = training_template_result.metrics.get("training_input_ids")
    raw_assistant_mask = training_template_result.metrics.get(
        "assistant_token_mask"
    )
    if raw_input_ids is None or raw_assistant_mask is None:
        raise ValueError("P5 result is missing the exact training IDs or assistant mask")
    input_ids = _flat_int_list(raw_input_ids)
    assistant_mask = _flat_int_list(raw_assistant_mask)
    labels = _assistant_only_labels(input_ids, assistant_mask)
    return input_ids, assistant_mask, labels


def _adapter_enabled_state(model: Any) -> bool:
    get_status = getattr(model, "get_model_status", None)
    if not callable(get_status):
        raise TypeError("PEFT model does not expose get_model_status")
    enabled = getattr(get_status(), "enabled", None)
    if type(enabled) is not bool:
        raise ValueError("PEFT adapter enabled state is not uniformly boolean")
    return enabled


def _run_with_adapters_disabled(
    model: Any, operation: Any
) -> tuple[Any, dict[str, bool]]:
    disable_adapter = getattr(model, "disable_adapter", None)
    if not callable(disable_adapter):
        raise TypeError("PEFT model does not expose disable_adapter")
    enabled_before = _adapter_enabled_state(model)
    with disable_adapter():
        disabled_inside = not _adapter_enabled_state(model)
        value = operation()
    enabled_after = _adapter_enabled_state(model)
    return value, {
        "adapter_enabled_before_reference": enabled_before,
        "adapter_disabled_inside_reference": disabled_inside,
        "adapter_restored_after_reference": enabled_after,
    }


def _selected_token_logps(
    *,
    torch: Any,
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    labels: Any,
) -> Any:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise ValueError("model did not return logits for the reference-policy probe")
    shifted_labels = labels[:, 1:]
    supervised = shifted_labels.ne(-100)
    supervised_count = int(supervised.sum().item())
    if supervised_count <= 0:
        raise ValueError("reference-policy probe has no supervised shifted token")
    selected_logits = logits[:, :-1, :][supervised]
    selected_targets = shifted_labels[supervised]
    chosen_logits = selected_logits.gather(
        -1, selected_targets.unsqueeze(-1)
    ).squeeze(-1)
    logps = (
        chosen_logits.float()
        - torch.logsumexp(selected_logits.float(), dim=-1)
    ).detach().cpu()
    del outputs, logits, selected_logits, chosen_logits
    return logps


def _gradient_statistics(
    torch: Any, named_parameters: list[tuple[str, Any]]
) -> dict[str, JsonValue]:
    gradient_parameter_count = 0
    gradient_none_count = 0
    nonzero_elements = 0
    element_count = 0
    maximum = 0.0
    norm_squared = 0.0
    all_finite = True
    for _, parameter in named_parameters:
        gradient = getattr(parameter, "grad", None)
        if gradient is None:
            gradient_none_count += 1
            continue
        gradient_parameter_count += 1
        element_count += int(gradient.numel())
        finite = bool(torch.isfinite(gradient).all().item())
        all_finite = all_finite and finite
        if not finite:
            continue
        nonzero_elements += int(torch.count_nonzero(gradient).item())
        maximum = max(maximum, float(gradient.detach().abs().max().item()))
        parameter_norm = float(gradient.detach().float().norm().item())
        norm_squared += parameter_norm * parameter_norm
    return {
        "gradient_parameter_count": gradient_parameter_count,
        "gradient_none_count": gradient_none_count,
        "gradient_element_count": element_count,
        "gradient_nonzero_element_count": nonzero_elements,
        "gradient_all_finite": all_finite,
        "gradient_max_abs": maximum if all_finite else None,
        "gradient_l2_norm": math.sqrt(norm_squared) if all_finite else None,
    }


def _parameter_update_statistics(
    torch: Any,
    named_parameters: list[tuple[str, Any]],
    before: dict[str, Any],
) -> dict[str, JsonValue]:
    changed_parameters = 0
    nonzero_elements = 0
    maximum = 0.0
    all_finite = True
    for name, parameter in named_parameters:
        delta = parameter.detach().float().cpu() - before[name]
        finite = bool(torch.isfinite(delta).all().item())
        all_finite = all_finite and finite
        if not finite:
            continue
        changed = int(torch.count_nonzero(delta).item())
        nonzero_elements += changed
        changed_parameters += int(changed > 0)
        maximum = max(maximum, float(delta.abs().max().item()))
    return {
        "updated_parameter_count": changed_parameters,
        "updated_element_count": nonzero_elements,
        "parameter_updates_all_finite": all_finite,
        "parameter_update_max_abs": maximum if all_finite else None,
    }


def _run_minimal_training_probe(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    probe: ProbeConfig,
    training_template_result: ProbeResult,
) -> MinimalTrainingResult:
    plan = _minimal_training_plan()
    metrics: dict[str, JsonValue] = {
        "training_started": False,
        "quality_claim": False,
        "optimizer_checkpoint_written": False,
        "separate_reference_model_loaded": False,
        "reference_policy_mechanism": "peft_disable_adapter_same_model",
    }
    checks: dict[str, bool] = {}
    optimizer: Any = None
    outputs: Any = None
    adapted_model: Any = model
    target_device_index = probe.target_cuda_device_index
    started = time.perf_counter()
    try:
        input_ids, assistant_mask, expected_labels = _p5_training_batch(
            training_template_result
        )
        effective_supervised_tokens = sum(
            label != -100 for label in expected_labels[1:]
        )
        metrics.update(
            {
                "batch_size": 1,
                "sequence_token_count": len(input_ids),
                "assistant_mask_one_count": sum(assistant_mask),
                "assistant_mask_zero_count": len(assistant_mask)
                - sum(assistant_mask),
                "effective_causal_supervised_token_count": effective_supervised_tokens,
                "input_ids_sha256": _artifact_sha256(input_ids),
                "assistant_mask_sha256": _artifact_sha256(assistant_mask),
                "expected_labels_sha256": _artifact_sha256(expected_labels),
            }
        )
        checks.update(
            {
                "batch_size_is_one": True,
                "sequence_within_token_cap": len(input_ids) <= MAX_P6_TOKENS,
                "p5_mask_reused_exactly": True,
                "effective_supervision_nonempty": effective_supervised_tokens > 0,
            }
        )

        unsloth = importlib.import_module("unsloth")
        sft_module = importlib.import_module("trl.trainer.sft_trainer")
        fast_language_model = getattr(unsloth, "FastLanguageModel")
        collator_class = getattr(sft_module, "DataCollatorForLanguageModeling")

        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if type(pad_token_id) is not int or pad_token_id < 0:
            raise ValueError("tokenizer has no usable pad or EOS token ID")
        collator = collator_class(
            pad_token_id=pad_token_id,
            completion_only_loss=False,
            padding_free=False,
        )
        batch = collator(
            [{"input_ids": input_ids, "assistant_masks": assistant_mask}]
        )
        collated_input_ids = _flat_int_list(batch["input_ids"])
        collated_attention_mask = _flat_int_list(batch["attention_mask"])
        collated_labels = _flat_int_list(batch["labels"])
        checks["trl_input_ids_match_p5"] = collated_input_ids == input_ids
        checks["trl_attention_mask_is_exact"] = collated_attention_mask == [
            1
        ] * len(input_ids)
        checks["trl_labels_match_assistant_mask"] = (
            collated_labels == expected_labels
        )
        if not all(
            checks[name]
            for name in (
                "trl_input_ids_match_p5",
                "trl_attention_mask_is_exact",
                "trl_labels_match_assistant_mask",
            )
        ):
            raise ValueError("TRL collator changed P5 IDs, attention, or assistant labels")
        metrics["trl_labels_sha256"] = _artifact_sha256(collated_labels)
        metrics["trl_collator_class"] = (
            f"{type(collator).__module__}.{type(collator).__qualname__}"
        )

        with torch.cuda.device(target_device_index):
            allocated_before = int(torch.cuda.memory_allocated(target_device_index))
            reserved_before = int(torch.cuda.memory_reserved(target_device_index))
            torch.cuda.reset_peak_memory_stats(target_device_index)
        metrics["cuda_allocated_before_p6_bytes"] = allocated_before
        metrics["cuda_reserved_before_p6_bytes"] = reserved_before

        torch.manual_seed(probe.seed)
        torch.cuda.manual_seed_all(probe.seed)
        adapted_model = fast_language_model.get_peft_model(
            model,
            r=P6_LORA_RANK,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=P6_LORA_RANK,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=probe.seed,
            max_seq_length=MAX_P6_TOKENS,
            use_rslora=False,
            modules_to_save=None,
            init_lora_weights=True,
            loftq_config=None,
        )
        prepared_model = fast_language_model.for_training(
            adapted_model, use_gradient_checkpointing=True
        )
        if prepared_model is not None:
            adapted_model = prepared_model
        if hasattr(adapted_model, "config"):
            adapted_model.config.use_cache = False

        named_trainable_parameters = [
            (name, parameter)
            for name, parameter in adapted_model.named_parameters()
            if parameter.requires_grad
        ]
        unexpected_trainable_names = [
            name
            for name, _ in named_trainable_parameters
            if "lora_" not in name.lower()
        ]
        trainable_parameter_count = sum(
            int(parameter.numel()) for _, parameter in named_trainable_parameters
        )
        trainable_parameter_bytes = sum(
            int(parameter.numel()) * int(parameter.element_size())
            for _, parameter in named_trainable_parameters
        )
        metrics.update(
            {
                "adapter_rank": P6_LORA_RANK,
                "adapter_alpha": P6_LORA_RANK,
                "adapter_target_modules": ["q_proj", "v_proj"],
                "trainable_tensor_count": len(named_trainable_parameters),
                "trainable_parameter_count": trainable_parameter_count,
                "trainable_parameter_bytes": trainable_parameter_bytes,
                "trainable_parameter_names_sha256": _artifact_sha256(
                    [name for name, _ in named_trainable_parameters]
                ),
                "unexpected_trainable_names": unexpected_trainable_names[:32],
            }
        )
        checks["trainable_adapter_parameters_present"] = bool(
            named_trainable_parameters
        )
        checks["only_lora_parameters_trainable"] = not unexpected_trainable_names
        if not all(
            checks[name]
            for name in (
                "trainable_adapter_parameters_present",
                "only_lora_parameters_trainable",
            )
        ):
            raise ValueError("P6 found no LoRA parameters or unexpected trainable base parameters")

        device = torch.device(f"cuda:{target_device_index}")
        cuda_input_ids = batch["input_ids"].to(device)
        cuda_attention_mask = batch["attention_mask"].to(device)
        cuda_labels = batch["labels"].to(device)
        metrics["training_started"] = True

        adapted_model.eval()
        with torch.inference_mode():
            reference_before, reference_checks_before = (
                _run_with_adapters_disabled(
                    adapted_model,
                    lambda: _selected_token_logps(
                        torch=torch,
                        model=adapted_model,
                        input_ids=cuda_input_ids,
                        attention_mask=cuda_attention_mask,
                        labels=cuda_labels,
                    ),
                )
            )
        checks.update(
            {
                f"reference_before_{name}": value
                for name, value in reference_checks_before.items()
            }
        )
        checks["reference_before_finite"] = bool(
            torch.isfinite(reference_before).all().item()
        )
        if not all(reference_checks_before.values()) or not checks[
            "reference_before_finite"
        ]:
            raise ValueError("reference adapter-disable pre-step check failed")

        adapted_model.train()
        adapted_model.zero_grad(set_to_none=True)
        optimizer = torch.optim.SGD(
            [parameter for _, parameter in named_trainable_parameters],
            lr=P6_LEARNING_RATE,
            momentum=0.0,
        )
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        trainable_parameter_ids = {
            id(parameter) for _, parameter in named_trainable_parameters
        }
        checks["optimizer_contains_exact_trainable_set"] = (
            optimizer_parameter_ids == trainable_parameter_ids
        )
        metrics["optimizer_class"] = (
            f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
        )
        metrics["optimizer_learning_rate"] = P6_LEARNING_RATE
        metrics["optimizer_momentum"] = 0.0
        metrics["optimizer_state_entries_before_step"] = len(optimizer.state)
        if not checks["optimizer_contains_exact_trainable_set"]:
            raise ValueError("P6 optimizer parameter set does not match LoRA trainables")

        forward_started = time.perf_counter()
        outputs = adapted_model(
            input_ids=cuda_input_ids,
            attention_mask=cuda_attention_mask,
            labels=cuda_labels,
            use_cache=False,
        )
        _synchronize_cuda(torch, target_device_index)
        metrics["forward_seconds"] = time.perf_counter() - forward_started
        loss = getattr(outputs, "loss", None)
        if loss is None:
            raise ValueError("model did not return a loss for the assistant-only batch")
        loss_value = float(loss.detach().float().cpu().item())
        checks["loss_is_finite"] = math.isfinite(loss_value)
        checks["loss_requires_grad"] = bool(getattr(loss, "requires_grad", False))
        if not checks["loss_is_finite"] or not checks["loss_requires_grad"]:
            raise ValueError("P6 loss is nonfinite or detached")
        metrics["assistant_only_loss"] = loss_value

        backward_started = time.perf_counter()
        loss.backward()
        _synchronize_cuda(torch, target_device_index)
        metrics["backward_seconds"] = time.perf_counter() - backward_started
        gradient_stats = _gradient_statistics(
            torch, named_trainable_parameters
        )
        metrics.update(gradient_stats)
        checks["all_trainables_have_gradients"] = (
            gradient_stats["gradient_none_count"] == 0
        )
        checks["gradients_are_finite"] = bool(
            gradient_stats["gradient_all_finite"]
        )
        checks["nonzero_adapter_gradient"] = (
            int(gradient_stats["gradient_nonzero_element_count"]) > 0
        )
        if not all(
            checks[name]
            for name in (
                "all_trainables_have_gradients",
                "gradients_are_finite",
                "nonzero_adapter_gradient",
            )
        ):
            raise ValueError("P6 adapter gradients are absent, zero, or nonfinite")

        parameter_before_step = {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in named_trainable_parameters
        }
        optimizer_started = time.perf_counter()
        optimizer.step()
        _synchronize_cuda(torch, target_device_index)
        metrics["optimizer_step_seconds"] = time.perf_counter() - optimizer_started
        metrics["optimizer_state_entries_after_step"] = len(optimizer.state)
        update_stats = _parameter_update_statistics(
            torch, named_trainable_parameters, parameter_before_step
        )
        metrics.update(update_stats)
        checks["optimizer_state_remains_empty"] = len(optimizer.state) == 0
        checks["parameter_updates_are_finite"] = bool(
            update_stats["parameter_updates_all_finite"]
        )
        checks["nonzero_adapter_parameter_update"] = (
            int(update_stats["updated_parameter_count"]) > 0
        )
        if not all(
            checks[name]
            for name in (
                "optimizer_state_remains_empty",
                "parameter_updates_are_finite",
                "nonzero_adapter_parameter_update",
            )
        ):
            raise ValueError("P6 adapter update or zero-state optimizer check failed")

        optimizer.zero_grad(set_to_none=True)
        adapted_model.zero_grad(set_to_none=True)
        del loss, outputs, parameter_before_step
        outputs = None
        torch.cuda.empty_cache()
        checks["training_graph_released_before_post_step_reference"] = True

        adapted_model.eval()
        with torch.inference_mode():
            reference_after, reference_checks_after = _run_with_adapters_disabled(
                adapted_model,
                lambda: _selected_token_logps(
                    torch=torch,
                    model=adapted_model,
                    input_ids=cuda_input_ids,
                    attention_mask=cuda_attention_mask,
                    labels=cuda_labels,
                ),
            )
            policy_after = _selected_token_logps(
                torch=torch,
                model=adapted_model,
                input_ids=cuda_input_ids,
                attention_mask=cuda_attention_mask,
                labels=cuda_labels,
            )
        checks.update(
            {
                f"reference_after_{name}": value
                for name, value in reference_checks_after.items()
            }
        )
        checks["reference_after_finite"] = bool(
            torch.isfinite(reference_after).all().item()
        )
        checks["policy_after_finite"] = bool(
            torch.isfinite(policy_after).all().item()
        )
        checks["reference_token_count_matches_batch"] = (
            int(reference_after.numel()) == effective_supervised_tokens
            and int(policy_after.numel()) == effective_supervised_tokens
        )
        reference_delta = (reference_after - reference_before).abs()
        policy_reference_delta = (policy_after - reference_after).abs()
        reference_max_delta = float(reference_delta.max().item())
        policy_reference_max_delta = float(policy_reference_delta.max().item())
        checks["reference_invariant_across_step"] = bool(
            torch.allclose(
                reference_before,
                reference_after,
                atol=P6_REFERENCE_ATOL,
                rtol=P6_REFERENCE_RTOL,
            )
        )
        checks["no_checkpoint_written"] = True
        if not all(reference_checks_after.values()) or not all(
            checks[name]
            for name in (
                "reference_after_finite",
                "policy_after_finite",
                "reference_token_count_matches_batch",
                "reference_invariant_across_step",
                "no_checkpoint_written",
            )
        ):
            raise ValueError("P6 reference-policy post-step check failed")
        metrics.update(
            {
                "reference_logprob_count": int(reference_after.numel()),
                "reference_before_sha256": _artifact_sha256(
                    reference_before.tolist()
                ),
                "reference_after_sha256": _artifact_sha256(
                    reference_after.tolist()
                ),
                "reference_pre_post_max_abs_delta": reference_max_delta,
                "reference_invariance_atol": P6_REFERENCE_ATOL,
                "reference_invariance_rtol": P6_REFERENCE_RTOL,
                "policy_reference_max_abs_logprob_delta_after": (
                    policy_reference_max_delta
                ),
            }
        )
        _synchronize_cuda(torch, target_device_index)
        metrics["peak_cuda_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(target_device_index)
        )
        metrics["peak_cuda_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(target_device_index)
        )
        metrics["total_seconds"] = time.perf_counter() - started
        metrics["checks"] = checks
        return MinimalTrainingResult(
            status="passed",
            executed=True,
            passed=True,
            plan=plan,
            metrics=metrics,
            error=None,
        )
    except Exception as exc:  # External CUDA/training errors are result data.
        metrics["checks"] = checks
        metrics["total_seconds"] = time.perf_counter() - started
        try:
            metrics["peak_cuda_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(target_device_index)
            )
            metrics["peak_cuda_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(target_device_index)
            )
        except Exception:
            pass
        return MinimalTrainingResult(
            status="failed",
            executed=True,
            passed=False,
            plan=plan,
            metrics=metrics,
            error=_error_text(exc),
        )
    finally:
        try:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            adapted_model.zero_grad(set_to_none=True)
            del outputs, optimizer
            torch.cuda.empty_cache()
        except Exception:
            pass


def _instrument_generation_blocks(template: str) -> str:
    if len(template) > MAX_TEMPLATE_SOURCE_CHARS:
        raise ValueError("training template source exceeds the character limit")
    if GENERATION_START_MARKER in template or GENERATION_END_MARKER in template:
        raise ValueError("training template already contains reserved mask markers")
    start_pattern = re.compile(r"{%-?\s*generation\s*-?%}")
    end_pattern = re.compile(r"{%-?\s*endgeneration\s*-?%}")
    start_matches = list(start_pattern.finditer(template))
    end_matches = list(end_pattern.finditer(template))
    if not start_matches or len(start_matches) != len(end_matches):
        raise ValueError("training template has unbalanced or absent generation blocks")
    if len(start_matches) > MAX_MESSAGES:
        raise ValueError("training template has too many generation blocks")

    # Preserve Jinja whitespace-control behavior. For a right-trimming start
    # tag, place the marker after the whitespace the tag must still consume.
    # For a left-trimming end tag, place it before that consumable whitespace.
    insertions: list[tuple[int, str]] = []
    for match in start_matches:
        position = match.end()
        if match.group(0).rstrip().endswith("-%}"):
            while position < len(template) and template[position].isspace():
                position += 1
        insertions.append((position, GENERATION_START_MARKER))
    for match in end_matches:
        position = match.start()
        if match.group(0).lstrip().startswith("{%-"):
            while position > 0 and template[position - 1].isspace():
                position -= 1
        insertions.append((position, GENERATION_END_MARKER))

    instrumented = template
    for position, marker in sorted(insertions, reverse=True):
        instrumented = instrumented[:position] + marker + instrumented[position:]
    return instrumented


def _remove_generation_markers(
    rendered: str,
) -> tuple[str, list[tuple[int, int]]]:
    clean_parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    clean_length = 0
    active_start: int | None = None
    while cursor < len(rendered):
        next_start = rendered.find(GENERATION_START_MARKER, cursor)
        next_end = rendered.find(GENERATION_END_MARKER, cursor)
        candidates = [position for position in (next_start, next_end) if position >= 0]
        if not candidates:
            tail = rendered[cursor:]
            clean_parts.append(tail)
            clean_length += len(tail)
            break
        marker_position = min(candidates)
        chunk = rendered[cursor:marker_position]
        clean_parts.append(chunk)
        clean_length += len(chunk)
        if marker_position == next_start:
            if active_start is not None:
                raise ValueError("nested generation markers are not supported")
            active_start = clean_length
            cursor = marker_position + len(GENERATION_START_MARKER)
        else:
            if active_start is None:
                raise ValueError("generation end marker has no start marker")
            spans.append((active_start, clean_length))
            active_start = None
            cursor = marker_position + len(GENERATION_END_MARKER)
    if active_start is not None or not spans:
        raise ValueError("instrumented render has incomplete generation markers")
    return "".join(clean_parts), spans


def _expected_assistant_mask(
    *,
    tokenizer: Any,
    training_template: str,
    trajectory: list[dict[str, JsonValue]],
    tools: list[dict[str, JsonValue]],
    template_kwargs: dict[str, JsonValue],
    rendered: str,
    offsets: list[tuple[int, int]],
) -> tuple[list[int], list[tuple[int, int]]]:
    if GENERATION_START_MARKER in rendered or GENERATION_END_MARKER in rendered:
        raise ValueError("rendered messages contain reserved mask markers")
    instrumented_template = _instrument_generation_blocks(training_template)
    instrumented_rendered = tokenizer.apply_chat_template(
        trajectory,
        tools=tools,
        chat_template=instrumented_template,
        add_generation_prompt=False,
        tokenize=False,
        **template_kwargs,
    )
    instrumented_limit = (
        MAX_PERSISTED_PROMPT_CHARS
        + MAX_MESSAGES * (len(GENERATION_START_MARKER) + len(GENERATION_END_MARKER))
    )
    if len(instrumented_rendered) > instrumented_limit:
        raise ValueError("instrumented training render exceeds the character limit")
    cleaned_rendered, spans = _remove_generation_markers(instrumented_rendered)
    if cleaned_rendered != rendered:
        raise ValueError("instrumented template did not reproduce the training render")
    expected_mask = [
        int(
            any(
                token_end > span_start and token_start < span_end
                for span_start, span_end in spans
            )
        )
        for token_start, token_end in offsets
    ]
    if not any(expected_mask):
        raise ValueError("generation blocks produced no assistant token span")
    return expected_mask, spans


def _prefix_diagnostic(
    *,
    tokenizer: Any,
    template: str | dict[str, str] | None,
    probe: ProbeConfig,
    tools: list[dict[str, JsonValue]],
) -> dict[str, JsonValue]:
    if not template:
        return {"status": "unavailable", "error": "native template absent"}
    try:
        trajectory = [
            message.model_dump(mode="json", exclude_none=True)
            for message in probe.training_trajectory
        ]
        prefix = _flat_int_list(
            tokenizer.apply_chat_template(
                trajectory[:-1],
                tools=tools,
                chat_template=template,
                add_generation_prompt=False,
                tokenize=True,
                **probe.chat_template_kwargs,
            )
        )
        full = _flat_int_list(
            tokenizer.apply_chat_template(
                trajectory,
                tools=tools,
                chat_template=template,
                add_generation_prompt=False,
                tokenize=True,
                **probe.chat_template_kwargs,
            )
        )
        if max(len(prefix), len(full)) > MAX_TEMPLATE_TOKENS:
            raise ValueError("native prefix diagnostic exceeds the token limit")
        return {
            "status": "measured",
            "prefix_preserved": full[: len(prefix)] == prefix,
            "prefix_token_count": len(prefix),
            "full_token_count": len(full),
        }
    except Exception as exc:
        return {"status": "error", "error": _error_text(exc)}


def _score_tool_output(
    output: str,
    *,
    expected_tool: str,
    probe: ProbeConfig,
    parser_module: Any,
) -> dict[str, JsonValue]:
    parsed = parser_module.parse_tool_calls(output)
    specs = {tool.function.name: tool for tool in probe.tools}
    normalized_calls: list[JsonValue] = []
    schema_valid_count = 0
    dispatchable_count = 0
    expected_tool_dispatchable = False
    for call in parsed.calls:
        spec = specs.get(call.name)
        schema_valid = spec is not None and _arguments_match_schema(
            call.arguments, spec.function.parameters
        )
        if schema_valid:
            schema_valid_count += 1
            # Schema acceptance only. No handler or gate is executed here.
            dispatchable_count += 1
            expected_tool_dispatchable = (
                expected_tool_dispatchable or call.name == expected_tool
            )
        normalized_calls.append(
            {
                "name": call.name,
                "arguments": call.arguments,
                "registered": spec is not None,
                "schema_valid": schema_valid,
                "dispatchable": schema_valid,
            }
        )
    issue_codes = [issue.code for issue in parsed.issues]
    strict_json = parsed.emitted_blocks > 0 and not parsed.issues and bool(parsed.calls)
    schema_valid_output = (
        strict_json
        and schema_valid_count == len(parsed.calls)
        and len(parsed.calls) > 0
    )
    exactly_one_expected_dispatchable_call = (
        parsed.emitted_blocks == 1
        and len(parsed.calls) == 1
        and not parsed.issues
        and schema_valid_count == 1
        and parsed.calls[0].name == expected_tool
    )
    return {
        "emitted_block_count": parsed.emitted_blocks,
        "parsed_call_count": len(parsed.calls),
        "parse_issue_codes": issue_codes,
        "strict_json_parse_success": strict_json,
        "registered_schema_valid_call_count": schema_valid_count,
        "registered_schema_valid_output": schema_valid_output,
        "dispatchable_call_count": dispatchable_count,
        "has_dispatchable_call": dispatchable_count > 0,
        "zero_tool_call": len(parsed.calls) == 0,
        "expected_tool_dispatchable": expected_tool_dispatchable,
        "exactly_one_expected_dispatchable_call": (
            exactly_one_expected_dispatchable_call
        ),
        "handler_or_gate_executed": False,
        "normalized_calls": normalized_calls,
    }


def _arguments_match_schema(
    arguments: object, parameters: FunctionParameters
) -> bool:
    if not isinstance(arguments, dict) or any(
        not isinstance(key, str) for key in arguments
    ):
        return False
    if not set(parameters.required).issubset(arguments):
        return False
    if parameters.additionalProperties is False and not set(arguments).issubset(
        parameters.properties
    ):
        return False
    for name, value in arguments.items():
        property_schema = parameters.properties.get(name)
        if property_schema is None:
            continue
        expected_type = property_schema.get("type")
        if not _json_type_matches(value, expected_type):
            return False
    return True


def _json_type_matches(value: object, expected: object) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return type(value) is int
    if expected == "number":
        return type(value) is int or (type(value) is float and math.isfinite(value))
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "null":
        return value is None
    return False


def _tool_payloads(probe: ProbeConfig) -> list[dict[str, JsonValue]]:
    return [
        {
            "type": tool.type,
            "function": tool.function.model_dump(mode="json"),
        }
        for tool in probe.tools
    ]


def _canonical_cuda_device(value: object) -> str:
    text = str(value).lower()
    if isinstance(value, int) or text.isdigit():
        return f"cuda:{int(value)}"
    return text


def _inspect_model_placement(
    model: Any, *, target_cuda_device_index: int
) -> tuple[dict[str, JsonValue], bool]:
    expected_device = f"cuda:{target_cuda_device_index}"
    raw_device_map = getattr(model, "hf_device_map", {})
    device_map = (
        {
            str(key): _canonical_cuda_device(value)
            for key, value in raw_device_map.items()
        }
        if isinstance(raw_device_map, dict)
        else {"unrecognized": str(raw_device_map)}
    )
    parameters = list(model.parameters())
    parameter_devices = sorted(
        {_canonical_cuda_device(parameter.device) for parameter in parameters}
    )
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in parameters})
    offload_targets = sorted(
        {
            target
            for target in [*device_map.values(), *parameter_devices]
            if target.lower() in {"cpu", "disk", "meta"}
            or target.lower().startswith(("cpu:", "disk:", "meta:"))
        }
    )
    every_parameter_on_target = bool(parameter_devices) and all(
        device == expected_device for device in parameter_devices
    )
    every_map_entry_on_target = bool(device_map) and all(
        device == expected_device for device in device_map.values()
    )
    placement_matches_target = (
        not offload_targets
        and every_parameter_on_target
        and every_map_entry_on_target
    )
    actual_dtype = (
        parameter_dtypes[0]
        if len(parameter_dtypes) == 1
        else f"mixed[{','.join(parameter_dtypes)}]"
    )
    metrics: dict[str, JsonValue] = {
        "device_map": device_map,
        "parameter_devices": parameter_devices,
        "actual_parameter_dtype": actual_dtype,
        "actual_parameter_dtypes": parameter_dtypes,
        "expected_cuda_device": expected_device,
        "every_parameter_on_target": every_parameter_on_target,
        "every_device_map_entry_on_target": every_map_entry_on_target,
        "offload_detected": bool(offload_targets),
        "offload_targets": offload_targets,
    }
    return metrics, placement_matches_target


def _quantization_value(config: object, name: str) -> object:
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _normalized_dtype(value: object) -> str | None:
    if value is None:
        return None
    return str(value).lower().removeprefix("torch.")


def _inspect_effective_quantization(
    model: Any,
    *,
    expected_quantization: str,
    expected_double_quant: bool,
    expected_compute_dtype: object,
) -> tuple[dict[str, JsonValue], bool]:
    effective_config = getattr(model, "quantization_config", None)
    if effective_config is None:
        effective_config = getattr(
            getattr(model, "config", None), "quantization_config", None
        )
    load_in_4bit = _quantization_value(effective_config, "load_in_4bit")
    quantization_type = _quantization_value(
        effective_config, "bnb_4bit_quant_type"
    )
    double_quant = _quantization_value(
        effective_config, "bnb_4bit_use_double_quant"
    )
    compute_dtype = _quantization_value(
        effective_config, "bnb_4bit_compute_dtype"
    )
    parameter_classes = sorted(
        {
            f"{type(parameter).__module__}.{type(parameter).__qualname__}"
            for parameter in model.parameters()
            if "4bit" in type(parameter).__name__.lower()
            or (
                "bitsandbytes" in type(parameter).__module__.lower()
                and "4" in type(parameter).__name__
            )
        }
    )
    module_classes = sorted(
        {
            f"{type(module).__module__}.{type(module).__qualname__}"
            for module in model.modules()
            if "4bit" in type(module).__name__.lower()
            or (
                "bitsandbytes" in type(module).__module__.lower()
                and "4" in type(module).__name__
            )
        }
    )
    class_evidence = (parameter_classes + module_classes)[:64]
    checks = {
        "model_reports_loaded_in_4bit": getattr(
            model, "is_loaded_in_4bit", False
        )
        is True,
        "effective_load_in_4bit": load_in_4bit is True,
        "effective_quantization_type_matches": (
            isinstance(quantization_type, str)
            and quantization_type.lower() == expected_quantization.lower()
        ),
        "effective_double_quant_matches": double_quant is expected_double_quant,
        "effective_compute_dtype_matches": (
            _normalized_dtype(compute_dtype)
            == _normalized_dtype(expected_compute_dtype)
        ),
        "four_bit_class_evidence_present": bool(class_evidence),
    }
    metrics: dict[str, JsonValue] = {
        "effective_quantization_checks": checks,
        "effective_load_in_4bit": _json_safe_metadata(load_in_4bit),
        "effective_quantization_type": _json_safe_metadata(quantization_type),
        "effective_double_quant": _json_safe_metadata(double_quant),
        "effective_compute_dtype": _json_safe_metadata(
            _normalized_dtype(compute_dtype)
        ),
        "four_bit_parameter_classes": parameter_classes[:32],
        "four_bit_module_classes": module_classes[:32],
    }
    return metrics, all(checks.values())


def _flat_int_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise TypeError("expected one flat integer token sequence")
    return value


def _bounded_text(value: str, limit: int) -> dict[str, JsonValue]:
    return {
        "text": value[:limit],
        "truncated": len(value) > limit,
        "character_count": len(value),
    }


def _artifact_sha256(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        encoded = value.encode("utf-8")
    else:
        encoded = json.dumps(
            _json_safe_metadata(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_safe_metadata(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe_metadata(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_metadata(item) for key, item in value.items()}
    return str(value)


def _synchronize_cuda(torch: Any, device_index: int) -> None:
    torch.cuda.synchronize(device_index)


def _resolved_revision(artifact: Any) -> str | None:
    candidates = [
        getattr(artifact, "_commit_hash", None),
        getattr(getattr(artifact, "config", None), "_commit_hash", None),
    ]
    init_kwargs = getattr(artifact, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        candidates.append(init_kwargs.get("_commit_hash"))
        candidates.append(init_kwargs.get("commit_hash"))
    for value in candidates:
        if isinstance(value, str) and value:
            return value
    return None


def _error_text(exc: BaseException) -> str:
    message = str(exc).strip()
    rendered = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
    if len(rendered) <= MAX_ERROR_CHARS:
        return rendered
    suffix = "...[truncated]"
    return rendered[: MAX_ERROR_CHARS - len(suffix)] + suffix


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan Qwen smoke probes offline by default. Actual tokenizer/model "
            "access requires --run-load together with --allow-download."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="candidate name to include; repeat to select more than one",
    )
    parser.add_argument(
        "--run-load",
        action="store_true",
        help="execute tokenizer, 4-bit CUDA load, and generation probes",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="acknowledge that executing probes may contact the model host and download files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    effective_argv = sys.argv[1:] if argv is None else list(argv)
    arguments = parser.parse_args(effective_argv)
    if arguments.run_load != arguments.allow_download:
        parser.error("--run-load and --allow-download must be supplied together")

    config_path: Path = arguments.config
    output_path: Path = arguments.output
    if config_path.resolve() == output_path.resolve():
        parser.error("--output must not overwrite --config")

    try:
        config_bytes = config_path.read_bytes()
        if len(config_bytes) > MAX_CONFIG_BYTES:
            raise SmokeConfigError(
                f"config exceeds the {MAX_CONFIG_BYTES}-byte size limit"
            )
        config = SmokeConfig.model_validate(_strict_json_loads(config_bytes))
        result = build_result(
            config,
            config_path=config_path,
            config_bytes=config_bytes,
            command=[sys.executable, str(Path(__file__).resolve()), *effective_argv],
            run_load=arguments.run_load,
            allow_download=arguments.allow_download,
            selected_names=arguments.candidate,
        )
        write_result_atomic(result, output_path)
    except (
        OSError,
        json.JSONDecodeError,
        _StrictJSONError,
        ValidationError,
        SmokeConfigError,
        SmokeResultError,
    ) as exc:
        parser.error(str(exc))

    mode = "executed" if arguments.run_load else "planned offline"
    print(f"{mode}: {len(result.candidates)} candidate(s); result={output_path}")
    if not arguments.run_load:
        print(
            "gated operations not run: tokenizer/repository access, 4-bit CUDA load, "
            "generation, and TRL/Unsloth imports"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
