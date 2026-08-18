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
MAX_CONFIG_BYTES: Final = 256 * 1024
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


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class CandidateConfig(StrictModel):
    name: str
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


class SmokeConfig(StrictModel):
    schema_version: Literal[1]
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


class CandidateResult(StrictModel):
    name: str
    role: Literal["primary_small", "scale_check"]
    model_id: str
    requested_revision: str
    resolved_revision: str | None
    probes: list[ProbeResult]


class RunOptions(StrictModel):
    dry_run: bool
    run_load: bool
    allow_download: bool
    selected_candidates: list[str]


class SmokeResult(StrictModel):
    schema_version: Literal[1]
    created_at_utc: str
    config_path: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    command: list[str]
    options: RunOptions
    hardware: HardwareFacts
    library_versions: dict[str, str | None]
    candidates: list[CandidateResult]

    @field_validator("created_at_utc")
    @classmethod
    def validate_utc_time(cls, value: str) -> str:
        if not value.endswith("Z"):
            raise ValueError("created_at_utc must use the UTC Z suffix")
        datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
        return value


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
        "artifact_access": "AutoModelForCausalLM.from_pretrained with the configured revision",
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
        "checks": [
            "all timed output token ID sequences are identical within each case",
            "at least one new token is produced across timed runs",
            "exactly one parsed call is registered, schema-valid, and names the expected tool",
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
            "trl.chat_template_utils.get_training_chat_template",
            "unsloth.FastLanguageModel",
            "unsloth.chat_templates.get_chat_template",
        ],
        "construct": "GRPOConfig only; no trainer and no training run",
        "hard_gate": "every planned import and configuration construction succeeds",
        "reference_policy_fixture": (
            "planned as a separate pre-compute unit fixture; this import smoke does not "
            "claim reference-policy correctness"
        ),
    }
    training_template_plan: dict[str, JsonValue] = {
        "template_source": "trl.chat_template_utils.get_training_chat_template",
        "record": [
            "training template SHA-256 separately from native template",
            "training rendered prompt SHA-256",
            "complete expected and returned assistant-token masks",
            "tokenized prefix before and after appending the tool observation",
        ],
        "hard_checks": [
            "assistant mask exactly equals all token spans emitted by TRL generation blocks",
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
            candidate_result = CandidateResult(
                name=candidate.name,
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
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        command=command,
        options=options,
        hardware=hardware,
        library_versions=library_versions,
        candidates=candidates,
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


def _execute_candidate(candidate: CandidateConfig, probe: ProbeConfig) -> CandidateResult:
    plans = {planned.name: planned.plan for planned in probe_plans(probe)}
    import_result = _run_training_stack_import_probe(
        plans["training_stack_imports"]
    )
    try:
        transformers = importlib.import_module("transformers")
    except (ImportError, RuntimeError) as exc:
        error = f"transformers runtime unavailable: {_error_text(exc)}"
        return CandidateResult(
            name=candidate.name,
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
        return CandidateResult(
            name=candidate.name,
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
        return CandidateResult(
            name=candidate.name,
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
        )

    if not torch.cuda.is_available():
        return CandidateResult(
            name=candidate.name,
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
        )

    target_cuda_device_index = probe.target_cuda_device_index
    if target_cuda_device_index >= torch.cuda.device_count():
        error = (
            f"configured CUDA device {target_cuda_device_index} is unavailable; "
            f"torch reports {torch.cuda.device_count()} visible device(s)"
        )
        return CandidateResult(
            name=candidate.name,
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
        model = transformers.AutoModelForCausalLM.from_pretrained(
            candidate.model_id,
            revision=candidate.revision,
            trust_remote_code=False,
            local_files_only=False,
            quantization_config=quantization_config,
            device_map={"": target_cuda_device_index},
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

    return CandidateResult(
        name=candidate.name,
        role=candidate.role,
        model_id=candidate.model_id,
        requested_revision=candidate.revision,
        resolved_revision=resolved_revision,
        probes=[
            template_result,
            load_result,
            generation_result,
            import_result,
            training_template_result,
        ],
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

        checks = {
            "timed_outputs_identical_within_each_case": all(
                deterministic_by_case.values()
            ),
            "nonzero_generated_tokens": token_total > 0,
            "positive_elapsed_time": seconds_total > 0,
            "every_output_is_strict_and_schema_valid": all(
                strict_tool_output_by_case.values()
            ),
            "every_output_has_exactly_one_expected_dispatchable_call": all(
                expected_dispatchable_by_case.values()
            ),
        }
        passed = all(checks.values())
        throughput = token_total / seconds_total if seconds_total > 0 else None
        return ProbeResult(
            name="deterministic_generation",
            status="passed" if passed else "failed",
            plan=plan,
            metrics={
                "checks": checks,
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
                "strict_json_parse_rate": rate("strict_json_parse_success"),
                "registered_schema_valid_output_rate": rate(
                    "registered_schema_valid_output"
                ),
                "dispatchable_call_output_rate": rate("has_dispatchable_call"),
                "zero_tool_call_rate": rate("zero_tool_call"),
                "retained_outputs": retained_outputs,
            },
            error=None if passed else "generation or strict tool-call hard gate failed",
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
        unsloth_chat = importlib.import_module("unsloth.chat_templates")
        imported["unsloth"] = True
        trl = importlib.import_module("trl")
        chat_utils = importlib.import_module("trl.chat_template_utils")
        imported["trl"] = True
        grpo_config_class = getattr(trl, "GRPOConfig")
        getattr(chat_utils, "get_training_chat_template")
        getattr(unsloth, "FastLanguageModel")
        getattr(unsloth_chat, "get_chat_template")
        imported.update({
            "trl.GRPOConfig": True,
            "trl.chat_template_utils.get_training_chat_template": True,
            "unsloth.FastLanguageModel": True,
            "unsloth.chat_templates.get_chat_template": True,
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
        chat_utils = importlib.import_module("trl.chat_template_utils")
        helper = getattr(chat_utils, "get_training_chat_template")
        patched_template = helper(tokenizer)
        training_template = patched_template or native_template
        if not isinstance(training_template, str) or not training_template:
            raise ValueError("TRL did not provide one usable string training template")
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
                "trl_patch_returned": patched_template is not None,
                "native_chat_template_sha256": _artifact_sha256(native_template),
                "training_chat_template_sha256": _artifact_sha256(training_template),
                "training_rendered_prompt_sha256": _text_sha256(rendered),
                "training_rendered_prompt": rendered,
                "training_rendered_token_count": len(full_ids),
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
