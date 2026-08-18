"""Versioned trajectory records and strict JSON Lines persistence."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from os import PathLike
from pathlib import Path
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator


TRAJECTORY_SCHEMA_VERSION: Final = 1
Pathish: TypeAlias = str | PathLike[str]
TrajectoryInput: TypeAlias = "TrajectoryRecord | Mapping[str, object]"


class TrajectoryRecord(BaseModel):
    """One fully auditable evaluation episode.

    Payload fields accept any strict JSON value. Non-JSON containers and
    non-finite floats are rejected so a write/read cycle cannot silently
    change their representation.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1]
    task_id: str = Field(min_length=1)
    run_idx: int = Field(ge=0)
    prompt: JsonValue
    raw_completion: JsonValue
    parsed_tool_calls: JsonValue
    sandbox_trace: JsonValue
    gate_events: JsonValue
    ground_truth: JsonValue
    reward_breakdown: JsonValue

    @field_validator(
        "prompt",
        "raw_completion",
        "parsed_tool_calls",
        "sandbox_trace",
        "gate_events",
        "ground_truth",
        "reward_breakdown",
        mode="before",
    )
    @classmethod
    def require_lossless_json_value(cls, value: object) -> object:
        _validate_json_value(value)
        return value


class TrajectoryJSONLError(ValueError):
    """Raised when a JSONL line is not a valid trajectory record."""


def write_trajectory_jsonl(
    records: Iterable[TrajectoryRecord | Mapping[str, object]],
    destination: Pathish,
) -> int:
    """Write records as UTF-8 JSONL, returning the number written.

    All records are validated before the destination is opened, avoiding a
    partially written file when an item is invalid.
    """

    validated = tuple(_revalidate_record(record) for record in records)
    lines = tuple(
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        for record in validated
    )

    path = Path(destination)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for line in lines:
            stream.write(line)
            stream.write("\n")
    return len(lines)


def read_trajectory_jsonl(source: Pathish) -> list[TrajectoryRecord]:
    """Read and validate every trajectory in a UTF-8 JSONL file.

    Blank lines are invalid JSONL. Parse and schema failures are wrapped with
    the source line number while preserving the original exception as cause.
    """

    path = Path(source)
    records: list[TrajectoryRecord] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise TrajectoryJSONLError(
                    f"invalid trajectory JSONL at line {line_number}: blank line"
                )
            try:
                payload = json.loads(line)
                records.append(TrajectoryRecord.model_validate(payload))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise TrajectoryJSONLError(
                    f"invalid trajectory JSONL at line {line_number}: {exc}"
                ) from exc
    return records


def _revalidate_record(
    record: TrajectoryRecord | Mapping[str, object],
) -> TrajectoryRecord:
    if isinstance(record, TrajectoryRecord):
        # ``frozen=True`` prevents field assignment, but JSON lists and objects
        # inside the model remain mutable. Dumping in Python mode preserves any
        # invalid post-construction value so validation rejects it instead of a
        # JSON-mode serializer silently coercing it (for example tuple -> list).
        record = record.model_dump(mode="python", warnings=False)
    return TrajectoryRecord.model_validate(record)


def _validate_json_value(value: object, path: str = "payload") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string object key")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise ValueError(f"{path} contains non-JSON value {value!r}")
