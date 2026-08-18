"""Versioned trajectory records and strict JSON Lines persistence."""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from collections.abc import Iterable, Mapping
from os import PathLike
from pathlib import Path
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator


TRAJECTORY_SCHEMA_VERSION: Final = 1
_REPLACE_LOCKS: Final = tuple(threading.Lock() for _ in range(64))
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

    Every record is validated *and encoded* before the destination is touched,
    so an invalid or unencodable item cannot damage an existing results file.
    The bytes are then written to a sibling temporary file and moved into place,
    making the replacement atomic for readers.
    """

    validated = tuple(_revalidate_record(record) for record in records)
    try:
        payloads = tuple(
            json.dumps(
                record.model_dump(mode="json"),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            for record in validated
        )
    except UnicodeEncodeError as exc:
        raise ValueError(
            "trajectory record contains text that UTF-8 cannot encode "
            "(most likely an unpaired surrogate): %s" % exc
        ) from exc

    path = Path(destination)
    temporary: Path | None = None
    try:
        # The temporary file must be unique even when worker threads share a
        # process and destination. Keeping it beside the destination preserves
        # the same-filesystem guarantee required by ``os.replace``.
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=".%s." % path.name,
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            for payload in payloads:
                stream.write(payload)
                stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        assert temporary is not None
        # Windows can reject two simultaneous replacements of one destination
        # even though both temporary files are distinct. Serialize only this
        # final in-process operation; each writer still prepares and fsyncs its
        # own complete artifact independently.
        normalized_path = os.path.normcase(str(path.resolve()))
        replace_lock = _REPLACE_LOCKS[hash(normalized_path) % len(_REPLACE_LOCKS)]
        with replace_lock:
            os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return len(payloads)


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
