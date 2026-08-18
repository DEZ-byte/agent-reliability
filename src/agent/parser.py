"""Parser for the model-neutral tagged JSON tool-call envelope."""

from __future__ import annotations

import json
import math
from typing import Any

from pydantic import ValidationError

from env.models import ParseIssue, ParseResult, ToolCall


OPEN_TAG = "<tool_call>"
CLOSE_TAG = "</tool_call>"


class _StrictJsonError(ValueError):
    pass


def _reject_constant(value: str) -> None:
    raise _StrictJsonError(f"non-standard JSON constant '{value}' is not allowed")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise _StrictJsonError(f"non-finite JSON number '{value}' is not allowed")
    return parsed


def _has_surrogate(text: str) -> bool:
    return any(0xD800 <= ord(char) <= 0xDFFF for char in text)


def _find_surrogate(value: Any, path: str = "tool_call") -> str | None:
    """Return the path of the first unpaired surrogate, if any.

    ``json`` accepts escapes such as ``ud800`` and yields a ``str`` holding a
    lone surrogate codepoint. UTF-8 cannot encode it, so it would later crash
    evidence hashing and result writing. It is rejected here instead, where it
    becomes reward-visible parse evidence.
    """

    if isinstance(value, str):
        return path if _has_surrogate(value) else None
    if isinstance(value, list):
        for index, item in enumerate(value):
            found = _find_surrogate(item, "%s[%d]" % (path, index))
            if found is not None:
                return found
        return None
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and _has_surrogate(key):
                return "%s key %r" % (path, key)
            found = _find_surrogate(item, "%s.%s" % (path, key))
            if found is not None:
                return found
        return None
    return None


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            # ``json`` accepts a lone surrogate in an escaped object key.
            # Keep strict-decoder errors ASCII-safe so constructing the
            # reward-visible ParseIssue cannot itself fail Unicode validation.
            raise _StrictJsonError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _strict_json_loads(payload: str) -> Any:
    return json.loads(
        payload,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
        parse_float=_parse_finite_float,
    )


def _decode_block(payload: str, block_index: int) -> tuple[ToolCall | None, ParseIssue | None]:
    if not payload.strip():
        return None, ParseIssue(
            block_index=block_index,
            code="empty_block",
            message="tool_call block is empty",
        )

    try:
        decoded: Any = _strict_json_loads(payload)
    except (json.JSONDecodeError, _StrictJsonError) as exc:
        position = f" at character {exc.pos}" if isinstance(exc, json.JSONDecodeError) else ""
        return None, ParseIssue(
            block_index=block_index,
            code="invalid_json",
            message=f"invalid JSON{position}: {exc}",
        )

    surrogate_path = _find_surrogate(decoded)
    if surrogate_path is not None:
        return None, ParseIssue(
            block_index=block_index,
            code="unpaired_surrogate",
            message="tool_call contains an unpaired surrogate at %s" % surrogate_path,
        )

    if not isinstance(decoded, dict):
        return None, ParseIssue(
            block_index=block_index,
            code="invalid_envelope",
            message="tool_call JSON must be an object",
        )

    expected_keys = {"name", "arguments"}
    if set(decoded) != expected_keys:
        return None, ParseIssue(
            block_index=block_index,
            code="invalid_envelope",
            message="tool_call object must contain exactly 'name' and 'arguments'",
        )

    try:
        call = ToolCall(
            call_id=f"call-{block_index}",
            name=decoded["name"],
            arguments=decoded["arguments"],
        )
    except ValidationError as exc:
        return None, ParseIssue(
            block_index=block_index,
            code="invalid_envelope",
            message=f"tool_call envelope failed validation: {exc.errors(include_url=False)}",
        )

    return call, None


def parse_tool_calls(completion: str) -> ParseResult:
    """Extract normalized tool calls while preserving every block failure.

    Text outside exact ``<tool_call>...</tool_call>`` tags is deliberately
    ignored. In particular, policy words or answer markers in prose cannot be
    mistaken for executed behavior.
    """

    calls: list[ToolCall] = []
    issues: list[ParseIssue] = []
    emitted_blocks = 0
    cursor = 0

    while cursor < len(completion):
        next_open = completion.find(OPEN_TAG, cursor)
        next_close = completion.find(CLOSE_TAG, cursor)

        if next_close != -1 and (next_open == -1 or next_close < next_open):
            issues.append(
                ParseIssue(
                    block_index=emitted_blocks,
                    code="unexpected_close_tag",
                    message="closing tool_call tag has no matching opening tag",
                    attached_to_block=False,
                )
            )
            cursor = next_close + len(CLOSE_TAG)
            continue

        if next_open == -1:
            break

        block_index = emitted_blocks
        emitted_blocks += 1
        payload_start = next_open + len(OPEN_TAG)
        payload_end = completion.find(CLOSE_TAG, payload_start)
        if payload_end == -1:
            issues.append(
                ParseIssue(
                    block_index=block_index,
                    code="unclosed_block",
                    message="opening tool_call tag has no matching closing tag",
                )
            )
            break

        call, issue = _decode_block(completion[payload_start:payload_end], block_index)
        if call is not None:
            calls.append(call)
        if issue is not None:
            issues.append(issue)
        cursor = payload_end + len(CLOSE_TAG)

    return ParseResult(emitted_blocks=emitted_blocks, calls=calls, issues=issues)
