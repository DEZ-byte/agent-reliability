"""Normalise a model's native tool-call dialect into the canonical form.

Models do not agree on how a tool call looks. Qwen3's chat template produces
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>`. Llama 3.1's produces
a bare JSON object using `parameters` rather than `arguments` and no tags at
all. Both are the model doing exactly what its own template asked.

The parser understands one dialect, and rightly so: it is the contract the
environment enforces, and loosening it would weaken the format checks every
measurement depends on. So the translation happens before parsing rather than
inside it.

This matters because the alternative is a false result. Scored without it, a
Llama emitting a perfectly correct calculator call gets zero, and the number
looks like a capability gap when it is a serialisation difference. That is the
same mistake D-060 recorded on the Qwen side, where a described format produced
a well-formed opening tag and an early stop, and it cost a full baseline run.

The translation is deliberately conservative. A completion that already carries
a `<tool_call>` tag is returned byte-identical, so no measurement recorded under
the canonical dialect can shift.
"""

from __future__ import annotations

import json
import re
from typing import Final

OPEN_TAG: Final = "<tool_call>"
CLOSE_TAG: Final = "</tool_call>"

# Llama 3.1 names the argument object `parameters`. The environment's schema and
# every recorded artifact use `arguments`.
_ARGUMENT_ALIASES: Final = ("arguments", "parameters")

# A bare JSON object at the start of a line, which is how the untagged dialects
# emit. Matching from the first brace to the last keeps nested objects intact.
_BARE_OBJECT: Final = re.compile(r"\{.*\}", re.DOTALL)


def looks_like_tool_call(payload: object) -> bool:
    """Whether a decoded object is a tool call in any dialect this accepts."""

    if not isinstance(payload, dict):
        return False
    if not isinstance(payload.get("name"), str):
        return False
    return any(
        isinstance(payload.get(alias), dict) for alias in _ARGUMENT_ALIASES
    )


def normalise_tool_dialect(completion: str) -> str:
    """Rewrite an untagged tool call into the canonical tagged form.

    Returns the input unchanged unless every one of these holds: there is no
    `<tool_call>` tag already, the text contains a JSON object, and that object
    is shaped like a tool call. Anything else is left exactly as the model wrote
    it, including prose, malformed JSON and partial calls, so the parser still
    sees and reports the failures it is there to catch.
    """

    if OPEN_TAG in completion or CLOSE_TAG in completion:
        return completion

    match = _BARE_OBJECT.search(completion)
    if match is None:
        return completion

    try:
        payload = json.loads(match.group(0))
    except ValueError:
        return completion

    if not looks_like_tool_call(payload):
        return completion

    arguments = next(
        payload[alias]
        for alias in _ARGUMENT_ALIASES
        if isinstance(payload.get(alias), dict)
    )
    canonical = json.dumps(
        {"name": payload["name"], "arguments": arguments},
        ensure_ascii=False,
        sort_keys=True,
    )
    return f"{OPEN_TAG}\n{canonical}\n{CLOSE_TAG}"


def template_uses_canonical_tags(chat_template: str | None) -> bool:
    """Whether this model's own template asks for `<tool_call>` tags.

    Read from the template rather than from a list of model names, so a new
    checkpoint is classified by what it was actually trained to emit.

    This is what makes the translation safe to apply. For a model whose template
    specifies the tags, bare JSON is a real format failure by that model's own
    convention and must keep counting as one; for a model whose template asks
    for bare JSON, the tags would be the anomaly. Judging every model against
    one family's convention would measure the convention, not the model.
    """

    return bool(chat_template) and OPEN_TAG in chat_template


__all__ = [
    "OPEN_TAG",
    "CLOSE_TAG",
    "looks_like_tool_call",
    "normalise_tool_dialect",
    "template_uses_canonical_tags",
]
