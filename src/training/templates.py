"""Patch a Qwen chat template so assistant tokens can be identified.

`return_assistant_tokens_mask=True` only works on a template that marks its
assistant span with Jinja's `{% generation %}` keyword. The native Qwen3
template has no such marker, and transformers does not raise when it is
missing: it logs once and returns an all-zero mask. Training against that mask
means every label is -100 and the run learns nothing while every loss curve
still looks plausible, so the patch and its guards are load-bearing.

The rewrite is deliberately paranoid about the template's shape. It refuses to
guess when the assistant branch is not exactly where it expects, because a
silently mis-placed marker produces a mask that is wrong rather than empty, and
a wrong mask trains on the prompt.

Moved out of the model compatibility probe at M2 so that probe and the
training pipeline patch the template the same way. Two implementations would be
two chances to disagree about which tokens are trained.
"""

from __future__ import annotations

import re
from typing import Final

MAX_TEMPLATE_SOURCE_CHARS: Final = 256 * 1024
MAX_TEMPLATE_TOKENS: Final = 8192

# Jinja tags the patched template must contain exactly once each.
GENERATION_OPEN: Final = "{% generation %}"
GENERATION_CLOSE: Final = "{% endgeneration %}"


def build_qwen_training_template(native_template: str) -> str:
    if len(native_template) > MAX_TEMPLATE_SOURCE_CHARS:
        raise ValueError("native Qwen template exceeds the character limit")
    generation_tags = re.findall(
        r"{%-?\s*(?:endgeneration|generation)\s*-?%}", native_template
    )
    if generation_tags:
        raise ValueError("native Qwen template already contains generation tags")

    role_access = r"(?:message\.role|message\[['\"]role['\"]\])"

    def branch(role: str) -> list[re.Match[str]]:
        pattern = re.compile(
            r"{%-?\s*elif\s+"
            + role_access
            + r"\s*==\s*(['\"])"
            + re.escape(role)
            + r"\1\s*-?%}"
        )
        return list(pattern.finditer(native_template))

    assistant_matches = branch("assistant")
    tool_matches = branch("tool")
    if len(assistant_matches) != 1 or len(tool_matches) != 1:
        raise ValueError(
            "native Qwen template must contain exactly one assistant branch and one tool branch"
        )
    assistant = assistant_matches[0]
    tool = tool_matches[0]
    if tool.start() <= assistant.end():
        raise ValueError("native Qwen assistant branch must precede its tool sibling")

    control_pattern = re.compile(r"{%-?\s*(.*?)\s*-?%}", re.DOTALL)
    depth = 0
    sibling_found = False
    block_starts = ("if ", "for ", "macro ", "block ", "filter ", "with ", "call ")
    block_ends = ("endif", "endfor", "endmacro", "endblock", "endfilter", "endwith", "endcall")
    for control in control_pattern.finditer(
        native_template, assistant.end(), tool.end()
    ):
        if control.start() == tool.start():
            if depth != 0:
                raise ValueError("native Qwen tool branch is not the assistant sibling")
            sibling_found = True
            break
        statement = control.group(1).strip()
        if statement.startswith(block_starts):
            depth += 1
        elif statement.startswith(block_ends):
            if depth == 0:
                raise ValueError("native Qwen assistant branch closes before the tool branch")
            depth -= 1
        elif depth == 0 and (
            statement.startswith("elif ") or statement == "else"
        ):
            raise ValueError("native Qwen assistant branch has an ambiguous sibling")
    if not sibling_found:
        raise ValueError("native Qwen tool branch is not the assistant sibling")

    generation_end_position = tool.start()
    if tool.group(0).lstrip().startswith("{%-"):
        while (
            generation_end_position > assistant.end()
            and native_template[generation_end_position - 1].isspace()
        ):
            generation_end_position -= 1

    return (
        native_template[: assistant.end()]
        + "{% generation %}"
        + native_template[assistant.end() : generation_end_position]
        + "{% endgeneration %}"
        + native_template[generation_end_position:]
    )


def template_marks_generation(template: str) -> bool:
    """Whether a template can produce a non-empty assistant mask at all.

    Callers use this as a precondition rather than discovering the problem as
    an all-zero mask further downstream.
    """

    return bool(re.search(r"{%-?\s*generation\s*-?%}", template))


__all__ = [
    "GENERATION_CLOSE",
    "GENERATION_OPEN",
    "MAX_TEMPLATE_SOURCE_CHARS",
    "MAX_TEMPLATE_TOKENS",
    "build_qwen_training_template",
    "template_marks_generation",
]
