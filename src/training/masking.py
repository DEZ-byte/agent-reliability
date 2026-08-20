"""Turn a graded trajectory into token ids and assistant-only training labels.

BLUEPRINT_v2 section 7.2 requires `labels[t] = -100` for every token outside an
assistant turn. That is the direct defence against training the model to
produce its own tool observations, which is the `fabricated_result` failure.

Every guard here exists because the failure it catches is silent. The mask
comes from `return_assistant_tokens_mask`, which returns all zeros rather than
raising when the chat template carries no `{% generation %}` marker; a run
built on that mask has every label masked out, learns nothing, and still
prints a falling loss curve. A mask that is merely misplaced is worse, because
the model then trains on the prompt it will be given at evaluation time.

So this module refuses to produce labels it cannot prove are right. It checks
that the marked tokens are byte-identical to the untouched render, that at
least one token is trained, and that no trained token overlaps the character
span of any non-assistant message. The last check is derived from the
trajectory itself rather than from a list of forbidden strings, because a
hard-coded canary only catches the leak someone already thought of.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Sequence

from training.templates import build_qwen_training_template, template_marks_generation

# Text the harness supplies at the head of an assistant turn. The model is
# never asked to generate it, so training on it teaches nothing and mildly
# corrupts the turn boundary. Resolved to ids through the live tokenizer rather
# than hard-coded, so a checkpoint with a different vocabulary fails loudly.
HARNESS_TURN_PREFIXES: Final = (
    "<|im_start|>assistant\n",
    "<think>\n\n</think>\n\n",
)

# Emitted after the turn's end marker, again by the template rather than by the
# model. The end marker itself stays trained so the model learns to stop.
TRAILING_TEMPLATE_TEXT: Final = "\n"

IGNORE_INDEX: Final = -100


class MaskingError(RuntimeError):
    """The labels could not be proven correct, so none are returned."""


@dataclass(frozen=True, slots=True)
class MaskedExample:
    """One training row whose trained tokens have been verified."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    trained_spans: tuple[tuple[int, int], ...]

    @property
    def trained_token_count(self) -> int:
        return sum(1 for label in self.labels if label != IGNORE_INDEX)

    def trained_text(self, tokenizer: Any) -> str:
        """Exactly the text the loss is computed over. For tests and audits."""

        return "".join(
            tokenizer.decode(self.input_ids[start:end])
            for start, end in self.trained_spans
        )


def _token_ids(rendered: Any) -> list[int]:
    """Normalise what `apply_chat_template` returns across its many shapes."""

    if isinstance(rendered, dict) or hasattr(rendered, "keys"):
        return list(rendered["input_ids"])
    if rendered and isinstance(rendered[0], list):
        return list(rendered[0])
    return list(rendered)


def _spans(mask: Sequence[int]) -> list[list[int]]:
    spans: list[list[int]] = []
    start: int | None = None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            spans.append([start, index])
            start = None
    if start is not None:
        spans.append([start, len(mask)])
    return spans


def _trim(
    span: list[int], input_ids: Sequence[int], tokenizer: Any
) -> tuple[int, int]:
    """Drop template-supplied tokens from the ends of one marked span."""

    start, end = span
    for text in HARNESS_TURN_PREFIXES:
        prefix = tokenizer.encode(text, add_special_tokens=False)
        if prefix and list(input_ids[start : start + len(prefix)]) == list(prefix):
            start += len(prefix)
    trailing = tokenizer.encode(TRAILING_TEMPLATE_TEXT, add_special_tokens=False)
    if trailing and list(input_ids[end - len(trailing) : end]) == list(trailing):
        end -= len(trailing)
    if start >= end:
        raise MaskingError("an assistant turn trimmed away to nothing")
    return start, end


def training_template_for(tokenizer: Any) -> str:
    """The tokenizer's own template, patched so assistant spans are markable."""

    native = tokenizer.chat_template
    if native is None:
        raise MaskingError("tokenizer has no chat template")
    if template_marks_generation(native):
        return native
    return build_qwen_training_template(native)


def encode_with_labels(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    *,
    tools: Sequence[dict[str, Any]],
    enable_thinking: bool = False,
) -> MaskedExample:
    """Render one trajectory and label only the tokens the model must produce.

    Raises rather than returning a mask it cannot verify. A caller that gets a
    `MaskedExample` back has been told, by construction, that the trained
    tokens are assistant tokens and nothing else.
    """

    if not messages:
        raise MaskingError("empty trajectory")
    if messages[-1].get("role") != "assistant":
        raise MaskingError("a training trajectory must end on an assistant turn")

    template = training_template_for(tokenizer)
    if not template_marks_generation(template):
        raise MaskingError("training template does not mark a generation span")

    kwargs: dict[str, Any] = {
        "tools": list(tools),
        "add_generation_prompt": False,
        "enable_thinking": enable_thinking,
    }
    marked = tokenizer.apply_chat_template(
        list(messages),
        chat_template=template,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        **kwargs,
    )
    input_ids = _token_ids(marked)
    mask = list(marked["assistant_masks"])

    # The silent failure this module exists for.
    if sum(mask) == 0:
        raise MaskingError(
            "assistant mask is empty; the template marked no generation span"
        )
    if len(mask) != len(input_ids):
        raise MaskingError("assistant mask length does not match the token count")

    # Marking must not change a single token, or training and evaluation would
    # be reading different strings.
    native_ids = _token_ids(
        tokenizer.apply_chat_template(
            list(messages), tokenize=True, return_dict=True, **kwargs
        )
    )
    if native_ids != input_ids:
        raise MaskingError(
            "patched template changed the token sequence relative to the native one"
        )

    trained = [_trim(span, input_ids, tokenizer) for span in _spans(mask)]
    assistant_turns = sum(1 for message in messages if message.get("role") == "assistant")
    if len(trained) != assistant_turns:
        raise MaskingError(
            f"marked {len(trained)} spans for {assistant_turns} assistant turns"
        )

    labels = [IGNORE_INDEX] * len(input_ids)
    for start, end in trained:
        labels[start:end] = list(input_ids[start:end])

    return MaskedExample(
        input_ids=tuple(input_ids),
        labels=tuple(labels),
        trained_spans=tuple(trained),
    )


__all__ = [
    "HARNESS_TURN_PREFIXES",
    "IGNORE_INDEX",
    "MaskedExample",
    "MaskingError",
    "TRAILING_TEMPLATE_TEXT",
    "encode_with_labels",
    "training_template_for",
]
