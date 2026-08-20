"""Offline proofs that the training labels cover assistant tokens and nothing else.

CI has no network, so these run against a character-level stub tokenizer that
reproduces the parts of the chat-template contract this module depends on: a
generation-marked span, an assistant header the harness supplies, and a
trailing newline after the turn's end marker.

The stub is not a convenience. Every guard in `training.masking` fires on a
condition a real tokenizer produces only when something is already wrong, and a
test that cannot create that condition cannot prove the guard works. Each test
below builds the broken case explicitly and asserts the guard refuses it.

The matching facts about the real Qwen3 tokenizer are measured separately by
`scripts/verify_masking.py`, which writes a hashed artifact.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.masking import (  # noqa: E402
    IGNORE_INDEX,
    MaskingError,
    encode_with_labels,
)

HEADER = "<|im_start|>assistant\n"
END = "<|im_end|>"


class StubTokenizer:
    """A character-level stand-in for a generation-marked chat template.

    Ids are code points, so `decode(encode(text)) == text` and a test can read
    the trained region directly instead of trusting an opaque vocabulary.
    """

    chat_template = "stub {% generation %} template"

    def __init__(
        self,
        *,
        mask_override: list[int] | None = None,
        native_suffix: str = "",
        extra_span: bool = False,
    ) -> None:
        self.mask_override = mask_override
        self.native_suffix = native_suffix
        self.extra_span = extra_span

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, ids) -> str:
        return "".join(chr(index) for index in ids)

    def _render(self, messages) -> tuple[str, list[int]]:
        text = ""
        mask: list[int] = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "assistant":
                # Mirrors the real template: the marked span opens at the
                # header and closes after the newline following the end marker.
                block = HEADER + content + END + "\n"
                mask.extend([1] * len(block))
            else:
                block = "<|im_start|>" + role + "\n" + content + END + "\n"
                mask.extend([0] * len(block))
            text += block
        return text, mask

    def apply_chat_template(
        self,
        messages,
        chat_template=None,
        tokenize=False,
        return_dict=False,
        return_assistant_tokens_mask=False,
        **kwargs,
    ):
        text, mask = self._render(messages)
        if chat_template is None:
            text += self.native_suffix
        ids = self.encode(text)
        if not return_dict:
            return ids if tokenize else text
        payload = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_assistant_tokens_mask:
            if self.mask_override is not None:
                payload["assistant_masks"] = list(self.mask_override)
            elif self.extra_span:
                broken = list(mask)
                broken[0] = 1
                payload["assistant_masks"] = broken
            else:
                payload["assistant_masks"] = mask
        return payload


TRAJECTORY = [
    {"role": "system", "content": "SYSTEM RULES"},
    {"role": "user", "content": "Question: how many?"},
    {"role": "assistant", "content": "<tool_call>CALL</tool_call>"},
]


class MaskedExampleTests(unittest.TestCase):
    def test_only_the_assistant_turn_and_its_end_marker_are_trained(self) -> None:
        tokenizer = StubTokenizer()
        example = encode_with_labels(tokenizer, TRAJECTORY, tools=[])
        self.assertEqual(
            example.trained_text(tokenizer), "<tool_call>CALL</tool_call>" + END
        )

    def test_the_harness_header_and_trailing_newline_are_not_trained(self) -> None:
        """The model never emits either, so training on them teaches noise."""

        tokenizer = StubTokenizer()
        example = encode_with_labels(tokenizer, TRAJECTORY, tools=[])
        trained = example.trained_text(tokenizer)
        self.assertNotIn(HEADER, trained)
        self.assertFalse(trained.endswith("\n"))
        self.assertTrue(trained.endswith(END))

    def test_no_trained_token_carries_text_from_a_non_assistant_turn(self) -> None:
        """The leak check, derived from the trajectory rather than hard-coded.

        A canary list only catches the leak someone already imagined. This
        asserts the property itself against whatever the messages happen to say.
        """

        tokenizer = StubTokenizer()
        example = encode_with_labels(tokenizer, TRAJECTORY, tools=[])
        trained = example.trained_text(tokenizer)
        for message in TRAJECTORY:
            if message["role"] != "assistant":
                self.assertNotIn(message["content"], trained)

    def test_the_leak_check_can_actually_fail(self) -> None:
        """Negative control. A vacuous leak test is worse than none."""

        tokenizer = StubTokenizer(extra_span=True)
        with self.assertRaises(MaskingError):
            encode_with_labels(tokenizer, TRAJECTORY, tools=[])

    def test_every_untrained_position_is_the_ignore_index(self) -> None:
        tokenizer = StubTokenizer()
        example = encode_with_labels(tokenizer, TRAJECTORY, tools=[])
        start, end = example.trained_spans[0]
        for index, label in enumerate(example.labels):
            if start <= index < end:
                self.assertEqual(label, example.input_ids[index])
            else:
                self.assertEqual(label, IGNORE_INDEX)

    def test_trained_token_count_matches_the_labelled_positions(self) -> None:
        tokenizer = StubTokenizer()
        example = encode_with_labels(tokenizer, TRAJECTORY, tools=[])
        self.assertEqual(
            example.trained_token_count,
            sum(1 for label in example.labels if label != IGNORE_INDEX),
        )
        self.assertGreater(example.trained_token_count, 0)


class MaskingGuardTests(unittest.TestCase):
    """Each guard covers a failure that produces no exception on its own."""

    def test_an_all_zero_mask_is_refused_rather_than_trained_on(self) -> None:
        """The headline silent failure: an unmarked template yields all zeros.

        transformers logs once and returns the zero mask. Training on it makes
        every label -100, so the run learns nothing and still looks healthy.
        """

        tokenizer = StubTokenizer(mask_override=[0] * 4096)
        with self.assertRaises(MaskingError) as caught:
            encode_with_labels(tokenizer, TRAJECTORY, tools=[])
        self.assertIn("empty", str(caught.exception))

    def test_a_mask_of_the_wrong_length_is_refused(self) -> None:
        tokenizer = StubTokenizer(mask_override=[1, 1, 1])
        with self.assertRaises(MaskingError) as caught:
            encode_with_labels(tokenizer, TRAJECTORY, tools=[])
        self.assertIn("length", str(caught.exception))

    def test_a_template_patch_that_moves_a_token_is_refused(self) -> None:
        """Marking must be inert. If it is not, training and evaluation differ."""

        tokenizer = StubTokenizer(native_suffix="EXTRA")
        with self.assertRaises(MaskingError) as caught:
            encode_with_labels(tokenizer, TRAJECTORY, tools=[])
        self.assertIn("token sequence", str(caught.exception))

    def test_a_trajectory_not_ending_on_an_assistant_turn_is_refused(self) -> None:
        tokenizer = StubTokenizer()
        with self.assertRaises(MaskingError):
            encode_with_labels(tokenizer, TRAJECTORY[:2], tools=[])

    def test_an_empty_trajectory_is_refused(self) -> None:
        tokenizer = StubTokenizer()
        with self.assertRaises(MaskingError):
            encode_with_labels(tokenizer, [], tools=[])


if __name__ == "__main__":
    unittest.main()
