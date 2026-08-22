"""A model is judged against the tool-call dialect its own template asks for.

Llama 3.1 emits a bare JSON object using `parameters`; Qwen3 emits
`<tool_call>` tags using `arguments`. Both are the model doing what its template
told it. Scoring one family against the other's convention measures the
convention, so the translation happens before parsing rather than inside it.

The translation is off unless the caller says the model needs it, and these
tests pin that default. Turning it on globally would change three completions
in the recorded episode logs, where an untrained Qwen emitted bare JSON: by
Qwen's own convention that is a real format failure, and quietly accepting it
would improve a frozen baseline and shrink every gain measured against it.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.dialects import (  # noqa: E402
    looks_like_tool_call,
    normalise_tool_dialect,
    template_uses_canonical_tags,
)
from agent.parser import parse_tool_calls  # noqa: E402

CANONICAL = '<tool_call>\n{"name": "calculator", "arguments": {"expression": "2+2"}}\n</tool_call>'
BARE_LLAMA = '{"name": "calculator", "parameters": {"expression": "3*60"}}'


class PassThroughTests(unittest.TestCase):
    """Anything already canonical, or not a tool call, is returned untouched."""

    def test_a_canonical_call_is_byte_identical(self) -> None:
        self.assertEqual(normalise_tool_dialect(CANONICAL), CANONICAL)

    def test_prose_is_untouched(self) -> None:
        text = "The answer is 72."
        self.assertEqual(normalise_tool_dialect(text), text)

    def test_malformed_json_is_left_for_the_parser_to_reject(self) -> None:
        """The parser reports format failures. Repairing them would hide them."""

        broken = '{"name": "calculator", '
        self.assertEqual(normalise_tool_dialect(broken), broken)

    def test_json_that_is_not_a_tool_call_is_untouched(self) -> None:
        self.assertEqual(normalise_tool_dialect('{"a": 1}'), '{"a": 1}')

    def test_a_stray_closing_tag_disables_translation(self) -> None:
        """Presence of either tag means the model was speaking the canonical
        dialect, however badly, so its output is judged in that dialect."""

        text = '{"name": "calculator", "parameters": {}}</tool_call>'
        self.assertEqual(normalise_tool_dialect(text), text)


class TranslationTests(unittest.TestCase):
    def test_a_bare_llama_call_becomes_parseable(self) -> None:
        """The point of the module: this is a correct call in Llama's dialect."""

        self.assertEqual(parse_tool_calls(BARE_LLAMA).emitted_blocks, 0)
        result = parse_tool_calls(normalise_tool_dialect(BARE_LLAMA))
        self.assertEqual(result.emitted_blocks, 1)
        self.assertEqual(result.issues, [])
        self.assertEqual(result.calls[0].name, "calculator")
        self.assertEqual(result.calls[0].arguments, {"expression": "3*60"})

    def test_parameters_is_renamed_to_arguments(self) -> None:
        payload = json.loads(
            normalise_tool_dialect(BARE_LLAMA)
            .removeprefix("<tool_call>\n")
            .removesuffix("\n</tool_call>")
        )
        self.assertEqual(set(payload), {"name", "arguments"})

    def test_an_already_correct_key_survives_translation(self) -> None:
        bare = '{"name": "calculator", "arguments": {"expression": "1+1"}}'
        result = parse_tool_calls(normalise_tool_dialect(bare))
        self.assertEqual(result.calls[0].arguments, {"expression": "1+1"})

    def test_surrounding_prose_does_not_block_translation(self) -> None:
        text = 'Let me compute.\n{"name": "calculator", "parameters": {"expression": "6*7"}}'
        self.assertEqual(parse_tool_calls(normalise_tool_dialect(text)).emitted_blocks, 1)


class ShapeTests(unittest.TestCase):
    def test_a_tool_call_needs_a_name_and_an_argument_object(self) -> None:
        self.assertTrue(looks_like_tool_call({"name": "c", "arguments": {}}))
        self.assertTrue(looks_like_tool_call({"name": "c", "parameters": {}}))
        self.assertFalse(looks_like_tool_call({"name": "c"}))
        self.assertFalse(looks_like_tool_call({"arguments": {}}))
        self.assertFalse(looks_like_tool_call({"name": 1, "arguments": {}}))
        self.assertFalse(looks_like_tool_call("not an object"))


class TemplateDetectionTests(unittest.TestCase):
    """Which dialect a model speaks is read from its template, not its name."""

    def test_a_template_naming_the_tag_is_canonical(self) -> None:
        self.assertTrue(template_uses_canonical_tags("... <tool_call> ..."))

    def test_a_template_without_the_tag_is_not(self) -> None:
        self.assertFalse(template_uses_canonical_tags("{{ messages }}"))

    def test_a_missing_template_is_not_canonical(self) -> None:
        self.assertFalse(template_uses_canonical_tags(None))


if __name__ == "__main__":
    unittest.main()
