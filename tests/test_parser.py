from __future__ import annotations

import unittest

from agent.parser import parse_tool_calls


class ToolCallParserTests(unittest.TestCase):
    def test_prose_and_answer_markers_do_not_create_calls(self) -> None:
        parsed = parse_tool_calls(
            "I authenticated the user in my reasoning. #### 12 #### 99"
        )

        self.assertEqual(parsed.emitted_blocks, 0)
        self.assertEqual(parsed.calls, [])
        self.assertEqual(parsed.issues, [])

    def test_parses_multiple_normalized_blocks_in_order(self) -> None:
        parsed = parse_tool_calls(
            '<tool_call>{"name":"authenticate","arguments":{"token":"ok"}}</tool_call>'
            "interstitial prose"
            '<tool_call>{"name":"modify","arguments":{"quantity":2}}</tool_call>'
        )

        self.assertEqual(parsed.emitted_blocks, 2)
        self.assertEqual([call.call_id for call in parsed.calls], ["call-0", "call-1"])
        self.assertEqual([call.name for call in parsed.calls], ["authenticate", "modify"])
        self.assertEqual(parsed.calls[1].arguments, {"quantity": 2})
        self.assertEqual(parsed.issues, [])

    def test_empty_block_is_preserved_as_a_failure(self) -> None:
        parsed = parse_tool_calls("<tool_call>   </tool_call>")

        self.assertEqual(parsed.emitted_blocks, 1)
        self.assertEqual(parsed.calls, [])
        self.assertEqual([issue.code for issue in parsed.issues], ["empty_block"])

    def test_invalid_json_does_not_hide_a_later_valid_block(self) -> None:
        parsed = parse_tool_calls(
            '<tool_call>{"name":</tool_call>'
            '<tool_call>{"name":"calculator","arguments":{"value":4}}</tool_call>'
        )

        self.assertEqual(parsed.emitted_blocks, 2)
        self.assertEqual([issue.code for issue in parsed.issues], ["invalid_json"])
        self.assertEqual(len(parsed.calls), 1)
        self.assertEqual(parsed.calls[0].call_id, "call-1")

    def test_non_standard_and_non_finite_json_numbers_are_rejected(self) -> None:
        for value in ("NaN", "Infinity", "-Infinity", "1e400"):
            with self.subTest(value=value):
                parsed = parse_tool_calls(
                    '<tool_call>{"name":"calculator","arguments":'
                    f'{{"value":{value}}}}}</tool_call>'
                )
                self.assertEqual(parsed.calls, [])
                self.assertEqual([issue.code for issue in parsed.issues], ["invalid_json"])

    def test_duplicate_json_keys_are_rejected(self) -> None:
        parsed = parse_tool_calls(
            '<tool_call>{"name":"safe","name":"modify","arguments":{}}</tool_call>'
        )

        self.assertEqual(parsed.calls, [])
        self.assertEqual([issue.code for issue in parsed.issues], ["invalid_json"])
        self.assertIn("duplicate", parsed.issues[0].message)

    def test_envelope_must_have_exact_name_and_arguments_keys(self) -> None:
        parsed = parse_tool_calls(
            '<tool_call>{"name":"calculator","arguments":{},"result":42}</tool_call>'
        )

        self.assertEqual([issue.code for issue in parsed.issues], ["invalid_envelope"])
        self.assertEqual(parsed.calls, [])

    def test_arguments_must_be_an_object(self) -> None:
        parsed = parse_tool_calls(
            '<tool_call>{"name":"calculator","arguments":[1,2]}</tool_call>'
        )

        self.assertEqual([issue.code for issue in parsed.issues], ["invalid_envelope"])

    def test_unclosed_and_unexpected_close_tags_are_reported(self) -> None:
        unclosed = parse_tool_calls('<tool_call>{"name":"x","arguments":{}}')
        unexpected = parse_tool_calls("text</tool_call>")

        self.assertEqual(unclosed.emitted_blocks, 1)
        self.assertEqual([issue.code for issue in unclosed.issues], ["unclosed_block"])
        self.assertEqual(unexpected.emitted_blocks, 0)
        self.assertEqual(
            [issue.code for issue in unexpected.issues],
            ["unexpected_close_tag"],
        )


if __name__ == "__main__":
    unittest.main()
