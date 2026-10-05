"""GRPO trains on the prompt the evaluator scores.

`build_prompt_dataset` pre-renders each train task with the evaluation system
prompt, user template and calculator schema, with thinking off. If any of that
drifted, GRPO would optimise a prompt that is never tested.

Everything here runs offline. The split loader and the `datasets` module are
replaced with stand-ins, so the real function runs end to end on one synthetic
task.
"""

from __future__ import annotations

import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts import train_grpo  # noqa: E402


class RecordingTokenizer:
    """Captures every `apply_chat_template` call instead of tokenising."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return "<rendered>"


SYNTHETIC_TASK = types.SimpleNamespace(
    task_id="gsm8k:train:0",
    question="Ken packed 3 boxes with 4 pens each. How many pens did he pack?",
    gold_answer=12.0,
)


class _ListDataset(list):
    """Stands in for `datasets.Dataset`: indexable rows, built from a list."""

    @classmethod
    def from_list(cls, rows):
        return cls(rows)


@contextmanager
def offline_inputs():
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.Dataset = _ListDataset
    with mock.patch.dict(sys.modules, {"datasets": fake_datasets}), mock.patch.object(
        train_grpo, "load_split", return_value=[SYNTHETIC_TASK]
    ) as loader:
        yield loader


class PromptDatasetTests(unittest.TestCase):
    def test_rows_come_from_the_train_split(self) -> None:
        with offline_inputs() as loader:
            dataset, tasks = train_grpo.build_prompt_dataset(RecordingTokenizer(), 1)
        loader.assert_called_once_with(train_grpo.SPLIT_MANIFEST_PATH, "train", limit=1)
        self.assertEqual(tasks, [SYNTHETIC_TASK])
        self.assertEqual(dataset[0]["task_id"], SYNTHETIC_TASK.task_id)
        self.assertEqual(dataset[0]["gold_answer"], SYNTHETIC_TASK.gold_answer)
        self.assertEqual(dataset[0]["prompt"], "<rendered>")

    def test_the_prompt_is_rendered_as_the_evaluator_renders_it(self) -> None:
        tokenizer = RecordingTokenizer()
        with offline_inputs():
            train_grpo.build_prompt_dataset(tokenizer, 1)
        (call,) = tokenizer.calls
        self.assertEqual(
            call["messages"],
            [
                {"role": "system", "content": train_grpo.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": train_grpo.USER_PROMPT.format(question=SYNTHETIC_TASK.question),
                },
            ],
        )
        self.assertEqual(call["tools"], [train_grpo.calculator_tool_schema()])
        self.assertFalse(call["tokenize"])
        self.assertTrue(call["add_generation_prompt"])
        self.assertFalse(call["enable_thinking"])


if __name__ == "__main__":
    unittest.main()
