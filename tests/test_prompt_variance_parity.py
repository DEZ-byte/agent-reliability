"""The variance probe must render the prompt the trainer will actually use.

The filter measures how hard each prompt is for the current policy, then
training selects on that measurement. If the two render the prompt even
slightly differently - a different chat template flag, thinking left on, the
tool schema omitted - the probe measures the difficulty of a string the policy
never sees, and the filter quietly selects on the wrong thing. Nothing would
fail; the run would simply train on a subset chosen for a different task.

That is the failure this file exists to catch, so both call sites are driven
through one recording tokenizer and their arguments compared.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts import probe_prompt_variance as probe  # noqa: E402
from scripts import train_grpo  # noqa: E402


class RecordingTokenizer:
    """Captures every `apply_chat_template` call instead of tokenising."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return "<rendered>"

    def __call__(self, text, **kwargs):
        return {"input_ids": [0]}


class RenderParityTests(unittest.TestCase):
    def _probe_call(self) -> dict:
        tokenizer = RecordingTokenizer()
        tools = [probe.calculator_tool_schema()]
        probe.render(tokenizer, "Ken had 2 boxes. How many?", tools)
        return tokenizer.calls[0]

    def _trainer_call(self) -> dict:
        tokenizer = RecordingTokenizer()
        train_grpo.build_prompt_dataset(tokenizer, 1)
        return tokenizer.calls[0]

    def test_both_pass_the_same_template_flags(self) -> None:
        probe_call = self._probe_call()
        trainer_call = self._trainer_call()
        for flag in ("tokenize", "add_generation_prompt", "enable_thinking"):
            self.assertEqual(
                probe_call[flag],
                trainer_call[flag],
                f"{flag} differs between the probe and the trainer",
            )

    def test_both_pass_the_same_tool_schema(self) -> None:
        self.assertEqual(self._probe_call()["tools"], self._trainer_call()["tools"])

    def test_both_build_the_same_message_roles_and_system_prompt(self) -> None:
        probe_messages = self._probe_call()["messages"]
        trainer_messages = self._trainer_call()["messages"]
        self.assertEqual(
            [m["role"] for m in probe_messages],
            [m["role"] for m in trainer_messages],
        )
        self.assertEqual(probe_messages[0]["content"], trainer_messages[0]["content"])

    def test_the_user_turn_is_the_same_template_around_the_question(self) -> None:
        """Same wrapper, different question: the only thing that may differ."""

        probe_user = self._probe_call()["messages"][1]["content"]
        trainer_user = self._trainer_call()["messages"][1]["content"]
        self.assertTrue(probe_user.startswith(train_grpo.USER_PROMPT.split("{")[0]))
        self.assertTrue(trainer_user.startswith(train_grpo.USER_PROMPT.split("{")[0]))


class RolloutParityTests(unittest.TestCase):
    """The probe must sample the way training samples, or it measures nothing.

    A probe run at a different temperature reports a different difficulty. The
    values are read from the same config block rather than restated, so this
    pins that the probe reads them at all.
    """

    def test_the_probe_reads_its_rollout_settings_from_the_grpo_config(self) -> None:
        config = probe.load_train_config(
            probe.TRAIN_CONFIG_PATH,
            require=["grpo.num_generations", "grpo.temperature", "grpo.top_p"],
        )
        grpo = config["grpo"]
        self.assertEqual(grpo["num_generations"], 8)
        self.assertGreaterEqual(grpo["temperature"], 0.7)
        self.assertLessEqual(grpo["temperature"], 0.85)

    def test_probe_and_trainer_read_the_same_config_file(self) -> None:
        self.assertEqual(probe.TRAIN_CONFIG_PATH, train_grpo.TRAIN_CONFIG_PATH)

    def test_probe_and_trainer_read_the_same_split_manifest(self) -> None:
        self.assertEqual(probe.SPLIT_MANIFEST_PATH, train_grpo.SPLIT_MANIFEST_PATH)

    def test_the_probe_seed_cannot_coincide_with_the_training_seed(self) -> None:
        """Distinct seed bases, the same rule generation and evaluation follow."""

        config = probe.load_train_config(probe.TRAIN_CONFIG_PATH)
        self.assertNotEqual(probe.PROBE_SEED_BASE, config["grpo"]["seed"])


class FakeTensor(list):
    """Just enough of a batch encoding to drive `generate_group`."""

    @property
    def shape(self):
        return (len(self), len(self[0]) if self else 0)


class FakeBatch(dict):
    def to(self, _device):
        return self


class FakeTorch:
    @staticmethod
    def manual_seed(_seed):
        return None

    @staticmethod
    def inference_mode():
        class _Noop:
            def __enter__(self):
                return None

            def __exit__(self, *_):
                return False

        return _Noop()


class GroupingTests(unittest.TestCase):
    """Each group must belong to the prompt it was generated for.

    `generate` returns one flat list for the whole batch, `num_return_sequences`
    rows per prompt in prompt order. Reshaping that wrongly does not raise; it
    scores prompt A's candidates against prompt B's gold answer, and every
    liveness verdict after that is measuring the wrong thing.
    """

    def _run(self, *, prompts, group_size):
        emitted = [
            f"{prompt}#{index}" for prompt in prompts for index in range(group_size)
        ]

        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 0
            padding_side = "left"

            def __call__(self, batch, **kwargs):
                return FakeBatch(input_ids=FakeTensor([[1, 2, 3] for _ in batch]))

            def decode(self, row, **kwargs):
                return row[0]

        class Model:
            def generate(self, **kwargs):
                # Rows are (prompt_prefix, payload); the prefix is sliced off
                # exactly as the real path slices the prompt tokens away.
                return [["", "", "", text] for text in emitted]

        return probe.generate_group(
            loaded=Model(),
            tokenizer=Tokenizer(),
            torch=FakeTorch(),
            prompts=prompts,
            group_size=group_size,
            temperature=0.8,
            top_p=0.95,
            max_new_tokens=32,
            seed=1,
        )

    def test_every_completion_lands_in_its_own_prompts_group(self) -> None:
        groups = self._run(prompts=["A", "B", "C"], group_size=4)
        self.assertEqual(len(groups), 3)
        for prompt, group in zip(["A", "B", "C"], groups):
            self.assertEqual(len(group), 4)
            for completion in group:
                self.assertTrue(
                    completion.startswith(prompt),
                    f"{completion!r} was paired with prompt {prompt!r}",
                )

    def test_a_single_prompt_batch_still_groups_correctly(self) -> None:
        groups = self._run(prompts=["A"], group_size=8)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 8)

    def test_the_group_size_matches_the_configured_number_of_generations(self) -> None:
        config = probe.load_train_config(probe.TRAIN_CONFIG_PATH)
        groups = self._run(
            prompts=["A", "B"], group_size=config["grpo"]["num_generations"]
        )
        self.assertEqual([len(g) for g in groups], [8, 8])


class AdapterIdentityTests(unittest.TestCase):
    """A filter is only valid for the weights it was probed against.

    Paths are not evidence of that. The probe may run on a rented GPU and the
    training on a laptop, so the same weights carry different paths; and a
    checkpoint directory retrained in place carries the same path with
    different weights. Both scripts therefore hash the adapter file, and this
    pins that they hash the same thing the same way.
    """

    def _write_adapter(self, root: Path, payload: bytes) -> Path:
        adapter = root / "checkpoint-75"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_model.safetensors").write_bytes(payload)
        return adapter

    def test_the_two_scripts_hash_identical_weights_identically(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            adapter = self._write_adapter(Path(raw), b"weights-a")
            self.assertEqual(
                probe.adapter_weights_sha256(adapter),
                train_grpo._adapter_weights_sha256(adapter),
            )

    def test_different_weights_hash_differently(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = probe.adapter_weights_sha256(
                self._write_adapter(root / "one", b"weights-a")
            )
            second = probe.adapter_weights_sha256(
                self._write_adapter(root / "two", b"weights-b")
            )
            self.assertNotEqual(first, second)

    def test_a_missing_adapter_reports_none_rather_than_raising(self) -> None:
        """Plan-only runs have no weights yet; that must not be an error."""

        self.assertIsNone(probe.adapter_weights_sha256(Path("does/not/exist")))
        self.assertIsNone(train_grpo._adapter_weights_sha256(Path("does/not/exist")))


if __name__ == "__main__":
    unittest.main()
