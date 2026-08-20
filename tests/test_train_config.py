"""The training config must match what was pre-registered, and admit what is not yet measured.

BLUEPRINT_v2 section 7.4 fixes several training values before any measurement
exists. Those are asserted here against the blueprint rather than against
themselves, so that editing the config to match a run cannot pass silently.

The remaining values are null on purpose. The risk with a null is that a script
falls back to a default nobody recorded, producing a run whose artifact names a
config that did not determine it. `PENDING` below is the full list of values
still awaiting measurement; a new null that nobody decided to add fails here.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.config import (  # noqa: E402
    TrainConfigError,
    config_hash_prefix,
    config_sha256,
    load_train_config,
    pending_keys,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "train_config.yaml"
EVAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "eval.yaml"

# Every value that is deliberately unset, with the measurement that will set it.
PENDING = {
    "retention.min_question_match_ratio",
    "retention.max_rows",
    "format_grounding.fraction",
}


class PreRegisteredValueTests(unittest.TestCase):
    """Asserted against BLUEPRINT_v2 section 7.4, not against the file itself."""

    def setUp(self) -> None:
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_the_config_is_strict_json(self) -> None:
        """A YAML 1.2 subset, so the standard library can load it (D-021)."""

        self.assertIsInstance(self.config, dict)
        self.assertEqual(self.config["schema_version"], 1)

    def test_lora_rank_and_alpha_match_the_blueprint(self) -> None:
        lora = self.config["sft"]["lora"]
        self.assertEqual(lora["r"], 16)
        self.assertEqual(lora["alpha"], 32)

    def test_training_is_four_bit(self) -> None:
        self.assertIs(self.config["sft"]["load_in_4bit"], True)

    def test_epochs_sit_inside_the_pre_registered_band(self) -> None:
        """Section 7.4 gives SFT two to three epochs."""

        self.assertIn(self.config["sft"]["epochs"], (2, 3))

    def test_every_saved_checkpoint_can_be_evaluated_on_dev(self) -> None:
        """Selection must not fall back to a cheaper signal than dev.

        If more checkpoints are saved than can be evaluated, the operator ranks
        the rest by training loss, which is selection on the training set.
        """

        sft = self.config["sft"]
        effective_batch = (
            sft["per_device_train_batch_size"] * sft["gradient_accumulation_steps"]
        )
        rows = 1000
        steps = (rows / effective_batch) * sft["epochs"]
        self.assertLessEqual(steps / sft["save_steps"], 12)
        self.assertEqual(self.config["selection"]["checkpoints_evaluated"], "all_saved")


class SelectionRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_checkpoints_are_selected_on_dev_only(self) -> None:
        self.assertEqual(self.config["selection"]["split"], "dev")

    def test_the_metric_and_rung_are_pinned_as_literals(self) -> None:
        """Written before any dev number, so a tie cannot be broken after it."""

        selection = self.config["selection"]
        self.assertEqual(selection["metric"], "pass^1")
        self.assertEqual(selection["rung"], "R0")


class GenerationConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_generation_seeds_cannot_collide_with_evaluation_seeds(self) -> None:
        """A shared seed base would make a training rollout reproduce an
        evaluation episode by accident, which is a leak nobody would see."""

        evaluation = json.loads(EVAL_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertNotEqual(
            self.config["generation"]["seed_base"],
            evaluation["runs"]["seed_base"],
        )

    def test_generation_runs_r0_only(self) -> None:
        """D-069: the loop ends on a successful call, so R1 cannot reach the
        dominant failure and yields almost no recovery trajectories."""

        self.assertEqual(self.config["generation"]["rung"], "R0")


class PendingValueTests(unittest.TestCase):
    def test_exactly_the_expected_values_are_still_unmeasured(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(set(pending_keys(config)), PENDING)

    def test_requiring_an_unmeasured_value_refuses_the_run(self) -> None:
        with self.assertRaises(TrainConfigError) as caught:
            load_train_config(CONFIG_PATH, require=["retention.max_rows"])
        self.assertIn("retention.max_rows", str(caught.exception))

    def test_requiring_only_measured_values_loads(self) -> None:
        config = load_train_config(
            CONFIG_PATH, require=["sft.lora.r", "generation.rung"]
        )
        self.assertEqual(config["sft"]["lora"]["r"], 16)

    def test_a_missing_key_is_an_error_rather_than_a_silent_none(self) -> None:
        with self.assertRaises(TrainConfigError):
            load_train_config(CONFIG_PATH, require=["sft.nonexistent"])


class ConfigHashTests(unittest.TestCase):
    def test_the_checkpoint_name_prefix_comes_from_the_config_hash(self) -> None:
        """Section 7.4 names checkpoints `<model>-<method>-<confighash>-step<N>`."""

        digest = config_sha256(CONFIG_PATH)
        self.assertEqual(len(digest), 64)
        self.assertEqual(config_hash_prefix(CONFIG_PATH), digest[:8])


if __name__ == "__main__":
    unittest.main()
