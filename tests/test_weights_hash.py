"""Every evaluation names the exact adapter weights it loaded.

A checkpoint directory keeps its name when it is retrained in place, so a path
says nothing about which weights produced a number. The hash does. It is taken
the way measure_weight_change.py always took it, SHA-256 over the whole
`adapter_model.safetensors`, so a new record matches an old one by value.

No model is loaded. Phase B and MMLU write their record before any load, and
checkpoint selection runs with a fake evaluation runner.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.provenance import (  # noqa: E402
    adapter_weights_sha256,
    pinned_revision,
    portable_path,
)
from scripts import run_phase_b_eval  # noqa: E402
from scripts import run_utility_eval  # noqa: E402
from scripts import select_checkpoint  # noqa: E402

MODEL = "Qwen/Qwen3-1.7B"
REGISTRY = PROJECT_ROOT / "configs" / "model_candidates.json"


def _adapter(directory: Path, weights: bytes) -> Path:
    directory.mkdir(parents=True)
    (directory / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": MODEL}), encoding="utf-8"
    )
    (directory / "adapter_model.safetensors").write_bytes(weights)
    return directory


class HashTests(unittest.TestCase):
    def test_the_hash_is_sha256_of_the_whole_weights_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _adapter(Path(tmp) / "a", b"some weights")
            self.assertEqual(
                adapter_weights_sha256(adapter),
                hashlib.sha256(b"some weights").hexdigest(),
            )

    def test_the_base_model_has_no_adapter_hash(self) -> None:
        self.assertIsNone(adapter_weights_sha256(None))

    def test_a_missing_weights_file_is_an_error_not_a_null(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                adapter_weights_sha256(Path(tmp))


class EvaluationRecordTests(unittest.TestCase):
    """The field lands in the JSON each evaluator writes."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.adapter = _adapter(self.root / "checkpoint-200", b"trained")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _record(self, module, argv: list[str]) -> dict:
        summary = self.root / "summary.json"
        with mock.patch.object(
            sys, "argv", [module.__name__, "--summary", str(summary), *argv]
        ), mock.patch("builtins.print"):
            module.main()
        return json.loads(summary.read_text(encoding="utf-8"))

    def _check(self, module, argv: list[str]) -> None:
        trained = self._record(module, argv + ["--adapter", str(self.adapter)])
        self.assertEqual(trained["weights_sha256"], hashlib.sha256(b"trained").hexdigest())
        self.assertEqual(trained["adapter"], portable_path(self.adapter, PROJECT_ROOT))

        base = self._record(module, argv)
        self.assertIsNone(base["weights_sha256"])
        self.assertIsNone(base["adapter"])
        self.assertEqual(base["model"]["revision"], pinned_revision(MODEL, REGISTRY))

    def test_phase_b(self) -> None:
        self._check(
            run_phase_b_eval,
            ["--model", MODEL, "--label", "x", "--episodes", str(self.root / "e.jsonl"),
             "--limit", "1"],
        )

    def test_mmlu(self) -> None:
        self._check(
            run_utility_eval,
            ["--model", MODEL, "--label", "x", "--responses", str(self.root / "r.jsonl")],
        )


class SelectionRecordTests(unittest.TestCase):
    """Each candidate checkpoint carries its own hash."""

    def test_every_candidate_names_its_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            weights = {"checkpoint-100": b"early", "checkpoint-200": b"late"}
            for name, data in weights.items():
                _adapter(run / name, data)
            summary = root / "selection.json"

            def runner(command, **kwargs):
                output = Path(command[command.index("--output") + 1])
                rung = command[command.index("--rung") + 1]
                score = 0.9 if "checkpoint-200" in command[command.index("--adapter") + 1] else 0.1
                output.write_text(
                    json.dumps({
                        "executed": True,
                        "results": [{"rungs": {rung: {
                            "metrics": {"pass^1": score}, "no_arithmetic_rate": 0.0,
                        }}}],
                    }),
                    encoding="utf-8",
                )
                return mock.Mock(returncode=0)

            argv = ["select_checkpoint", "--adapter-dir", str(run), "--base-model", MODEL,
                    "--summary", str(summary), "--scratch", str(root / "scratch")]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                select_checkpoint.subprocess, "run", side_effect=runner
            ), mock.patch.object(
                select_checkpoint, "require_clean_worktree", return_value="0" * 40
            ), mock.patch("builtins.print"):
                select_checkpoint.main()

            record = json.loads(summary.read_text(encoding="utf-8"))
        hashes = {c["checkpoint"]: c["weights_sha256"] for c in record["candidates"]}
        self.assertEqual(
            hashes, {name: hashlib.sha256(data).hexdigest() for name, data in weights.items()}
        )
        self.assertEqual(record["selected"]["checkpoint"], "checkpoint-200")
        self.assertEqual(record["selected"]["weights_sha256"], hashlib.sha256(b"late").hexdigest())


if __name__ == "__main__":
    unittest.main()
