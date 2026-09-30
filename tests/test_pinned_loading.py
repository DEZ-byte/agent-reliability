"""Every model load names the pinned base revision, or it does not happen.

A result is only reproducible if the weights under it are. The Hub's default
branch can move, and without `use_exact_model_name` Unsloth swaps the name for a
pre-quantised mirror that has no pinned revision at all. Handed an adapter
directory, Unsloth also drops `revision` for the base, so a script that loads
the adapter directory ran on whatever the default branch held that day.

Nothing here loads a model. A fake `unsloth` records what each script would
have passed and stops the run there.
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.provenance import (  # noqa: E402
    UnpinnedModelError,
    pinned_load_kwargs,
    pinned_revision,
)
from scripts import probe_prompt_variance  # noqa: E402
from scripts import run_phase_b_eval  # noqa: E402
from scripts import run_utility_eval  # noqa: E402
from scripts import train_grpo  # noqa: E402

REGISTRY = PROJECT_ROOT / "configs" / "model_candidates.json"
MODEL = "Qwen/Qwen3-1.7B"
# The frozen smoke script is pinned by digest and cannot be edited.
FROZEN = {"smoke_models.py"}


def _registered_revision(model_id: str) -> str:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    for entries in registry["roles"].values():
        for entry in entries:
            if entry["id"] == model_id:
                return entry["revision"]
    raise AssertionError(f"{model_id} is not in the registry")


def _adapter(root: Path, base: str = MODEL) -> Path:
    adapter = root / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base}), encoding="utf-8"
    )
    return adapter


class _Stop(Exception):
    """Raised by the fake loader so the script goes no further."""


class RegistryTests(unittest.TestCase):
    def test_a_registered_model_gets_its_pinned_revision(self) -> None:
        self.assertEqual(
            pinned_revision(MODEL, REGISTRY), _registered_revision(MODEL)
        )

    def test_an_unregistered_model_is_refused(self) -> None:
        with self.assertRaises(UnpinnedModelError):
            pinned_revision("Qwen/Not-A-Model", REGISTRY)

    def test_a_registered_model_with_no_revision_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            path.write_text(
                json.dumps({"roles": {"x": [{"id": MODEL, "revision": None}]}}),
                encoding="utf-8",
            )
            with self.assertRaises(UnpinnedModelError):
                pinned_revision(MODEL, path)

    def test_the_kwargs_name_the_base_its_revision_and_the_exact_name(self) -> None:
        kwargs = pinned_load_kwargs(MODEL, REGISTRY)
        self.assertEqual(kwargs["model_name"], MODEL)
        self.assertEqual(kwargs["revision"], _registered_revision(MODEL))
        self.assertIs(kwargs["use_exact_model_name"], True)
        self.assertNotIn("tokenizer_name", kwargs)

    def test_an_adapter_keeps_its_tokenizer_and_never_becomes_the_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _adapter(Path(tmp))
            kwargs = pinned_load_kwargs(MODEL, REGISTRY, adapter)
        self.assertEqual(kwargs["model_name"], MODEL)
        self.assertEqual(kwargs["tokenizer_name"], str(adapter))

    def test_an_adapter_trained_on_another_base_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = _adapter(Path(tmp), base="Qwen/Qwen3-4B")
            with self.assertRaises(UnpinnedModelError):
                pinned_load_kwargs(MODEL, REGISTRY, adapter)

    def test_a_directory_with_no_adapter_config_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(UnpinnedModelError):
                pinned_load_kwargs(MODEL, REGISTRY, Path(tmp))


class ScriptWiringTests(unittest.TestCase):
    """Each script hands the pinned kwargs to the loader it really calls."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.adapter = _adapter(self.root)
        self.calls: list[dict] = []

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _fakes(self) -> dict[str, types.ModuleType]:
        calls = self.calls

        class FastLanguageModel:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                calls.append({"args": args, **kwargs})
                raise _Stop

        unsloth = types.ModuleType("unsloth")
        unsloth.FastLanguageModel = FastLanguageModel
        trl = types.ModuleType("trl")
        trl.GRPOConfig = trl.GRPOTrainer = object
        return {"unsloth": unsloth, "torch": types.ModuleType("torch"), "trl": trl}

    def _run(self, module, argv: list[str]) -> dict:
        # The worktree guard has its own tests; here it must not stop the run
        # before the loader is reached.
        with mock.patch.dict(sys.modules, self._fakes()), mock.patch.object(
            sys, "argv", [module.__name__, *argv]
        ), mock.patch.object(module, "require_clean_worktree", return_value="0" * 40):
            with self.assertRaises(_Stop):
                module.main()
        self.assertEqual(len(self.calls), 1)
        return self.calls[0]

    def _assert_pinned(self, call: dict) -> None:
        self.assertEqual(call["args"], ())
        self.assertEqual(call["model_name"], MODEL)
        self.assertEqual(call["revision"], _registered_revision(MODEL))
        self.assertIs(call["use_exact_model_name"], True)

    def _common(self) -> list[str]:
        return [
            "--model",
            MODEL,
            "--summary",
            str(self.root / "s.json"),
            "--run-load",
            "--allow-download",
        ]

    def test_phase_b_eval_with_and_without_an_adapter(self) -> None:
        extra = ["--label", "x", "--episodes", str(self.root / "e.jsonl"), "--limit", "1"]
        self._assert_pinned(self._run(run_phase_b_eval, self._common() + extra))
        self.calls.clear()
        call = self._run(
            run_phase_b_eval, self._common() + extra + ["--adapter", str(self.adapter)]
        )
        self._assert_pinned(call)
        self.assertEqual(call["tokenizer_name"], str(self.adapter))

    def test_utility_eval_with_and_without_an_adapter(self) -> None:
        extra = ["--label", "x", "--responses", str(self.root / "r.jsonl")]
        with mock.patch.object(run_utility_eval, "load_questions", return_value=[]):
            self._assert_pinned(self._run(run_utility_eval, self._common() + extra))
            self.calls.clear()
            call = self._run(
                run_utility_eval,
                self._common() + extra + ["--adapter", str(self.adapter)],
            )
        self._assert_pinned(call)
        self.assertEqual(call["tokenizer_name"], str(self.adapter))

    def test_train_grpo(self) -> None:
        extra = ["--adapter", str(self.adapter), "--output-dir", str(self.root / "out")]
        self._assert_pinned(self._run(train_grpo, self._common() + extra))

    def test_probe_prompt_variance(self) -> None:
        extra = ["--adapter", str(self.adapter), "--output", str(self.root / "o.jsonl")]
        self._assert_pinned(self._run(probe_prompt_variance, self._common() + extra))

    def test_an_unregistered_model_never_reaches_the_loader(self) -> None:
        argv = self._common() + ["--label", "x", "--episodes", str(self.root / "e.jsonl")]
        argv[1] = "Qwen/Not-A-Model"
        with mock.patch.dict(sys.modules, self._fakes()), mock.patch.object(
            sys, "argv", ["run_phase_b_eval", *argv]
        ):
            with self.assertRaises(UnpinnedModelError):
                run_phase_b_eval.main()
        self.assertEqual(self.calls, [])


class EveryLoaderIsPinnedTests(unittest.TestCase):
    """A static sweep, so a new script cannot quietly load an unpinned model."""

    def test_every_fast_language_model_load_is_pinned(self) -> None:
        checked = 0
        for path in sorted((PROJECT_ROOT / "scripts").glob("*.py")):
            if path.name in FROZEN:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "from_pretrained"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "FastLanguageModel"
                ):
                    continue
                checked += 1
                where = f"{path.name}:{node.lineno}"
                helper = [
                    k
                    for k in node.keywords
                    if k.arg is None
                    and isinstance(k.value, ast.Call)
                    and getattr(k.value.func, "id", None) == "pinned_load_kwargs"
                ]
                if helper:
                    continue
                named = {k.arg: k.value for k in node.keywords if k.arg}
                self.assertIn("revision", named, where)
                exact = named.get("use_exact_model_name")
                self.assertTrue(
                    isinstance(exact, ast.Constant) and exact.value is True, where
                )
        # Eight scripts load a model this way today; finding none would mean
        # the sweep stopped looking, not that every load is pinned.
        self.assertGreaterEqual(checked, 8)


if __name__ == "__main__":
    unittest.main()
