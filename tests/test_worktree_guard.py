"""Every command that writes a `source_commit` refuses a dirty tree first.

`source_commit` is only true if the code that ran is the code in that commit.
The guard itself is tested in test_provenance.py. This file checks the wiring:
each command calls it on the path that really runs, puts the HEAD it returns in
the artifact, and leaves plan-only runs alone.
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

from evaluation.provenance import DirtyWorktreeError  # noqa: E402
from commands import build_sft  # noqa: E402
from commands import compare_models  # noqa: E402
from commands import evaluate_gsm8k  # noqa: E402
from commands import evaluate_mmlu  # noqa: E402
from commands import evaluate_order_support  # noqa: E402
from commands import generate_teacher  # noqa: E402
from commands import probe_contamination  # noqa: E402
from commands import select_checkpoint  # noqa: E402
from commands import train_grpo  # noqa: E402
from commands import train_sft  # noqa: E402
from commands import verify_masking  # noqa: E402

MODEL = "Qwen/Qwen3-1.7B"
HEAD = "c0ffee" * 6 + "c0ff"

COMMANDS_DIR = PROJECT_ROOT / "src" / "commands"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Every command module that writes a source_commit, and so must call the guard.
GUARDED = {
    "backfill_tokens.py",
    "build_sft.py",
    "compare_mmlu.py",
    "compare_models.py",
    "evaluate_gsm8k.py",
    "evaluate_mmlu.py",
    "evaluate_order_support.py",
    "failure_taxonomy.py",
    "generate_teacher.py",
    "order_support_gates.py",
    "order_support_templates.py",
    "probe_contamination.py",
    "select_checkpoint.py",
    "train_grpo.py",
    "train_sft.py",
    "verify_masking.py",
    "weight_change.py",
}
# Everything in scripts/. These must run on any tree (CI, and the manifest
# builder, which runs before the artifacts are committed), or only dispatch to
# a command module and write nothing themselves.
CLIS = {"data.py", "train.py", "evaluate.py", "analyze.py"}
NEVER = {"build_artifact_manifest.py", "run_tests_offline.py", *CLIS}


class _Refused(Exception):
    """Raised by the fake guard, standing in for a dirty tree."""


class _Loaded(Exception):
    """Raised by the fake loader: the run got past the guard."""


def _fake_modules() -> dict[str, types.ModuleType]:
    class FastLanguageModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise _Loaded

    unsloth = types.ModuleType("unsloth")
    unsloth.FastLanguageModel = FastLanguageModel
    trl = types.ModuleType("trl")
    trl.GRPOConfig = trl.GRPOTrainer = trl.SFTConfig = trl.SFTTrainer = object
    datasets = types.ModuleType("datasets")
    datasets.Dataset = object
    return {
        "unsloth": unsloth,
        "torch": types.ModuleType("torch"),
        "trl": trl,
        "datasets": datasets,
    }


# -- each command's wiring, with a fake guard --------------------------------


class ScriptWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.adapter = self.root / "adapter"
        self.adapter.mkdir()
        (self.adapter / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": MODEL}), encoding="utf-8"
        )
        (self.adapter / "adapter_model.safetensors").write_bytes(b"weights")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _main(self, module, argv: list[str], guard: mock.Mock):
        with mock.patch.dict(sys.modules, _fake_modules()), mock.patch.object(
            sys, "argv", [module.__name__, *argv]
        ), mock.patch.object(module, "require_clean_worktree", guard), mock.patch(
            "builtins.print"
        ):
            return module.main()

    def _check_run_path(self, module, argv: list[str]) -> None:
        """Planning never asks; a real run asks before any model loads."""

        planned = mock.Mock(side_effect=AssertionError("plan-only must not guard"))
        self._main(module, argv, planned)
        planned.assert_not_called()

        refused = mock.Mock(side_effect=_Refused)
        with self.assertRaises(_Refused):
            self._main(module, argv + ["--run-load", "--allow-download"], refused)
        refused.assert_called_once_with(module.PROJECT_ROOT)

    def test_evaluate_order_support(self) -> None:
        self._check_run_path(
            evaluate_order_support,
            ["--model", MODEL, "--label", "x", "--summary", str(self.root / "s.json"),
             "--episodes", str(self.root / "e.jsonl"), "--limit", "1"],
        )

    def test_evaluate_mmlu(self) -> None:
        with mock.patch.object(evaluate_mmlu, "load_questions") as questions:
            self._check_run_path(
                evaluate_mmlu,
                ["--model", MODEL, "--label", "x", "--summary", str(self.root / "s.json"),
                 "--responses", str(self.root / "r.jsonl")],
            )
        questions.assert_not_called()

    def test_train_grpo(self) -> None:
        self._check_run_path(
            train_grpo,
            ["--adapter", str(self.adapter), "--model", MODEL,
             "--output-dir", str(self.root / "out"), "--summary", str(self.root / "s.json")],
        )

    def test_train_sft(self) -> None:
        dataset = self.root / "rows.jsonl"
        dataset.write_text("{}\n", encoding="utf-8")
        self._check_run_path(
            train_sft,
            ["--dataset", str(dataset), "--model", MODEL,
             "--output-dir", str(self.root / "out"), "--summary", str(self.root / "s.json")],
        )

    def test_generate_teacher(self) -> None:
        with mock.patch.object(generate_teacher, "load_split") as tasks:
            self._check_run_path(
                generate_teacher,
                ["--model", MODEL, "--candidates", str(self.root / "c.jsonl"),
                 "--summary", str(self.root / "s.json")],
            )
        tasks.assert_not_called()

    def test_evaluate_gsm8k(self) -> None:
        with mock.patch.object(evaluate_gsm8k, "_load_tasks") as tasks:
            self._check_run_path(
                evaluate_gsm8k,
                ["--candidate", MODEL, "--output", str(self.root / "o.json"),
                 "--episodes", str(self.root / "e.jsonl")],
            )
        tasks.assert_not_called()

    def test_probe_contamination(self) -> None:
        with mock.patch.object(probe_contamination, "_load_tasks", return_value=[]), \
                mock.patch.object(probe_contamination, "_measure") as measure:
            self._check_run_path(
                probe_contamination,
                ["--candidate", MODEL, "--output", str(self.root / "o.json")],
            )
        measure.assert_not_called()

    def test_verify_masking(self) -> None:
        refused = mock.Mock(side_effect=_Refused)
        with mock.patch.object(verify_masking, "verify") as verify:
            with self.assertRaises(_Refused):
                self._main(
                    verify_masking,
                    ["--output", str(self.root / "o.json"), "--candidate", MODEL],
                    refused,
                )
        verify.assert_not_called()

    def test_build_sft_guards_the_build_but_not_the_check(self) -> None:
        rows = [{"task_id": "t", "run_index": 0, "messages": [], "messages_sha256": "x"}]
        argv = ["--candidates", str(self.root / "c.jsonl"),
                "--dataset", str(self.root / "d.jsonl"),
                "--summary", str(self.root / "s.json"),
                "--manifest", str(self.root / "m.json")]
        with mock.patch.object(build_sft, "read_candidates", return_value=[]), \
                mock.patch.object(build_sft, "select", return_value=(rows, {})), \
                mock.patch.object(build_sft, "check_split_membership"):
            unused = mock.Mock(side_effect=AssertionError("--check must not guard"))
            self._main(build_sft, argv + ["--check"], unused)
            unused.assert_not_called()

            refused = mock.Mock(side_effect=_Refused)
            with self.assertRaises(_Refused):
                self._main(build_sft, argv, refused)
        self.assertFalse((self.root / "d.jsonl").exists())
        self.assertFalse((self.root / "m.json").exists())

    def test_compare_models_records_the_guards_head(self) -> None:
        paths = []
        for arm in ("base", "treated"):
            path = self.root / f"{arm}.jsonl"
            rows = [
                {"candidate": MODEL, "rung": "R0", "task_id": f"t{t}", "run_index": r,
                 "correct": (t + r) % 2 == 0, "completions": [f"{arm}{t}{r}"]}
                for t in range(3)
                for r in range(2)
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            paths.append(path)
        summary = self.root / "s.json"
        argv = ["--baseline-episodes", str(paths[0]), "--treatment-episodes", str(paths[1]),
                "--baseline-label", "b", "--treatment-label", "t",
                "--rung", "R0", "--k", "1", "--summary", str(summary)]
        self._main(compare_models, argv, mock.Mock(return_value=HEAD))
        self.assertEqual(json.loads(summary.read_text(encoding="utf-8"))["source_commit"], HEAD)

    def test_select_checkpoint_records_the_guards_head(self) -> None:
        adapter_dir = self.root / "run"
        (adapter_dir / "checkpoint-1").mkdir(parents=True)
        (adapter_dir / "checkpoint-1" / "adapter_model.safetensors").write_bytes(b"a")
        summary = self.root / "s.json"
        argv = ["--adapter-dir", str(adapter_dir), "--base-model", MODEL,
                "--summary", str(summary), "--scratch", str(self.root / "scratch")]
        fake_score = {"checkpoint": "checkpoint-1", "path": "p", "score": 0.5,
                      "no_arithmetic_rate": 0.0, "artifact": "a"}
        with mock.patch.object(select_checkpoint, "score", return_value=fake_score):
            self._main(select_checkpoint, argv, mock.Mock(return_value=HEAD))
        self.assertEqual(json.loads(summary.read_text(encoding="utf-8"))["source_commit"], HEAD)

    def test_select_checkpoint_refuses_scratch_inside_the_repository(self) -> None:
        argv = ["--adapter-dir", str(self.root), "--base-model", MODEL,
                "--summary", str(self.root / "s.json"),
                "--scratch", str(PROJECT_ROOT / "results" / "scratch")]
        guard = mock.Mock(return_value=HEAD)
        with self.assertRaises(DirtyWorktreeError):
            self._main(select_checkpoint, argv, guard)
        guard.assert_not_called()
        self.assertFalse((PROJECT_ROOT / "results" / "scratch").exists())


# -- a sweep, so a new command cannot skip the guard ------------------------


def _guard_assignments(tree: ast.AST) -> int:
    """How many times the guard's return value is stored as source_commit."""

    found = 0
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None) == "require_clean_worktree"
        ):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "source_commit":
                found += 1
            elif (
                isinstance(target, ast.Subscript)
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "source_commit"
            ):
                found += 1
    return found


class EveryWriterIsGuardedTests(unittest.TestCase):
    def test_every_command_that_writes_a_source_commit_stores_the_guards_head(self) -> None:
        commands = sorted(COMMANDS_DIR.glob("*.py"))
        writers = {
            path.name
            for path in commands
            if '"source_commit"' in path.read_text(encoding="utf-8")
        }
        self.assertEqual(writers, GUARDED)
        for name in sorted(GUARDED):
            tree = ast.parse((COMMANDS_DIR / name).read_text(encoding="utf-8"))
            self.assertEqual(_guard_assignments(tree), 1, name)

    def test_scripts_holds_only_tools_that_never_guard(self) -> None:
        # A new script in scripts/ would escape the sweep above, so the folder
        # is pinned to the tools that are known not to need the guard.
        self.assertEqual({path.name for path in SCRIPTS_DIR.glob("*.py")}, NEVER)

    def test_tools_that_must_run_on_any_tree_never_guard(self) -> None:
        for name in sorted(NEVER):
            text = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
            self.assertNotIn("require_clean_worktree", text, name)

    def test_the_clis_write_no_source_commit(self) -> None:
        for name in sorted(CLIS):
            text = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
            self.assertNotIn("source_commit", text, name)


if __name__ == "__main__":
    unittest.main()
