"""Every script that writes a `source_commit` refuses a dirty tree first.

`source_commit` is only true if the code that ran is the code in that commit.
The guard itself is tested in test_provenance.py. This file checks the wiring:
each script calls it on the path that really runs, puts the HEAD it returns in
the artifact, and leaves plan-only runs alone.

The pipeline gets a real throwaway repository. Its stages are subprocesses that
each refuse a dirty tree, so one stage writing into the repository would block
the next. The chain is driven end to end with fake stages to show it cannot.
"""

from __future__ import annotations

import ast
import io
import json
import subprocess
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
    DirtyWorktreeError,
    require_clean_worktree,
    require_outside_worktree,
)
from scripts import build_sft_dataset  # noqa: E402
from scripts import compare_arms  # noqa: E402
from scripts import generate_sft_trajectories  # noqa: E402
from scripts import probe_contamination  # noqa: E402
from scripts import probe_prompt_variance  # noqa: E402
from scripts import run_phase_a_baseline  # noqa: E402
from scripts import run_phase_b_eval  # noqa: E402
from scripts import run_primary_arm_pipeline as pipeline  # noqa: E402
from scripts import run_utility_eval  # noqa: E402
from scripts import select_checkpoint  # noqa: E402
from scripts import train_grpo  # noqa: E402
from scripts import train_sft  # noqa: E402
from scripts import verify_masking  # noqa: E402

MODEL = "Qwen/Qwen3-1.7B"
HEAD = "c0ffee" * 6 + "c0ff"

# Every script that writes a source_commit, and so must call the guard.
GUARDED = {
    "analyse_phase_b_gates.py",
    "build_sft_dataset.py",
    "compare_arms.py",
    "compare_utility.py",
    "generate_sft_trajectories.py",
    "measure_weight_change.py",
    "probe_contamination.py",
    "probe_prompt_variance.py",
    "run_phase_a_baseline.py",
    "run_phase_b_eval.py",
    "run_primary_arm_pipeline.py",
    "run_utility_eval.py",
    "select_checkpoint.py",
    "train_grpo.py",
    "train_sft.py",
    "verify_masking.py",
}
# These write no source_commit, or must run on any tree (CI, the smoke probe,
# and the manifest builder, which runs before the artifacts are committed).
NEVER = {
    "build_artifact_manifest.py",
    "probe_smoke_environment.py",
    "run_tests_offline.py",
    "smoke_models.py",
}


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


class _TempRepo:
    """A throwaway git repository with one commit."""

    def __init__(self, root: Path) -> None:
        self.root = root
        for args in (
            ("init", "-q"),
            ("config", "core.autocrlf", "false"),
            ("config", "user.name", "Test"),
            ("config", "user.email", "test@example.invalid"),
        ):
            self.git(*args)
        (root / "code.py").write_text("x = 1\n", encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "start")

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=self.root, capture_output=True, text=True, check=True
        ).stdout.strip()


# -- the pipeline chain, against a real repository --------------------------

OUTPUT_FILES = ("--summary", "--output", "--episodes", "--candidates", "--dataset", "--manifest")
OUTPUT_DIRS = ("--output-dir", "--scratch")


class PipelineChainTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        (base / "repo").mkdir()
        self.repo = _TempRepo(base / "repo")
        self.run_dir = base / "run"
        self.stages: list[str] = []

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _args(self, run_dir: Path, *extra: str) -> list[str]:
        return [
            "--run-dir", str(run_dir),
            "--teacher", "Qwen/Qwen3-4B",
            "--teacher-deviation", "none",
            "--student", MODEL,
            "--seed", "7",
            "--skip-comparator",
            *extra,
        ]

    def _child(self, leak_from: str | None = None):
        """A fake stage: refuses a dirty tree, then writes its outputs."""

        def popen(command, **kwargs):
            script = Path(command[1]).name
            self.stages.append(script)
            process = types.SimpleNamespace(wait=lambda: 0)
            try:
                require_clean_worktree(self.repo.root)
            except DirtyWorktreeError as error:
                process.stdout, process.returncode = io.StringIO(f"{error}\n"), 1
                return process
            for flag, value in zip(command, command[1:]):
                path = Path(value)
                if flag in OUTPUT_DIRS:
                    path.mkdir(parents=True, exist_ok=True)
                elif flag in OUTPUT_FILES:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    payload = {
                        "executed": True,
                        "results": [],
                        "selected": {"path": str(path.parent / "checkpoint-1")},
                        "comparisons": [],
                        "treatment_label": "t",
                        "baseline_label": "b",
                    }
                    path.write_text(
                        json.dumps(payload) if path.suffix == ".json" else "",
                        encoding="utf-8",
                    )
            if script == leak_from:
                (self.repo.root / "results").mkdir(exist_ok=True)
                (self.repo.root / "results" / "leaked.json").write_text("{}", encoding="utf-8")
            process.stdout, process.returncode = io.StringIO("done\n"), 0
            return process

        return popen

    def _main(self, argv: list[str], popen) -> int:
        # Only the pipeline's own view of subprocess is replaced. The guard
        # inside each fake stage still runs real git.
        fake_subprocess = types.SimpleNamespace(
            run=subprocess.run, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT, Popen=popen
        )
        with mock.patch.object(pipeline, "PROJECT_ROOT", self.repo.root), mock.patch.object(
            pipeline, "subprocess", fake_subprocess
        ), mock.patch("builtins.print"):
            return pipeline.main(argv)

    def test_a_chain_that_writes_outside_the_repository_passes_every_guard(self) -> None:
        self.assertEqual(self._main(self._args(self.run_dir), self._child()), 0)
        self.assertEqual(
            self.stages,
            [
                "generate_sft_trajectories.py",
                "build_sft_dataset.py",
                "run_phase_a_baseline.py",
                "train_sft.py",
                "select_checkpoint.py",
                "run_phase_a_baseline.py",
                "compare_arms.py",
            ],
        )
        (summary,) = (self.run_dir / "results").glob("summary-*.json")
        recorded = json.loads(summary.read_text(encoding="utf-8"))["source_commit"]
        self.assertEqual(recorded, self.repo.git("rev-parse", "HEAD"))

    def test_a_stage_that_writes_into_the_repository_blocks_the_next_one(self) -> None:
        """The trap the entry checks exist to prevent."""

        code = self._main(
            self._args(self.run_dir), self._child(leak_from="build_sft_dataset.py")
        )
        self.assertEqual(code, 1)
        self.assertEqual(self.stages[-1], "run_phase_a_baseline.py")
        self.assertEqual(len(self.stages), 3)

    def test_a_run_dir_inside_the_repository_is_refused_before_any_stage(self) -> None:
        with self.assertRaises(DirtyWorktreeError):
            self._main(self._args(self.repo.root / "runs"), self._child())
        self.assertEqual(self.stages, [])

    def test_a_dirty_tree_is_refused_before_any_stage(self) -> None:
        (self.repo.root / "stray.py").write_text("y = 1\n", encoding="utf-8")
        with self.assertRaises(DirtyWorktreeError):
            self._main(self._args(self.run_dir), self._child())
        self.assertEqual(self.stages, [])

    def test_a_dry_run_needs_no_clean_tree(self) -> None:
        (self.repo.root / "stray.py").write_text("y = 1\n", encoding="utf-8")
        self.assertEqual(self._main(self._args(self.run_dir, "--dry-run"), self._child()), 0)
        self.assertEqual(self.stages, [])

    def test_an_output_directory_outside_the_repository_is_allowed(self) -> None:
        require_outside_worktree(self.run_dir, self.repo.root)
        with self.assertRaises(DirtyWorktreeError):
            require_outside_worktree(self.repo.root / "results", self.repo.root)


# -- each script's wiring, with a fake guard ---------------------------------


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

    def test_run_phase_b_eval(self) -> None:
        self._check_run_path(
            run_phase_b_eval,
            ["--model", MODEL, "--label", "x", "--summary", str(self.root / "s.json"),
             "--episodes", str(self.root / "e.jsonl"), "--limit", "1"],
        )

    def test_run_utility_eval(self) -> None:
        with mock.patch.object(run_utility_eval, "load_questions") as questions:
            self._check_run_path(
                run_utility_eval,
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

    def test_probe_prompt_variance(self) -> None:
        self._check_run_path(
            probe_prompt_variance,
            ["--adapter", str(self.adapter), "--model", MODEL,
             "--output", str(self.root / "o.jsonl"), "--summary", str(self.root / "s.json")],
        )

    def test_generate_sft_trajectories(self) -> None:
        with mock.patch.object(generate_sft_trajectories, "load_split") as tasks:
            self._check_run_path(
                generate_sft_trajectories,
                ["--model", MODEL, "--candidates", str(self.root / "c.jsonl"),
                 "--summary", str(self.root / "s.json")],
            )
        tasks.assert_not_called()

    def test_run_phase_a_baseline(self) -> None:
        with mock.patch.object(run_phase_a_baseline, "_load_tasks") as tasks:
            self._check_run_path(
                run_phase_a_baseline,
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

    def test_build_sft_dataset_guards_the_build_but_not_the_check(self) -> None:
        rows = [{"task_id": "t", "run_index": 0, "messages": [], "messages_sha256": "x"}]
        argv = ["--candidates", str(self.root / "c.jsonl"),
                "--dataset", str(self.root / "d.jsonl"),
                "--summary", str(self.root / "s.json"),
                "--manifest", str(self.root / "m.json")]
        with mock.patch.object(build_sft_dataset, "read_candidates", return_value=[]), \
                mock.patch.object(build_sft_dataset, "select", return_value=(rows, {})), \
                mock.patch.object(build_sft_dataset, "check_split_membership"):
            unused = mock.Mock(side_effect=AssertionError("--check must not guard"))
            self._main(build_sft_dataset, argv + ["--check"], unused)
            unused.assert_not_called()

            refused = mock.Mock(side_effect=_Refused)
            with self.assertRaises(_Refused):
                self._main(build_sft_dataset, argv, refused)
        self.assertFalse((self.root / "d.jsonl").exists())
        self.assertFalse((self.root / "m.json").exists())

    def test_compare_arms_records_the_guards_head(self) -> None:
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
        self._main(compare_arms, argv, mock.Mock(return_value=HEAD))
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


# -- a sweep, so a new script cannot skip the guard -------------------------


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
    def test_every_script_that_writes_a_source_commit_stores_the_guards_head(self) -> None:
        scripts = sorted((PROJECT_ROOT / "scripts").glob("*.py"))
        writers = {
            path.name
            for path in scripts
            if '"source_commit"' in path.read_text(encoding="utf-8")
        }
        self.assertEqual(writers - NEVER, GUARDED)
        for name in sorted(GUARDED):
            tree = ast.parse((PROJECT_ROOT / "scripts" / name).read_text(encoding="utf-8"))
            self.assertEqual(_guard_assignments(tree), 1, name)

    def test_tools_that_must_run_on_any_tree_never_guard(self) -> None:
        for name in sorted(NEVER):
            text = (PROJECT_ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertNotIn("require_clean_worktree", text, name)


if __name__ == "__main__":
    unittest.main()
