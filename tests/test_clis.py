"""The four scripts/ CLIs list their subcommands and hand each one its argv.

The CLIs only dispatch. Each subcommand is a module in `src/commands`, imported
only when it is chosen, so asking for a CPU analysis never imports the GPU
stack.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from commands import cli  # noqa: E402

SCRIPTS = PROJECT_ROOT / "scripts"
CLIS = ("data", "train", "evaluate", "analyze")
CPU_COMMANDS = (
    "compare_models",
    "compare_mmlu",
    "failure_taxonomy",
    "order_support_gates",
    "order_support_templates",
    "backfill_tokens",
    "plots",
)
GPU_STACK = ("torch", "unsloth", "trl", "transformers")


def _load(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(f"cli_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=False,
    )


class HelpTests(unittest.TestCase):
    def test_help_lists_every_subcommand_with_its_help(self) -> None:
        for name in CLIS:
            commands = _load(name).COMMANDS
            result = _run(str(SCRIPTS / f"{name}.py"), "--help")
            with self.subTest(cli=name):
                self.assertEqual(result.returncode, 0, result.stderr)
                for subcommand, (_, help_text) in commands.items():
                    self.assertIn(subcommand, result.stdout)
                    self.assertIn(help_text, result.stdout)

    def test_a_subcommand_prints_its_own_help(self) -> None:
        result = _run(str(SCRIPTS / "analyze.py"), "failure-taxonomy", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage: analyze.py failure-taxonomy", result.stdout)
        self.assertIn("--output", result.stdout)

    def test_an_unknown_subcommand_exits_non_zero(self) -> None:
        result = _run(str(SCRIPTS / "analyze.py"), "no-such-command")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown subcommand", result.stderr)

    def test_no_subcommand_exits_non_zero(self) -> None:
        result = _run(str(SCRIPTS / "train.py"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("subcommands:", result.stderr)


class RegistryTests(unittest.TestCase):
    def test_every_subcommand_names_a_command_module_with_a_main(self) -> None:
        # Parsed, not imported: the GPU modules import torch at load time.
        for name in CLIS:
            for subcommand, (target, _) in _load(name).COMMANDS.items():
                if not isinstance(target, str):
                    continue
                with self.subTest(cli=name, subcommand=subcommand):
                    path = PROJECT_ROOT / "src" / "commands" / f"{target}.py"
                    self.assertTrue(path.is_file(), path.name)
                    tree = ast.parse(path.read_text(encoding="utf-8"))
                    functions = {
                        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
                    }
                    self.assertIn("main", functions)

    def test_every_analyze_subcommand_answers_help(self) -> None:
        # analyze.py is the CPU CLI, so each of its modules can be loaded here.
        # verify-manifest runs in its own process and has its own test below.
        for subcommand, (target, _) in _load("analyze").COMMANDS.items():
            if not isinstance(target, str):
                continue
            out = io.StringIO()
            with self.subTest(subcommand=subcommand):
                with mock.patch.object(sys, "argv", ["analyze.py"]), \
                        contextlib.redirect_stdout(out), \
                        self.assertRaises(SystemExit) as exited:
                    cli.run("analyze.py", "test", {subcommand: (target, "")}, [subcommand, "--help"])
                self.assertEqual(exited.exception.code, 0)
                self.assertIn(f"usage: analyze.py {subcommand}", out.getvalue())

    def test_verify_manifest_passes_on_the_check_exit_code(self) -> None:
        result = _run(str(SCRIPTS / "analyze.py"), "verify-manifest")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("manifest reproduces exactly", result.stdout)

    def test_cpu_commands_never_import_the_gpu_stack(self) -> None:
        probe = (
            "import importlib, json, sys\n"
            f"sys.path.insert(0, {str(PROJECT_ROOT / 'src')!r})\n"
            f"for name in {CPU_COMMANDS!r}:\n"
            "    importlib.import_module('commands.' + name)\n"
            f"print(json.dumps(sorted(m for m in {GPU_STACK!r} if m in sys.modules)))\n"
        )
        result = _run("-c", probe)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [])


class DispatchTests(unittest.TestCase):
    def _fake(self, main) -> types.ModuleType:
        module = types.ModuleType("commands.fake")
        module.main = main
        return module

    def test_the_rest_of_argv_reaches_the_module_untouched(self) -> None:
        seen: list[list[str]] = []
        fake = self._fake(lambda argv: seen.append(argv) or 7)
        commands = {"go": ("fake", "a fake command")}
        with mock.patch.object(cli.importlib, "import_module", return_value=fake) as load, \
                mock.patch.object(sys, "argv", ["x.py"]):
            code = cli.run("x.py", "test", commands, ["go", "--flag", "value", "rest"])
            self.assertEqual(sys.argv[0], "x.py go")
        load.assert_called_once_with("commands.fake")
        self.assertEqual(seen, [["--flag", "value", "rest"]])
        self.assertEqual(code, 7)

    def test_an_unchosen_module_is_never_imported(self) -> None:
        fake = self._fake(lambda argv: 0)
        commands = {"go": ("fake", "chosen"), "other": ("other", "not chosen")}
        with mock.patch.object(cli.importlib, "import_module", return_value=fake) as load, \
                mock.patch.object(sys, "argv", ["x.py"]):
            cli.run("x.py", "test", commands, ["go"])
        load.assert_called_once_with("commands.fake")

    def test_a_refusal_exits_1_with_one_line(self) -> None:
        class Refused(RuntimeError):
            pass

        def main(argv):
            raise Refused("inputs do not match")

        fake = self._fake(main)
        fake.REFUSALS = (Refused,)
        with mock.patch.object(cli.importlib, "import_module", return_value=fake), \
                mock.patch.object(sys, "argv", ["x.py"]), \
                mock.patch.object(sys, "stderr") as stderr:
            code = cli.run("x.py", "test", {"go": ("fake", "")}, ["go"])
        self.assertEqual(code, 1)
        written = "".join(call.args[0] for call in stderr.write.call_args_list)
        self.assertEqual(written, "refused: inputs do not match\n")

    def test_other_errors_still_raise(self) -> None:
        def main(argv):
            raise ValueError("a bug, not a refusal")

        with mock.patch.object(cli.importlib, "import_module", return_value=self._fake(main)), \
                mock.patch.object(sys, "argv", ["x.py"]):
            with self.assertRaises(ValueError):
                cli.run("x.py", "test", {"go": ("fake", "")}, ["go"])

    def test_a_function_target_gets_the_rest_of_argv(self) -> None:
        seen: list[list[str]] = []
        commands = {"check": (lambda argv: seen.append(argv) or 3, "a function")}
        self.assertEqual(cli.run("x.py", "test", commands, ["check", "-v"]), 3)
        self.assertEqual(seen, [["-v"]])


if __name__ == "__main__":
    unittest.main()
