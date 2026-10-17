from __future__ import annotations

import errno
import multiprocessing.context
import os
import unittest
from unittest import mock

import env.sandbox as sandbox_module
from env.sandbox import (
    SandboxUnavailable,
    SandboxViolation,
    _receive_payload,
    memory_limit_supported,
    run_code,
)


# Stand-in workers. A spawned worker imports its target by name, so these
# must live at module scope.
def _exit_without_result(*args: object) -> None:
    os._exit(3)


def _report_limit_setup_failure(source: str, connection, *limits: object) -> None:
    sandbox_module._send_payload(
        connection,
        {
            "kind": "violation",
            "reason": "limit_setup_failed",
            "detail": "OSError: refused",
        },
    )
    connection.close()


class _FailingConnection:
    def __init__(self, error: OSError) -> None:
        self._error = error

    def recv_bytes(self, maxlength: int | None = None) -> bytes:
        raise self._error


def _posix_cpu_limit_supported() -> bool:
    if os.name != "posix":
        return False
    try:
        import resource

        return hasattr(resource, "RLIMIT_CPU")
    except ImportError:
        return False


class SandboxTests(unittest.TestCase):
    def test_captures_output_and_final_expression(self) -> None:
        result = run_code('print("hello")\neprint("warning")\n6 * 7')

        self.assertTrue(result.succeeded)
        self.assertEqual(result.stdout, "hello\n")
        self.assertEqual(result.stderr, "warning\n")
        self.assertEqual(result.value_repr, "42")

    def test_runtime_exception_is_a_tool_result(self) -> None:
        result = run_code('print("before")\n1 / 0')

        self.assertFalse(result.succeeded)
        self.assertEqual(result.stdout, "before\n")
        self.assertEqual(result.exception_type, "ZeroDivisionError")
        self.assertIn("division by zero", result.exception_message or "")
        self.assertIn("ZeroDivisionError", result.stderr)

    def test_invalid_syntax_is_a_violation(self) -> None:
        with self.assertRaises(SandboxViolation) as caught:
            run_code("17 * (23")

        self.assertEqual(caught.exception.reason, "invalid_syntax")

    def test_a_return_outside_a_function_is_a_violation(self) -> None:
        """It parses, then fails to compile inside the worker: still the model's fault."""

        with self.assertRaises(SandboxViolation) as caught:
            run_code("return 1")

        self.assertEqual(caught.exception.reason, "worker_internal_error")

    def test_import_statements_are_rejected(self) -> None:
        samples = ("import os", "from os import path")
        for source in samples:
            with self.subTest(source=source):
                with self.assertRaises(SandboxViolation) as caught:
                    run_code(source)
                self.assertEqual(caught.exception.reason, "forbidden_syntax")

    def test_dangerous_builtins_are_not_available(self) -> None:
        samples = (
            'open("secret.txt")',
            'eval("1 + 1")',
            'exec("answer = 1")',
            'compile("1", "x", "eval")',
            '__import__("os")',
        )
        for source in samples:
            with self.subTest(source=source):
                with self.assertRaises(SandboxViolation):
                    run_code(source)

    def test_dunder_and_frame_escape_probes_are_rejected(self) -> None:
        samples = (
            "(1).__class__",
            'getattr((), "__class__")',
            "(item for item in ()).gi_frame.f_builtins",
        )
        for source in samples:
            with self.subTest(source=source):
                with self.assertRaises(SandboxViolation):
                    run_code(source)

    def test_output_limit_fires(self) -> None:
        with self.assertRaises(SandboxViolation) as caught:
            run_code('print("x" * 100)', max_output_chars=16)

        self.assertEqual(caught.exception.reason, "output_limit")

    def test_final_value_limit_fires(self) -> None:
        with self.assertRaises(SandboxViolation) as caught:
            run_code('"x" * 100', max_output_chars=16)

        self.assertEqual(caught.exception.reason, "output_limit")

    def test_exception_message_limit_fires(self) -> None:
        with self.assertRaises(SandboxViolation) as caught:
            run_code('assert False, "x" * 100', max_output_chars=16)

        self.assertEqual(caught.exception.reason, "output_limit")

    def test_total_protocol_limit_fires(self) -> None:
        with self.assertRaises(SandboxViolation) as caught:
            run_code(
                'print(chr(0x1F600) * 400_000, end="")',
                max_output_chars=500_000,
            )

        self.assertEqual(caught.exception.reason, "protocol_limit")

    def test_source_limit_fires_before_execution(self) -> None:
        with self.assertRaises(SandboxViolation) as caught:
            run_code("value = 123", max_source_chars=8)

        self.assertEqual(caught.exception.reason, "source_limit")

    def test_timeout_kills_worker(self) -> None:
        with self.assertRaises(SandboxViolation) as caught:
            run_code(
                "while True:\n    pass",
                timeout_seconds=1.0,
                cpu_time_seconds=None,
            )

        self.assertEqual(caught.exception.reason, "timeout")

    @unittest.skipUnless(
        _posix_cpu_limit_supported(), "POSIX RLIMIT_CPU is unavailable"
    )
    def test_posix_cpu_limit_fires(self) -> None:
        with self.assertRaises(SandboxViolation) as caught:
            run_code(
                "while True:\n    pass",
                timeout_seconds=10.0,
                cpu_time_seconds=1,
            )

        self.assertEqual(caught.exception.reason, "cpu_limit")

    @unittest.skipUnless(memory_limit_supported(), "no supported memory limiter")
    def test_memory_limit_fires(self) -> None:
        source = (
            "chunks = []\n"
            "while True:\n"
            "    chunks.append(bytearray(1024 * 1024))\n"
        )
        with self.assertRaises(SandboxViolation) as caught:
            run_code(
                source,
                timeout_seconds=5.0,
                memory_limit_bytes=96 * 1024 * 1024,
                cpu_time_seconds=4,
            )

        self.assertEqual(caught.exception.reason, "memory_limit")


class SandboxUnavailableTests(unittest.TestCase):
    """A broken runner is not the model's mistake and must not look like one."""

    def test_unavailable_is_not_a_kind_of_violation(self) -> None:
        # Callers treat SandboxViolation as the model's failure.
        self.assertFalse(issubclass(SandboxUnavailable, SandboxViolation))

    def test_a_worker_that_cannot_start_is_unavailable(self) -> None:
        with mock.patch.object(
            multiprocessing.context.SpawnProcess,
            "start",
            side_effect=OSError("blocked by policy"),
        ):
            with self.assertRaises(SandboxUnavailable) as caught:
                run_code("6 * 7")

        self.assertEqual(caught.exception.reason, "worker_start_failed")

    def test_a_worker_that_dies_without_a_result_is_unavailable(self) -> None:
        with mock.patch.object(sandbox_module, "_worker_main", _exit_without_result):
            with self.assertRaises(SandboxUnavailable) as caught:
                run_code("6 * 7")

        self.assertEqual(caught.exception.reason, "worker_terminated")
        self.assertIn("exit code 3", str(caught.exception))

    def test_a_worker_that_cannot_set_its_limits_is_unavailable(self) -> None:
        with mock.patch.object(
            sandbox_module, "_worker_main", _report_limit_setup_failure
        ):
            with self.assertRaises(SandboxUnavailable) as caught:
                run_code("6 * 7")

        self.assertEqual(caught.exception.reason, "limit_setup_failed")

    def test_only_an_oversized_message_is_a_protocol_violation(self) -> None:
        with self.assertRaises(SandboxViolation) as caught:
            _receive_payload(_FailingConnection(OSError("bad message length")), 16)
        self.assertEqual(caught.exception.reason, "protocol_limit")

        with self.assertRaises(SandboxUnavailable) as caught_pipe:
            _receive_payload(_FailingConnection(OSError(errno.EIO, "I/O error")), 16)
        self.assertEqual(caught_pipe.exception.reason, "pipe_read_failed")


if __name__ == "__main__":
    unittest.main()
