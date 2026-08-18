"""Best-effort resource sandbox for small, generated Python snippets.

This module deliberately does not claim to provide a security or trust boundary.
It combines conservative syntax checks, a small builtin allow-list, a spawned
worker process, and operating-system resource controls to contain ordinary model
mistakes and common Python escape attempts.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import gc
import io
import json
import math
import multiprocessing
import os
import signal
import time
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any

__all__ = [
    "SandboxResult",
    "SandboxViolation",
    "memory_limit_supported",
    "run_code",
]


_MIB = 1024 * 1024
_DEFAULT_MEMORY_LIMIT_BYTES = 256 * _MIB
_DEFAULT_CPU_TIME_SECONDS = 2
_DEFAULT_FILE_SIZE_LIMIT_BYTES = 1 * _MIB
_DEFAULT_MAX_OUTPUT_CHARS = 64 * 1024
_DEFAULT_MAX_SOURCE_CHARS = 64 * 1024
_MAX_PROTOCOL_PAYLOAD_BYTES = 4 * _MIB
_RESULT_NAME = "_sandbox_last_value"

_FORBIDDEN_NAMES = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "dir",
        "eval",
        "exec",
        "getattr",
        "globals",
        "help",
        "input",
        "locals",
        "open",
        "setattr",
        "vars",
    }
)

# These non-dunder attributes expose interpreter frames or code objects and can
# bypass a builtin allow-list. Dunder attributes are rejected separately.
_FORBIDDEN_ATTRIBUTES = frozenset(
    {
        "ag_code",
        "ag_frame",
        "cr_code",
        "cr_frame",
        "f_back",
        "f_builtins",
        "f_code",
        "f_globals",
        "f_locals",
        "gi_code",
        "gi_frame",
        "mro",
        "tb_frame",
    }
)


class SandboxViolation(RuntimeError):
    """A rejected program or a sandbox resource/control failure.

    ``reason`` is a stable machine-readable category. ``detail`` is intended
    for logs and may contain platform-specific information.
    """

    def __init__(self, reason: str, detail: str | None = None) -> None:
        self.reason = reason
        self.detail = detail
        message = reason if detail is None else f"{reason}: {detail}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class SandboxResult:
    """Captured outcome from a completed sandbox worker.

    Worker results cross the process boundary as bounded JSON rather than
    pickle. The value of a final expression is returned only as text in
    ``value_repr``.
    """

    stdout: str
    stderr: str
    value_repr: str | None = None
    exception_type: str | None = None
    exception_message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.exception_type is None


class _OutputLimitExceeded(RuntimeError):
    pass


class _CappedTextBuffer(io.StringIO):
    def __init__(self, limit: int) -> None:
        super().__init__()
        self._limit = limit
        self._written = 0

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("sandbox output must be text")
        remaining = self._limit - self._written
        if len(text) > remaining:
            if remaining > 0:
                super().write(text[:remaining])
                self._written += remaining
            raise _OutputLimitExceeded(
                f"captured output exceeded {self._limit} characters"
            )
        written = super().write(text)
        self._written += written
        return written


def _contains_dunder(text: str) -> bool:
    return "__" in text


def _parse_and_validate(source: str) -> ast.Module:
    if not isinstance(source, str):
        raise TypeError("source must be a string")

    try:
        tree = ast.parse(source, filename="<sandbox>", mode="exec")
    except SyntaxError as exc:
        location = f"line {exc.lineno}" if exc.lineno is not None else "unknown line"
        raise SandboxViolation("invalid_syntax", f"{location}: {exc.msg}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SandboxViolation(
                "forbidden_syntax", f"imports are not allowed (line {node.lineno})"
            )

        if isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES:
                raise SandboxViolation(
                    "forbidden_name",
                    f"{node.id!r} is unavailable (line {node.lineno})",
                )
            if _contains_dunder(node.id):
                raise SandboxViolation(
                    "dunder_access",
                    f"dunder name {node.id!r} is unavailable (line {node.lineno})",
                )

        if isinstance(node, ast.Attribute):
            if _contains_dunder(node.attr):
                raise SandboxViolation(
                    "dunder_access",
                    f"dunder attribute {node.attr!r} is unavailable (line {node.lineno})",
                )
            if node.attr in _FORBIDDEN_ATTRIBUTES:
                raise SandboxViolation(
                    "introspection_access",
                    f"attribute {node.attr!r} is unavailable (line {node.lineno})",
                )

        # Without getattr in the allow-list these strings are not directly
        # useful, but rejecting them also blocks common obfuscated escape probes.
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _contains_dunder(node.value)
        ):
            raise SandboxViolation(
                "dunder_access",
                f"dunder string literal is unavailable (line {node.lineno})",
            )

    return tree


def _instrument_last_expression(tree: ast.Module) -> ast.Module:
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        expression = tree.body[-1]
        replacement = ast.Assign(
            targets=[ast.Name(id=_RESULT_NAME, ctx=ast.Store())],
            value=expression.value,
        )
        ast.copy_location(replacement, expression)
        tree.body[-1] = replacement
        ast.fix_missing_locations(tree)
    return tree


def _safe_builtins() -> dict[str, Any]:
    allowed = (
        "abs",
        "all",
        "any",
        "bool",
        "bytearray",
        "bytes",
        "chr",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "int",
        "isinstance",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    )
    return {name: getattr(builtins, name) for name in allowed}


def _apply_posix_limits(
    memory_limit_bytes: int | None,
    cpu_time_seconds: int | None,
    file_size_limit_bytes: int | None,
) -> None:
    if os.name != "posix":
        return

    import resource

    if memory_limit_bytes is not None and hasattr(resource, "RLIMIT_AS"):
        resource.setrlimit(
            resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes)
        )
    if cpu_time_seconds is not None and hasattr(resource, "RLIMIT_CPU"):
        resource.setrlimit(resource.RLIMIT_CPU, _cpu_limits(resource, cpu_time_seconds))
    if file_size_limit_bytes is not None and hasattr(resource, "RLIMIT_FSIZE"):
        resource.setrlimit(
            resource.RLIMIT_FSIZE,
            (file_size_limit_bytes, file_size_limit_bytes),
        )


def _cpu_limits(resource_module: Any, requested_soft: int) -> tuple[int, int]:
    current_soft, current_hard = resource_module.getrlimit(resource_module.RLIMIT_CPU)
    infinity = resource_module.RLIM_INFINITY

    soft = requested_soft
    if current_soft != infinity:
        soft = min(soft, current_soft)
    if current_hard != infinity:
        # Preserve one second between the soft and hard limits whenever the
        # inherited hard limit allows it, so SIGXCPU is observable before kill.
        soft = min(soft, max(0, current_hard - 1))

    hard = soft + 1
    if current_hard != infinity:
        hard = min(hard, current_hard)
    return int(soft), int(hard)


def _send_payload(connection: Connection, payload: dict[str, Any]) -> None:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        encoded = json.dumps(
            {
                "kind": "violation",
                "reason": "worker_protocol_error",
                "detail": f"could not encode payload: {type(exc).__name__}",
            },
            separators=(",", ":"),
        ).encode("utf-8")

    if len(encoded) > _MAX_PROTOCOL_PAYLOAD_BYTES:
        encoded = json.dumps(
            {
                "kind": "violation",
                "reason": "protocol_limit",
                "detail": (
                    f"encoded payload exceeded {_MAX_PROTOCOL_PAYLOAD_BYTES} bytes"
                ),
            },
            separators=(",", ":"),
        ).encode("utf-8")

    try:
        connection.send_bytes(encoded)
    except (BrokenPipeError, EOFError, MemoryError, OSError):
        # The parent may already have killed the worker for a resource breach.
        pass


def _receive_payload(
    connection: Connection, max_output_chars: int
) -> dict[str, Any]:
    try:
        encoded = connection.recv_bytes(_MAX_PROTOCOL_PAYLOAD_BYTES)
    except OSError as exc:
        raise SandboxViolation(
            "protocol_limit",
            f"worker message exceeded {_MAX_PROTOCOL_PAYLOAD_BYTES} bytes",
        ) from exc

    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise SandboxViolation(
            "worker_protocol_error", f"invalid JSON payload: {type(exc).__name__}"
        ) from exc

    if not isinstance(payload, dict):
        raise SandboxViolation("worker_protocol_error", "payload must be an object")

    kind = payload.get("kind")
    if kind == "violation":
        reason = payload.get("reason")
        detail = payload.get("detail")
        if not isinstance(reason, str) or not reason:
            raise SandboxViolation(
                "worker_protocol_error", "violation reason must be non-empty text"
            )
        if detail is not None and not isinstance(detail, str):
            raise SandboxViolation(
                "worker_protocol_error", "violation detail must be text or null"
            )
        return payload

    if kind != "result":
        raise SandboxViolation("worker_protocol_error", "unknown payload kind")

    for field in ("stdout", "stderr"):
        value = payload.get(field)
        if not isinstance(value, str) or len(value) > max_output_chars:
            raise SandboxViolation(
                "worker_protocol_error", f"invalid or oversized {field} field"
            )
    for field in ("value_repr", "exception_message"):
        value = payload.get(field)
        if value is not None and (
            not isinstance(value, str) or len(value) > max_output_chars
        ):
            raise SandboxViolation(
                "worker_protocol_error", f"invalid or oversized {field} field"
            )
    exception_type = payload.get("exception_type")
    if exception_type is not None and (
        not isinstance(exception_type, str) or len(exception_type) > 256
    ):
        raise SandboxViolation(
            "worker_protocol_error", "invalid exception_type field"
        )
    return payload


def _worker_main(
    source: str,
    connection: Connection,
    memory_limit_bytes: int | None,
    cpu_time_seconds: int | None,
    file_size_limit_bytes: int | None,
    max_output_chars: int,
) -> None:
    """Spawn target. It must remain at module scope for Windows."""

    stdout_buffer = _CappedTextBuffer(max_output_chars)
    stderr_buffer = _CappedTextBuffer(max_output_chars)
    scope: dict[str, Any] | None = None

    try:
        try:
            _apply_posix_limits(
                memory_limit_bytes,
                cpu_time_seconds,
                file_size_limit_bytes,
            )
        except (OSError, ValueError) as exc:
            _send_payload(
                connection,
                {
                    "kind": "violation",
                    "reason": "limit_setup_failed",
                    "detail": f"{type(exc).__name__}: {exc}",
                },
            )
            return

        tree = _instrument_last_expression(_parse_and_validate(source))
        code = builtins.compile(tree, "<sandbox>", "exec", dont_inherit=True)

        def eprint(*values: Any, sep: str = " ", end: str = "\n") -> None:
            builtins.print(*values, sep=sep, end=end, file=stderr_buffer)

        scope = {
            "__builtins__": _safe_builtins(),
            "eprint": eprint,
        }

        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
                stderr_buffer
            ):
                builtins.exec(code, scope, scope)
        except MemoryError:
            scope.clear()
            gc.collect()
            _send_payload(
                connection,
                {
                    "kind": "violation",
                    "reason": "memory_limit",
                    "detail": "worker allocation exceeded its address-space limit",
                },
            )
            return
        except _OutputLimitExceeded as exc:
            scope.clear()
            _send_payload(
                connection,
                {
                    "kind": "violation",
                    "reason": "output_limit",
                    "detail": str(exc),
                },
            )
            return
        except BaseException as exc:
            exception_type = builtins.type(exc).__name__
            exception_message = builtins.str(exc)
            if len(exception_message) > max_output_chars:
                scope.clear()
                _send_payload(
                    connection,
                    {
                        "kind": "violation",
                        "reason": "output_limit",
                        "detail": (
                            "exception message exceeded "
                            f"{max_output_chars} characters"
                        ),
                    },
                )
                return
            try:
                formatted_traceback = "Traceback (most recent call last):\n"
                formatted_traceback += "".join(
                    traceback.format_tb(exc.__traceback__, limit=20)
                )
                formatted_traceback += (
                    f"{exception_type}: {exception_message}\n"
                )
                stderr_buffer.write(formatted_traceback)
            except _OutputLimitExceeded:
                pass
            _send_payload(
                connection,
                {
                    "kind": "result",
                    "stdout": stdout_buffer.getvalue(),
                    "stderr": stderr_buffer.getvalue(),
                    "value_repr": None,
                    "exception_type": exception_type,
                    "exception_message": exception_message,
                },
            )
            return

        value_repr = None
        if _RESULT_NAME in scope:
            value_repr = builtins.repr(scope[_RESULT_NAME])
            if len(value_repr) > max_output_chars:
                scope.clear()
                _send_payload(
                    connection,
                    {
                        "kind": "violation",
                        "reason": "output_limit",
                        "detail": (
                            "final value representation exceeded "
                            f"{max_output_chars} characters"
                        ),
                    },
                )
                return
        _send_payload(
            connection,
            {
                "kind": "result",
                "stdout": stdout_buffer.getvalue(),
                "stderr": stderr_buffer.getvalue(),
                "value_repr": value_repr,
                "exception_type": None,
                "exception_message": None,
            },
        )
    except MemoryError:
        if scope is not None:
            scope.clear()
        gc.collect()
        _send_payload(
            connection,
            {
                "kind": "violation",
                "reason": "memory_limit",
                "detail": "worker exhausted memory while preparing or reporting a result",
            },
        )
    except SandboxViolation as exc:
        _send_payload(
            connection,
            {"kind": "violation", "reason": exc.reason, "detail": exc.detail},
        )
    except BaseException as exc:
        _send_payload(
            connection,
            {
                "kind": "violation",
                "reason": "worker_internal_error",
                "detail": f"{builtins.type(exc).__name__}: {exc}",
            },
        )
    finally:
        connection.close()


def _windows_private_usage_bytes(process_id: int) -> int | None:
    if os.name != "nt":
        return None

    try:
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCountersEx),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        process_query_limited_information = 0x1000
        process_vm_read = 0x0010
        handle = kernel32.OpenProcess(
            process_query_limited_information | process_vm_read,
            False,
            process_id,
        )
        if not handle:
            return None
        try:
            counters = ProcessMemoryCountersEx()
            counters.cb = ctypes.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            ):
                return None
            return int(counters.PrivateUsage or counters.WorkingSetSize)
        finally:
            kernel32.CloseHandle(handle)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def memory_limit_supported() -> bool:
    """Return whether this platform has an enforceable memory mechanism."""

    if os.name == "nt":
        return _windows_private_usage_bytes(os.getpid()) is not None
    if os.name == "posix":
        try:
            import resource

            return hasattr(resource, "RLIMIT_AS")
        except ImportError:
            return False
    return False


def _positive_int_or_none(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer or None")
    return value


def _exit_violation(exit_code: int | None) -> SandboxViolation:
    if exit_code is not None and exit_code < 0:
        terminating_signal = -exit_code
        if terminating_signal == getattr(signal, "SIGXCPU", object()):
            return SandboxViolation("cpu_limit", "worker exceeded RLIMIT_CPU")
        if terminating_signal == getattr(signal, "SIGXFSZ", object()):
            return SandboxViolation("file_size_limit", "worker exceeded RLIMIT_FSIZE")
        return SandboxViolation(
            "worker_terminated", f"worker received signal {terminating_signal}"
        )
    return SandboxViolation(
        "worker_terminated", f"worker exited without a result (exit code {exit_code})"
    )


def run_code(
    source: str,
    *,
    timeout_seconds: float = 3.0,
    memory_limit_bytes: int | None = _DEFAULT_MEMORY_LIMIT_BYTES,
    cpu_time_seconds: int | None = _DEFAULT_CPU_TIME_SECONDS,
    file_size_limit_bytes: int | None = _DEFAULT_FILE_SIZE_LIMIT_BYTES,
    max_output_chars: int = _DEFAULT_MAX_OUTPUT_CHARS,
    max_source_chars: int = _DEFAULT_MAX_SOURCE_CHARS,
) -> SandboxResult:
    """Execute ``source`` in a fresh spawned worker and return captured output.

    Static-policy and resource failures raise :class:`SandboxViolation`.
    Ordinary exceptions raised by the snippet are captured in
    :class:`SandboxResult`, allowing callers to present a tool observation.

    The timeout is wall-clock time and is enforced by ``Process.kill()``.
    POSIX workers also receive RLIMIT_AS, RLIMIT_CPU, and RLIMIT_FSIZE. On
    Windows the parent polls the worker's private bytes and kills it when the
    requested memory limit is exceeded. Source, captured fields, and the JSON
    protocol message all have independent size bounds.
    """

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a positive finite number")

    memory_limit_bytes = _positive_int_or_none(
        "memory_limit_bytes", memory_limit_bytes
    )
    cpu_time_seconds = _positive_int_or_none("cpu_time_seconds", cpu_time_seconds)
    file_size_limit_bytes = _positive_int_or_none(
        "file_size_limit_bytes", file_size_limit_bytes
    )
    max_output_chars = _positive_int_or_none("max_output_chars", max_output_chars)
    max_source_chars = _positive_int_or_none("max_source_chars", max_source_chars)
    assert max_output_chars is not None
    assert max_source_chars is not None

    if not isinstance(source, str):
        raise TypeError("source must be a string")
    if len(source) > max_source_chars:
        raise SandboxViolation(
            "source_limit",
            f"source has {len(source)} characters; maximum is {max_source_chars}",
        )
    _parse_and_validate(source)

    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process: multiprocessing.Process | None = None
    process_started = False
    payload: dict[str, Any] | None = None
    termination_reason: SandboxViolation | None = None

    try:
        process = context.Process(
            target=_worker_main,
            args=(
                source,
                child_connection,
                memory_limit_bytes,
                cpu_time_seconds,
                file_size_limit_bytes,
                max_output_chars,
            ),
            name="python-sandbox-worker",
            daemon=True,
        )
        process.start()
        process_started = True
        child_connection.close()
        deadline = time.monotonic() + float(timeout_seconds)

        while process.is_alive():
            now = time.monotonic()
            if now >= deadline:
                process.kill()
                termination_reason = SandboxViolation(
                    "timeout", f"wall-clock limit of {timeout_seconds:g}s exceeded"
                )
                break

            if os.name == "nt" and memory_limit_bytes is not None:
                private_usage = _windows_private_usage_bytes(process.pid)
                if private_usage is not None and private_usage > memory_limit_bytes:
                    process.kill()
                    termination_reason = SandboxViolation(
                        "memory_limit",
                        f"private bytes {private_usage} exceeded {memory_limit_bytes}",
                    )
                    break

            if payload is None and parent_connection.poll(0):
                try:
                    payload = _receive_payload(parent_connection, max_output_chars)
                except EOFError:
                    pass
                except SandboxViolation as exc:
                    process.kill()
                    termination_reason = exc
                    break

            process.join(min(0.01, max(0.0, deadline - now)))

        process.join()

        if termination_reason is not None:
            raise termination_reason

        if payload is None and parent_connection.poll(0.1):
            try:
                payload = _receive_payload(parent_connection, max_output_chars)
            except EOFError:
                pass

        if payload is None:
            raise _exit_violation(process.exitcode)

        if payload.get("kind") == "violation":
            raise SandboxViolation(payload["reason"], payload.get("detail"))

        return SandboxResult(
            stdout=payload["stdout"],
            stderr=payload["stderr"],
            value_repr=payload.get("value_repr"),
            exception_type=payload.get("exception_type"),
            exception_message=payload.get("exception_message"),
        )
    finally:
        child_connection.close()
        if process_started and process is not None and process.is_alive():
            process.kill()
            process.join()
        parent_connection.close()
