"""Record a bounded, model-free compatibility probe for the Phase-A ML lane.

The parent process inventories the lock and installed distributions.  A child
process imports Unsloth before the libraries it patches, then checks CUDA and
the shared reliability-kernel import.  Running the risky imports in a child
lets the parent persist a failure record when a native extension exits early.
No model repository is opened and no model weights are loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final


SCHEMA_VERSION: Final = 1
LANE_ID: Final = "phase-a-windows-unsloth-trl024"
PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT: Final = PROJECT_ROOT / "results" / "smoke_environment.json"
LOCK_PATH: Final = PROJECT_ROOT / "requirements-smoke.lock"
CONFIG_PATH: Final = PROJECT_ROOT / "configs" / "model_smoke.json"
CHILD_MARKER: Final = "__SMOKE_ENV_CHILD_JSON__="
MAX_CAPTURE_CHARS: Final = 16 * 1024
MAX_CONFIG_BYTES: Final = 256 * 1024
EXPECTED_PACKAGES: Final = {
    "accelerate": "1.14.0",
    "bitsandbytes": "0.50.1",
    "huggingface-hub": "1.27.0",
    "peft": "0.20.0",
    "pip": "25.0.1",
    "pydantic": "2.13.4",
    "torch": "2.11.0+cu128",
    "torchvision": "0.26.0+cu128",
    "transformers": "5.5.0",
    "triton-windows": "3.6.0.post26",
    "trl": "0.24.0",
    "unsloth": "2026.8.18",
    "unsloth-zoo": "2026.8.12",
    "uv": "0.12.5",
    "xformers": "0.0.35",
}
EXPECTED_EDITABLE_PROJECT: Final = {
    "internalizing-agent-reliability": "0.1.0",
}
SOURCE_PATHS: Final = (
    "pyproject.toml",
    "requirements-smoke.in",
    "requirements-smoke.lock",
    "configs/model_smoke.json",
    "scripts/smoke_models.py",
    "scripts/probe_smoke_environment.py",
)
_LOCK_PIN_RE: Final = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s\\]+)"
)
_SENSITIVE_ENV_RE: Final = re.compile(
    r"(?i)(?:token|secret|password|passwd|credential|authorization|cookie|api[_-]?key)"
)
_SENSITIVE_CAPTURE_PATTERNS: Final = (
    re.compile(r"\bhf_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bgh(?:p|o|u|s|r)_[A-Za-z0-9_]{16,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*"),
    re.compile(
        r"(?i)\b(token|secret|password|passwd|authorization|api[_-]?key)"
        r"\s*[:=]\s*([^\s,;]+)"
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded(value: str) -> str:
    if len(value) <= MAX_CAPTURE_CHARS:
        return value
    return "...[truncated]\n" + value[-MAX_CAPTURE_CHARS:]


def _sanitize_capture(value: str) -> str:
    sanitized = value
    for pattern in _SENSITIVE_CAPTURE_PATTERNS:
        if pattern.groups == 2:
            sanitized = pattern.sub(lambda match: f"{match.group(1)}=[REDACTED]", sanitized)
        else:
            sanitized = pattern.sub("[REDACTED]", sanitized)
    return _bounded(sanitized)


def _run_command(
    command: list[str],
    *,
    timeout_seconds: float,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return {
            "status": "timeout",
            "returncode": None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "stdout_tail": _sanitize_capture(stdout),
            "stderr_tail": _sanitize_capture(stderr),
        }
    except OSError as error:
        return {
            "status": "failed",
            "returncode": None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "stdout_tail": "",
            "stderr_tail": _sanitize_capture(f"{type(error).__name__}: {error}"),
        }
    return {
        "status": "passed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "stdout_tail": _sanitize_capture(completed.stdout),
        "stderr_tail": _sanitize_capture(completed.stderr),
    }


def _git_source() -> dict[str, Any]:
    commit_result = _run_command(
        ["git", "rev-parse", "HEAD"],
        timeout_seconds=10,
    )
    status_result = _run_command(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        timeout_seconds=10,
    )
    commit = commit_result["stdout_tail"].strip()
    valid_commit = commit if len(commit) == 40 else None
    clean = status_result["status"] == "passed" and not status_result[
        "stdout_tail"
    ].strip()
    return {
        "git_commit": valid_commit,
        "git_worktree_clean": clean,
        "git_status_check": {
            "status": status_result["status"],
            "returncode": status_result["returncode"],
        },
    }


def _strict_json_load(path: Path) -> Any:
    raw = path.read_bytes()
    if len(raw) > MAX_CONFIG_BYTES:
        raise ValueError(f"configuration exceeds {MAX_CONFIG_BYTES} bytes")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    return json.loads(
        raw,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )


def _configured_lock_sha256() -> str:
    payload = _strict_json_load(CONFIG_PATH)
    try:
        expected = payload["lane"]["expected_lock_sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError("model smoke config has no lane lock identity") from error
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("model smoke config lock SHA-256 is invalid")
    return expected


def _source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative_path in SOURCE_PATHS:
        path = PROJECT_ROOT / relative_path
        if path.is_file():
            hashes[relative_path] = sha256_file(path)
    return hashes


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in EXPECTED_PACKAGES:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _lock_environment_consistency() -> dict[str, Any]:
    locked_versions: dict[str, str] = {}
    for line in LOCK_PATH.read_text(encoding="utf-8").splitlines():
        match = _LOCK_PIN_RE.match(line)
        if match:
            locked_versions[_normalized_distribution_name(match.group(1))] = match.group(2)
    for name, version in EXPECTED_PACKAGES.items():
        locked_versions[_normalized_distribution_name(name)] = version
    expected_versions = {
        **locked_versions,
        **{
            _normalized_distribution_name(name): version
            for name, version in EXPECTED_EDITABLE_PROJECT.items()
        },
    }

    installed_versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            installed_versions[_normalized_distribution_name(name)] = distribution.version

    missing = sorted(name for name in expected_versions if name not in installed_versions)
    mismatches = {
        name: {"expected": expected, "actual": installed_versions.get(name)}
        for name, expected in sorted(expected_versions.items())
        if name in installed_versions and installed_versions[name] != expected
    }
    unexpected = sorted(set(installed_versions).difference(expected_versions))
    return {
        "status": (
            "passed"
            if not missing and not mismatches and not unexpected
            else "failed"
        ),
        "locked_distribution_count": len(locked_versions),
        "editable_distribution_count": len(EXPECTED_EDITABLE_PROJECT),
        "installed_distribution_count": len(installed_versions),
        "missing_expected_distributions": missing,
        "version_mismatches": mismatches,
        "unexpected_distributions": unexpected,
    }


def _nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    result = _run_command(command, timeout_seconds=15)
    facts: dict[str, Any] = {"command": result}
    if result["status"] != "passed":
        return facts
    rows = [line.strip() for line in result["stdout_tail"].splitlines() if line.strip()]
    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 4:
            continue
        name, memory_mib, driver_version, compute_capability = fields
        try:
            parsed_memory_mib = int(memory_mib)
        except ValueError:
            continue
        parsed_rows.append(
            {
                "name": name,
                "memory_mib": parsed_memory_mib,
                "driver_version": driver_version,
                "compute_capability": compute_capability,
            }
        )
    facts["gpus"] = parsed_rows
    return facts


def _child_payload() -> dict[str, Any]:
    import unsloth  # noqa: F401  # Must precede patched libraries.
    import bitsandbytes
    import torch
    import transformers
    import triton
    import trl
    import xformers
    from agent import parser as agent_parser

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(device_count):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    return {
        "imports": {
            "unsloth": True,
            "torch": True,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
            "bitsandbytes": bitsandbytes.__version__,
            "xformers": xformers.__version__,
            "triton": triton.__version__,
            "agent.parser": callable(agent_parser.parse_tool_calls),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "cuda_available": cuda_available,
            "device_count": device_count,
            "devices": devices,
        },
    }


def _run_import_probe(timeout_seconds: float) -> dict[str, Any]:
    safe_names = {
        "ALLUSERSPROFILE",
        "APPDATA",
        "COMMONPROGRAMFILES",
        "COMMONPROGRAMFILES(X86)",
        "COMMONPROGRAMW6432",
        "COMSPEC",
        "DRIVERDATA",
        "HOMEDRIVE",
        "HOMEPATH",
        "LOCALAPPDATA",
        "NUMBER_OF_PROCESSORS",
        "OS",
        "PATH",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "PROCESSOR_IDENTIFIER",
        "PROCESSOR_LEVEL",
        "PROCESSOR_REVISION",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "PROGRAMW6432",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERDOMAIN",
        "USERNAME",
        "USERPROFILE",
        "WINDIR",
    }
    environment = {
        key: value
        for key, value in os.environ.items()
        if (
            key.upper() in safe_names
            or key.upper().startswith(("CUDA", "NVIDIA", "TORCH", "TRITON"))
        )
        and not _SENSITIVE_ENV_RE.search(key)
    }
    environment["HF_HOME"] = str(PROJECT_ROOT / ".hf-cache-smoke")
    environment["TRITON_CACHE_DIR"] = str(PROJECT_ROOT / ".triton-cache-smoke")
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["HF_DATASETS_OFFLINE"] = "1"
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    environment["PYTHONUTF8"] = "1"
    command = [sys.executable, str(Path(__file__).resolve()), "--child"]
    result = _run_command(
        command,
        timeout_seconds=timeout_seconds,
        environment=environment,
    )
    payload: dict[str, Any] | None = None
    for line in reversed(result["stdout_tail"].splitlines()):
        if line.startswith(CHILD_MARKER):
            try:
                decoded = json.loads(line.removeprefix(CHILD_MARKER))
            except json.JSONDecodeError:
                break
            if isinstance(decoded, dict):
                payload = decoded
            break
    result["details"] = payload
    result["environment_policy"] = {
        "inherited_key_count": len(environment),
        "sensitive_named_variables_removed": True,
        "huggingface_offline_flags_set": True,
        "captured_output_redacted": True,
    }
    if result["status"] == "passed" and payload is None:
        result["status"] = "failed"
    return result


def _dependency_check() -> dict[str, Any]:
    uv_name = "uv.exe" if os.name == "nt" else "uv"
    uv_path = Path(sys.executable).with_name(uv_name)
    command = [str(uv_path), "pip", "check", "--python", sys.executable]
    environment = dict(os.environ)
    environment.setdefault("UV_CACHE_DIR", str(PROJECT_ROOT / ".uv-cache-smoke"))
    return _run_command(command, timeout_seconds=30, environment=environment)


def _report_passes(report: dict[str, Any]) -> bool:
    packages = report["packages"]
    imports = report["import_probe"]
    details = imports.get("details") or {}
    torch_facts = details.get("torch") or {}
    return bool(
        packages == EXPECTED_PACKAGES
        and report["lock"]["matches_configured_sha256"] is True
        and report["environment_consistency"]["status"] == "passed"
        and report["dependency_check"]["status"] == "passed"
        and report["source"]["git_worktree_clean"] is True
        and imports["status"] == "passed"
        and torch_facts.get("cuda_available") is True
        and torch_facts.get("device_count") == 1
        and details.get("imports", {}).get("agent.parser") is True
        and bool(report["nvidia_smi"].get("gpus"))
    )


def build_report(timeout_seconds: float) -> dict[str, Any]:
    if not LOCK_PATH.is_file():
        raise FileNotFoundError(f"missing smoke lock: {LOCK_PATH}")
    actual_lock_sha256 = sha256_file(LOCK_PATH)
    expected_lock_sha256 = _configured_lock_sha256()
    git_source = _git_source()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "lane_id": LANE_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
        "scope": {
            "model_weights_loaded": False,
            "model_repository_access": "not_requested_by_probe",
            "huggingface_offline_mode_enforced_for_import_child": True,
            "m6_environment_factory_in_scope": False,
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "lock": {
            "path": LOCK_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": actual_lock_sha256,
            "configured_sha256": expected_lock_sha256,
            "matches_configured_sha256": actual_lock_sha256
            == expected_lock_sha256,
        },
        "source": {
            **git_source,
            "file_sha256": _source_hashes(),
        },
        "packages": _package_versions(),
        "environment_consistency": _lock_environment_consistency(),
        "dependency_check": _dependency_check(),
        "nvidia_smi": _nvidia_smi(),
        "import_probe": _run_import_probe(timeout_seconds),
    }
    report["status"] = "passed" if _report_passes(report) else "failed"
    return report


def write_json_atomic(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialized)
            temporary_name = handle.name
        os.replace(temporary_name, output)
    finally:
        if temporary_name is not None:
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0 or args.timeout_seconds > 600:
        parser.error("--timeout-seconds must be in (0, 600]")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.child:
        print(CHILD_MARKER + json.dumps(_child_payload(), sort_keys=True, allow_nan=False))
        return 0
    report = build_report(args.timeout_seconds)
    write_json_atomic(report, args.output.resolve())
    print(
        json.dumps(
            {
                "lane_id": report["lane_id"],
                "lock_sha256": report["lock"]["sha256"],
                "output": str(args.output.resolve()),
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
