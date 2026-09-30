"""Run the unit suite with the network switched off.

CI went red for twenty runs because four tests reached for the Hugging Face
Hub. A suite that needs the network passes or fails with the weather, and a
test that quietly reads a local model cache passes on the author's machine and
nowhere else. This runner makes both impossible to miss:

- every outbound connection and every DNS lookup raises, except loopback,
  which some standard-library machinery uses internally;
- the Hugging Face libraries are told they are offline, and their cache points
  at an empty temporary directory, so nothing cached on this machine can stand
  in for a download.

Before running anything it checks that the block actually works, so a green
result cannot come from a guard that silently failed to install.

Usage matches `python -m unittest discover`, with any extra flags passed on:

    python scripts/run_tests_offline.py -v
"""

from __future__ import annotations

import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
LOOPBACK: Final = frozenset({"127.0.0.1", "::1", "localhost"})


class NetworkBlocked(OSError):
    """A test tried to reach the network."""


def _host(address: object) -> object:
    return address[0] if isinstance(address, tuple) else address


def _refuse(host: object) -> None:
    raise NetworkBlocked(f"network access is blocked in tests (tried {host!r})")


def block_network() -> None:
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def connect(self, address):
        # Unix-domain sockets take a path, not a host, and never leave the box.
        if self.family == getattr(socket, "AF_UNIX", None) or _host(address) in LOOPBACK:
            return real_connect(self, address)
        _refuse(_host(address))

    def connect_ex(self, address):
        if self.family == getattr(socket, "AF_UNIX", None) or _host(address) in LOOPBACK:
            return real_connect_ex(self, address)
        _refuse(_host(address))

    def getaddrinfo(host, *args, **kwargs):
        if host is None or host in LOOPBACK:
            return real_getaddrinfo(host, *args, **kwargs)
        _refuse(host)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.getaddrinfo = getaddrinfo


def isolate_hugging_face(cache_root: str) -> None:
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "1"
    os.environ["HF_HOME"] = cache_root
    os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_root, "datasets")
    os.environ["HF_HUB_CACHE"] = os.path.join(cache_root, "hub")


def assert_blocked() -> None:
    try:
        socket.create_connection(("pypi.org", 443), timeout=1)
    except NetworkBlocked:
        return
    except OSError as error:
        raise SystemExit(f"network guard did not engage: {error!r}") from error
    raise SystemExit("network guard did not engage: a connection succeeded")


def main(argv: list[str]) -> int:
    block_network()
    assert_blocked()
    # `python -m unittest` puts the working directory first on the path, and
    # some tests import `scripts.<name>` from there. Running this file directly
    # would put `scripts/` there instead, so match what unittest does.
    sys.path[0] = str(PROJECT_ROOT)
    with tempfile.TemporaryDirectory(prefix="empty-hf-cache-") as cache_root:
        isolate_hugging_face(cache_root)
        os.chdir(PROJECT_ROOT)
        program = unittest.main(
            module=None,
            argv=["run_tests_offline", "discover", "-s", "tests", *argv],
            exit=False,
        )
    return 0 if program.result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
