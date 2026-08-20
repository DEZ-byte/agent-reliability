"""Load the frozen training configuration and refuse the unmeasured parts.

`configs/train_config.yaml` holds two kinds of value. Pre-registered ones come
from BLUEPRINT_v2 section 7.4 and were fixed before any measurement. Measured
ones are null until the measurement that sets them exists.

A null that scripts quietly tolerate is worse than no field at all, because a
run would proceed with a default nobody recorded and the artifact would name a
config that did not actually determine the run. So a caller names the keys it
needs and gets an error listing every one that is still unset, rather than
discovering the problem as a strange result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final, Iterable

CONFIG_RELATIVE_PATH: Final = "configs/train_config.yaml"


class TrainConfigError(RuntimeError):
    """The configuration cannot determine the run that was asked for."""


def config_sha256(path: Path) -> str:
    """The hash written into every artifact this config produced."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_hash_prefix(path: Path) -> str:
    """The `<confighash>` in a checkpoint name, per section 7.4."""

    return config_sha256(path)[:8]


def _lookup(config: dict[str, Any], dotted: str) -> Any:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise TrainConfigError(f"missing configuration key {dotted!r}")
        node = node[part]
    return node


def pending_keys(config: dict[str, Any]) -> list[str]:
    """Every value still awaiting the measurement that sets it."""

    found: list[str] = []

    def walk(node: Any, prefix: str) -> None:
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            if key.startswith("_"):
                continue
            path = f"{prefix}.{key}".lstrip(".")
            if value is None:
                found.append(path)
            else:
                walk(value, path)

    walk(config, "")
    return sorted(found)


def load_train_config(
    path: Path, *, require: Iterable[str] = ()
) -> dict[str, Any]:
    """Read the config, insisting that the named keys have been measured.

    `require` is the set of keys the caller actually depends on. A script that
    only generates trajectories does not need the retention thresholds, so it
    does not have to wait for them to be measured.
    """

    config = json.loads(Path(path).read_text(encoding="utf-8"))
    unset = [key for key in require if _lookup(config, key) is None]
    if unset:
        raise TrainConfigError(
            "these values are not measured yet, so this run cannot be "
            "reproduced from the config that names it: " + ", ".join(sorted(unset))
        )
    return config


__all__ = [
    "CONFIG_RELATIVE_PATH",
    "TrainConfigError",
    "config_hash_prefix",
    "config_sha256",
    "load_train_config",
    "pending_keys",
]
