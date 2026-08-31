"""Measure how far GRPO moved the policy away from the SFT checkpoint it started from.

The README argues that two GRPO nulls across a tenfold learning-rate range are
harder to dismiss than one, and that argument rests on the weights having
actually moved further at the higher rate. Those figures were computed once,
by hand, and recorded only in a decision log that is not public. A reader of
the repository could not check them, which is exactly the kind of gap this
project is otherwise careful about.

This recomputes them from the adapters on disk and writes an artifact, so the
number is reproducible from files anyone can hash.

Two quantities, because for LoRA they answer different questions.

The adapter-parameter change is the plain one: how much did the numbers in the
adapter file move, relative to their own size. It is easy to state and easy to
check.

The effective-delta change is the one that matters to the model. A LoRA adapter
does not act through A and B separately; it acts through the product, scaled by
alpha over r. Two adapters can differ substantially in their factors while
producing nearly the same delta, or barely differ while producing a different
one, so the product is computed per module and compared directly.

Both are reported. Where they disagree, the effective delta is the honest
answer to "did the policy change", and the parameter change is the honest
answer to "did training touch anything".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SCHEMA_VERSION: Final = 1
WEIGHTS_NAME: Final = "adapter_model.safetensors"
CONFIG_NAME: Final = "adapter_config.json"


class WeightChangeError(RuntimeError):
    """The comparison could not be made as described."""


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() or "unknown"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(adapter: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    weights = adapter / WEIGHTS_NAME
    if not weights.is_file():
        raise WeightChangeError(f"no {WEIGHTS_NAME} under {adapter}")
    return load_file(str(weights))


def _scaling(adapter: Path) -> float:
    """alpha / r, the factor LoRA applies to B @ A."""

    config_path = adapter / CONFIG_NAME
    if not config_path.is_file():
        raise WeightChangeError(f"no {CONFIG_NAME} under {adapter}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    r = config.get("r")
    alpha = config.get("lora_alpha")
    if not r or alpha is None:
        raise WeightChangeError(f"{config_path} does not declare r and lora_alpha")
    return float(alpha) / float(r)


def _module_pairs(tensors: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Map each adapted module to its (A, B) tensor names."""

    pairs: dict[str, dict[str, str]] = {}
    for name in tensors:
        if ".lora_A." in name:
            pairs.setdefault(name.split(".lora_A.")[0], {})["A"] = name
        elif ".lora_B." in name:
            pairs.setdefault(name.split(".lora_B.")[0], {})["B"] = name
    return {
        module: (halves["A"], halves["B"])
        for module, halves in pairs.items()
        if "A" in halves and "B" in halves
    }


def compare(before: Path, after: Path) -> dict[str, Any]:
    """Relative change from one adapter to another, two ways."""

    import torch

    start = _load(before)
    end = _load(after)
    missing = set(start) ^ set(end)
    if missing:
        raise WeightChangeError(
            f"adapters do not share the same tensors; {len(missing)} differ"
        )

    # Plain parameter change, over every adapter tensor at once.
    num = 0.0
    den = 0.0
    for name, tensor in start.items():
        a = tensor.to(torch.float64)
        b = end[name].to(torch.float64)
        num += float(torch.sum((b - a) ** 2))
        den += float(torch.sum(a**2))
    parameter_change = math.sqrt(num) / math.sqrt(den) if den else 0.0

    # Effective delta, module by module: scaling * (B @ A).
    scale_before = _scaling(before)
    scale_after = _scaling(after)
    pairs = _module_pairs(start)
    if not pairs:
        raise WeightChangeError("no lora_A/lora_B pairs found in the adapter")
    num_eff = 0.0
    den_eff = 0.0
    for module, (a_name, b_name) in sorted(pairs.items()):
        delta_before = scale_before * (
            start[b_name].to(torch.float64) @ start[a_name].to(torch.float64)
        )
        delta_after = scale_after * (
            end[b_name].to(torch.float64) @ end[a_name].to(torch.float64)
        )
        num_eff += float(torch.sum((delta_after - delta_before) ** 2))
        den_eff += float(torch.sum(delta_before**2))
    effective_change = math.sqrt(num_eff) / math.sqrt(den_eff) if den_eff else 0.0

    return {
        "adapter_parameter_relative_change": parameter_change,
        "effective_delta_relative_change": effective_change,
        "modules_compared": len(pairs),
        "tensors_compared": len(start),
        "lora_scaling": {"before": scale_before, "after": scale_after},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path, help="SFT checkpoint")
    parser.add_argument(
        "--after",
        required=True,
        nargs="+",
        type=Path,
        help="one or more GRPO checkpoints to compare against it",
    )
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    comparisons = []
    for after in args.after:
        measured = compare(args.before, after)
        comparisons.append(
            {
                "after": str(after),
                "after_weights_sha256": _sha256_file(after / WEIGHTS_NAME),
                **measured,
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "kind": "weight_change",
        "method": (
            "Relative Frobenius norm of the difference, computed in float64. "
            "The parameter figure is taken over every adapter tensor at once. "
            "The effective figure is taken over the per-module LoRA product "
            "(alpha / r) * B @ A, which is what the base weights actually see."
        ),
        "before": str(args.before),
        "before_weights_sha256": _sha256_file(args.before / WEIGHTS_NAME),
        "comparisons": comparisons,
        "executed": True,
        "source_commit": _git_commit(),
        "platform": {"python": platform.python_version(), "system": platform.system()},
    }

    path = Path(args.summary)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(
        (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )
    os.replace(temporary, path)
    print(json.dumps({"summary": str(path), "comparisons": comparisons}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
