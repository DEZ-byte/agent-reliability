"""Score every saved checkpoint on dev and pick the winner by the pinned rule.

BLUEPRINT_v2 section 7.4: evaluate on the dev split only, freeze the dev
winner, and run it on test exactly once. The rule this script applies - which
split, which rung, which metric - is read from `configs/train_config.yaml`,
where it was written before any dev number existed.

That matters more than it looks. With several checkpoints and one obvious
number per checkpoint, the temptation is to glance at the results and pick.
Doing it by hand is where "best on dev" quietly becomes "best on whichever
metric happened to favour the checkpoint I expected", and nothing in the
artifact would show the difference. Here the rule is applied by code, every
candidate's score is recorded whether it won or lost, and ties break toward the
earliest checkpoint rather than toward whichever the operator prefers.

The final checkpoint is included as a candidate. Nothing distinguishes it from
the intermediate ones except that training stopped there.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.config import config_sha256, load_train_config  # noqa: E402

TRAIN_CONFIG_PATH: Final = PROJECT_ROOT / "configs" / "train_config.yaml"
EVAL_CONFIGS: Final = {
    "dev": PROJECT_ROOT / "configs" / "eval_dev.yaml",
    "test": PROJECT_ROOT / "configs" / "eval.yaml",
}
RUNNER: Final = PROJECT_ROOT / "scripts" / "run_phase_a_baseline.py"
SCHEMA_VERSION: Final = 1


class SelectionError(RuntimeError):
    """Selection could not be carried out as pinned."""


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() or "unknown"


def discover_checkpoints(adapter_dir: Path) -> list[Path]:
    """Every distinct saved adapter, intermediate and final, in training order.

    The final directory is a candidate like any other. Treating it as special
    would make "best on dev" mean "the last one unless something beat it".

    Distinct by weight hash, though. `save_model` writes the final adapter to
    the top level as well as to its last step directory, and those are the same
    weights under two names. Scoring both wastes an evaluation and, worse,
    reports five candidates where four exist, so a reader counting checkpoints
    would misjudge how much the selection actually chose between.
    """

    found: list[tuple[int, Path]] = []
    for child in sorted(adapter_dir.glob("checkpoint-*")):
        if (child / "adapter_model.safetensors").is_file():
            try:
                step = int(child.name.split("-")[-1])
            except ValueError:
                continue
            found.append((step, child))
    ordered = [path for _, path in sorted(found)]
    if (adapter_dir / "adapter_model.safetensors").is_file():
        ordered.append(adapter_dir)
    if not ordered:
        raise SelectionError(f"no adapter found under {adapter_dir}")

    distinct: list[Path] = []
    seen: set[str] = set()
    for path in ordered:
        digest = hashlib.sha256(
            (path / "adapter_model.safetensors").read_bytes()
        ).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        distinct.append(path)
    return distinct


def score(
    *,
    checkpoint: Path,
    base_model: str,
    split: str,
    rung: str,
    metric: str,
    output_dir: Path,
    limit: int | None,
) -> dict[str, Any]:
    """Run one checkpoint through the ordinary evaluation runner."""

    tag = checkpoint.name.replace(".", "-")
    result_path = output_dir / f"select-{tag}.json"
    episodes_path = output_dir / f"select-{tag}.jsonl"
    command = [
        sys.executable,
        str(RUNNER),
        "--config",
        str(EVAL_CONFIGS[split]),
        "--output",
        str(result_path),
        "--episodes",
        str(episodes_path),
        "--candidate",
        base_model,
        "--adapter",
        str(checkpoint),
        "--rung",
        rung,
        "--run-load",
        "--allow-download",
    ]
    if limit:
        command += ["--limit", str(limit)]

    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise SelectionError(f"evaluation failed for {checkpoint}")

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    entry = payload["results"][0]
    if "error" in entry:
        raise SelectionError(f"{checkpoint}: {entry['error']}")
    metrics = entry["rungs"][rung]["metrics"]
    if metric not in metrics:
        raise SelectionError(f"{metric} not produced by the runner")
    return {
        "checkpoint": checkpoint.name,
        "path": str(checkpoint),
        "score": metrics[metric],
        "no_arithmetic_rate": entry["rungs"][rung]["no_arithmetic_rate"],
        "artifact": str(result_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_train_config(
        TRAIN_CONFIG_PATH,
        require=[
            "selection.split",
            "selection.rung",
            "selection.metric",
            "selection.checkpoints_evaluated",
        ],
    )
    rule = config["selection"]
    if rule["split"] != "dev":
        raise SelectionError(
            f"selection split is {rule['split']!r}; section 7.4 allows dev only"
        )
    if rule["checkpoints_evaluated"] != "all_saved":
        raise SelectionError(
            "this script evaluates every saved checkpoint; the config asks for "
            f"{rule['checkpoints_evaluated']!r}"
        )

    args.scratch.mkdir(parents=True, exist_ok=True)
    checkpoints = discover_checkpoints(args.adapter_dir)

    scored = [
        score(
            checkpoint=checkpoint,
            base_model=args.base_model,
            split=rule["split"],
            rung=rule["rung"],
            metric=rule["metric"],
            output_dir=args.scratch,
            limit=args.limit,
        )
        for checkpoint in checkpoints
    ]

    # Ties break toward the earliest checkpoint. Any tie-break is arbitrary;
    # this one is at least fixed in advance and independent of the scores.
    best = max(range(len(scored)), key=lambda index: (scored[index]["score"], -index))

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "kind": "checkpoint_selection",
        "train_config_sha256": config_sha256(TRAIN_CONFIG_PATH),
        "rule": {
            "split": rule["split"],
            "rung": rule["rung"],
            "metric": rule["metric"],
            "tie_break": "earliest_checkpoint",
            "note": (
                "Pinned in configs/train_config.yaml before any dev number "
                "existed. The winner runs on test exactly once."
            ),
        },
        "base_model": args.base_model,
        "candidates": scored,
        "selected": scored[best],
        "source_commit": _git_commit(),
        "platform": {"python": platform.python_version(), "system": platform.system()},
    }

    path = Path(args.summary)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(
        (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )
    os.replace(temporary, path)

    print(
        json.dumps(
            {
                "summary": str(path),
                "candidates": {row["checkpoint"]: row["score"] for row in scored},
                "selected": scored[best]["checkpoint"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
