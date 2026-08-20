"""Run the Phase A R0/R1 baseline and record pass^k with confidence intervals.

This produces the first reliability numbers in the project. It plugs a real
checkpoint into the framework-neutral rung core in `src/evaluation/rungs.py`,
which is what keeps the R0-versus-R1 contrast identical to the one already
tested on CPU.

Offline by default. A measured run needs --run-load with --allow-download and
refuses to start on a dirty worktree, so every artifact names the exact source
that produced it.

Every accuracy figure is reported beside its no-arithmetic rate, which D-064
makes mandatory: a model that can solve these problems in its head may pass the
remembered answer to the tool instead of computing it, and that scores correct.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from env.phase_a import (  # noqa: E402
    PhaseATask,
    build_phase_a_registry,
    calculator_tool_schema,
)
from env.splits import load_split  # noqa: E402
from agent.gates import GateEngine  # noqa: E402
from evaluation.metrics import compute_pass_metrics  # noqa: E402
from evaluation.rungs import (  # noqa: E402
    SYSTEM_PROMPT,
    USER_PROMPT,
    run_episode,
)

EVAL_CONFIG_PATH: Final = PROJECT_ROOT / "configs" / "eval.yaml"
REGISTRY_PATH: Final = PROJECT_ROOT / "configs" / "model_candidates.json"
SCHEMA_VERSION: Final = 1
MAX_SEQUENCE_TOKENS: Final = 2048

MEASURED_ROLES: Final = (
    "primary_small",
    "scale_check",
    "cross_family_check",
    "scaffolded_comparator",
)


class BaselineError(RuntimeError):
    """Raised when the run cannot honestly proceed."""


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise BaselineError("git " + " ".join(args) + " failed")
    return completed.stdout.strip()


def _require_clean_worktree() -> None:
    if _git("status", "--porcelain"):
        raise BaselineError(
            "refusing to measure on a dirty worktree; commit first so the "
            "artifact names the exact source that produced it"
        )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_eval_config() -> dict[str, Any]:
    return json.loads(EVAL_CONFIG_PATH.read_text(encoding="utf-8"))


def _load_tasks(config: dict[str, Any], limit: int | None) -> list[PhaseATask]:
    """Delegate to the shared loader so evaluation and training agree.

    This used to assume every frozen task came from upstream `test`, which is
    true for dev and test and false for train. One loader now derives the
    upstream split per task and checks every content hash, so the training data
    generator cannot disagree with the evaluator about what a task is.
    """

    return load_split(
        PROJECT_ROOT / config["phase_a"]["split_manifest"],
        config["phase_a"]["split"],
        limit=limit,
    )


def _candidates(selected: list[str]) -> list[dict[str, str]]:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    out: list[dict[str, str]] = []
    for role in MEASURED_ROLES:
        for entry in registry["roles"].get(role, []):
            if entry.get("selection_status") == "rejected":
                continue
            out.append({"role": role, "id": entry["id"], "revision": entry["revision"]})
    if selected:
        out = [c for c in out if c["id"] in selected]
    return out


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def _bootstrap_pass_k(
    rows: list[list[int]], k: int, resamples: int, confidence: float, seed: int
) -> tuple[float, float] | None:
    """Resample tasks with replacement, recomputing pass^k each time.

    Tasks are the unit of resampling, matching the hierarchical bootstrap
    section 10.2 asks for: the uncertainty that matters is which tasks landed in
    the split, not which token a run happened to sample.
    """

    if not rows:
        return None
    rng = random.Random(seed)
    count = len(rows)
    estimates: list[float] = []
    for _ in range(resamples):
        sample = [rows[rng.randrange(count)] for _ in range(count)]
        estimates.append(compute_pass_metrics(sample, k).pass_power_k)
    estimates.sort()
    lower = estimates[int((1 - confidence) / 2 * resamples)]
    upper = estimates[min(resamples - 1, int((1 + confidence) / 2 * resamples))]
    return (lower, upper)


def _make_policy(model: Any, tokenizer: Any, torch: Any, config: dict[str, Any], seed: int):
    decoding = config["decoding"]
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    tools = [calculator_tool_schema()]

    def policy(messages: list[dict[str, str]]) -> str:
        # Render through the model's own tool interface. These checkpoints were
        # trained on it, and the smoke probe measured 100% strict validity
        # under it; a described format produced unterminated calls instead.
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=False, add_generation_prompt=True
            )
        inputs = tokenizer(rendered, return_tensors="pt").to("cuda:0")
        torch.manual_seed(seed)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=decoding["max_new_tokens"],
                do_sample=True,
                temperature=decoding["temperature"],
                top_p=decoding["top_p"],
                pad_token_id=pad,
            )
        return tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    return policy


def _measure(
    candidate: dict[str, str],
    tasks: list[PhaseATask],
    config: dict[str, Any],
    episodes_out,
) -> dict[str, Any]:
    from unsloth import FastLanguageModel  # patches transformers; import first

    import torch
    import transformers

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=candidate["id"],
        revision=candidate["revision"],
        max_seq_length=MAX_SEQUENCE_TOKENS,
        dtype=compute_dtype,
        load_in_4bit=True,
        trust_remote_code=False,
        device_map={"": 0},
        quantization_config=quantization,
        local_files_only=False,
        use_exact_model_name=True,
        fast_inference=False,
        random_state=config["runs"]["seed_base"],
        disable_log_stats=True,
    )
    FastLanguageModel.for_inference(model)

    n_runs = config["runs"]["tier_1_n"]
    seed_base = config["runs"]["seed_base"]
    cap = config["episode"]["environment_turn_cap"]
    registry = build_phase_a_registry()
    gate_engine = GateEngine.from_mapping({})

    per_rung: dict[str, Any] = {}
    for rung in config["phase_a"]["rungs"]:
        rows: list[list[int]] = []
        laundered = 0
        total_episodes = 0
        decisions = 0
        for task in tasks:
            outcomes: list[int] = []
            for run_index in range(n_runs):
                policy = _make_policy(
                    model, tokenizer, torch, config, seed_base + run_index
                )
                result = run_episode(
                    task=task,
                    registry=registry,
                    gate_engine=gate_engine,
                    policy=policy,
                    rung=rung,
                    run_index=run_index,
                    environment_turn_cap=cap,
                )
                outcomes.append(1 if result.correct else 0)
                total_episodes += 1
                decisions += result.counters.policy_model_decision_count
                if result.answered_without_arithmetic:
                    laundered += 1
                episodes_out.write(
                    json.dumps(
                        {"candidate": candidate["id"], **result.to_json()},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            rows.append(outcomes)

        metrics: dict[str, Any] = {}
        for k in config["reporting"]["k_values"]:
            if k > n_runs:
                continue
            computed = compute_pass_metrics(rows, k)
            interval = _bootstrap_pass_k(
                rows,
                k,
                config["reporting"]["bootstrap_resamples"],
                config["reporting"]["confidence"],
                seed_base + k,
            )
            metrics[f"pass^{k}"] = computed.pass_power_k
            metrics[f"pass@{k}"] = computed.pass_at_k
            metrics[f"pass^{k}_ci95"] = list(interval) if interval else None

        solved = sum(1 for row in rows if sum(row) > 0)
        per_rung[rung] = {
            "tasks": len(rows),
            "runs_per_task": n_runs,
            "episodes": total_episodes,
            "model_decisions": decisions,
            "metrics": metrics,
            "no_arithmetic_rate": laundered / total_episodes if total_episodes else None,
            "no_arithmetic_episodes": laundered,
            "tasks_solved_at_least_once": solved,
        }

    del model
    torch.cuda.empty_cache()
    return {"candidate": candidate, "rungs": per_rung}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", required=True, help="trajectory JSONL path")
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-load", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    if args.run_load and not args.allow_download:
        parser.error("--run-load and --allow-download must be supplied together")

    config = _load_eval_config()
    candidates = _candidates(args.candidate)
    if args.candidate and not candidates:
        parser.error("no registry candidate matched --candidate")

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "kind": "phase_a_baseline",
        "eval_config_sha256": _sha256_file(EVAL_CONFIG_PATH),
        "split_manifest_sha256": _sha256_file(
            PROJECT_ROOT / config["phase_a"]["split_manifest"]
        ),
        "registry_sha256": _sha256_file(REGISTRY_PATH),
        "prompt_sha256": {
            "system": _sha256_text(SYSTEM_PROMPT),
            "user": _sha256_text(USER_PROMPT),
        },
        "decoding": config["decoding"],
        "candidates_planned": candidates,
        "executed": bool(args.run_load),
        "results": [],
    }

    if not args.run_load:
        result["note"] = "planned offline; no checkpoint was loaded"
        Path(args.output).write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps({"planned_candidates": len(candidates)}))
        return 0

    _require_clean_worktree()
    result["source_commit"] = _git("rev-parse", "HEAD")
    result["platform"] = platform.platform()

    tasks = _load_tasks(config, args.limit)
    result["task_count"] = len(tasks)

    episodes_path = Path(args.episodes)
    with episodes_path.open("w", encoding="utf-8", newline="\n") as episodes_out:
        for candidate in candidates:
            try:
                result["results"].append(
                    _measure(candidate, tasks, config, episodes_out)
                )
            except Exception as exc:  # Model and runtime errors are result data.
                result["results"].append(
                    {
                        "candidate": candidate,
                        "error": type(exc).__name__ + ": " + str(exc)[:400],
                    }
                )

    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    path = Path(args.output)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(payload.encode("utf-8"))
    os.replace(temporary, path)

    summary = {}
    for entry in result["results"]:
        if "error" in entry:
            summary[entry["candidate"]["id"]] = entry["error"]
            continue
        summary[entry["candidate"]["id"]] = {
            rung: {
                "pass^1": data["metrics"].get("pass^1"),
                "pass^4": data["metrics"].get("pass^4"),
                "no_arithmetic_rate": data["no_arithmetic_rate"],
            }
            for rung, data in entry["rungs"].items()
        }
    print(json.dumps({"output": str(path), "summary": summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
