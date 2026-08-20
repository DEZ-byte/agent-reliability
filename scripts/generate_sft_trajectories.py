"""Roll out a policy over the frozen train split and keep what the grader passes.

BLUEPRINT_v2 section 5.2: the teacher plays the agent on the Phase A train
split, every trajectory is graded by the deterministic grader, and only passing
trajectories are kept. The grader doubles as the data filter, which is what
makes this rejection sampling rather than distillation of whatever the teacher
happened to say.

Two things this script deliberately does not do.

It does not decide what to keep. It records the grader's verdict and the
laundering verdict for every candidate, passing or not, and writes them all.
Selection, capping and de-duplication happen in the dataset builder, against
thresholds pinned in `configs/train_config.yaml`. Splitting it this way means a
threshold can be changed and the dataset rebuilt without paying for generation
again, and it means the rejected rows stay inspectable instead of never having
existed.

It does not touch dev or test. The split is an argument but the default is
train, and the generated rows carry their task ids so a later build can be
checked against the frozen manifest.

Each row stores the fully materialised message list and its hash. Re-deriving
that list later from tool events loses the exact observation strings, and a
training row that cannot be rebuilt byte-for-byte cannot be verified at all.
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

from agent.gates import GateEngine  # noqa: E402
from env.phase_a import (  # noqa: E402
    PhaseATask,
    build_phase_a_registry,
    calculator_tool_schema,
    is_answering_event,
)
from env.splits import load_split  # noqa: E402
from evaluation.policy import build_policy  # noqa: E402
from evaluation.rungs import (  # noqa: E402
    SYSTEM_PROMPT,
    USER_PROMPT,
    run_episode,
)
from training.config import config_sha256, load_train_config  # noqa: E402
from training.retention import laundering_verdict  # noqa: E402

TRAIN_CONFIG_PATH: Final = PROJECT_ROOT / "configs" / "train_config.yaml"
SPLIT_MANIFEST_PATH: Final = PROJECT_ROOT / "configs" / "splits" / "phase_a_gsm8k.json"
REGISTRY_PATH: Final = PROJECT_ROOT / "configs" / "model_candidates.json"

SCHEMA_VERSION: Final = 1
MAX_SEQUENCE_TOKENS: Final = 4096


class GenerationError(RuntimeError):
    """Generation could not proceed under the configuration it was given."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() or "unknown"


def _resolve_model(model_id: str, revision: str | None) -> dict[str, str]:
    """Prefer the pinned registry entry; fall back to an explicit revision.

    A teacher that is not yet in the registry can still be probed, but only by
    naming its revision, because an unpinned generation run cannot be
    reproduced and its output cannot honestly be called frozen.
    """

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    for entries in registry["roles"].values():
        for entry in entries:
            if entry["id"] == model_id:
                return {"id": model_id, "revision": entry["revision"], "source": "registry"}
    if not revision:
        raise GenerationError(
            f"{model_id} is not in configs/model_candidates.json, so --revision "
            "is required: an unpinned run cannot be reproduced"
        )
    return {"id": model_id, "revision": revision, "source": "explicit"}


def _messages(task: PhaseATask, completion: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(question=task.question)},
        {"role": "assistant", "content": completion},
    ]


def _answering_expression(events) -> str | None:
    for event in reversed(events):
        if is_answering_event(event):
            expression = event.call.arguments.get("expression")
            return expression if isinstance(expression, str) else None
    return None


def _load_model(model: dict[str, str], seed: int):
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
    loaded, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model["id"],
        revision=model["revision"],
        max_seq_length=MAX_SEQUENCE_TOKENS,
        dtype=compute_dtype,
        load_in_4bit=True,
        trust_remote_code=False,
        device_map={"": 0},
        quantization_config=quantization,
        local_files_only=False,
        use_exact_model_name=True,
        fast_inference=False,
        random_state=seed,
        disable_log_stats=True,
    )
    FastLanguageModel.for_inference(loaded)
    return loaded, tokenizer, torch


def generate(
    *,
    model: dict[str, str],
    tasks: list[PhaseATask],
    config: dict[str, Any],
    rows_out,
) -> dict[str, Any]:
    generation = config["generation"]
    seed_base = generation["seed_base"]
    loaded, tokenizer, torch = _load_model(model, seed_base)

    registry = build_phase_a_registry()
    gate_engine = GateEngine.from_mapping({})
    tools = [calculator_tool_schema()]
    rung = generation["rung"]
    runs = generation["runs_per_task"]

    counts = {
        "episodes": 0,
        "graded_correct": 0,
        "laundered": 0,
        "usable": 0,
        "tasks_with_at_least_one_usable": 0,
    }
    laundering_reasons: dict[str, int] = {}

    for task in tasks:
        usable_here = 0
        for run_index in range(runs):
            policy = build_policy(
                model=loaded,
                tokenizer=tokenizer,
                torch=torch,
                tools=tools,
                temperature=generation["temperature"],
                top_p=generation["top_p"],
                max_new_tokens=generation["max_new_tokens"],
                seed=seed_base + run_index,
                enable_thinking=generation["enable_thinking"],
            )
            episode = run_episode(
                task=task,
                registry=registry,
                gate_engine=gate_engine,
                policy=policy,
                rung=rung,
                run_index=run_index,
            )
            counts["episodes"] += 1

            expression = _answering_expression(episode.tool_events)
            verdict = None
            if episode.correct and expression is not None:
                verdict = laundering_verdict(
                    expression=expression,
                    question=task.question,
                    gold_answer=task.gold_answer,
                )

            messages = _messages(task, episode.completions[-1] if episode.completions else "")
            payload = json.dumps(messages, sort_keys=True, ensure_ascii=False)

            if episode.correct:
                counts["graded_correct"] += 1
            if verdict is not None:
                key = verdict.reason or "retained"
                laundering_reasons[key] = laundering_reasons.get(key, 0) + 1
                if verdict.laundered:
                    counts["laundered"] += 1

            usable = bool(episode.correct and verdict is not None and not verdict.laundered)
            if usable:
                counts["usable"] += 1
                usable_here += 1

            rows_out.write(
                json.dumps(
                    {
                        "model": model["id"],
                        "task_id": task.task_id,
                        "template_id": task.template_id,
                        "run_index": run_index,
                        "rung": rung,
                        "correct": episode.correct,
                        "terminal_reason": episode.terminal_reason,
                        "expression": expression,
                        "laundered": None if verdict is None else verdict.laundered,
                        "laundering_reason": None if verdict is None else verdict.reason,
                        "usable": usable,
                        "messages": messages,
                        "messages_sha256": _sha256_text(payload),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        if usable_here:
            counts["tasks_with_at_least_one_usable"] += 1

    del loaded
    torch.cuda.empty_cache()

    total = counts["episodes"] or 1
    return {
        "model": model,
        "counts": counts,
        "laundering_reasons": laundering_reasons,
        "task_coverage": counts["tasks_with_at_least_one_usable"] / len(tasks)
        if tasks
        else None,
        "usable_rate": counts["usable"] / total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face repo id")
    parser.add_argument("--revision", default=None, help="required if unregistered")
    parser.add_argument("--candidates", required=True, help="candidate JSONL path")
    parser.add_argument("--summary", required=True, help="summary artifact path")
    parser.add_argument("--split", default="train", choices=("train", "dev", "test"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-load", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    if args.split != "train":
        # Not forbidden, because a coverage probe on dev is legitimate, but it
        # must be a deliberate keystroke rather than a default.
        print(
            json.dumps({"warning": f"generating from the {args.split} split"}),
            file=sys.stderr,
        )
    if args.run_load and not args.allow_download:
        parser.error("--run-load and --allow-download must be supplied together")

    # Retention thresholds are the builder's business, so generation does not
    # wait for them to be measured.
    config = load_train_config(
        TRAIN_CONFIG_PATH,
        require=[
            "generation.rung",
            "generation.runs_per_task",
            "generation.seed_base",
        ],
    )
    model = _resolve_model(args.model, args.revision)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "kind": "sft_trajectory_candidates",
        "train_config_sha256": config_sha256(TRAIN_CONFIG_PATH),
        "split_manifest_sha256": _sha256_file(SPLIT_MANIFEST_PATH),
        "prompt_sha256": {
            "system": _sha256_text(SYSTEM_PROMPT),
            "user": _sha256_text(USER_PROMPT),
        },
        "generation": config["generation"],
        "split": args.split,
        "model_planned": model,
        "executed": bool(args.run_load),
        "source_commit": _git_commit(),
        "platform": {"python": platform.python_version(), "system": platform.system()},
    }

    if not args.run_load:
        Path(args.summary).write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps({"planned_model": model, "executed": False}))
        return 0

    tasks = load_split(SPLIT_MANIFEST_PATH, args.split, limit=args.limit)
    result["task_count"] = len(tasks)

    candidates_path = Path(args.candidates)
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    with candidates_path.open("w", encoding="utf-8") as rows_out:
        result["result"] = generate(
            model=model, tasks=tasks, config=config, rows_out=rows_out
        )
    result["candidates_sha256"] = _sha256_file(candidates_path)

    path = Path(args.summary)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(
        (json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )
    os.replace(temporary, path)

    print(
        json.dumps(
            {
                "summary": str(path),
                "candidates": str(candidates_path),
                "counts": result["result"]["counts"],
                "task_coverage": result["result"]["task_coverage"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
