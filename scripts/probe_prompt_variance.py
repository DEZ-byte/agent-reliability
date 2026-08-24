"""Measure which train prompts a GRPO run could actually learn from.

The first GRPO run here spent roughly a quarter of its steps on groups where
all eight candidates scored alike. A group-relative advantage is zero in that
state, so those steps updated nothing. Raising the learning rate tenfold made
no difference, which is what amplifying a zero gradient looks like.

This script measures the problem instead of inferring it. It replays the GRPO
rollout exactly - same policy, same prompt rendering, same temperature, same
group size, same execution-backed reward - and records, per prompt, how many
of the G candidates were correct. A prompt where none are correct and a prompt
where all are correct both teach nothing. Everything between them does.

Nothing here is tuned. The criterion is DAPO's and it has no free parameter:
keep a prompt when at least one candidate is right and at least one is wrong.

What this is not: DAPO's dynamic sampling, which resamples inside the training
loop so a prompt that becomes solvable mid-run rejoins the batch. This is
GRESO's cheaper approximation, justified by their measurement that over 90% of
dead prompts stay dead. The difference is real and any result built on this
should name it. See `src/training/prompt_variance.py` for why the loop is not
being touched: unsloth rewrites `GRPOTrainer` on import, and over-generating
candidates does not fit an 8 GB card.

The output feeds `scripts/train_grpo.py --prompt-filter`. Rows are written as
they are computed, so `--resume` can pick up a probe that a Colab session
killed halfway.
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

from agent.dialects import template_uses_canonical_tags  # noqa: E402
from agent.gates import GateEngine  # noqa: E402
from env.phase_a import build_phase_a_registry, calculator_tool_schema  # noqa: E402
from env.splits import load_split  # noqa: E402
from evaluation.rungs import SYSTEM_PROMPT, USER_PROMPT  # noqa: E402
from training.config import config_hash_prefix, config_sha256, load_train_config  # noqa: E402
from training.grpo_reward import score_completion  # noqa: E402
from training.prompt_variance import (  # noqa: E402
    SCHEMA_KIND,
    GroupVerdict,
    classify_group,
    summarise,
)

TRAIN_CONFIG_PATH: Final = PROJECT_ROOT / "configs" / "train_config.yaml"
SPLIT_MANIFEST_PATH: Final = PROJECT_ROOT / "configs" / "splits" / "phase_a_gsm8k.json"
REGISTRY_PATH: Final = PROJECT_ROOT / "configs" / "model_candidates.json"
SCHEMA_VERSION: Final = 1

# Distinct from the GRPO training seed so a probe rollout and a training
# rollout can never silently coincide, the same reason generation and
# evaluation carry different seed bases in configs/train_config.yaml.
PROBE_SEED_BASE: Final = 20260824


class ProbeError(RuntimeError):
    """The probe was refused because something about it could not be verified."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() or "unknown"


def adapter_weights_sha256(adapter: Path) -> str | None:
    """The hash of the adapter this probe measured, or None if it is not there.

    Recorded so the trainer can check it is filtering on a probe of the same
    weights. A path is not enough: the probe may have run on a rented GPU and
    the training on a laptop, and a checkpoint directory retrained in place
    keeps its name while changing what is inside it.
    """

    weights = adapter / "adapter_model.safetensors"
    if not weights.is_file():
        return None
    return hashlib.sha256(weights.read_bytes()).hexdigest()


def _revision_for(model_id: str) -> str:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    for entries in registry["roles"].values():
        for entry in entries:
            if entry["id"] == model_id:
                return entry["revision"]
    raise ProbeError(f"{model_id} is not pinned in configs/model_candidates.json")


def already_probed(path: Path) -> dict[str, dict[str, Any]]:
    """Rows from an earlier invocation, keyed by task id.

    A truncated final line is dropped rather than repaired: a killed process
    can leave half a JSON object, and re-probing one prompt is cheaper than
    reasoning about what half a record meant.
    """

    if not path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if "task_id" in row and "liveness" in row:
            rows[row["task_id"]] = row
    return rows


def render(tokenizer, question: str, tools) -> str:
    """The prompt string GRPO training will continue, byte for byte.

    `build_prompt_dataset` in train_grpo.py renders this way. If the two ever
    diverge the probe measures the difficulty of a prompt the policy never
    sees, and the filter silently selects on the wrong thing.
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(question=question)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        tools_in_user_message=True,
    )


def generate_group(
    *,
    loaded,
    tokenizer,
    torch,
    prompts: list[str],
    group_size: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
) -> list[list[str]]:
    """`group_size` sampled completions for each prompt, one forward pass."""

    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda:0")
    prompt_length = inputs["input_ids"].shape[1]
    torch.manual_seed(seed)
    with torch.inference_mode():
        generated = loaded.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=group_size,
            pad_token_id=pad,
        )
    decoded = [
        tokenizer.decode(row[prompt_length:], skip_special_tokens=True)
        for row in generated
    ]
    # `generate` returns prompts in order, group_size rows each.
    return [
        decoded[index * group_size : (index + 1) * group_size]
        for index in range(len(prompts))
    ]


def batches(items: list, size: int) -> list[list]:
    if size < 1:
        raise ProbeError(f"batch size must be >= 1, got {size}")
    return [items[start : start + size] for start in range(0, len(items), size)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="SFT checkpoint to probe")
    parser.add_argument("--model", required=True, help="base model id")
    parser.add_argument("--output", required=True, help="per-prompt verdict JSONL")
    parser.add_argument("--summary", required=True, help="summary artifact path")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help=(
            "prompts per forward pass; each one expands to num_generations "
            "sequences, so 2 means 16 in flight on an 8 GB card"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep verdicts already in --output and probe only what is missing",
    )
    parser.add_argument("--run-load", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    if args.run_load and not args.allow_download:
        parser.error("--run-load and --allow-download must be supplied together")

    config = load_train_config(
        TRAIN_CONFIG_PATH,
        require=["grpo.num_generations", "grpo.temperature", "grpo.top_p"],
    )
    grpo = config["grpo"]
    revision = _revision_for(args.model)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "kind": SCHEMA_KIND,
        "train_config_sha256": config_sha256(TRAIN_CONFIG_PATH),
        "config_hash_prefix": config_hash_prefix(TRAIN_CONFIG_PATH),
        "model": {"id": args.model, "revision": revision},
        "adapter": str(args.adapter),
        "adapter_weights_sha256": adapter_weights_sha256(Path(args.adapter)),
        "criterion": {
            "rule": "keep a prompt when 0 < correct < group_size",
            "source": "DAPO arXiv 2503.14476, Dynamic Sampling",
            "free_parameters": 0,
            "applied": "before training, not inside the loop",
            "approximation": (
                "GRESO arXiv 2506.02177 measured that over 90% of "
                "zero-variance prompts stay zero-variance, which is what "
                "makes a one-off probe a usable stand-in for resampling"
            ),
        },
        "rollout": {
            "group_size": grpo["num_generations"],
            "temperature": grpo["temperature"],
            "top_p": grpo["top_p"],
            "max_new_tokens": grpo["max_completion_length"],
            "seed_base": PROBE_SEED_BASE,
        },
        "prompt_sha256": {
            "system": _sha256_text(SYSTEM_PROMPT),
            "user": _sha256_text(USER_PROMPT),
        },
        "executed": bool(args.run_load),
        "source_commit": _git_commit(),
        "platform": {"python": platform.python_version(), "system": platform.system()},
    }

    if not args.run_load:
        Path(args.summary).write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps({"planned": str(args.summary), "executed": False}))
        return 0

    import unsloth  # noqa: F401  # must precede trl/transformers; it rewrites both
    from unsloth import FastLanguageModel

    import torch

    loaded, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter),
        max_seq_length=grpo["max_prompt_length"] + grpo["max_completion_length"],
        dtype=None,
        load_in_4bit=config["sft"]["load_in_4bit"],
        trust_remote_code=False,
    )
    FastLanguageModel.for_inference(loaded)

    normalise_dialect = not template_uses_canonical_tags(tokenizer.chat_template)
    result["normalise_dialect"] = normalise_dialect
    registry = build_phase_a_registry()
    gate_engine = GateEngine.from_mapping({})
    tools = [calculator_tool_schema()]

    tasks = load_split(SPLIT_MANIFEST_PATH, "train", limit=args.limit)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = already_probed(output_path) if args.resume else {}
    if not args.resume and output_path.exists():
        output_path.unlink()
    pending = [task for task in tasks if task.task_id not in existing]
    result["resume"] = {"reused": len(existing), "probed": len(pending)}

    verdicts: list[GroupVerdict] = []
    for row in existing.values():
        verdicts.append(
            GroupVerdict(
                task_id=row["task_id"],
                group_size=row["group_size"],
                correct=row["correct"],
                total_std=row["total_std"],
                liveness=row["liveness"],
            )
        )

    with output_path.open("a", encoding="utf-8") as handle:
        for index, chunk in enumerate(batches(pending, args.batch_size)):
            groups = generate_group(
                loaded=loaded,
                tokenizer=tokenizer,
                torch=torch,
                prompts=[render(tokenizer, task.question, tools) for task in chunk],
                group_size=grpo["num_generations"],
                temperature=grpo["temperature"],
                top_p=grpo["top_p"],
                max_new_tokens=grpo["max_completion_length"],
                seed=PROBE_SEED_BASE + index,
            )
            for task, completions in zip(chunk, groups):
                scores = [
                    score_completion(
                        text,
                        gold_answer=float(task.gold_answer),
                        question=task.question,
                        registry=registry,
                        gate_engine=gate_engine,
                        normalise_dialect=normalise_dialect,
                    )
                    for text in completions
                ]
                verdict = classify_group(scores, task_id=task.task_id)
                verdicts.append(verdict)
                handle.write(
                    json.dumps(verdict.as_row(), ensure_ascii=False) + "\n"
                )
            handle.flush()
            done = len(verdicts)
            print(f"[probe] {done}/{len(tasks)} prompts", flush=True)

    result["summary"] = summarise(verdicts)
    result["prompts"] = [verdict.as_row() for verdict in verdicts]

    path = Path(args.summary)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(
        (json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )
    os.replace(temporary, path)

    print(json.dumps({"summary": str(path), **result["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
