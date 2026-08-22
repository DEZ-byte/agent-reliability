"""Train the Phase A GRPO arm from the SFT checkpoint, on execution-backed reward.

Section 7.1 requires GRPO to initialise from the SFT policy rather than from
base; GRPO-from-base exists only as a separate labelled ablation and is not run
here. The reward is the same composite the evaluator grades with, so the
constraint trained against cannot drift from the one measured against.

Every configuration value is passed explicitly and then re-read from the
constructed trainer. Importing unsloth replaces `GRPOTrainer` and changes
`GRPOConfig` defaults: on this stack `loss_type` defaults to `dapo` where
section 7.2 specifies token-level GRPO, and `temperature` defaults to 1.0 where
section 7.3 wants 0.7 to 0.85. A config that omits a field does not get the
documented default, it gets unsloth's.

Two things are logged every step because section 7.3 says a silent failure hides
in each. Zero-variance groups contribute exactly no gradient, since advantages
are group-relative, so a run can look busy while learning nothing. And the
laundering rate is the thing this reward cannot see: section 7.0 scores a call
that restates a remembered answer at full accuracy, measured rather than
penalised, and GRPO is the sharpest possible version of that pressure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.dialects import template_uses_canonical_tags  # noqa: E402
from env.phase_a import calculator_tool_schema  # noqa: E402
from env.splits import load_split  # noqa: E402
from evaluation.rungs import SYSTEM_PROMPT, USER_PROMPT  # noqa: E402
from training.config import (  # noqa: E402
    config_hash_prefix,
    config_sha256,
    load_train_config,
)
from training.grpo_reward import make_reward_function  # noqa: E402

TRAIN_CONFIG_PATH: Final = PROJECT_ROOT / "configs" / "train_config.yaml"
SPLIT_MANIFEST_PATH: Final = PROJECT_ROOT / "configs" / "splits" / "phase_a_gsm8k.json"
REGISTRY_PATH: Final = PROJECT_ROOT / "configs" / "model_candidates.json"
SCHEMA_VERSION: Final = 1


class GRPOError(RuntimeError):
    """The run was refused because something about it could not be verified."""


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


def _revision_for(model_id: str) -> str:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    for entries in registry["roles"].values():
        for entry in entries:
            if entry["id"] == model_id:
                return entry["revision"]
    raise GRPOError(f"{model_id} is not pinned in configs/model_candidates.json")


def build_prompt_dataset(tokenizer: Any, limit: int | None):
    """One row per train task, rendered exactly as the evaluator renders it.

    The prompt is pre-rendered rather than left conversational so that the
    string the policy continues during training is byte-identical to the one it
    continues at evaluation. `gold_answer` and `question` ride along as columns;
    TRL passes every extra column through to the reward function, expanded to
    one entry per generation.
    """

    from datasets import Dataset

    tools = [calculator_tool_schema()]
    tasks = load_split(SPLIT_MANIFEST_PATH, "train", limit=limit)
    rows = []
    for task in tasks:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(question=task.question)},
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        rows.append(
            {
                "prompt": rendered,
                "gold_answer": float(task.gold_answer),
                "question": task.question,
                "task_id": task.task_id,
            }
        )
    return Dataset.from_list(rows), tasks


def _assert_resolved(trainer: Any, grpo: dict[str, Any]) -> dict[str, Any]:
    """Re-read what the trainer resolved. What was passed is not evidence."""

    args = trainer.args
    resolved = {
        name: getattr(args, name, None)
        for name in (
            "num_generations",
            "beta",
            "epsilon",
            "loss_type",
            "importance_sampling_level",
            "scale_rewards",
            "temperature",
            "top_p",
            "max_prompt_length",
            "max_completion_length",
            "learning_rate",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "seed",
        )
    }
    for name in (
        "num_generations",
        "loss_type",
        "importance_sampling_level",
        "temperature",
        "beta",
    ):
        if resolved[name] != grpo[name]:
            raise GRPOError(
                f"resolved {name}={resolved[name]!r} does not match the pinned "
                f"{grpo[name]!r}; the stack overrode it"
            )
    if resolved["temperature"] > 0.85 or resolved["temperature"] < 0.7:
        raise GRPOError(
            "rollout temperature is outside section 7.3's 0.7-0.85 band; "
            "candidates that collapse to identical strings give a zero-variance "
            "group and no gradient"
        )
    return resolved


def summarise_health(health: list[dict[str, Any]]) -> dict[str, Any]:
    """What the groups actually looked like, including the ways they were dead."""

    if not health:
        return {"groups": 0}
    totals = [h["std"]["total"] for h in health]
    return {
        "groups": len(health),
        "zero_variance_fraction": sum(1 for h in health if h["zero_variance"])
        / len(health),
        "mean_total_reward_std": statistics.fmean(totals),
        "mean_component_std": {
            component: statistics.fmean([h["std"][component] for h in health])
            for component in ("accuracy", "format", "gate", "efficiency")
        },
        "mean_correct_fraction": statistics.fmean(
            [h["correct_fraction"] for h in health]
        ),
        "mean_laundered_fraction": statistics.fmean(
            [h["laundered_fraction"] for h in health]
        ),
        "mean_no_tool_call_fraction": statistics.fmean(
            [h["no_tool_call_fraction"] for h in health]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="SFT checkpoint to start from")
    parser.add_argument("--model", required=True, help="base model id")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="smoke runs only")
    parser.add_argument("--run-load", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    if args.run_load and not args.allow_download:
        parser.error("--run-load and --allow-download must be supplied together")

    config = load_train_config(
        TRAIN_CONFIG_PATH,
        require=["grpo.num_generations", "grpo.beta", "grpo.loss_type"],
    )
    grpo = config["grpo"]
    if grpo["init_from"] != "sft":
        raise GRPOError(
            "section 7.1 initialises GRPO from the SFT checkpoint; "
            "GRPO-from-base is a separate labelled ablation"
        )
    revision = _revision_for(args.model)
    config_hash = config_hash_prefix(TRAIN_CONFIG_PATH)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "kind": "grpo_run",
        "train_config_sha256": config_sha256(TRAIN_CONFIG_PATH),
        "config_hash_prefix": config_hash,
        "model": {"id": args.model, "revision": revision},
        "init_from_adapter": str(args.adapter),
        "checkpoint_name": f"{args.model.split('/')[-1]}-grpo-{config_hash}",
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
        print(json.dumps({"planned": result["checkpoint_name"], "executed": False}))
        return 0

    import unsloth  # noqa: F401  # must precede trl/transformers; it rewrites both
    from unsloth import FastLanguageModel

    import torch
    from trl import GRPOConfig, GRPOTrainer

    # Loading the adapter directory gives back the SFT policy with its adapter
    # already attached and trainable, which is what section 7.1 asks GRPO to
    # continue from.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter),
        max_seq_length=grpo["max_prompt_length"] + grpo["max_completion_length"],
        dtype=None,
        load_in_4bit=config["sft"]["load_in_4bit"],
        trust_remote_code=False,
    )

    dataset, tasks = build_prompt_dataset(tokenizer, args.limit)
    longest = max(len(tokenizer(row["prompt"])["input_ids"]) for row in dataset)
    if longest > grpo["max_prompt_length"]:
        raise GRPOError(
            f"a prompt is {longest} tokens against max_prompt_length "
            f"{grpo['max_prompt_length']}; truncation removes the system prompt "
            "from the left and would train against a different task"
        )
    result["dataset"] = {"rows": len(dataset), "longest_prompt_tokens": longest}

    health: list[dict[str, Any]] = []
    reward = make_reward_function(
        normalise_dialect=not template_uses_canonical_tags(tokenizer.chat_template),
        health_log=health,
    )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=grpo["num_generations"],
        beta=grpo["beta"],
        epsilon=grpo["epsilon"],
        loss_type=grpo["loss_type"],
        importance_sampling_level=grpo["importance_sampling_level"],
        scale_rewards=grpo["scale_rewards"],
        temperature=grpo["temperature"],
        top_p=grpo["top_p"],
        max_prompt_length=grpo["max_prompt_length"],
        max_completion_length=grpo["max_completion_length"],
        learning_rate=grpo["learning_rate"],
        per_device_train_batch_size=grpo["per_device_train_batch_size"],
        gradient_accumulation_steps=grpo["gradient_accumulation_steps"],
        max_steps=args.max_steps or grpo["max_steps"],
        save_steps=grpo["save_steps"],
        save_strategy="steps",
        seed=grpo["seed"],
        logging_steps=1,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[reward],
        processing_class=tokenizer,
    )
    result["resolved"] = _assert_resolved(trainer, grpo)

    train_output = trainer.train()
    result["train"] = {
        "global_step": int(train_output.global_step),
        "training_loss": float(train_output.training_loss),
    }
    result["group_health"] = summarise_health(health)
    trainer.save_model(args.output_dir)
    result["adapter_dir"] = str(Path(args.output_dir).resolve())

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
                "checkpoint": result["checkpoint_name"],
                "steps": result["train"]["global_step"],
                "health": result["group_health"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
