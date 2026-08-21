"""Train the Phase A SFT adapter on pre-masked rows, asserting the stack's traps shut.

Every value here is set explicitly, including ones that look like defaults.
Importing unsloth rewrites `trl.SFTTrainer` and silently changes `SFTConfig`
defaults - on this stack the learning rate moves from 2e-5 to 5e-5 and bf16
flips off - so a config that omits a field does not get the documented default,
it gets unsloth's. Assertions below re-read the resolved values from the
trainer rather than trusting what was passed.

The rows arrive already tokenised, with labels built by `training.masking`.
That is deliberate: `assistant_only_loss=True` is silently non-functional here,
because unsloth's dataset preparation never produces an assistant mask, so the
flag would train on everything while appearing to do the opposite.

Three traps get explicit guards.

`packing=True` drops a custom `labels` column with no warning and falls back to
labels=input_ids, which trains on the prompt. It is forced off and asserted.

Nothing truncates a pre-tokenised row under unsloth. A row longer than
`max_length` is therefore dropped rather than cut, because the assistant span
sits at the end of the row and right-truncation removes precisely the tokens
the loss is computed over, leaving a row that still trains and teaches nothing.

An `assistant_masks` column would overwrite the supplied labels after they are
loaded. The dataset is asserted to carry only the three columns it should.
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

from env.phase_a import calculator_tool_schema  # noqa: E402
from training.masking import IGNORE_INDEX  # noqa: E402
from training.config import (  # noqa: E402
    config_hash_prefix,
    config_sha256,
    load_train_config,
)

TRAIN_CONFIG_PATH: Final = PROJECT_ROOT / "configs" / "train_config.yaml"
REGISTRY_PATH: Final = PROJECT_ROOT / "configs" / "model_candidates.json"
SCHEMA_VERSION: Final = 1

DATASET_COLUMNS: Final = ("input_ids", "attention_mask", "labels")


class TrainingError(RuntimeError):
    """The run was refused because something about it could not be verified."""


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


def _revision_for(model_id: str) -> str:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    for entries in registry["roles"].values():
        for entry in entries:
            if entry["id"] == model_id:
                return entry["revision"]
    raise TrainingError(f"{model_id} is not pinned in configs/model_candidates.json")


def _read_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise TrainingError("dataset is empty")
    return rows


def encode_rows(
    rows: list[dict[str, Any]], tokenizer: Any, *, max_length: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Mask every row, dropping any that will not fit.

    Dropping rather than truncating is the whole point. The trained span is the
    final assistant turn, so cutting a row from the right removes exactly the
    tokens the loss is computed over and leaves a row that trains on nothing
    while still counting as a row.
    """

    from training.masking import encode_with_labels

    tools = [calculator_tool_schema()]
    encoded: list[dict[str, Any]] = []
    dropped: list[str] = []
    trained_tokens = 0

    for row in rows:
        example = encode_with_labels(tokenizer, row["messages"], tools=tools)
        if len(example.input_ids) > max_length:
            dropped.append(row["task_id"])
            continue
        trained_tokens += example.trained_token_count
        encoded.append(
            {
                "input_ids": list(example.input_ids),
                "attention_mask": [1] * len(example.input_ids),
                "labels": list(example.labels),
            }
        )

    if not encoded:
        raise TrainingError("every row exceeded max_length; nothing to train on")

    stats = {
        "rows_in": len(rows),
        "rows_trained": len(encoded),
        "rows_dropped_over_max_length": len(dropped),
        "dropped_task_ids": dropped[:20],
        "trained_tokens_total": trained_tokens,
        "row_tokens_max": max(len(row["input_ids"]) for row in encoded),
    }
    return encoded, stats


def _assert_stack_traps_are_shut(trainer: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Re-read what the trainer actually resolved, not what was passed to it.

    Importing unsloth changes SFTConfig defaults and replaces the trainer, so
    the only trustworthy source for a value is the constructed object.
    """

    args = trainer.args
    sft = config["sft"]
    resolved = {
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "packing": getattr(args, "packing", None),
        "max_length": getattr(args, "max_length", None),
        "seed": args.seed,
        "save_steps": args.save_steps,
        "data_collator": type(trainer.data_collator).__name__,
        "padding_side": getattr(trainer.processing_class, "padding_side", None),
    }

    if resolved["packing"]:
        raise TrainingError(
            "packing is on; it drops a custom labels column and trains on the prompt"
        )
    if abs(resolved["learning_rate"] - sft["learning_rate"]) > 1e-12:
        raise TrainingError(
            "resolved learning rate "
            f"{resolved['learning_rate']} does not match the pinned "
            f"{sft['learning_rate']}; unsloth overrode it"
        )
    if resolved["max_length"] != sft["max_seq_length"]:
        raise TrainingError(
            f"resolved max_length {resolved['max_length']} does not match the "
            f"pinned {sft['max_seq_length']}"
        )
    columns = set(trainer.train_dataset.column_names)
    if columns != set(DATASET_COLUMNS):
        raise TrainingError(
            "dataset columns are " + ", ".join(sorted(columns)) +
            "; an assistant_masks or completion_mask column would overwrite labels"
        )
    return resolved


def _verify_first_batch(
    trainer: Any,
    encoded: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    tokenizer: Any,
) -> dict[str, Any]:
    """Pull a real batch and check the labels that actually reach the loss.

    Asserting against the collator class proves nothing: the object that runs is
    chosen by unsloth at dataset-preparation time, and unsloth also auto-enables
    padding-free batching, which concatenates several rows into one sequence
    with `position_ids` rather than padding them into a rectangle. The dataset
    is shuffled too. So neither the batch's row count nor its order can be
    predicted, and a check that assumes either reports a mismatch on a correct
    run - which is worse than no check, because it teaches you to ignore it.

    What is stable is the content. Every trained token must belong to some row's
    trained span, whole and in order, and the decoded trained region must
    contain no text from any prompt. That holds under shuffling, under
    padding-free packing, and under any batch size.
    """

    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch["labels"].reshape(-1)
    trained = [int(value) for value in labels[labels != -100].tolist()]
    if not trained:
        raise TrainingError(
            "the first batch has no trained tokens; every label is masked out"
        )

    spans = [
        [value for value in row["labels"] if value != IGNORE_INDEX]
        for row in encoded
    ]
    position = 0
    covered: list[int] = []
    while position < len(trained):
        for index, span in enumerate(spans):
            if span and trained[position : position + len(span)] == span:
                covered.append(index)
                position += len(span)
                break
        else:
            raise TrainingError(
                "the first batch contains trained tokens that are not a whole "
                f"row's assistant span, starting at offset {position}"
            )

    decoded = tokenizer.decode(trained)
    leaked = sorted(
        {
            message["role"]
            for row in rows
            for message in row["messages"]
            if message["role"] != "assistant" and message["content"] in decoded
        }
    )
    if leaked:
        raise TrainingError(
            "prompt text from these roles reached the trained region: "
            + ", ".join(leaked)
        )

    return {
        "batch_trained_tokens": len(trained),
        "rows_covered": covered,
        "every_trained_token_is_a_whole_assistant_span": True,
        "prompt_roles_leaked_into_trained_region": leaked,
        "padding_free": bool(getattr(trainer.args, "padding_free", False)),
        "label_dtype": str(batch["labels"].dtype),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
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
        require=[
            "sft.epochs",
            "sft.learning_rate",
            "sft.lora.r",
            "sft.max_seq_length",
        ],
    )
    sft = config["sft"]
    revision = _revision_for(args.model)
    config_hash = config_hash_prefix(TRAIN_CONFIG_PATH)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "kind": "sft_run",
        "train_config_sha256": config_sha256(TRAIN_CONFIG_PATH),
        "config_hash_prefix": config_hash,
        "dataset_sha256": _sha256_file(Path(args.dataset)),
        "model": {"id": args.model, "revision": revision},
        "checkpoint_name": f"{args.model.split('/')[-1]}-sft-{config_hash}",
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
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    rows = _read_dataset(Path(args.dataset))
    if args.limit:
        rows = rows[: args.limit]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        revision=revision,
        max_seq_length=sft["max_seq_length"],
        dtype=None,
        load_in_4bit=sft["load_in_4bit"],
        trust_remote_code=False,
        use_exact_model_name=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=sft["lora"]["r"],
        lora_alpha=sft["lora"]["alpha"],
        lora_dropout=sft["lora"]["dropout"],
        target_modules=sft["lora"]["target_modules"],
        use_gradient_checkpointing="unsloth",
        random_state=sft["seed"],
    )

    encoded, encode_stats = encode_rows(
        rows, tokenizer, max_length=sft["max_seq_length"]
    )
    result["dataset_stats"] = encode_stats

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=sft["epochs"],
        learning_rate=sft["learning_rate"],
        lr_scheduler_type=sft["lr_scheduler_type"],
        warmup_ratio=sft["warmup_ratio"],
        per_device_train_batch_size=sft["per_device_train_batch_size"],
        gradient_accumulation_steps=sft["gradient_accumulation_steps"],
        max_length=sft["max_seq_length"],
        seed=sft["seed"],
        save_steps=sft["save_steps"],
        save_strategy="steps",
        logging_steps=1,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        # Both are silently non-functional on this stack and both would
        # overwrite the supplied labels if they were not off.
        packing=False,
        assistant_only_loss=False,
        completion_only_loss=False,
        report_to=[],
        **({"max_steps": args.max_steps} if args.max_steps else {}),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(encoded),
        processing_class=tokenizer,
    )

    result["resolved"] = _assert_stack_traps_are_shut(trainer, config)
    result["first_batch"] = _verify_first_batch(trainer, encoded, rows, tokenizer)

    train_output = trainer.train()
    result["train"] = {
        "global_step": int(train_output.global_step),
        "training_loss": float(train_output.training_loss),
    }
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
                "loss": result["train"]["training_loss"],
                "rows_trained": encode_stats["rows_trained"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
