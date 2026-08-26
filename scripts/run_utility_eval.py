"""Measure what fine-tuning cost outside the task it was fine-tuned for.

Every result this project has published was measured on GSM8K with a
calculator, which is the task the model was trained on. That answers whether
training worked and says nothing about what it broke. This runs the same
checkpoints over 400 held-out general-knowledge questions with no tool in
sight, and reports two numbers.

Accuracy is the obvious one: does the model still know things.

The tool-call rate is the one worth watching. Every training example was a
single tool call, so the pressure to emit one on anything question-shaped is
real, and this benchmark offers no tools at all. A `<tool_call>` block here is
a habit that has escaped its context. A model can hold its accuracy and still
become unusable outside its training task, and folding the two together would
hide exactly that.

Decoded greedily, one pass per question. This asks what the model knows rather
than how reliably it answers, so `pass^k` is the wrong instrument and sampling
four times would cost four times as much to answer a question nobody asked.

No tools are passed to the chat template, deliberately. Handing the model a
tool block and then counting tool calls would measure obedience; withholding it
measures habit.
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

from evaluation.utility import CHOICE_LABELS, score_completion, summarise  # noqa: E402
from evaluation.utility_split import MMLU_MANIFEST_NAME, load_questions  # noqa: E402

SPLIT_MANIFEST_PATH: Final = PROJECT_ROOT / "configs" / "splits" / MMLU_MANIFEST_NAME
REGISTRY_PATH: Final = PROJECT_ROOT / "configs" / "model_candidates.json"
SCHEMA_VERSION: Final = 1

SYSTEM_PROMPT: Final = (
    "Answer the multiple-choice question. Reply with the letter of the correct "
    "option and nothing else."
)
USER_PROMPT: Final = "{question}\n\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n\nAnswer:"

# 128 left a sixth of answers cut off mid-reasoning. That is not knowledge
# lost, and if one arm reasons at greater length than another it would bias
# the comparison, so the budget is wider and what still gets cut is counted.
MAX_NEW_TOKENS: Final = 320
MAX_SEQUENCE_TOKENS: Final = 2048


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


def _revision_for(model_id: str) -> str | None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    for entries in registry["roles"].values():
        for entry in entries:
            if entry["id"] == model_id:
                return entry["revision"]
    return None


def render(tokenizer, question) -> str:
    """The prompt, with no tool block attached."""

    choices = list(question.choices) + [""] * (len(CHOICE_LABELS) - len(question.choices))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT.format(
                question=question.question,
                a=choices[0],
                b=choices[1],
                c=choices[2],
                d=choices[3],
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def by_subject(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per-subject accuracy, so a broad loss can be told from a narrow one."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["subject"], []).append(row)
    return {
        subject: {
            "questions": len(group),
            "accuracy": sum(1 for r in group if r["correct"]) / len(group),
        }
        for subject, group in sorted(grouped.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--label", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--run-load", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    if args.run_load and not args.allow_download:
        parser.error("--run-load and --allow-download must be supplied together")

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "kind": "utility_eval",
        "label": args.label,
        "model": {"id": args.model, "revision": _revision_for(args.model)},
        "adapter": args.adapter,
        "benchmark": "mmlu",
        "split_manifest_sha256": hashlib.sha256(
            SPLIT_MANIFEST_PATH.read_bytes()
        ).hexdigest(),
        "decoding": {"greedy": True, "max_new_tokens": MAX_NEW_TOKENS},
        "tools_offered": False,
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
        print(json.dumps({"planned": args.label, "executed": False}))
        return 0

    questions = load_questions(SPLIT_MANIFEST_PATH, limit=args.limit)

    import unsloth  # noqa: F401  # must precede transformers; it rewrites it
    from unsloth import FastLanguageModel

    import torch

    loaded, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter or args.model),
        max_seq_length=MAX_SEQUENCE_TOKENS,
        dtype=None,
        load_in_4bit=True,
        trust_remote_code=False,
    )
    FastLanguageModel.for_inference(loaded)

    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    rows: list[dict[str, Any]] = []
    responses_path = Path(args.responses)
    responses_path.parent.mkdir(parents=True, exist_ok=True)

    with responses_path.open("w", encoding="utf-8") as handle:
        for start in range(0, len(questions), args.batch_size):
            chunk = questions[start : start + args.batch_size]
            prompts = [render(tokenizer, q) for q in chunk]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda:0")
            prompt_length = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                generated = loaded.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=pad,
                )
            for question, row in zip(chunk, generated):
                new_tokens = row[prompt_length:]
                # A generation that used its whole budget without emitting a
                # stop token was cut off rather than finished.
                truncated = (
                    len(new_tokens) >= MAX_NEW_TOKENS
                    and int(new_tokens[-1]) != tokenizer.eos_token_id
                )
                completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
                score = score_completion(
                    completion, gold_index=question.gold_index, truncated=truncated
                )
                record = {
                    "task_id": question.task_id,
                    "subject": question.subject,
                    "gold": CHOICE_LABELS[question.gold_index],
                    "extracted": score.extracted,
                    "correct": score.correct,
                    "emitted_tool_call": score.emitted_tool_call,
                    "truncated": truncated,
                    "completion": completion,
                }
                rows.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[utility] {min(start + args.batch_size, len(questions))}/"
                f"{len(questions)}",
                flush=True,
            )

    scores = [
        score_completion(
            r["completion"],
            gold_index=CHOICE_LABELS.index(r["gold"]),
            truncated=r["truncated"],
        )
        for r in rows
    ]
    result["summary"] = summarise(scores)
    result["by_subject"] = by_subject(rows)

    path = Path(args.summary)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(
        (json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )
    os.replace(temporary, path)

    print(json.dumps({"summary": str(path), "label": args.label, **result["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
