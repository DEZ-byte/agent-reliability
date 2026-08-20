"""Measure how much of the Phase A test split a base model already remembers.

BLUEPRINT_v2 section 5.4 requires this before any Phase A baseline is read.
Each task is put to the model with no tool available and no way to compute, so
a correct answer means the model recalled it.

Offline by default. Loading a checkpoint needs --run-load and --allow-download,
matching scripts/smoke_models.py, and a measured run refuses to start on a dirty
worktree so every artifact names the exact source that produced it.

This records a diagnostic, never a task score. A high recall rate does not
invalidate a later baseline; it changes what that baseline means, which is why
it is measured first rather than argued about afterwards.
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

from env.phase_a import ANSWER_TOLERANCE, parse_gsm8k_answer  # noqa: E402
from evaluation.contamination import recall_rate, score_recall  # noqa: E402

SPLIT_PATH: Final = PROJECT_ROOT / "configs" / "splits" / "phase_a_gsm8k.json"
REGISTRY_PATH: Final = PROJECT_ROOT / "configs" / "model_candidates.json"
SCHEMA_VERSION: Final = 1
MAX_NEW_TOKENS: Final = 256
MAX_SEQUENCE_TOKENS: Final = 1024
SEED: Final = 20260820

PROMPT: Final = (
    "Answer the question with a single final number and nothing else.\n"
    "You have no tools and must not show working.\n\n"
    "Question: {question}\n"
    "Final answer:"
)

MEASURED_ROLES: Final = (
    "primary_small",
    "scale_check",
    "cross_family_check",
    "scaffolded_comparator",
)


class ProbeError(RuntimeError):
    """Raised when the probe cannot honestly proceed."""


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ProbeError("git " + " ".join(args) + " failed")
    return completed.stdout.strip()


def _require_clean_worktree() -> None:
    if _git("status", "--porcelain"):
        raise ProbeError(
            "refusing to measure on a dirty worktree; commit first so the "
            "artifact names the exact source that produced it"
        )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_digest(question: str, answer: str) -> str:
    payload = json.dumps(
        {"question": question, "answer": answer}, sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_tasks(limit: int | None) -> list[dict[str, Any]]:
    """Read the frozen test split, verifying each item against its hash."""

    manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    dataset = manifest["dataset"]
    from datasets import load_dataset

    rows = load_dataset(
        dataset["id"],
        dataset["config"],
        split="test",
        revision=dataset["revision"],
    )
    tasks: list[dict[str, Any]] = []
    for entry in manifest["splits"]["test"]:
        row = rows[entry["source_index"]]
        digest = _content_digest(row["question"], row["answer"])
        if digest != entry["content_sha256"]:
            raise ProbeError(
                entry["task_id"]
                + " does not match its recorded content hash; the pinned "
                "revision or the manifest has moved"
            )
        tasks.append(
            {
                "task_id": entry["task_id"],
                "question": row["question"],
                "gold_answer": parse_gsm8k_answer(row["answer"]),
            }
        )
    return tasks[:limit] if limit else tasks


def _candidates() -> list[dict[str, str]]:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    out: list[dict[str, str]] = []
    for role in MEASURED_ROLES:
        for entry in registry["roles"].get(role, []):
            if entry.get("selection_status") == "rejected":
                continue
            out.append(
                {"role": role, "id": entry["id"], "revision": entry["revision"]}
            )
    return out


def _plan(task_count: int, candidates: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "purpose": (
            "no-tool recall over the frozen Phase A test split; diagnostic "
            "only, never a task score"
        ),
        "task_count": task_count,
        "candidates": candidates,
        "prompt_template": PROMPT,
        "decoding": {
            "seed": SEED,
            "greedy": True,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "answer_tolerance": ANSWER_TOLERANCE,
        "gated_operations": ["checkpoint download and load", "generation"],
    }


def _measure(candidate: dict[str, str], tasks: list[dict[str, Any]]) -> dict[str, Any]:
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
        random_state=SEED,
        disable_log_stats=True,
    )
    FastLanguageModel.for_inference(model)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    probes = []
    for task in tasks:
        messages = [
            {"role": "user", "content": PROMPT.format(question=task["question"])}
        ]
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = tokenizer(rendered, return_tensors="pt").to("cuda:0")
        torch.manual_seed(SEED)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        completion = tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        probes.append(
            score_recall(
                task_id=task["task_id"],
                gold_answer=task["gold_answer"],
                completion=completion,
                tolerance=ANSWER_TOLERANCE,
            )
        )

    del model
    torch.cuda.empty_cache()

    return {
        "candidate": candidate,
        "task_count": len(probes),
        "recall_rate": recall_rate(probes),
        "recalled_task_count": sum(1 for probe in probes if probe.recalled),
        "probes": [probe.model_dump(mode="json") for probe in probes],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-load", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    if args.run_load and not args.allow_download:
        parser.error("--run-load and --allow-download must be supplied together")

    candidates = _candidates()
    if args.candidate:
        candidates = [c for c in candidates if c["id"] in args.candidate]
        if not candidates:
            parser.error("no registry candidate matched --candidate")

    tasks = _load_tasks(args.limit) if args.run_load else []
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "kind": "phase_a_no_tool_recall",
        "plan": _plan(len(tasks), candidates),
        "split_manifest_sha256": _sha256_file(SPLIT_PATH),
        "registry_sha256": _sha256_file(REGISTRY_PATH),
        "executed": bool(args.run_load),
        "results": [],
    }

    if not args.run_load:
        result["note"] = "planned offline; no checkpoint was loaded"
        Path(args.output).write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"planned_candidates": len(candidates)}))
        return 0

    _require_clean_worktree()
    result["source_commit"] = _git("rev-parse", "HEAD")
    result["platform"] = platform.platform()

    for candidate in candidates:
        try:
            result["results"].append(_measure(candidate, tasks))
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

    summary = {
        entry["candidate"]["id"]: entry.get("recall_rate", entry.get("error"))
        for entry in result["results"]
    }
    print(json.dumps({"output": str(path), "recall_rate": summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
