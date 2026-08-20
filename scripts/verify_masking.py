"""Prove on the real checkpoints that training labels cover assistant tokens only.

`tests/test_masking.py` proves the logic against a stub, because CI has no
network. That leaves one thing unproven: that the real Qwen3 chat template
behaves the way the stub says it does. This script closes that gap and writes
the answer down as a hashed artifact rather than as a claim.

Three properties are measured per checkpoint.

1. The mask is non-empty. `return_assistant_tokens_mask` returns all zeros
   instead of raising when a template carries no `{% generation %}` marker, so
   an unguarded pipeline trains on nothing and still reports a falling loss.
2. Marking is inert: the patched template produces byte-identical token ids to
   the native one, so training and evaluation read the same string.
3. The training row begins with exactly the token sequence the evaluator feeds
   the model at generation time, and the first trained token sits exactly where
   generation begins. This is the property that makes an SFT checkpoint
   comparable to its own baseline; if it fails, the trained model is being
   asked a different question than the one it was measured on.

Tokenizers only. No weights are loaded, so this runs anywhere in seconds.
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
from evaluation.rungs import SYSTEM_PROMPT, USER_PROMPT  # noqa: E402
from training.masking import (  # noqa: E402
    IGNORE_INDEX,
    MaskingError,
    encode_with_labels,
    training_template_for,
)

REGISTRY_PATH: Final = PROJECT_ROOT / "configs" / "model_candidates.json"
SCHEMA_VERSION: Final = 1

# Roles whose checkpoints are actually trained in Phase A.
TRAINED_ROLES: Final = ("primary_small", "scale_check")

# A representative retained trajectory: one question, one calculator call. The
# question is fixed rather than sampled so the artifact is reproducible without
# downloading the dataset.
FIXTURE_QUESTION: Final = (
    "Ken put 2 pounds of jelly beans in a box and twice as many pounds of "
    "brownies. How many pounds is the box?"
)
FIXTURE_CALL: Final = (
    '<tool_call>\n{"name": "calculator", "arguments": {"expression": "2 + 2*2"}}'
    "\n</tool_call>"
)


class VerificationError(RuntimeError):
    """A checkpoint failed a property the training pipeline depends on."""


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


def _candidates(selected: list[str]) -> list[dict[str, str]]:
    """Every trainable candidate, carrying whether it was actually selected.

    Rejected checkpoints are still measured. Qwen2.5 routes an assistant turn
    whose tool call sits in `content` through an earlier generic branch than the
    one the patch marks, so its mask comes back empty and the guard refuses it.
    That is worth recording rather than hiding, but it cannot gate a lane whose
    checkpoints D-048 already rejected on licence.
    """

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    out: list[dict[str, str]] = []
    for role in TRAINED_ROLES:
        for entry in registry["roles"].get(role, []):
            if selected and entry["id"] not in selected:
                continue
            out.append(
                {
                    "role": role,
                    "id": entry["id"],
                    "revision": entry["revision"],
                    "selection_status": entry.get("selection_status", "unknown"),
                }
            )
    return out


def _messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(question=FIXTURE_QUESTION)},
        {"role": "assistant", "content": FIXTURE_CALL},
    ]


def _token_ids(rendered: Any) -> list[int]:
    if isinstance(rendered, dict) or hasattr(rendered, "keys"):
        return list(rendered["input_ids"])
    return list(rendered)


def verify(candidate: dict[str, str]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        candidate["id"], revision=candidate["revision"]
    )
    tools = [calculator_tool_schema()]
    messages = _messages()

    example = encode_with_labels(tokenizer, messages, tools=tools)

    # The exact render the evaluator performs before generating.
    inference_ids = _token_ids(
        tokenizer.apply_chat_template(
            messages[:-1],
            tools=tools,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    first_trained = example.trained_spans[0][0]
    prefix_matches = list(example.input_ids[: len(inference_ids)]) == inference_ids
    starts_at_generation = first_trained == len(inference_ids)

    trained_text = example.trained_text(tokenizer)
    leaked = [
        message["role"]
        for message in messages
        if message["role"] != "assistant" and message["content"] in trained_text
    ]

    native = tokenizer.chat_template
    patched = training_template_for(tokenizer)

    result = {
        "candidate": candidate,
        "native_template_sha256": _sha256_text(native),
        "training_template_sha256": _sha256_text(patched),
        "native_template_marks_generation": native == patched,
        "total_tokens": len(example.input_ids),
        "trained_tokens": example.trained_token_count,
        "ignored_tokens": sum(
            1 for label in example.labels if label == IGNORE_INDEX
        ),
        "trained_span_count": len(example.trained_spans),
        "trained_text": trained_text,
        "trained_text_sha256": _sha256_text(trained_text),
        "inference_prompt_tokens": len(inference_ids),
        "training_row_starts_with_inference_prompt": prefix_matches,
        "first_trained_token_is_first_generated_token": starts_at_generation,
        "non_assistant_roles_leaked_into_trained_region": leaked,
    }
    result["passed"] = bool(
        example.trained_token_count > 0
        and prefix_matches
        and starts_at_generation
        and not leaked
        and len(example.trained_spans) == 1
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate", action="append", default=[])
    args = parser.parse_args()

    candidates = _candidates(args.candidate)
    if not candidates:
        parser.error("no registry candidate matched --candidate")

    results: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            results.append(verify(candidate))
        except (MaskingError, VerificationError) as error:
            results.append({"candidate": candidate, "passed": False, "error": str(error)})

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "kind": "masking_verification",
        "purpose": (
            "Evidence that assistant-token-only loss masking holds on the real "
            "checkpoints, and that a training row begins with exactly the token "
            "sequence the evaluator generates from."
        ),
        "prompt_sha256": {
            "system": _sha256_text(SYSTEM_PROMPT),
            "user": _sha256_text(USER_PROMPT),
        },
        "fixture_sha256": _sha256_text(FIXTURE_QUESTION + FIXTURE_CALL),
        "source_commit": _git_commit(),
        "platform": {
            "python": platform.python_version(),
            "system": platform.system(),
        },
        "results": results,
        # Only the selected bundle gates. A rejected checkpoint's result is
        # recorded so the difference is visible, never so it can block or
        # flatter the lane that is actually trained.
        "gated_on": [
            r["candidate"]["id"]
            for r in results
            if r["candidate"].get("selection_status") == "selected"
        ],
        "status": "passed"
        if all(
            r.get("passed")
            for r in results
            if r["candidate"].get("selection_status") == "selected"
        )
        else "failed",
    }

    path = Path(args.output)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(
        (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )
    os.replace(temporary, path)

    print(
        json.dumps(
            {
                "output": str(path),
                "status": payload["status"],
                "summary": {
                    r["candidate"]["id"]: {
                        "trained_tokens": r.get("trained_tokens"),
                        "total_tokens": r.get("total_tokens"),
                        "passed": r.get("passed"),
                    }
                    for r in results
                },
            }
        )
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
