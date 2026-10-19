"""Generated-token counts and unique parameter counts, for the cost column.

The cost of one attempt is the tokens it generated times the model's parameters
in billions. The GSM8K evaluation counts generated tokens as it generates. Three
older runs predate that code, so their counts are rebuilt here from the decoded
completions in their episode logs.

The rebuild rule is `min(len(encode(text)) + 1, max_new_tokens)` per decision.
Decoding drops the end-of-sequence token, so it is added back, and a reply that
hit the limit stopped without one. The rule is only trusted after it gives
back every token total the evaluation recorded for the same tokenizer.

Nothing here imports a tokenizer library. The caller passes `encode`.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

Encode = Callable[[str], Sequence[int]]

# Each parameter counted once, in billions, rounded to two places. Qwen3-1.7B
# ties its output head to its embedding, but the checkpoint also stores the
# head as a copy; counting that copy gives 2.03, which overstates the model.
UNIQUE_PARAMS_B: Final[dict[str, float]] = {
    "Qwen/Qwen3-1.7B": 1.72,
    "Qwen/Qwen3-4B": 4.02,
    "meta-llama/Llama-3.1-8B-Instruct": 8.03,
}

# The weight a model with tied embeddings stores twice, or does not store at all.
TIED_HEAD: Final = "lm_head.weight"


class TokenCountError(RuntimeError):
    """The logs, summaries or checkpoints do not support the counts."""


@dataclass(frozen=True)
class RungCount:
    """Generated tokens for one model at one rung."""

    episodes: int
    decisions: int
    tokens_total: int

    @property
    def tokens_per_attempt(self) -> float:
        return self.tokens_total / self.episodes


def decision_tokens(text: str, encode: Encode, max_new_tokens: int) -> int:
    """Tokens one decision generated: the decoded text, plus the dropped EOS, capped."""

    return min(len(encode(text)) + 1, max_new_tokens)


def rebuild_rung(
    rows: Iterable[Mapping[str, Any]],
    *,
    candidate: str,
    rung: str,
    encode: Encode,
    max_new_tokens: int,
) -> RungCount:
    """Sum the rule over every decision of one candidate's episodes at one rung.

    A log can hold more than one model under the same task ids, so rows are
    picked by the candidate they record. Each completion is one decision; at R1
    the retry, when it fired, is the second completion.
    """

    episodes = decisions = total = 0
    for row in rows:
        if row.get("candidate") != candidate or row.get("rung") != rung:
            continue
        completions = row["completions"]
        logged = row["counters"]["policy_model_decision_count"]
        if len(completions) != logged:
            raise TokenCountError(
                f"{candidate} {rung} {row['task_id']} run {row['run_index']}: "
                f"{len(completions)} completions but {logged} logged decisions"
            )
        episodes += 1
        decisions += len(completions)
        total += sum(decision_tokens(text, encode, max_new_tokens) for text in completions)
    if episodes == 0:
        raise TokenCountError(f"the log holds no {rung} rows for {candidate}")
    return RungCount(episodes=episodes, decisions=decisions, tokens_total=total)


def recorded_rung(stored: Mapping[str, Any]) -> RungCount | None:
    """The counts a summary recorded for one rung, or None if it has none."""

    total = stored.get("generated_tokens_total")
    if total is None:
        return None
    count = RungCount(
        episodes=stored["episodes"], decisions=stored["model_decisions"], tokens_total=total
    )
    if stored.get("generated_tokens_per_episode") != count.tokens_per_attempt:
        raise TokenCountError("the recorded tokens per episode do not equal total / episodes")
    return count


def check_rung(rebuilt: RungCount, stored: Mapping[str, Any], label: str) -> None:
    """Refuse a rebuild whose episodes or decisions differ from the summary."""

    expected = (stored.get("episodes"), stored.get("model_decisions"))
    if (rebuilt.episodes, rebuilt.decisions) != expected:
        raise TokenCountError(
            f"{label}: the log gives {rebuilt.episodes} episodes and {rebuilt.decisions} "
            f"decisions, the summary {expected[0]} and {expected[1]}"
        )


def validate(checks: Sequence[Mapping[str, Any]]) -> None:
    """Refuse unless every rebuilt total equals the recorded one.

    Each check holds `label`, `recorded` and `rebuilt`.
    """

    if not checks:
        raise TokenCountError("no recorded totals to validate the rule against")
    wrong = [check for check in checks if check["rebuilt"] != check["recorded"]]
    if wrong:
        detail = "; ".join(
            f"{check['label']}: recorded {check['recorded']}, rebuilt {check['rebuilt']}"
            for check in wrong
        )
        raise TokenCountError(
            f"the rebuild rule does not reproduce {len(wrong)} of {len(checks)} "
            f"recorded totals, so it cannot stand in for them: {detail}"
        )


def cost(tokens_per_attempt: float, params_b: float) -> float:
    """Generated tokens per attempt times parameters in billions."""

    return tokens_per_attempt * params_b


def safetensors_shapes(path: Path) -> dict[str, list[int]]:
    """Each tensor's shape, read from a safetensors header without the weights."""

    with Path(path).open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    return {name: entry["shape"] for name, entry in header.items() if name != "__metadata__"}


def parameter_counts(snapshot: Path) -> dict[str, Any]:
    """Stored and unique parameter counts for a checkpoint folder.

    Stored counts every tensor in the safetensors files. Unique drops the
    output head when config.json ties it to the embedding, because the loaded
    model then holds one matrix for both.
    """

    snapshot = Path(snapshot)
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    tied = bool(config.get("tie_word_embeddings", False))
    shapes: dict[str, list[int]] = {}
    files = sorted(snapshot.glob("*.safetensors"))
    if not files:
        raise TokenCountError(f"no safetensors files in the {snapshot.name} snapshot")
    for file in files:
        shapes.update(safetensors_shapes(file))

    def size(shape: Sequence[int]) -> int:
        product = 1
        for dim in shape:
            product *= dim
        return product

    stored = sum(size(shape) for shape in shapes.values())
    unique = stored - (size(shapes[TIED_HEAD]) if tied and TIED_HEAD in shapes else 0)
    return {
        "stored": stored,
        "unique": unique,
        "tie_word_embeddings": tied,
        "head_stored_separately": TIED_HEAD in shapes,
    }


def check_rounded(model: str, unique: int) -> None:
    """Refuse if the rounded constant is not the exact count to two places."""

    rounded = UNIQUE_PARAMS_B[model]
    if round(unique / 1e9, 2) != rounded:
        raise TokenCountError(
            f"{model}: the checkpoint holds {unique} unique parameters, "
            f"which does not round to the constant {rounded}B"
        )
