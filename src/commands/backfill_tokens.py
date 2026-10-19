"""Record generated-token counts and cost for every model row in the README.

The GSM8K evaluation counts generated tokens as it generates. Three rows ran
before that code existed: the untrained Qwen3-1.7B, the untrained Qwen3-4B and
SFT run 1. Their counts are rebuilt here by re-tokenizing the decoded
completions in their episode logs, with each model's pinned tokenizer.

The rebuild is a reconstruction, not a count taken at generation time, so it is
checked first. Applied to the four Qwen3-1.7B runs that did record their
counts, it must give back all eight recorded rung totals exactly, or nothing is
written. The rule is not applied to Llama-3.1-8B, whose recorded counts are
kept; the artifact shows what the rule would have given for it.

Cost is tokens per attempt times unique parameters in billions. The parameter
constants are checked against the safetensors headers of the pinned snapshots
when those are in the local cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.provenance import (  # noqa: E402
    DirtyWorktreeError,
    UnpinnedModelError,
    pinned_revision,
    require_clean_worktree,
)
from evaluation.token_counts import (  # noqa: E402
    UNIQUE_PARAMS_B,
    Encode,
    RungCount,
    TokenCountError,
    check_rounded,
    check_rung,
    cost,
    parameter_counts,
    rebuild_rung,
    recorded_rung,
    validate,
)

SCHEMA_VERSION: Final = 1
RUNGS: Final = ("R0", "R1")
RESULTS_DIR: Final = PROJECT_ROOT / "results"
REGISTRY: Final = PROJECT_ROOT / "configs" / "model_candidates.json"
RECORDED: Final = "recorded"
REBUILT: Final = "rebuilt_from_logs"

QWEN_SMALL: Final = "Qwen/Qwen3-1.7B"
QWEN_TEACHER: Final = "Qwen/Qwen3-4B"
LLAMA: Final = "meta-llama/Llama-3.1-8B-Instruct"

# One entry per model row of the README cost table, in its order. A recorded
# row with an episode log is one the rebuild rule is validated on.
ROWS: Final = (
    {
        "row": "Qwen3-1.7B, untrained",
        "model": QWEN_SMALL,
        "summary": "baseline-phase_a-3cc174f.json",
        "episodes": "episodes-phase_a-3cc174f.jsonl",
        "source": REBUILT,
    },
    {
        "row": "Llama-3.1-8B + retry",
        "model": LLAMA,
        "summary": "comparator-8b-2889b6d.json",
        "episodes": "episodes-comparator-8b-2889b6d.jsonl",
        "source": RECORDED,
    },
    {
        "row": "Qwen3-1.7B, SFT run 1",
        "seed": 20260820,
        "model": QWEN_SMALL,
        "summary": "sft-test-qwen3-1.7b-c16530e.json",
        "episodes": "episodes-sft-test-qwen3-1.7b-c16530e.jsonl",
        "source": REBUILT,
    },
    {
        "row": "Qwen3-1.7B, SFT run 2",
        "seed": 20260823,
        "model": QWEN_SMALL,
        "summary": "sft-test-qwen3-1.7b-seed20260823-67488ea.json",
        "episodes": "episodes-sft-test-qwen3-1.7b-seed20260823-67488ea.jsonl",
        "source": RECORDED,
    },
    {
        "row": "Qwen3-1.7B, SFT run 3",
        "seed": 20260824,
        "model": QWEN_SMALL,
        "summary": "sft-test-qwen3-1.7b-seed20260824-51f16bf.json",
        "episodes": "episodes-sft-test-qwen3-1.7b-seed20260824-51f16bf.jsonl",
        "source": RECORDED,
    },
    {
        "row": "Qwen3-1.7B, SFT + GRPO, lr 1e-6",
        "model": QWEN_SMALL,
        "summary": "grpo-test-qwen3-1.7b-f6138ee.json",
        "episodes": "episodes-grpo-test-qwen3-1.7b-f6138ee.jsonl",
        "source": RECORDED,
    },
    {
        "row": "Qwen3-1.7B, SFT + GRPO, lr 1e-5",
        "model": QWEN_SMALL,
        "summary": "grpo-test-lr1e5-ce5f2ca.json",
        "episodes": "episodes-grpo-test-lr1e5-ce5f2ca.jsonl",
        "source": RECORDED,
    },
    {
        "row": "Qwen3-4B, untrained (the teacher)",
        "model": QWEN_TEACHER,
        "summary": "baseline-phase_a-3cc174f.json",
        "episodes": "episodes-phase_a-3cc174f.jsonl",
        "source": REBUILT,
    },
)

METHOD: Final = {
    RECORDED: (
        "Counted at generation time: the length of each generate() call's new "
        "tokens, which includes the end-of-sequence token, summed over every decision."
    ),
    REBUILT: (
        "Reconstructed after the fact, not counted at generation time. Each decoded "
        "completion in the episode log is re-tokenized with the model's pinned "
        "tokenizer, add_special_tokens=False, and counted as "
        "min(tokens + 1, max_new_tokens): decoding drops the end-of-sequence token, "
        "and a reply that hit the limit has none. At R1 the retry completion counts "
        "as its own decision."
    ),
    "validation": (
        "Before anything is written, the rule is applied to every Qwen3-1.7B run "
        "that recorded its counts and has its episode log, and must give back each "
        "recorded rung total exactly. The rule is not applied to Llama-3.1-8B; "
        "rule_on_llama shows what it gives there, beside the recorded totals that are kept."
    ),
    "tokens_per_attempt": "tokens_total divided by episodes; one episode is one attempt.",
    "cost": "tokens_per_attempt times the model's unique parameters in billions (rounded).",
}

PARAMETER_CONVENTION: Final = (
    "Each parameter is counted once. A model whose config ties the output head to "
    "the embedding holds one matrix for both, so a stored copy of the head is not "
    "counted again. stored counts every tensor in the safetensors files; unique "
    "drops that copy. rounded_billions is the constant cost uses."
)


class BackfillError(TokenCountError):
    """An input is missing or does not match what this command expects."""


# Refusals exit 1 with a one-line message instead of a traceback.
REFUSALS: Final = (TokenCountError, DirtyWorktreeError, UnpinnedModelError)


def load_encoder(model: str, revision: str) -> Encode:
    """The pinned tokenizer's encode, read from the local cache only."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=True)
    return lambda text: tokenizer.encode(text, add_special_tokens=False)


def snapshot_dir(model: str, revision: str) -> Path | None:
    """The cached snapshot folder for a pinned revision, or None if it is not cached."""

    from huggingface_hub import try_to_load_from_cache

    found = try_to_load_from_cache(model, "config.json", revision=revision)
    return Path(found).parent if isinstance(found, str) else None


def tool_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("transformers", "tokenizers"):
        try:
            module = __import__(name)
        except ImportError:
            versions[name] = None
        else:
            versions[name] = getattr(module, "__version__", None)
    return versions


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BackfillError(f"missing input {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BackfillError(f"missing episode log {path.name}")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def stored_rungs(summary: dict[str, Any], model: str, name: str) -> dict[str, Any]:
    """The summary's per-rung results for one model."""

    results = [r for r in summary.get("results", []) if r["candidate"]["id"] == model]
    if summary.get("executed") is not True or len(results) != 1:
        raise BackfillError(f"{name} holds no single executed result for {model}")
    rungs = results[0].get("rungs", {})
    missing = [rung for rung in RUNGS if rung not in rungs]
    if missing:
        raise BackfillError(f"{name} has no {', '.join(missing)} result for {model}")
    return rungs


def parameters(models: list[str]) -> dict[str, Any]:
    """Exact and rounded parameter counts per model, checked when the snapshot is cached."""

    out: dict[str, Any] = {"convention": PARAMETER_CONVENTION, "models": {}}
    for model in models:
        revision = pinned_revision(model, REGISTRY)
        entry: dict[str, Any] = {"revision": revision, "rounded_billions": UNIQUE_PARAMS_B[model]}
        snapshot = snapshot_dir(model, revision)
        if snapshot is None:
            entry.update(stored=None, unique=None, checked_against_checkpoint=False)
        else:
            counts = parameter_counts(snapshot)
            check_rounded(model, counts["unique"])
            entry.update(counts, checked_against_checkpoint=True)
        out["models"][model] = entry
    return out


def rung_entry(count: RungCount, source: str, params_b: float) -> dict[str, Any]:
    return {
        "tokens_per_attempt": count.tokens_per_attempt,
        "tokens_total": count.tokens_total,
        "episodes": count.episodes,
        "decisions": count.decisions,
        "source": source,
        "cost": cost(count.tokens_per_attempt, params_b),
    }


def build(episodes_dir: Path, max_new_tokens: int) -> dict[str, Any]:
    """Validate the rule, then count every row. Returns the artifact body."""

    summaries: dict[str, dict[str, Any]] = {}
    logs: dict[str, list[dict[str, Any]]] = {}
    inputs: dict[str, str] = {}

    def summary(name: str) -> dict[str, Any]:
        if name not in summaries:
            path = RESULTS_DIR / name
            summaries[name] = read_json(path)
            inputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            limit = summaries[name].get("decoding", {}).get("max_new_tokens")
            if limit != max_new_tokens:
                raise BackfillError(
                    f"{name} ran with max_new_tokens={limit}, not {max_new_tokens}"
                )
        return summaries[name]

    def log(name: str) -> list[dict[str, Any]]:
        if name not in logs:
            path = episodes_dir / name
            logs[name] = read_rows(path)
            inputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return logs[name]

    encoders: dict[str, Encode] = {}

    def encoder(model: str) -> Encode:
        if model not in encoders:
            encoders[model] = load_encoder(model, pinned_revision(model, REGISTRY))
        return encoders[model]

    def rebuild(spec: dict[str, Any], rung: str) -> RungCount:
        return rebuild_rung(
            log(spec["episodes"]),
            candidate=spec["model"],
            rung=rung,
            encode=encoder(spec["model"]),
            max_new_tokens=max_new_tokens,
        )

    # 1. The rule must reproduce every recorded Qwen total before it is used.
    checks: list[dict[str, Any]] = []
    llama: list[dict[str, Any]] = []
    for spec in ROWS:
        if spec["source"] != RECORDED:
            continue
        stored = stored_rungs(summary(spec["summary"]), spec["model"], spec["summary"])
        for rung in RUNGS:
            recorded = recorded_rung(stored[rung])
            if recorded is None:
                raise BackfillError(f"{spec['summary']} recorded no {rung} token counts")
            rebuilt = rebuild(spec, rung)
            check_rung(rebuilt, stored[rung], f"{spec['row']} {rung}")
            check = {
                "label": f"{spec['row']} {rung}",
                "summary": spec["summary"],
                "rung": rung,
                "recorded": recorded.tokens_total,
                "rebuilt": rebuilt.tokens_total,
            }
            if spec["model"] == LLAMA:
                llama.append({**check, "difference": rebuilt.tokens_total - recorded.tokens_total})
            else:
                checks.append({**check, "match": rebuilt.tokens_total == recorded.tokens_total})
    validate(checks)

    # 2. Count every row: recorded where the summary has it, rebuilt otherwise.
    models = sorted({spec["model"] for spec in ROWS})
    params = parameters(models)
    rows: list[dict[str, Any]] = []
    for spec in ROWS:
        stored = stored_rungs(summary(spec["summary"]), spec["model"], spec["summary"])
        params_b = UNIQUE_PARAMS_B[spec["model"]]
        rungs: dict[str, Any] = {}
        for rung in RUNGS:
            if spec["source"] == RECORDED:
                count = recorded_rung(stored[rung])
            else:
                if recorded_rung(stored[rung]) is not None:
                    raise BackfillError(f"{spec['summary']} already records {rung} token counts")
                count = rebuild(spec, rung)
                check_rung(count, stored[rung], f"{spec['row']} {rung}")
            rungs[rung] = rung_entry(count, spec["source"], params_b)
        row = {"row": spec["row"], "model": spec["model"], "summary": spec["summary"]}
        if "seed" in spec:
            row["seed"] = spec["seed"]
        rows.append({**row, "unique_params_b": params_b, "rungs": rungs})

    return {
        "max_new_tokens": max_new_tokens,
        "parameters": params,
        "validation": {"qwen": checks, "rule_on_llama": llama},
        "inputs": [{"file": name, "sha256": inputs[name]} for name in sorted(inputs)],
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--episodes-dir",
        type=Path,
        default=RESULTS_DIR,
        help="folder holding the episode logs (default: results/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="artifact path (default: results/token-counts-<commit>.json)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="the generation cap the evaluations used; every summary read must "
        "record this value (default: 256)",
    )
    args = parser.parse_args(argv)

    source_commit = require_clean_worktree(PROJECT_ROOT)
    body = build(args.episodes_dir, args.max_new_tokens)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "token_counts",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_commit": source_commit,
        "platform": {
            "python": platform.python_version(),
            "system": platform.system(),
            **tool_versions(),
        },
        "method": METHOD,
        **body,
    }
    path = args.output or RESULTS_DIR / f"token-counts-{source_commit[:7]}.json"
    temporary = path.with_suffix(".tmp")
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    temporary.write_bytes(text.encode("utf-8"))
    os.replace(temporary, path)

    for check in body["validation"]["qwen"]:
        print(
            f"validated {check['label']}: "
            f"recorded {check['recorded']}, rebuilt {check['rebuilt']}"
        )
    for row in body["rows"]:
        r0, r1 = row["rungs"]["R0"], row["rungs"]["R1"]
        print(
            f"{row['row']}: R0 {r0['tokens_per_attempt']:.2f}, R1 {r1['tokens_per_attempt']:.2f} "
            f"tokens/attempt ({r1['source']}), R1 cost {r1['cost']:.1f}"
        )
    print(f"wrote {path.name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except REFUSALS as error:
        print(f"refused: {error}", file=sys.stderr)
        raise SystemExit(1)
