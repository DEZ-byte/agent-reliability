"""Record token counts and cost for every GSM8K row, plus multi-turn token counts.

The GSM8K evaluation counts generated tokens as it generates. Three rows ran
before that code existed: the untrained Qwen3-1.7B, the untrained Qwen3-4B and
SFT run 1. Their counts are rebuilt here by re-tokenizing the decoded
completions in their episode logs, with each model's pinned tokenizer.

The rebuild is a reconstruction, not a count taken at generation time, so it is
checked first. Applied to the four Qwen3-1.7B runs that did record their
counts, it must give back all eight recorded rung totals exactly, or nothing is
written. The rule is not applied to Llama-3.1-8B, whose recorded counts are
kept; the artifact shows what the rule would have given for it.

The multi-turn order-support runs (untrained, SFT run 3, SFT + GRPO) recorded
each turn's prompt length but not the tokens each turn generated. The same rule
rebuilds those, summed over every turn of an episode, at the cap the summary
records for that evaluation. The rule was validated only on GSM8K at its
256-token cap; there are no recorded multi-turn totals to check it against.

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
    MultiTurnCount,
    RungCount,
    TokenCountError,
    check_rounded,
    check_rung,
    cost,
    parameter_counts,
    rebuild_multiturn,
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

# The multi-turn order-support runs, one per checkpoint, all on Qwen3-1.7B.
# Each log holds one model only, so its rows carry no candidate field.
MULTITURN: Final = (
    {
        "label": "base",
        "row": "Qwen3-1.7B, untrained",
        "summary": "order-support-multiturn-base-3cacc39.json",
        "episodes": "episodes-order-support-multiturn-base-3cacc39.jsonl",
    },
    {
        "label": "sft",
        "row": "Qwen3-1.7B, SFT run 3",
        "summary": "order-support-multiturn-sft-3cacc39.json",
        "episodes": "episodes-order-support-multiturn-sft-3cacc39.jsonl",
    },
    {
        "label": "grpo",
        "row": "Qwen3-1.7B, SFT + GRPO, lr 1e-6",
        "summary": "order-support-multiturn-grpo-3cacc39.json",
        "episodes": "episodes-order-support-multiturn-grpo-3cacc39.jsonl",
    },
)
MULTITURN_MODEL: Final = QWEN_SMALL

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
        "as its own decision. In the multi-turn order-support runs each turn is one "
        "decision, and max_new_tokens is the cap that evaluation's summary records."
    ),
    "validation": (
        "Before anything is written, the rule is applied to every Qwen3-1.7B run "
        "that recorded its counts and has its episode log, and must give back each "
        "recorded rung total exactly. The rule is not applied to Llama-3.1-8B; "
        "rule_on_llama shows what it gives there, beside the recorded totals that are kept. "
        "This check covers GSM8K only, at a 256-token cap. The order-support "
        "multi-turn counts apply the same rule at that evaluation's own cap, and no "
        "recorded generated totals exist there to check them against."
    ),
    "tokens_per_attempt": "tokens_total divided by episodes; one episode is one attempt.",
    "cost": "tokens_per_attempt times the model's unique parameters in billions (rounded).",
    "order_support_multiturn": (
        "generated is rebuilt from the logs with the rule above, summed over every "
        "turn of an episode. prompt is recorded: the prompt (prefill) length of "
        "each turn's generate() call, logged as prompt_tokens_per_turn, summed the "
        "same way. Each turn's prompt holds the whole conversation so far. "
        "turns_at_cap counts turns whose rebuilt count equals the cap: the "
        "replies that used the whole budget. cost is generated tokens per episode times "
        "unique parameters in billions."
    ),
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


def multiturn_limit(summary: dict[str, Any], name: str, expected: int) -> int:
    """The per-turn cap a multi-turn summary records; refuse a missing or other value."""

    limit = summary.get("rollout", {}).get("max_new_tokens")
    if limit is None:
        raise BackfillError(f"{name} records no max_new_tokens")
    if limit != expected:
        raise BackfillError(f"{name} ran with max_new_tokens={limit}, not {expected}")
    return int(limit)


def check_multiturn(
    count: MultiTurnCount, summary: dict[str, Any], spec: dict[str, Any]
) -> None:
    """Refuse a log that does not match its summary's label, model, episodes or turns."""

    name = spec["summary"]
    if summary.get("executed") is not True or summary.get("mode") != "multi_turn":
        raise BackfillError(f"{name} is not an executed multi-turn run")
    if summary.get("label") != spec["label"]:
        raise BackfillError(f"{name} is labeled {summary.get('label')}, not {spec['label']}")
    model = summary.get("model", {}).get("id")
    if model != MULTITURN_MODEL:
        raise BackfillError(f"{name} ran {model}, not {MULTITURN_MODEL}")
    histogram = summary["multi_turn"]["turns_used"]
    turns = sum(int(used) * n for used, n in histogram.items())
    expected = (summary.get("episodes"), turns)
    if (count.episodes, count.turns) != expected:
        raise BackfillError(
            f"{name}: the log gives {count.episodes} episodes and {count.turns} turns, "
            f"the summary {expected[0]} and {expected[1]}"
        )


def multiturn_entry(
    spec: dict[str, Any], count: MultiTurnCount, limit: int, params_b: float
) -> dict[str, Any]:
    return {
        "row": spec["row"],
        "model": MULTITURN_MODEL,
        "summary": spec["summary"],
        "log": spec["episodes"],
        "max_new_tokens": limit,
        "episodes": count.episodes,
        "turns": count.turns,
        "generated": {
            "tokens_total": count.generated_total,
            "tokens_per_episode": count.generated_per_episode,
            "tokens_per_turn": count.generated_per_turn,
            "turns_at_cap": count.turns_at_cap,
            "source": REBUILT,
        },
        "prompt": {
            "tokens_total": count.prompt_total,
            "tokens_per_episode": count.prompt_per_episode,
            "tokens_per_turn": count.prompt_per_turn,
            "source": RECORDED,
        },
        "unique_params_b": params_b,
        "cost": cost(count.generated_per_episode, params_b),
    }


def build(
    episodes_dir: Path, max_new_tokens: int, order_support_max_new_tokens: int = 384
) -> dict[str, Any]:
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

    # 3. The multi-turn order-support runs: generated rebuilt, prompt recorded.
    multiturn: dict[str, Any] = {}
    for spec in MULTITURN:
        path = RESULTS_DIR / spec["summary"]
        stored = read_json(path)
        inputs[spec["summary"]] = hashlib.sha256(path.read_bytes()).hexdigest()
        limit = multiturn_limit(stored, spec["summary"], order_support_max_new_tokens)
        count = rebuild_multiturn(
            log(spec["episodes"]), encode=encoder(MULTITURN_MODEL), max_new_tokens=limit
        )
        check_multiturn(count, stored, spec)
        multiturn[spec["label"]] = multiturn_entry(
            spec, count, limit, UNIQUE_PARAMS_B[MULTITURN_MODEL]
        )

    return {
        "max_new_tokens": max_new_tokens,
        "parameters": params,
        "validation": {"qwen": checks, "rule_on_llama": llama},
        "inputs": [{"file": name, "sha256": inputs[name]} for name in sorted(inputs)],
        "rows": rows,
        "order_support_multiturn": multiturn,
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
        help="the generation cap the GSM8K evaluations used; every GSM8K summary "
        "read must record this value (default: 256)",
    )
    parser.add_argument(
        "--order-support-max-new-tokens",
        type=int,
        default=384,
        help="the per-turn generation cap of the multi-turn order-support runs; "
        "every multi-turn summary must record this value (default: 384)",
    )
    args = parser.parse_args(argv)

    source_commit = require_clean_worktree(PROJECT_ROOT)
    body = build(args.episodes_dir, args.max_new_tokens, args.order_support_max_new_tokens)
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
    for label, entry in body["order_support_multiturn"].items():
        generated, prompt = entry["generated"], entry["prompt"]
        print(
            f"order support, multi-turn, {label}: {entry['turns']} turns, "
            f"{generated['tokens_per_episode']:.2f} generated/episode ({generated['source']}), "
            f"{prompt['tokens_per_episode']:.2f} prompt/episode ({prompt['source']}), "
            f"cost {entry['cost']:.1f}"
        )
    print(f"wrote {path.name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except REFUSALS as error:
        print(f"refused: {error}", file=sys.stderr)
        raise SystemExit(1)
