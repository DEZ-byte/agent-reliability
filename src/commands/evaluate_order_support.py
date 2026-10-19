"""Run a checkpoint on order support to test whether GSM8K tool training transfers.

The SFT checkpoints were trained on GSM8K word problems with one calculator
tool. GSM8K tests them on more of the same, so it cannot tell apart two
explanations of the gain: the model learned to use tools, or it learned to use
the calculator.

Order support has three different tools, in a domain with no arithmetic. A gain
that holds here is transfer. A gain that disappears was specific to the
training task.

The headline number is `pass^k` on the same terms as GSM8K. The diagnostics
separate the ways a model can fail:

- No tool call means tool calling did not transfer.
- An `update_order` call on every task means the model writes without
  verifying.
- Passing the `fulfil` requests while failing the `refuse` requests shows the
  same thing, which a benchmark with no refusals would score as success.
- The gate violation rate counts writes made before the tool rules allowed
  them.

Graded in `GateMode.AUDIT`. Enforcement would block the unauthorized writes and
hide them; audit lets each write execute so it can be counted.

`--mode single_turn` (the default) is the run above: every tool call is written
in one generation and executed afterward. `--mode multi_turn` runs the closed
loop in `evaluation/multiturn.py`: the same first prompt, then each turn's tool
results go back to the model before its next reply, for up to `--max-turns`
replies. It writes its own artifact kind, `order_support_multiturn_eval`, and
records how many turns had a prompt too long to leave room for a full reply
within the model's context length.
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

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.dialects import normalize_tool_dialect, template_uses_canonical_tags  # noqa: E402
from agent.gates import GateEngine, GateMode  # noqa: E402
from agent.parser import parse_tool_calls  # noqa: E402
from env.order_support import (  # noqa: E402
    INTENT_FULFIL,
    INTENT_REFUSE,
    UPDATE_ORDER_TOOL_NAME,
    build_order_support_registry,
    grade_episode,
    initial_state,
    order_support_tool_schemas,
    touched_the_order,
)
from env.order_support_tasks import load_split  # noqa: E402
from evaluation.metrics import compute_pass_metrics  # noqa: E402
from evaluation.multiturn import (  # noqa: E402
    DEFAULT_MAX_TURNS,
    evaluate_episode,
    run_episode,
    summarize_multi_turn,
)
from evaluation.provenance import (  # noqa: E402
    adapter_weights_sha256,
    pinned_load_kwargs,
    pinned_revision,
    portable_path,
    require_clean_worktree,
)
from training.rewards import score_episode  # noqa: E402

SPLIT_MANIFEST_PATH: Final = PROJECT_ROOT / "configs" / "splits" / "phase_b_orders.json"
GATES_PATH: Final = PROJECT_ROOT / "configs" / "gates.yaml"
REGISTRY_PATH: Final = PROJECT_ROOT / "configs" / "model_candidates.json"
SCHEMA_VERSION: Final = 1

# Deliberately parallel to the GSM8K prompt in wording and shape, so a
# difference in score reflects the task rather than a differently written
# instruction. It names no tool: which tools exist arrives through the chat
# template's tool block, exactly as for GSM8K.
SYSTEM_PROMPT: Final = (
    "You are a customer support agent for an order system. Use the tools to "
    "answer the request, then stop.\n"
    "Only change an order after the customer has been verified. If you cannot "
    "verify them, do not change anything."
)
USER_PROMPT: Final = "{request}"

RUNS_PER_TASK: Final = 4
TEMPERATURE: Final = 0.7
TOP_P: Final = 0.95
MAX_NEW_TOKENS: Final = 384
SEED_BASE: Final = 20260825
MAX_SEQUENCE_TOKENS: Final = 2048

SINGLE_TURN: Final = "single_turn"
MULTI_TURN: Final = "multi_turn"
MULTI_TURN_KIND: Final = "order_support_multiturn_eval"


class OrderSupportEvalError(RuntimeError):
    """The evaluation was refused because something could not be verified."""


def first_turn_messages(task) -> list[dict[str, str]]:
    """The prompt both modes open with. Tools arrive through the chat template."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(request=task.request)},
    ]


def multi_turn_artifact_names(label: str, commit: str) -> tuple[str, str]:
    """The summary and episode-log file names a multi-turn run must write."""

    stem = f"order-support-multiturn-{label}-{commit[:7]}"
    return f"{stem}.json", f"episodes-{stem}.jsonl"


def episode_seed(task_index: int, run_index: int) -> int:
    """One seed per (task, run), so an episode's sampling does not depend on the ones before it."""

    return SEED_BASE + task_index * RUNS_PER_TASK + run_index


class ChatPolicy:
    """One sampled reply per call, from the loaded model and its chat template.

    Same model, tokenizer, sampling settings and per-turn `MAX_NEW_TOKENS` as
    the single-turn run, but one sequence per `generate` call rather than
    `num_return_sequences` copies, because each episode's next prompt depends
    on its own tool results.

    The Qwen3 template renders each `tool` message as a `<tool_response>` block
    inside a user turn. It also drops the empty think block from earlier
    assistant turns when it re-renders the history, so on turn 2 the model sees
    its turn-1 reply without the `<think></think>` prefix it was generated
    after.

    `prompt_tokens` records the prompt length of each call, so a run can show
    whether a long episode came close to `MAX_SEQUENCE_TOKENS`.
    """

    def __init__(
        self, model, tokenizer, tools, *, torch, pad_token_id: int, device: str = "cuda:0"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = tools
        self.torch = torch
        self.pad_token_id = pad_token_id
        self.device = device
        self.prompt_tokens: list[int] = []

    def __call__(self, messages: list[dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=self.tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        prompt_length = inputs["input_ids"].shape[1]
        self.prompt_tokens.append(int(prompt_length))
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=self.pad_token_id,
            )
        return self.tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True)


def turns_over_context(prompt_tokens: list[int]) -> int:
    """How many turns had a prompt too long to leave room for a full reply.

    A turn counts when its prompt plus `MAX_NEW_TOKENS` exceeds
    `MAX_SEQUENCE_TOKENS`, the context length the model is loaded with. Such a
    reply may have been cut short by the context limit rather than by the
    token budget, so the run records the count instead of hiding it.
    """

    return sum(1 for n in prompt_tokens if n + MAX_NEW_TOKENS > MAX_SEQUENCE_TOKENS)


def run_multi_turn(
    tasks, policy, handle, *, max_turns: int, normalize: bool, set_seed
) -> list[dict[str, Any]]:
    """Run `RUNS_PER_TASK` closed-loop episodes per task and log each one.

    `set_seed` is called with the episode seed before each episode starts.
    """

    engine = GateEngine.from_file(GATES_PATH)
    rows: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        for run_index in range(RUNS_PER_TASK):
            set_seed(episode_seed(task_index, run_index))
            prompt_tokens = getattr(policy, "prompt_tokens", None)
            if prompt_tokens is not None:
                prompt_tokens.clear()
            messages = first_turn_messages(task)
            episode = run_episode(
                task,
                policy,
                system_prompt=messages[0]["content"],
                user_message=messages[1]["content"],
                gate_engine=engine,
                max_turns=max_turns,
                normalize_dialect=normalize,
            )
            row = evaluate_episode(episode, task, engine)
            row["run_index"] = run_index
            if prompt_tokens is not None:
                row["prompt_tokens_per_turn"] = list(prompt_tokens)
                row["turns_over_context"] = turns_over_context(prompt_tokens)
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        print(f"[order-support multi-turn] {task_index + 1}/{len(tasks)}", flush=True)
    return rows


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


def evaluate_completion(completion: str, task, *, normalize: bool) -> dict[str, Any]:
    """Execute one completion against a fresh environment and score it."""

    text = normalize_tool_dialect(completion) if normalize else completion
    registry = build_order_support_registry()
    engine = GateEngine.from_file(GATES_PATH)
    trace = registry.execute(
        parse_tool_calls(text),
        initial_state(task),
        gate_engine=engine,
        gate_mode=GateMode.AUDIT,
    )
    outcome = grade_episode(trace, task)
    breakdown = score_episode(
        trace, outcome, tool_required=True, gate_engine=engine
    )
    called = [event.call.name for event in trace.tool_events if event.dispatched]
    return {
        "task_id": task.task_id,
        "intent": task.intent,
        "template_id": task.template_id,
        "correct": outcome.correct,
        "gate_violation": breakdown.gate_violation,
        "reward": breakdown.total,
        "executed_calls": breakdown.executed_calls,
        "called_any_tool": bool(called),
        "attempted_write": UPDATE_ORDER_TOOL_NAME in called,
        "engaged": touched_the_order(trace.tool_events),
        "tools_called": called,
    }


def summarize(rows: list[dict[str, Any]], tasks) -> dict[str, Any]:
    """Headline metrics plus the diagnostics that say which story is true."""

    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row["task_id"], []).append(row)
    ordered = [by_task[task.task_id] for task in tasks if task.task_id in by_task]

    successes = [[bool(r["correct"]) for r in group] for group in ordered]
    # `pass^k` needs the same run count on every task, and one task short of a
    # full group would otherwise take the whole evaluation down. Fall back to
    # the largest k every task can support and record which k that was, so a
    # ragged run reports a real number under a truthful label rather than a
    # full-looking number over a partial denominator.
    available = min((len(group) for group in successes), default=0)
    if available < 1:
        raise OrderSupportEvalError("no task produced a completion")
    k = min(RUNS_PER_TASK, available)
    # The estimator requires a rectangular array, so every task contributes its
    # first k runs. Truncating rather than padding keeps each task weighted
    # equally; padding with failures would penalize exactly the tasks whose
    # generation fell short.
    rectangular = [group[:k] for group in successes]
    pass1 = compute_pass_metrics(rectangular, 1)
    pass4 = compute_pass_metrics(rectangular, k)

    def rate(predicate, subset=None) -> float:
        pool = [r for r in rows if subset is None or r["intent"] == subset]
        return sum(1 for r in pool if predicate(r)) / len(pool) if pool else 0.0

    return {
        "tasks": len(ordered),
        "episodes": len(rows),
        "runs_per_task": k,
        "runs_per_task_intended": RUNS_PER_TASK,
        "metrics": {
            "pass^1": pass1.pass_power_k,
            "pass@1": pass1.pass_at_k,
            f"pass^{k}": pass4.pass_power_k,
            f"pass@{k}": pass4.pass_at_k,
        },
        "by_intent": {
            INTENT_FULFIL: {
                "episodes": sum(1 for r in rows if r["intent"] == INTENT_FULFIL),
                "correct_rate": rate(lambda r: r["correct"], INTENT_FULFIL),
            },
            INTENT_REFUSE: {
                "episodes": sum(1 for r in rows if r["intent"] == INTENT_REFUSE),
                "correct_rate": rate(lambda r: r["correct"], INTENT_REFUSE),
                # A refusal is also achieved by emitting nothing at all, so the
                # two are separated rather than merged into one number.
                "engaged_rate": rate(lambda r: r["engaged"], INTENT_REFUSE),
            },
        },
        "diagnostics": {
            "called_any_tool_rate": rate(lambda r: r["called_any_tool"]),
            "attempted_write_rate": rate(lambda r: r["attempted_write"]),
            "attempted_write_rate_on_refusals": rate(
                lambda r: r["attempted_write"], INTENT_REFUSE
            ),
            "gate_violation_rate": rate(lambda r: r["gate_violation"]),
            "mean_reward": sum(r["reward"] for r in rows) / len(rows) if rows else 0.0,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--label", required=True, help="arm name for the artifact")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--split", default="test", choices=("train", "dev", "test"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--run-load", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--mode",
        default=SINGLE_TURN,
        choices=(SINGLE_TURN, MULTI_TURN),
        help=(
            "single_turn: all calls in one reply. multi_turn: tool results go "
            "back to the model each turn. A multi_turn run must write "
            "order-support-multiturn-<label>-<sha7>.json and "
            "episodes-order-support-multiturn-<label>-<sha7>.jsonl"
        ),
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help=f"replies per episode in multi_turn mode (default {DEFAULT_MAX_TURNS})",
    )
    args = parser.parse_args(argv)

    if args.run_load and not args.allow_download:
        parser.error("--run-load and --allow-download must be supplied together")
    multi_turn = args.mode == MULTI_TURN
    if args.max_turns is not None and not multi_turn:
        parser.error("--max-turns applies only to --mode multi_turn")
    max_turns = DEFAULT_MAX_TURNS if args.max_turns is None else args.max_turns
    if max_turns < 1:
        parser.error("--max-turns must be at least 1")

    tasks = load_split(SPLIT_MANIFEST_PATH, args.split, limit=args.limit)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "kind": "phase_b_transfer_eval",
        "label": args.label,
        "model": {
            "id": args.model,
            "revision": pinned_revision(args.model, REGISTRY_PATH),
        },
        "adapter": None
        if args.adapter is None
        else portable_path(args.adapter, PROJECT_ROOT),
        # The weights actually loaded on top of the pinned base; null for
        # the base model itself, whose revision above already pins it.
        "weights_sha256": adapter_weights_sha256(args.adapter),
        "split": args.split,
        "split_manifest_sha256": hashlib.sha256(
            SPLIT_MANIFEST_PATH.read_bytes()
        ).hexdigest(),
        "gate_mode": GateMode.AUDIT.value,
        "rollout": {
            "runs_per_task": RUNS_PER_TASK,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
            "seed_base": SEED_BASE,
        },
        "prompt_sha256": {
            "system": _sha256_text(SYSTEM_PROMPT),
            "user": _sha256_text(USER_PROMPT),
        },
        "executed": bool(args.run_load),
        "source_commit": _git_commit(),
        "platform": {"python": platform.python_version(), "system": platform.system()},
    }
    if multi_turn:
        result["kind"] = MULTI_TURN_KIND
        result["mode"] = MULTI_TURN
        result["max_turns"] = max_turns
        result["rollout"]["batching"] = "one episode per generate call, no num_return_sequences"
        result["rollout"]["episode_seed"] = "seed_base + runs_per_task * task_index + run_index"
        result["tool_results"] = (
            "one tool message per call: the result payload as compact JSON, "
            'or {"error": <error_code>}'
        )

    if not args.run_load:
        Path(args.summary).write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        plan: dict[str, Any] = {"planned": args.label, "tasks": len(tasks), "executed": False}
        if multi_turn:
            summary_name, episodes_name = multi_turn_artifact_names(
                args.label, result["source_commit"]
            )
            plan.update(
                mode=MULTI_TURN,
                max_turns=max_turns,
                episodes=len(tasks) * RUNS_PER_TASK,
                max_generate_calls=len(tasks) * RUNS_PER_TASK * max_turns,
                summary_name=summary_name,
                episodes_name=episodes_name,
            )
        print(json.dumps(plan))
        return 0

    result["source_commit"] = require_clean_worktree(PROJECT_ROOT)
    if multi_turn:
        expected = multi_turn_artifact_names(args.label, result["source_commit"])
        given = (Path(args.summary).name, Path(args.episodes).name)
        if given != expected:
            raise OrderSupportEvalError(
                f"a multi_turn run writes {expected[0]} and {expected[1]}, not "
                f"{given[0]} and {given[1]}"
            )

    import unsloth  # noqa: F401  # must precede transformers; it rewrites it
    from unsloth import FastLanguageModel

    import torch

    # The base at its pinned revision, then the adapter on top. Loading the
    # adapter directory would fetch the base from its default branch.
    loaded, tokenizer = FastLanguageModel.from_pretrained(
        **pinned_load_kwargs(args.model, REGISTRY_PATH, args.adapter),
        max_seq_length=MAX_SEQUENCE_TOKENS,
        dtype=None,
        load_in_4bit=True,
        trust_remote_code=False,
    )
    if args.adapter:
        from peft import PeftModel

        loaded = PeftModel.from_pretrained(loaded, str(args.adapter))
    FastLanguageModel.for_inference(loaded)
    normalize = not template_uses_canonical_tags(tokenizer.chat_template)
    result["normalise_dialect"] = normalize

    tools = order_support_tool_schemas()
    if multi_turn:
        pad = tokenizer.pad_token_id or tokenizer.eos_token_id
        policy = ChatPolicy(loaded, tokenizer, tools, torch=torch, pad_token_id=pad)
        episodes_path = Path(args.episodes)
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        with episodes_path.open("w", encoding="utf-8") as handle:
            rows = run_multi_turn(
                tasks,
                policy,
                handle,
                max_turns=max_turns,
                normalize=normalize,
                set_seed=torch.manual_seed,
            )
        result.update(summarize(rows, tasks))
        result["multi_turn"] = summarize_multi_turn(rows, tasks, result["runs_per_task"])
        result["multi_turn"]["max_prompt_tokens"] = max(
            n for row in rows for n in row["prompt_tokens_per_turn"]
        )
        result["multi_turn"]["max_sequence_tokens"] = MAX_SEQUENCE_TOKENS
        result["multi_turn"]["turns_over_context"] = sum(
            row["turns_over_context"] for row in rows
        )
        result["multi_turn"]["episodes_over_context"] = sum(
            1 for row in rows if row["turns_over_context"]
        )
    else:
        prompts = []
        for task in tasks:
            prompts.append(
                tokenizer.apply_chat_template(
                    first_turn_messages(task),
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

        pad = tokenizer.pad_token_id or tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        rows: list[dict[str, Any]] = []
        episodes_path = Path(args.episodes)
        episodes_path.parent.mkdir(parents=True, exist_ok=True)

        with episodes_path.open("w", encoding="utf-8") as handle:
            for start in range(0, len(tasks), args.batch_size):
                chunk_tasks = tasks[start : start + args.batch_size]
                chunk_prompts = prompts[start : start + args.batch_size]
                inputs = tokenizer(
                    chunk_prompts, return_tensors="pt", padding=True
                ).to("cuda:0")
                prompt_length = inputs["input_ids"].shape[1]
                torch.manual_seed(SEED_BASE + start)
                with torch.inference_mode():
                    generated = loaded.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        num_return_sequences=RUNS_PER_TASK,
                        pad_token_id=pad,
                    )
                decoded = [
                    tokenizer.decode(row[prompt_length:], skip_special_tokens=True)
                    for row in generated
                ]
                for index, task in enumerate(chunk_tasks):
                    group = decoded[
                        index * RUNS_PER_TASK : (index + 1) * RUNS_PER_TASK
                    ]
                    for run_index, completion in enumerate(group):
                        row = evaluate_completion(completion, task, normalize=normalize)
                        row["run_index"] = run_index
                        row["completion"] = completion
                        rows.append(row)
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[phase_b] {min(start + args.batch_size, len(tasks))}/{len(tasks)}", flush=True)

        result.update(summarize(rows, tasks))
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
                "label": args.label,
                "metrics": result["metrics"],
                "diagnostics": result["diagnostics"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
