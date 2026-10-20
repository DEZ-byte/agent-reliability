"""The README's numbers must be the numbers in the frozen artifacts.

Every table row the README prints is rebuilt here from the files in results/,
including the one token-counts artifact and the one order-support mode
comparison, and looked up in the README with its whitespace collapsed.
Numbers in the prose are checked in context: each must appear close to a
phrase that says what it is. The last number test lists
every number left in the README's prose, tables and diagram and fails on any
that no table row or claim below accounts for, so a new number cannot be added
without a source.

Two kinds of value are not in results/ and are pinned here instead: the top-k
defaults in the model generation configs, and the release URL. The worked
retry example is replayed through the sandbox.
"""

from __future__ import annotations

import functools
import hashlib
from datetime import datetime
import json
import re
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.gates import GateEngine  # noqa: E402
from agent.parser import parse_tool_calls  # noqa: E402
from env.gsm8k import build_gsm8k_registry, evaluate_expression  # noqa: E402
from env.order_support import order_support_tool_schemas  # noqa: E402
from evaluation.rungs import _failure_observation  # noqa: E402

RESULTS = PROJECT_ROOT / "results"
README_PATH = PROJECT_ROOT / "README.md"
RESULTS_README_PATH = RESULTS / "README.md"
MINUS = "\u2212"  # the typographic minus the README uses
EN_DASH = "\u2013"  # used for ranges in tables


def collapse(text: str) -> str:
    # Whitespace is collapsed so a figure wrapped across two lines still matches.
    return " ".join(text.split())


README_TEXT = README_PATH.read_text(encoding="utf-8")
README = collapse(README_TEXT)

# top_k in the generation_config.json of the Qwen3 revisions pinned in
# configs/model_candidates.json. Llama-3.1-8B-Instruct's sets none, so
# transformers applies its own default at generate time.
QWEN_TOP_K = 20
LIBRARY_DEFAULT_TOP_K = 50

QWEN_SMALL = "Qwen/Qwen3-1.7B"
QWEN_TEACHER = "Qwen/Qwen3-4B"
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"

BASELINE = "baseline-phase_a-3cc174f.json"
COMPARATOR = "comparator-8b-2889b6d.json"
SFT_TESTS = (
    "sft-test-qwen3-1.7b-c16530e.json",
    "sft-test-qwen3-1.7b-seed20260823-67488ea.json",
    "sft-test-qwen3-1.7b-seed20260824-51f16bf.json",
)
SFT_RUNS = (
    "sft-run-qwen3-1.7b-e0ed20b.json",
    "sft-run-qwen3-1.7b-seed20260823-a7bb0e7.json",
    "sft-run-qwen3-1.7b-seed20260824-1b938aa.json",
)
SFT_SELECTIONS = (
    "sft-selection-qwen3-1.7b-72ff036.json",
    "sft-selection-qwen3-1.7b-seed20260823-0abb401.json",
    "sft-selection-qwen3-1.7b-seed20260824-fbb9cd4.json",
)
H1 = (
    "h1-comparison-2889b6d.json",
    "h1-comparison-seed20260823-2889b6d.json",
    "h1-comparison-seed20260824-2889b6d.json",
)
TEACHER = tuple(f"sft-vs-teacher-run{run}-d692d43.json" for run in (1, 2, 3))
# (learning rate as printed, paired comparison, training record, test summary)
GRPO = (
    ("1e-6", "grpo-vs-sft-c364562.json", "grpo-run-qwen3-1.7b-a498a7b.json", "grpo-test-qwen3-1.7b-f6138ee.json"),
    ("1e-5", "grpo-lr1e5-vs-sft-91a2de9.json", "grpo-run-qwen3-1.7b-lr1e5-3d7e90f.json", "grpo-test-lr1e5-ce5f2ca.json"),
)
FAILURE_MODES = "failure-modes-2f70608.json"
MMLU = "utility-comparison-26ce399.json"
MMLU_ARMS = ("utility-base-001023e.json", "utility-sft-bb6764e.json", "utility-grpo-76a1e3f.json")
TRANSFER = ("phase_b-base-10f9d97.json", "phase_b-sft-aa59330.json", "phase_b-grpo-42e347c.json")
TRANSFER_GATES = "phase_b-gates-8d5825b.json"
TRANSFER_TEMPLATES = "phase_b-templates-2f70608.json"
# The same three checkpoints in the closed-loop multi-turn mode, in TRANSFER's order.
ORDER_LABELS = ("base", "sft", "grpo")
MULTI_TURN = tuple(f"order-support-multiturn-{label}-3cacc39.json" for label in ORDER_LABELS)
GSM8K_SPLITS = PROJECT_ROOT / "configs" / "splits" / "phase_a_gsm8k.json"
ORDER_SPLITS = PROJECT_ROOT / "configs" / "splits" / "phase_b_orders.json"
# GPU runs whose platform record names the Python version.
GPU_RUNS = SFT_RUNS + tuple(g[2] for g in GRPO) + TRANSFER + MMLU_ARMS
RELEASE_URL = "https://github.com/DEZ-byte/agent-reliability/releases/tag/v1.0.0"
# The multi-turn order-support logs came later, on their own release.
MULTI_TURN_RELEASE_URL = "https://github.com/DEZ-byte/agent-reliability/releases/tag/v1.1.0"
# Generated tokens per GSM8K row at (R0, R1), 600 attempts each, as first
# published in token-counts-3301c34.json. Adding the multi-turn runs to the
# token counts must leave these unchanged.
GSM8K_TOKEN_TOTALS = {
    "Qwen3-1.7B, untrained": (27560, 31249),
    "Llama-3.1-8B + retry": (17003, 17664),
    "Qwen3-1.7B, SFT run 1": (21247, 21725),
    "Qwen3-1.7B, SFT run 2": (21509, 22264),
    "Qwen3-1.7B, SFT run 3": (21251, 21489),
    "Qwen3-1.7B, SFT + GRPO, lr 1e-6": (21495, 22215),
    "Qwen3-1.7B, SFT + GRPO, lr 1e-5": (21506, 21960),
    "Qwen3-4B, untrained (the teacher)": (21153, 22137),
}


# --- loading ---------------------------------------------------------------


@functools.cache
def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def the_one(pattern: str, kind: str, command: str) -> dict:
    """The single results/ file matching pattern, which must be of this kind."""

    found = sorted(RESULTS.glob(pattern))
    if len(found) != 1:
        raise AssertionError(
            f"expected exactly one results/{pattern}, found {len(found)}; "
            f"run `python scripts/analyze.py {command}` on a clean tree"
        )
    payload = json.loads(found[0].read_text(encoding="utf-8"))
    if payload.get("kind") != kind:
        raise AssertionError(f"{found[0].name} is not a {kind} artifact")
    return payload


@functools.cache
def token_counts() -> dict:
    return the_one("token-counts-*.json", "token_counts", "backfill-tokens")


@functools.cache
def modes() -> dict:
    return the_one("order-support-modes-*.json", "order_support_mode_comparison", "order-support-modes")


def mode(label: str, which: str) -> dict:
    """One checkpoint's per-mode counts; which is single_turn or multi_turn."""

    return modes()["models"][label][which]


def mode_comparison(label: str, metric: str, k: int) -> dict:
    for entry in modes()["models"][label]["comparisons"]:
        if (entry["metric"], entry["k"]) == (metric, k):
            return entry
    raise KeyError((label, metric, k))


def strict_counts(label: str) -> dict:
    """Multi-turn episodes per intent: total, correct, and strict (clean_correct)."""

    by_intent = load(f"order-support-multiturn-{label}-3cacc39.json")["multi_turn"]["by_intent"]
    return {
        intent: {
            "episodes": entry["episodes"],
            "correct": round(entry["correct_rate"] * entry["episodes"]),
            "clean": round(entry["clean_correct_rate"] * entry["episodes"]),
        }
        for intent, entry in by_intent.items()
    }


def params_b(model: str) -> float:
    return token_counts()["parameters"]["models"][model]["rounded_billions"]


def rung(name: str, candidate: str, which: str) -> dict:
    for result in load(name)["results"]:
        if result["candidate"]["id"] == candidate:
            return result["rungs"][which]
    raise KeyError((name, candidate))


def comparison(name: str, which: str, k: int) -> dict:
    for entry in load(name)["comparisons"]:
        if entry["rung"] == which and entry["k"] == k:
            return entry
    raise KeyError((name, which, k))


def failures(arm: str, which: str = "R0") -> dict:
    return load(FAILURE_MODES)["arms"][arm]["rungs"][which]


def mmlu_comparison(treatment: str, scope: str) -> dict:
    for entry in load(MMLU)["comparisons"]:
        if (entry["treatment"], entry["baseline"], entry["scope"]) == (treatment, "base", scope):
            return entry
    raise KeyError((treatment, scope))


def templates(arm: str) -> dict:
    return load(TRANSFER_TEMPLATES)["arms"][arm]["by_template"]


# --- formatting, as the README prints numbers -------------------------------


def f3(value: float) -> str:
    return f"{value:.3f}"


def signed(value: float, places: int = 3) -> str:
    return ("+" if value >= 0 else MINUS) + f"{abs(value):.{places}f}"


def interval(bounds: list[float]) -> str:
    low, high = bounds
    return f"[{signed(low)}, {signed(high)}]"


def pvalue(p: float) -> str:
    return f"{p:.2g}"


def p3(p: float) -> str:
    # The GRPO p-values print to three decimals, so 0.125 stays 0.125.
    return f"{p:.3f}"


def turns(value: float) -> str:
    # Mean turns print to three decimals without trailing zeros, as in the
    # figure legend, so 1.095 is printed as is rather than rounded either way.
    return f"{value:.3f}".rstrip("0").rstrip(".")


def created(name: str) -> datetime:
    return datetime.fromisoformat(load(name)["created_at_utc"].replace("Z", "+00:00"))


def percent(share: float) -> str:
    return f"{round(100 * share)}%"


def span(values: list[int]) -> str:
    low, high = min(values), max(values)
    return str(low) if low == high else f"{low}{EN_DASH}{high}"


def row(*cells: str) -> str:
    return "| " + " | ".join(cells) + " |"


# --- what the README states ---------------------------------------------------


def gsm8k_rows() -> list[str]:
    """The main R1 table: one row per token-counts row."""

    rows = []
    for entry in token_counts()["rows"]:
        r1 = rung(entry["summary"], entry["model"], "R1")["metrics"]
        counts = entry["rungs"]["R1"]
        tokens = f"{counts['tokens_per_attempt']:.1f}"
        if counts["source"] == "rebuilt_from_logs":
            tokens += " \u2020"
        rows.append(
            row(entry["row"], f3(r1["pass^1"]), f3(r1["pass^4"]), f3(r1["pass@4"]), tokens, f"{counts['cost']:.1f}")
        )
    return rows


def seed_rows() -> list[str]:
    rows = []
    for run, (test, train, selection) in enumerate(zip(SFT_TESTS, SFT_RUNS, SFT_SELECTIONS), start=1):
        r0 = rung(test, QWEN_SMALL, "R0")["metrics"]
        selected = load(selection)["selected"]
        rows.append(
            row(
                f"Run {run}",
                str(load(train)["resolved"]["seed"]),
                selected["checkpoint"],
                f"{selected['score']:.4f}",
                f3(r0["pass^1"]),
                f3(r0["pass^4"]),
            )
        )
    return rows


def paired_rows() -> list[str]:
    rows = []
    for against, names in (("Llama-3.1-8B + retry", H1), ("Qwen3-4B teacher", TEACHER)):
        for run, name in enumerate(names, start=1):
            one, four = comparison(name, "R1", 1), comparison(name, "R1", 4)
            rows.append(
                row(
                    f"SFT run {run} vs. {against}",
                    f"{signed(one['difference'])} {interval(one['difference_ci95'])}",
                    f"{signed(four['difference'])} {interval(four['difference_ci95'])}",
                    pvalue(four["p_permutation_two_sided"]),
                )
            )
    return rows


FAILURE_COLUMNS = (
    ("base-1.7b",),
    ("sft-run1", "sft-run2", "sft-run3"),
    ("grpo-1e-6", "grpo-1e-5"),
    ("llama-8b",),
    ("teacher-4b",),
)


def failure_rows() -> list[str]:
    rows = []
    for label, field in (
        ("No call, or a call that did not parse", "no_call_or_unparsed"),
        ("Call parsed but failed to run", "call_failed_to_run"),
        ("Call ran, wrong value", "wrong_value"),
        ("Total", "total"),
    ):
        cells = [span([failures(arm)["failures"][field] for arm in arms]) for arms in FAILURE_COLUMNS]
        rows.append(row(label, *cells))
    retry_cells = []
    for arms in FAILURE_COLUMNS:
        retries = [failures(arm, "R1")["retry"] for arm in arms]
        retry_cells.append(
            f"{span([r['fired'] for r in retries])} / {span([r['rescued'] for r in retries])}"
        )
    rows.append(row("R1 retries fired / rescued", *retry_cells))
    return rows


def grpo_rows() -> list[str]:
    rows = []
    for rate, paired, run, test in GRPO:
        one, four = comparison(paired, "R0", 1), comparison(paired, "R0", 4)
        health = load(run)["group_health"]
        identical = round(health["zero_variance_fraction"] * health["groups"])
        step = load(test)["adapter"]["path"].replace("\\", "/").rsplit("checkpoint-", 1)[1]
        rows.append(
            row(
                rate,
                step,
                f"{signed(one['difference'])} {interval(one['difference_ci95'])}",
                p3(one["p_permutation_two_sided"]),
                p3(one["p_sign_test_exact"]),
                f"{signed(four['difference'])} {interval(four['difference_ci95'])}",
                f"{identical} of {health['groups']} ({percent(health['zero_variance_fraction'])})",
            )
        )
    return rows


def mmlu_rows() -> list[str]:
    summaries = [load(name)["summary"] for name in MMLU_ARMS]
    accuracy = [f"{100 * s['accuracy']:.2f}% ({round(s['accuracy'] * s['questions'])})" for s in summaries]
    return [
        row("Accuracy", *accuracy),
        row("Answers cut off at the token limit", *(f"{100 * s['truncated_rate']:.2f}%" for s in summaries)),
        row("Tool-call rate", *(f3(s["tool_call_rate"]) for s in summaries)),
    ]


def transfer_rows(which: str) -> list[str]:
    """One order-support table; which is single_turn or multi_turn."""

    summaries = [load(name) for name in (TRANSFER if which == "single_turn" else MULTI_TURN)]
    counts = [mode(label, which) for label in ORDER_LABELS]
    fulfil, refuse = (counts[0]["episodes_by_intent"][intent] for intent in ("fulfil", "refuse"))
    rows = [
        row(
            f"Completable requests completed ({fulfil} attempts)",
            *(f3(c["completion_rate_on_fulfil"]) for c in counts),
        ),
        row(
            f"Refusal-required requests refused ({refuse} attempts), with unauthenticated writes",
            *(
                f"{f3(c['refusal_rate_on_refuse'])} ({c['write_classes']['refuse']['unauthenticated_write']})"
                for c in counts
            ),
        ),
        row("Calls `get_order` at least once (share of attempts)", *(f3(c["lookup_rate"]) for c in counts)),
        row(
            "Strict pass^1 (no tool rule broken)",
            *(f3(mode_comparison(label, "clean_correct", 1)[f"{which}_pass_power_k"]) for label in ORDER_LABELS),
        ),
        row("pass^1 / pass^4", *(f"{f3(s['metrics']['pass^1'])} / {f3(s['metrics']['pass^4'])}" for s in summaries)),
    ]
    if which == "multi_turn":
        rows.append(
            row(
                f"Mean turns used (at most {summaries[0]['max_turns']})",
                *(turns(s["multi_turn"]["mean_turns_used"]) for s in summaries),
            )
        )
    return rows


ORDER_ROWS = (("Untrained 1.7B", "base"), ("SFT run 3", "sft"), ("SFT + GRPO, lr 1e-6", "grpo"))


def mode_difference_rows() -> list[str]:
    rows = []
    for name, label in ORDER_ROWS:
        cells = []
        for metric, k in (("correct", 1), ("correct", 4), ("clean_correct", 1)):
            entry = mode_comparison(label, metric, k)
            cells.append(f"{signed(entry['difference'])} {interval(entry['difference_ci95'])}")
        rows.append(row(name, *cells))
    return rows


def all_rows() -> list[str]:
    return (
        gsm8k_rows() + seed_rows() + paired_rows() + failure_rows() + grpo_rows() + mmlu_rows()
        + transfer_rows("single_turn") + transfer_rows("multi_turn") + mode_difference_rows()
    )


def claims() -> list[tuple[str, str]]:
    """(number as printed, a phrase it must appear near), section by section."""

    out: list[tuple[str, str]] = []

    # Key findings: SFT against the scaffolded 8B and against the teacher.
    vs_llama = [comparison(name, "R1", 4) for name in H1]
    out += [(signed(c["difference"]), "exclude zero") for c in vs_llama]
    run3 = vs_llama[2]
    out += [(interval(run3["difference_ci95"]), "run 3"), (pvalue(run3["p_permutation_two_sided"]), "permutation p")]
    rows = {entry["row"]: entry for entry in token_counts()["rows"]}
    llama_cost = rows["Llama-3.1-8B + retry"]["rungs"]["R1"]["cost"]
    sft_costs = [rows[f"Qwen3-1.7B, SFT run {run}"]["rungs"]["R1"]["cost"] for run in (1, 2, 3)]
    ratios = [cost / llama_cost for cost in sft_costs]
    out += [
        (f"{percent(min(ratios))} to {percent(max(ratios))}", "of an 8B attempt"),
        (f"{min(sft_costs):.1f} to {max(sft_costs):.1f} vs. {llama_cost:.1f} tokens", "of an 8B attempt"),
    ]
    teacher = rung(BASELINE, QWEN_TEACHER, "R1")["metrics"]["pass^4"]
    sft4 = [rung(name, QWEN_SMALL, "R1")["metrics"]["pass^4"] for name in SFT_TESTS]
    out += [(f"{f3(teacher)} vs. {f3(min(sft4))} to {f3(max(sft4))}", "teacher")]
    sft_fired = [failures(f"sft-run{run}", "R1")["retry"]["fired"] for run in (1, 2, 3)]
    attempts = failures("sft-run1", "R1")["episodes"]
    out.append((
        f"{min(sft_fired)} to {max(sft_fired)} of {attempts} attempts "
        f"({100 * min(sft_fired) / attempts:.1f}% to {100 * max(sft_fired) / attempts:.1f}%)",
        "retry",
    ))
    sft4_r0 = [rung(name, QWEN_SMALL, "R0")["metrics"]["pass^4"] for name in SFT_TESTS]
    llama4 = rung(COMPARATOR, LLAMA, "R1")["metrics"]["pass^4"]
    out.append((f"({f3(min(sft4_r0))} to {f3(max(sft4_r0))}) already exceeds", "R0 pass^4"))
    out.append((f"R1 pass^4 of {f3(llama4)}", "Llama-3.1-8B"))

    # Key findings: GRPO.
    pinned, late_rate = (comparison(paired, "R0", 1) for _, paired, _, _ in GRPO)
    out.append((f"{signed(pinned['difference'])} at the pinned learning rate", GRPO[0][0]))
    out.append((f"{signed(late_rate['difference'])} in an exploratory follow-up", GRPO[1][0]))
    late = comparison(GRPO[1][1], "R0", 1)
    out.append((f"p = {p3(late['p_sign_test_exact'])}", "sign test"))
    out.append((f"p = {p3(late['p_permutation_two_sided'])}", "permutation"))
    out.append((interval(late["difference_ci95"]), "exploratory"))
    out.append((f"{late['tasks_improved']} improved, {late['tasks_regressed']} regressed", "sign test"))
    bonferroni = load(GRPO[1][1])["multiplicity"]["bonferroni_alpha"]
    out.append((f"Bonferroni threshold of {bonferroni:.3f}", "comparison file"))
    correct = [load(run)["group_health"]["mean_correct_fraction"] for _, _, run, _ in GRPO]
    out.append((f"{percent(min(correct))} to {percent(max(correct))} of training rollouts", "already correct"))
    shares = [load(run)["group_health"]["zero_variance_fraction"] for _, _, run, _ in GRPO]
    out.append((f"{percent(min(shares))} to {percent(max(shares))}", "rollout groups"))
    out.append((
        f"{percent(shares[0])} (lr = {GRPO[0][0]}) to {percent(shares[1])} (lr = {GRPO[1][0]})",
        "8-rollout groups",
    ))
    out.append((str(load(GRPO[0][2])["resolved"]["num_generations"]), "-rollout groups"))

    # Key findings: order support, single-turn against multi-turn.
    single = {label: mode(label, "single_turn") for label in ORDER_LABELS}
    multi = {label: mode(label, "multi_turn") for label in ORDER_LABELS}
    summary = dict(zip(ORDER_LABELS, (load(name) for name in MULTI_TURN)))
    strict = {label: mode_comparison(label, "clean_correct", 1) for label in ORDER_LABELS}
    mean_turns = {label: turns(summary[label]["multi_turn"]["mean_turns_used"]) for label in ORDER_LABELS}
    first = {label: multi[label]["first_reply_update_order"] for label in ORDER_LABELS}
    later = {
        label: multi[label]["later_write_after_failed_authenticate"]["by_intent"]["refuse"]
        for label in ORDER_LABELS
    }
    episodes = multi["base"]["episodes"]
    n_refuse = multi["base"]["episodes_by_intent"]["refuse"]
    grpo_lookups = round(multi["grpo"]["lookup_rate"] * episodes)

    def refused(label: str) -> str:
        unauthenticated = multi[label]["write_classes"]["refuse"]["unauthenticated_write"]
        return (
            f"{f3(multi[label]['refusal_rate_on_refuse'])} of refusal-required requests "
            f"({unauthenticated} of {n_refuse}"
        )

    base_unauthenticated = multi["base"]["write_classes"]["refuse"]["unauthenticated_write"]
    sft_unauthenticated = multi["sft"]["write_classes"]["refuse"]["unauthenticated_write"]
    out += [
        (f"completes {f3(single['base']['completion_rate_on_fulfil'])} of completable requests", "open-loop"),
        (f"the untrained model averages {mean_turns['base']} turns", "closed-loop"),
        (f"completes {f3(multi['base']['completion_rate_on_fulfil'])} of completable requests", "averages"),
        (
            f"refuses {f3(multi['base']['refusal_rate_on_refuse'])} of refusal-required requests with "
            f"{base_unauthenticated} unauthenticated writes",
            "completes",
        ),
        (f"pass^4 = {f3(summary['base']['metrics']['pass^4'])}", "strict policy-compliant"),
        (f"pass^1 = {f3(strict['base']['multi_turn_pass_power_k'])}", "strict policy-compliant"),
        (
            f"only {f3(single['sft']['refusal_rate_on_refuse'])} of refusal-required requests in single-turn mode "
            f"and {f3(multi['sft']['refusal_rate_on_refuse'])} in closed-loop mode",
            "SFT run 3",
        ),
        (f"{sft_unauthenticated} of {n_refuse} attempts execute unauthenticated writes", "closed-loop mode"),
        (f"averages {mean_turns['sft']} turns", "horizon collapse"),
        (
            f"{first['sft']['by_intent']['refuse']['episodes']} of its {sft_unauthenticated} unauthenticated "
            "writes come from a first turn",
            "horizon collapse",
        ),
        (f"calls `get_order` in {grpo_lookups} of {episodes} episodes", "almost identically"),
        (f"reaches a strict pass^1 of {f3(strict['grpo']['multi_turn_pass_power_k'])}", "almost identically"),
        (f"refuses {refused('sft')} attempts wrote without authentication)", "get_order"),
        (f"strict pass^1 of {f3(strict['sft']['multi_turn_pass_power_k'])}", "refuses"),
    ]
    # Key findings: what strict pass^1 counts, per intent.
    base_strict, sft_strict = strict_counts("base"), strict_counts("sft")
    strict_total = base_strict["fulfil"]["clean"] + base_strict["refuse"]["clean"]
    asked = base_strict["refuse"]["correct"] - base_strict["refuse"]["clean"]
    out += [
        (f"({base_strict['fulfil']['correct']} of {base_strict['fulfil']['episodes']})", "completable requests"),
        (f"({strict_total} of {episodes})", "strict policy-compliant"),
        (f"rejects {asked} correct refusals", "missing postal code"),
        (f"{strict_total + asked} of {episodes} ({f3((strict_total + asked) / episodes)})", "without that requirement"),
        (
            f"strict pass^1 of {f3(strict['sft']['multi_turn_pass_power_k'])} is "
            f"{sft_strict['fulfil']['clean'] + sft_strict['refuse']['clean']} of {episodes}",
            "none of its",
        ),
        (f"none of its {sft_strict['fulfil']['correct']} completions performs the lookup", "Its strict pass^1"),
        (f"only {sft_strict['refuse']['correct']} of its {n_refuse} refusal-required attempts", "refused"),
    ]
    mmlu_base, mmlu_sft = (load(name)["summary"]["accuracy"] for name in MMLU_ARMS[:2])
    out.append((f"{100 * mmlu_base:.2f}% before SFT and {100 * mmlu_sft:.2f}% after", "MMLU"))
    halted = [templates("base")[t] for t in ("address_change_verified", "address_change_wrong_postcode")]
    lone = sum(t["call_sequences"].get("authenticate", 0) for t in halted)
    silent = templates("base")["address_change_no_postcode"]

    # Figure captions.
    out.append((f"{failures('base-1.7b')['episodes']} test attempts per model", "failure taxonomy"))

    # Pipeline diagram.
    sizes = {name: len(ids) for name, ids in json.loads(GSM8K_SPLITS.read_text(encoding="utf-8"))["splits"].items()}
    counts = load("sft-candidates-a2dbe77.json")["result"]["counts"]
    stats = load("sft-dataset-54218c4.json")["selection_stats"]
    out += [
        (f"{sizes['train']:,} tasks", "GSM8K train split"),
        (f"{counts['episodes']:,} attempts", "teacher"),
        (f"{counts['graded_correct']:,} correct", "Execution filter"),
        (f"{counts['laundered']} restated", "dropped"),
        (f"{stats['selected']} rows", "one per task"),
        (", ".join(str(load(name)["resolved"]["seed"]) for name in SFT_RUNS), "seeds"),
        (f"{sizes['dev']} tasks", "dev split"),
        (f"group size {load(GRPO[0][2])['resolved']['num_generations']}", "GRPO"),
    ]
    gsm8k_r1 = rung(BASELINE, QWEN_SMALL, "R1")
    order = load(TRANSFER[0])
    out += [
        (f"{gsm8k_r1['tasks']} test tasks x {gsm8k_r1['runs_per_task']}", "GSM8K"),
        (f"{load(MMLU)['comparisons'][0]['questions']} questions", "zero-shot"),
        (f"{order['tasks']} requests x {order['runs_per_task']}", "order support"),
    ]

    # Methodology.
    out.append((f"{sizes['train']:,} train, {sizes['dev']} dev and {sizes['test']} test tasks", "splits"))
    out.append(("230", "test task"))
    decoding = load(BASELINE)["decoding"]
    out += [
        (str(decoding["temperature"]), "temperature"),
        (str(decoding["top_p"]), "top-p"),
        (str(decoding["max_new_tokens"]), "new tokens per generation"),
    ]
    run3_paired = comparison(H1[2], "R1", 4)
    out += [
        (f"{run3_paired['bootstrap_replicates']:,} replicates", "bootstrap"),
        (f"{run3_paired['permutation_resamples']:,} resamples", "permutation"),
        (f"{round(100 * load(MMLU)['confidence'])}%", "interval"),
    ]
    models = token_counts()["parameters"]["models"]
    small = models[QWEN_SMALL]
    out += [
        (f"{params_b(QWEN_SMALL)} for Qwen3-1.7B", "unique"),
        (f"{params_b(QWEN_TEACHER)} for Qwen3-4B", "unique"),
        (f"{params_b(LLAMA)} for Llama-3.1-8B", "unique"),
        (f"{small['stored']:,} values", "stores"),
        (f"{small['unique']:,} unique parameters", "ties"),
        (f"min(tokens + 1, {token_counts()['max_new_tokens']})", "re-tokenizing"),
        (f"all {len(token_counts()['validation']['qwen'])} recorded R0 and R1 totals", "exactly"),
    ]

    # GSM8K results.
    llama = rung(COMPARATOR, LLAMA, "R1")
    out += [
        (f"{gsm8k_r1['tasks']} test tasks, {gsm8k_r1['runs_per_task']} attempts each", "R1"),
        (f"{f3(llama['metrics']['pass@4'])} of tasks at least once", "Llama-3.1-8B"),
        (f"only {f3(llama['metrics']['pass^4'])} on all four", "Llama-3.1-8B"),
        (f"{params_b(LLAMA) / params_b(QWEN_SMALL):.1f} times as many parameters", "fewest tokens"),
    ]
    trained = [rung(name, QWEN_SMALL, "R1") for name in SFT_TESTS + tuple(g[3] for g in GRPO)]
    fired = [r["model_decisions"] - r["episodes"] for r in trained]
    out.append((f"{min(fired)} to {max(fired)} times in {trained[0]['episodes']} attempts", "retry"))
    out.append((f"against {gsm8k_r1['model_decisions'] - gsm8k_r1['episodes']}", "untrained 1.7B"))
    seed4 = [rung(name, QWEN_SMALL, "R0")["metrics"]["pass^4"] for name in SFT_TESTS]
    out.append((f3(max(seed4) - min(seed4)), "varies by"))

    # Failure taxonomy.
    sft_arms = [failures(f"sft-run{run}")["failures"] for run in (1, 2, 3)]
    wrong = [arm["wrong_value"] / arm["total"] for arm in sft_arms]
    untrained = failures("base-1.7b")["failures"]
    out += [
        (f"{percent(min(wrong))} to {percent(max(wrong))}", "of SFT failures"),
        (percent(untrained["wrong_value"] / untrained["total"]), "before training"),
        (str(failures("base-1.7b")["episodes"]), "per model"),
    ]
    n = failures("base-1.7b")["episodes"]

    def share(count: int) -> str:
        return f"{100 * count / n:.1f}%"

    def format_errors(arm: dict) -> int:
        return arm["no_call_or_unparsed"] + arm["call_failed_to_run"]

    base_errors = format_errors(untrained)
    sft_errors = [format_errors(arm) for arm in sft_arms]
    sft_wrong = [arm["wrong_value"] for arm in sft_arms]
    out += [
        (f"{base_errors} of {n} attempts ({share(base_errors)})", "format, parsing and sandbox"),
        (
            f"{min(sft_errors)} to {max(sft_errors)} of {n} ({share(min(sft_errors))} to {share(max(sft_errors))})",
            "per SFT run",
        ),
        (
            f"{min(sft_wrong)} to {max(sft_wrong)} of {n} attempts ({share(min(sft_wrong))} to {share(max(sft_wrong))})",
            "incorrect value",
        ),
    ]
    rescued = [failures(f"sft-run{run}", "R1")["retry"]["rescued"] for run in (1, 2, 3)]
    base_retry = failures("base-1.7b", "R1")["retry"]
    out += [
        (f"{min(rescued)} to {max(rescued)} attempts", "per SFT run"),
        (f"{base_retry['rescued']} of {base_retry['fired']}", "untrained model"),
    ]

    # GRPO.
    run = load(GRPO[0][2])
    dev = [load(name)["selected"]["score"] for name in SFT_SELECTIONS]
    out += [
        (f"({max(dev):.4f})", "highest dev pass^1"),
        (f"permutation p is {p3(late['p_permutation_two_sided'])}", "1e-5"),
        (f"{run['train']['global_step']} steps", "two prompts per step"),
        (f"{run['resolved']['num_generations']} completions per prompt", "two prompts per step"),
    ]
    spread = [load(r)["group_health"]["mean_component_std"] for _, _, r, _ in GRPO]
    out.append((f"{f3(spread[0]['accuracy'])} and {f3(spread[1]['accuracy'])}", "standard deviation"))
    others = max(s[term] for s in spread for term in ("format", "gate", "efficiency"))
    out.append((f"at most {f3(others)}", "any other term"))
    late_test = rung(GRPO[1][3], QWEN_SMALL, "R0")
    net = round(late["difference"] * late_test["episodes"])
    out += [
        (f"about {net} attempts out of {late_test['episodes']}", "per-attempt accuracy"),
        (f"{late['tasks_improved']} tasks that improved against {late['tasks_regressed']} that regressed",
         "sign test"),
        (f"Bonferroni threshold of {bonferroni:.3f}", "permutation p also exceeds"),
    ]

    # MMLU.
    everything = mmlu_comparison("sft", "all_questions")
    readable = mmlu_comparison("sft", "readable_in_both")
    rows_mmlu = load(MMLU)["rows"]
    unreadable = [q for q in rows_mmlu if any(q["arms"][a]["extracted"] is None for a in ("base", "sft"))]
    cut = [
        q
        for q in unreadable
        if any(q["arms"][a]["extracted"] is None and q["arms"][a]["truncated"] for a in ("base", "sft"))
    ]
    out += [
        (f"{everything['questions']} questions, zero-shot", "greedy"),
        (f"at most {load(MMLU_ARMS[0])['decoding']['max_new_tokens']} new tokens", "no tool offered"),
        (signed(everything["difference"], 4), "paired difference"),
        (interval(everything["difference_ci95"]), "95% CI"),
        (f"about {round(100 * max(-everything['difference_ci95'][0], everything['difference_ci95'][1]))} points",
         "cannot"),
        (f"{len(unreadable)} questions", "lack a parseable answer"),
        (f"{len(cut)} of them", "token limit"),
        (f"{readable['questions']} questions", "both models produced"),
        (f"{signed(readable['difference'])} {interval(readable['difference_ci95'])}", "both models produced"),
    ]

    # Order-support transfer.
    test_tasks = json.loads(ORDER_SPLITS.read_text(encoding="utf-8"))["splits"]["test"]

    def n_template(template: str) -> int:
        return sum(1 for t in test_tasks if t["template_id"] == template)

    verified = n_template("address_change_verified")
    no_postcode, wrong_postcode = n_template("address_change_no_postcode"), n_template("address_change_wrong_postcode")
    out += [
        (f"{len(test_tasks)} test requests", "Order-support transfer"),
        (f"{verified} completable", "address changes"),
        (f"{no_postcode + wrong_postcode} that must be refused", "address changes"),
        (f"{no_postcode} omit the postal code", "refused"),
        (f"{wrong_postcode} supply the wrong one", "refused"),
    ]
    max_turns = summary["base"]["max_turns"]
    out.append((f"after {max_turns} generations (turns)", "episode terminates"))
    runs = summary["base"]["runs_per_task"]
    out.append((
        f"the {n_refuse} refusal-required attempts are {runs * no_postcode} without a postal code "
        f"and {runs * wrong_postcode} with the wrong one",
        "counts as refused",
    ))

    # Order-support transfer: the paired differences.
    floor = 1 / (modes()["permutation_resamples"] + 1)
    out += [
        (f"{modes()['bootstrap_replicates']:,} replicates", "multi-turn minus single-turn"),
        (pvalue(floor), "add-one floor"),
        (f"{modes()['permutation_resamples']:,} resamples", "add-one floor"),
        (pvalue(modes()["multiplicity"]["bonferroni_alpha"]), "Bonferroni threshold"),
    ]

    # Order-support transfer: the untrained model.
    out.append((f"{lone} of {sum(t['episodes'] for t in halted)} episodes", "postal code"))
    out.append((f"{silent['call_sequences']['(none)']} of its {silent['episodes']} generations", "no tool call"))
    base_fulfil = multi["base"]["write_classes"]["fulfil"]
    base_writes = base_fulfil["clean_write"] + base_fulfil["verified_write_without_lookup"]
    out += [
        (f"refusal rate of {f3(single['base']['refusal_rate_on_refuse'])} therefore reflects", "absence of writes"),
        (f"makes {base_writes} write attempts on completable requests while still refusing all {n_refuse}",
         "closed loop"),
        (f"a pass^4 of {f3(summary['base']['metrics']['pass^4'])}", "fed back"),
        (
            f"strict pass^1 rises from {f3(strict['base']['single_turn_pass_power_k'])} single-turn to "
            f"{f3(strict['base']['multi_turn_pass_power_k'])}, a gain of {f3(strict['base']['difference'])}",
            "a pass^4 of",
        ),
        (
            f"{base_writes} write attempts on completable requests, "
            f"{base_fulfil['verified_write_without_lookup']} skipped",
            "get_order",
        ),
    ]
    # Order-support transfer: the untrained model's strict pass^1, per intent.
    fulfil, refuse = base_strict["fulfil"], base_strict["refuse"]
    out += [
        (
            f"{f3(strict['base']['multi_turn_pass_power_k'])} is {strict_total} of {episodes} attempts: "
            f"{fulfil['clean']} of {fulfil['episodes']} completable and {refuse['clean']} of {refuse['episodes']} "
            "refusal-required",
            "strict pass^1",
        ),
        (f"Of the {fulfil['episodes'] - fulfil['clean']} completable attempts", "strict criterion"),
        (
            f"{fulfil['episodes'] - fulfil['correct']} did not complete the change and "
            f"{fulfil['correct'] - fulfil['clean']} completed it without",
            "get_order",
        ),
        (f"All {asked} refusal-required attempts that fail", "postal code"),
        (f"{strict_total + asked} of {episodes} ({f3((strict_total + asked) / episodes)})", "counting them"),
    ]

    # Order-support transfer: SFT and GRPO.
    grpo_refuse = multi["grpo"]["write_classes"]["refuse"]["unauthenticated_write"]
    grpo_refused = f"{f3(multi['grpo']['refusal_rate_on_refuse'])} ({grpo_refuse} of {n_refuse}"
    grpo_strict = strict_counts("grpo")
    out += [
        (
            f"which is {sft_strict['fulfil']['clean'] + sft_strict['refuse']['clean']} of {episodes}: "
            f"{sft_strict['fulfil']['clean']} of its {sft_strict['fulfil']['correct']} completions pass the lookup "
            f"rule, and {sft_strict['refuse']['correct']} of {n_refuse} refusal-required attempts",
            "strict pass^1",
        ),
        (
            f"({grpo_strict['fulfil']['clean'] + grpo_strict['refuse']['clean']} of {episodes}: "
            f"{grpo_strict['fulfil']['clean']} completions with the lookup and "
            f"{grpo_strict['refuse']['clean']} refusals)",
            "strict pass^1",
        ),
    ]
    out += [
        (f"averages {mean_turns['sft']} turns per episode", "finish the task in one turn"),
        (f"in {first['sft']['episodes']} of {episodes} episodes its first turn", "finish the task in one turn"),
        (f"{first['sft']['by_intent']['refuse']['episodes']} of the {n_refuse} refusal-required", "first turn"),
        (
            f"in {later['sft']['wrote_after_failed_authenticate']} of the {later['sft']['waited']} "
            "refusal-required attempts where SFT run 3 waited",
            "does not explain all of it",
        ),
        (
            f"in {later['grpo']['wrote_after_failed_authenticate']} of the {later['grpo']['waited']} "
            "refusal-required attempts that waited for it",
            "failed `authenticate`",
        ),
        (f"in its first turn in {first['base']['episodes']} of {episodes} episodes", "untrained"),
        (f"{mean_turns['grpo']} turns on average", "GRPO"),
        (f"`update_order` in {first['grpo']['episodes']} of {episodes} episodes", "GRPO"),
        (f"refusal rate of {grpo_refused} attempts wrote without authentication)", "GRPO"),
        (f"strict pass^1 of {f3(strict['grpo']['multi_turn_pass_power_k'])}", "first-turn"),
        (f"pass^4 of {f3(summary['sft']['metrics']['pass^4'])}", "exactly"),
        (f"{verified} completable requests out of {len(test_tasks)}", "pass^4"),
    ]
    sft_fulfil = load(TRANSFER_GATES)["arms"]["sft"]["write_classes"]["fulfil"]
    completed = sft_fulfil["clean_write"] + sft_fulfil["verified_write_without_lookup"]
    out.append((f"{sft_fulfil['verified_write_without_lookup']} of SFT run 3's {completed}", "get_order"))
    sft_multi_fulfil = multi["sft"]["write_classes"]["fulfil"]
    sft_skipped = sft_multi_fulfil["verified_write_without_lookup"]
    out.append((f"all {sft_skipped} of its writes on completable requests", "closed loop"))
    none = {arm: templates(arm)["address_change_no_postcode"] for arm in ("sft", "grpo")}
    out.append((f"{none['sft']['wrote']} of {none['sft']['episodes']}", "without a postal code"))
    out.append((f"GRPO on {none['grpo']['wrote']}", "without a postal code"))

    # Order-support transfer: tokens per multi-turn episode.
    tokens = token_counts()["order_support_multiturn"]
    generated = [f"{tokens[label]['generated']['tokens_per_episode']:.1f}" for label in ORDER_LABELS]
    prompt = [f"{tokens[label]['prompt']['tokens_per_episode']:,.0f}" for label in ORDER_LABELS]
    out += [
        (f"{generated[0]} for the untrained model, {generated[1]} for SFT run 3 and {generated[2]} for SFT + GRPO",
         "reconstructed from the logs"),
        (f"{tokens['base']['max_new_tokens']}-token cap", "no turn reached"),
        (
            f"{prompt[0]} prompt tokens per episode, against {prompt[1]} for SFT run 3 and {prompt[2]} for SFT + GRPO",
            "logged at generation time",
        ),
    ]

    # Limitations.
    untrained_same = load("sft-comparison-qwen3-1.7b-89a7fbb.json")["sampling_degeneracy"]["R1"]["baseline"]
    teacher_same = load(TEACHER[0])["sampling_degeneracy"]["R1"]["baseline"]
    out += [
        (f"{untrained_same['tasks_with_identical_completions']} of {untrained_same['tasks']}", "identical text"),
        (f"did on {teacher_same['tasks_with_identical_completions']}", "teacher"),
        (f"top-k {QWEN_TOP_K}", "Qwen generation configs"),
        (f"default of {LIBRARY_DEFAULT_TOP_K}", "Llama"),
        (f"{100 * load(MMLU_ARMS[0])['summary']['truncated_rate']:.2f}%", "320-token limit"),
        (f"almost {round(-100 * readable['difference_ci95'][0])} points", "both models"),
    ]

    # Footer: the software the GPU runs used, and the Python versions CI tests.
    platforms = {load(name)["platform"]["python"] for name in GPU_RUNS}
    (python,) = {".".join(version.split(".")[:2]) for version in platforms}
    lock = (PROJECT_ROOT / "requirements-gpu.lock").read_text(encoding="utf-8")
    torch = re.search(r"/torch-(\d+\.\d+)\.\d+%2Bcu(\d+)(\d)-", lock)
    ci = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    tested = re.search(r"python-version: \[(.*)\]", ci).group(1).replace('"', "").split(", ")
    out += [
        (f"Python {python}", "GPU runs"),
        (f"PyTorch {torch.group(1)}", "GPU runs"),
        (f"CUDA {torch.group(2)}.{torch.group(3)}", "GPU runs"),
        (" or ".join(tested), "verify on CPU"),
    ]
    return out


# Numbers that name things rather than measure them.
STRUCTURAL = {
    "50%": "the hypothetical example in the opening paragraph",
    "1": "figure numbers, run names, and the +1 for the end-of-sequence token",
    "2": "figure number",
    "3.1": "the Llama 3.1 Community License",
    "4060": "the GPU the runs used",
}

FENCE = re.compile(r"^```(\w*)\n.*?^```$", re.MULTILINE | re.DOTALL)
NUMBER = re.compile(r"(?<![\w.^@\-])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?!\w|\.\d)")


def numbers_in(text: str) -> set[str]:
    return set(NUMBER.findall(text))


def readme_numbers() -> set[str]:
    """Numbers in the README's prose, tables and diagram."""

    def drop_code(block: re.Match) -> str:
        return block.group(0) if block.group(1) == "mermaid" else " "

    text = FENCE.sub(drop_code, README_TEXT)  # code blocks hold data, not claims
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)  # display math
    text = re.sub(r"\$[^$\n]+\$", " ", text)  # inline math
    text = re.sub(r"`[^`\n]+`", " ", text)  # inline code
    text = re.sub(r"\]\([^)]*\)", "]", text)  # link targets
    return numbers_in(text)


def near(number: str, context: str, window: int = 200) -> bool:
    pattern = re.compile(r"(?<![\d.^@])" + re.escape(number) + r"(?!\d|\.\d)")
    return any(
        context in README[max(0, match.start() - window) : match.end() + window]
        for match in pattern.finditer(README)
    )


# --- tests ---------------------------------------------------------------------


class TableTests(unittest.TestCase):
    """Every table row, rebuilt from the artifacts."""

    def check(self, rows: list[str]) -> None:
        for expected in rows:
            with self.subTest(row=expected):
                self.assertTrue(expected in README, f"row not in the README: {expected}")

    def test_gsm8k_table(self) -> None:
        self.check(gsm8k_rows())

    def test_seed_table(self) -> None:
        self.check(seed_rows())

    def test_paired_comparisons(self) -> None:
        self.check(paired_rows())

    def test_failure_taxonomy(self) -> None:
        self.check(failure_rows())

    def test_grpo_table(self) -> None:
        self.check(grpo_rows())

    def test_mmlu_table(self) -> None:
        self.check(mmlu_rows())

    def test_single_turn_transfer_table(self) -> None:
        self.check(transfer_rows("single_turn"))

    def test_multi_turn_transfer_table(self) -> None:
        self.check(transfer_rows("multi_turn"))

    def test_mode_differences(self) -> None:
        self.check(mode_difference_rows())


class ClaimTests(unittest.TestCase):
    def test_every_prose_number_sits_by_its_meaning(self) -> None:
        for number, context in claims():
            with self.subTest(number=number, context=context):
                self.assertTrue(near(number, context), f"{number!r} not found near {context!r}")

    def test_no_number_without_a_source(self) -> None:
        sourced = set(STRUCTURAL)
        for text in all_rows():
            sourced |= numbers_in(text)
        for number, _ in claims():
            sourced |= numbers_in(number)
        self.assertEqual(readme_numbers() - sourced, set())


class EvidenceTests(unittest.TestCase):
    """What the words around the numbers assert about the data."""

    def test_token_counts_artifact(self) -> None:
        payload = token_counts()
        for model, entry in payload["parameters"]["models"].items():
            self.assertEqual(entry["rounded_billions"], round(entry["unique"] / 1e9, 2), model)
        self.assertTrue(all(check["match"] for check in payload["validation"]["qwen"]))
        checked = payload["validation"]["qwen"]
        self.assertEqual(len({check["summary"] for check in checked}), 4, "the README says four runs")
        self.assertEqual({check["rung"] for check in checked}, {"R0", "R1"})
        for item in payload["inputs"]:
            if item["file"].endswith(".json"):
                digest = hashlib.sha256((RESULTS / item["file"]).read_bytes()).hexdigest()
                self.assertEqual(digest, item["sha256"], item["file"])
        rebuilt = set()
        for entry in payload["rows"]:
            self.assertEqual(entry["unique_params_b"], params_b(entry["model"]))
            for which, counts in entry["rungs"].items():
                self.assertAlmostEqual(counts["cost"], counts["tokens_per_attempt"] * params_b(entry["model"]))
                self.assertAlmostEqual(counts["tokens_per_attempt"], counts["tokens_total"] / counts["episodes"])
                summary = rung(entry["summary"], entry["model"], which)
                if counts["source"] == "recorded":
                    self.assertEqual(counts["tokens_total"], summary["generated_tokens_total"])
                else:
                    self.assertNotIn("generated_tokens_total", summary)
                    rebuilt.add(entry["row"])
        self.assertEqual(len(rebuilt), 3, "the README says three rows were rebuilt")
        totals = {
            entry["row"]: tuple(entry["rungs"][which]["tokens_total"] for which in ("R0", "R1"))
            for entry in payload["rows"]
        }
        self.assertEqual(totals, GSM8K_TOKEN_TOTALS)

    def test_sft_beats_llama_and_trails_the_teacher(self) -> None:
        for name in H1:
            for k in (1, 4):
                self.assertGreater(comparison(name, "R1", k)["difference_ci95"][0], 0.0, (name, k))
        for name in TEACHER:
            self.assertLess(comparison(name, "R1", 4)["difference_ci95"][1], 0.0, name)
        low, high = comparison(TEACHER[2], "R1", 1)["difference_ci95"]
        self.assertTrue(low < 0.0 < high, "only run 3's pass^1 interval includes zero")
        for name in TEACHER[:2]:
            self.assertLess(comparison(name, "R1", 1)["difference_ci95"][1], 0.0, name)

    def test_the_teacher_is_best(self) -> None:
        teacher = rung(BASELINE, QWEN_TEACHER, "R1")["metrics"]
        others = [rung(COMPARATOR, LLAMA, "R1")["metrics"]] + [
            rung(name, QWEN_SMALL, "R1")["metrics"] for name in SFT_TESTS + tuple(g[3] for g in GRPO)
        ]
        for metric in ("pass^1", "pass^4"):
            self.assertGreater(teacher[metric], max(m[metric] for m in others), metric)

    def test_llama_writes_the_fewest_tokens(self) -> None:
        tokens = {e["row"]: e["rungs"]["R1"]["tokens_per_attempt"] for e in token_counts()["rows"]}
        self.assertEqual(min(tokens, key=tokens.get), "Llama-3.1-8B + retry")

    def test_trained_r0_and_r1_pass4_are_identical(self) -> None:
        for name in SFT_TESTS + tuple(g[3] for g in GRPO):
            r0, r1 = (rung(name, QWEN_SMALL, which)["metrics"]["pass^4"] for which in ("R0", "R1"))
            self.assertEqual(r0, r1, name)

    def test_seed_spread_exceeds_both_grpo_pass4_changes(self) -> None:
        seed4 = [rung(name, QWEN_SMALL, "R0")["metrics"]["pass^4"] for name in SFT_TESTS]
        changes = [comparison(paired, "R0", 4)["difference"] for _, paired, _, _ in GRPO]
        self.assertGreater(max(seed4) - min(seed4), max(abs(c) for c in changes))

    def test_grpo_setup(self) -> None:
        run3 = load(SFT_TESTS[2])["adapter"]["path"].replace("\\", "/")
        for _, paired, run, _ in GRPO:
            record = load(run)
            self.assertEqual(record["init_from_adapter"], run3)
            self.assertEqual(record["prompt_filter"], {"applied": False})
            self.assertEqual(record["group_health"]["groups"], 2 * record["train"]["global_step"], "two per step")
            self.assertEqual(record["dataset"]["rows"], 1000)
            self.assertEqual(record["resolved"]["beta"], 0.0, "the README says there is no KL term")
        four = comparison(GRPO[1][1], "R0", 4)["difference_ci95"]
        self.assertTrue(four[0] < 0.0 < four[1], "the 1e-5 pass^4 interval includes zero")

    def test_grpo_starts_from_the_best_dev_run_after_the_sft_tests(self) -> None:
        dev = [load(name)["selected"]["score"] for name in SFT_SELECTIONS]
        self.assertEqual(dev.index(max(dev)), 2, "run 3 has the highest dev pass^1")
        self.assertEqual(dev.count(max(dev)), 1)
        last_sft_test = max(created(name) for name in SFT_TESTS)
        for _, _, run, _ in GRPO:
            self.assertGreater(created(run), last_sft_test, run)

    def test_the_1e5_run_is_a_late_override(self) -> None:
        pinned, late = (load(run)["learning_rate"] for _, _, run, _ in GRPO)
        self.assertEqual(pinned, {"pinned_in_config": 1e-6, "used": 1e-6, "overridden": False})
        self.assertEqual(late, {"pinned_in_config": 1e-6, "used": 1e-5, "overridden": True})
        # A run record's created_at_utc is taken when training starts.
        self.assertGreater(created(GRPO[1][2]), created(GRPO[0][1]), "started after the 1e-6 comparison")

    def test_decoding(self) -> None:
        for name in (BASELINE, COMPARATOR) + SFT_TESTS + tuple(g[3] for g in GRPO):
            decoding = load(name)["decoding"]
            self.assertEqual((decoding["temperature"], decoding["top_p"]), (0.7, 0.95), name)
            self.assertNotIn("top_k", decoding, "the README says top-k was not overridden")

    def test_mmlu_setup(self) -> None:
        for name in MMLU_ARMS:
            payload = load(name)
            self.assertFalse(payload["tools_offered"], name)
            self.assertTrue(payload["decoding"]["greedy"], name)
        self.assertEqual(load(MMLU_ARMS[1])["adapter"], load(TRANSFER[1])["adapter"], "SFT run 3 in both")

    def test_transfer_setup(self) -> None:
        tools = sorted(schema["function"]["name"] for schema in order_support_tool_schemas())
        self.assertEqual(tools, ["authenticate", "get_order", "update_order"])
        for name in TRANSFER:
            self.assertEqual(load(name)["gate_mode"], "audit")
        self.assertEqual(load(TRANSFER[1])["weights_sha256"], load(SFT_TESTS[2])["adapter"]["weights_sha256"])
        self.assertEqual(load(TRANSFER[2])["weights_sha256"], load(GRPO[0][3])["adapter"]["weights_sha256"])

    def test_gpu_runs_were_on_windows(self) -> None:
        for name in GPU_RUNS:
            self.assertEqual(load(name)["platform"]["system"], "Windows", name)
        for name in (BASELINE, COMPARATOR) + SFT_TESTS + tuple(g[3] for g in GRPO):
            self.assertTrue(load(name)["platform"].startswith("Windows"), name)

    def test_multi_turn_runs_match_the_single_turn_runs(self) -> None:
        self.assertEqual(sorted(p.name for p in RESULTS.glob("order-support-multiturn-*.json")), sorted(MULTI_TURN))
        for single_name, multi_name in zip(TRANSFER, MULTI_TURN):
            single, multi = load(single_name), load(multi_name)
            with self.subTest(run=multi_name):
                self.assertEqual(multi["kind"], "order_support_multiturn_eval")
                self.assertTrue(multi["executed"])
                self.assertEqual((multi["mode"], multi["max_turns"], multi["gate_mode"]), ("multi_turn", 4, "audit"))
                self.assertEqual(multi["model"]["id"], QWEN_SMALL)
                for key in ("weights_sha256", "split", "split_manifest_sha256", "tasks", "runs_per_task"):
                    self.assertEqual(multi[key], single[key], key)
                for key in ("runs_per_task", "temperature", "top_p", "max_new_tokens", "seed_base"):
                    self.assertEqual(multi["rollout"][key], single["rollout"][key], key)

    def test_the_mode_comparison_reads_these_files(self) -> None:
        payload = modes()
        for label, single_name, multi_name in zip(ORDER_LABELS, TRANSFER, MULTI_TURN):
            for which, name in (("single_turn", single_name), ("multi_turn", multi_name)):
                entry = payload["inputs"][label][which]["summary"]
                with self.subTest(label=label, which=which):
                    self.assertEqual(entry["file"], name)
                    self.assertEqual(entry["sha256"], hashlib.sha256((RESULTS / name).read_bytes()).hexdigest())
                    self.assertEqual(entry["source_commit"], load(name)["source_commit"])
            self.assertEqual(payload["models"][label]["weights_sha256"], load(multi_name)["weights_sha256"])

    def test_the_mode_comparison_agrees_with_the_summaries(self) -> None:
        gates = load(TRANSFER_GATES)["arms"]
        for label, single_name, multi_name in zip(ORDER_LABELS, TRANSFER, MULTI_TURN):
            for which, name in (("single_turn", single_name), ("multi_turn", multi_name)):
                summary, counts = load(name), mode(label, which)
                with self.subTest(label=label, which=which):
                    by_intent = summary["by_intent"]
                    self.assertEqual(counts["completion_rate_on_fulfil"], by_intent["fulfil"]["correct_rate"])
                    self.assertEqual(counts["refusal_rate_on_refuse"], by_intent["refuse"]["correct_rate"])
                    for k in (1, 4):
                        self.assertAlmostEqual(
                            mode_comparison(label, "correct", k)[f"{which}_pass_power_k"],
                            summary["metrics"][f"pass^{k}"],
                        )
            self.assertEqual(mode(label, "single_turn")["write_classes"], gates[label]["write_classes"], label)
            multi = load(multi_name)["multi_turn"]
            stored = {intent: entry["write_classes"] for intent, entry in multi["by_intent"].items()}
            self.assertEqual(mode(label, "multi_turn")["write_classes"], stored, label)
            self.assertAlmostEqual(
                mode_comparison(label, "clean_correct", 1)["multi_turn_pass_power_k"], multi["clean_metrics"]["pass^1"]
            )
            self.assertEqual(mode(label, "multi_turn")["mean_turns_used"], multi["mean_turns_used"])

    def test_mode_comparison_statistics(self) -> None:
        payload = modes()
        comparisons = [entry for label in ORDER_LABELS for entry in payload["models"][label]["comparisons"]]
        self.assertEqual(len(comparisons), payload["multiplicity"]["comparisons_reported"])
        self.assertEqual(len(comparisons), 9, "the README says nine comparisons")
        self.assertAlmostEqual(payload["multiplicity"]["bonferroni_alpha"], 0.05 / 9)
        floor = 1 / (payload["permutation_resamples"] + 1)
        for entry in comparisons:
            self.assertEqual(entry["difference_is"], "multi_turn minus single_turn")
            self.assertEqual(entry["tasks_compared"], 150)
            self.assertEqual(entry["bootstrap_replicates"], payload["bootstrap_replicates"])
            low, high = entry["difference_ci95"]
            if low > 0.0 or high < 0.0:
                self.assertAlmostEqual(entry["p_permutation_two_sided"], floor, msg=entry["rng_stream"])

    def test_the_closed_loop_claims(self) -> None:
        base, sft, grpo = (mode(label, "multi_turn") for label in ORDER_LABELS)
        # The untrained model never wrote single-turn, and never wrote in a first reply.
        self.assertEqual(load(TRANSFER[0])["diagnostics"]["attempted_write_rate"], 0.0)
        self.assertEqual(base["first_reply_update_order"]["episodes"], 0)
        self.assertEqual(sum(v for k, v in base["write_classes"]["refuse"].items() if k != "no_write"), 0)
        self.assertEqual(sft["lookup_rate"], 0.0, "SFT never calls get_order in the closed loop")
        tasks = json.loads(ORDER_SPLITS.read_text(encoding="utf-8"))["splits"]["test"]
        completable = sum(1 for t in tasks if t["intent"] == "fulfil")
        for label, counts, name in (("sft", sft, MULTI_TURN[1]), ("grpo", grpo, MULTI_TURN[2])):
            with self.subTest(label=label):
                first = counts["first_reply_update_order"]
                self.assertEqual(first["dispatched"], first["episodes"], "every first-reply write ran")
                self.assertEqual(counts["completion_rate_on_fulfil"], 1.0)
                self.assertEqual(round(load(name)["metrics"]["pass^4"] * len(tasks)), completable)
                self.assertEqual(
                    counts["write_classes"]["fulfil"]["verified_write_without_lookup"]
                    + counts["write_classes"]["fulfil"]["clean_write"],
                    counts["episodes_by_intent"]["fulfil"],
                )
                change = mode_comparison(label, "correct", 4)
                self.assertEqual(change["tasks_regressed"], 0, "pass^4 rose only on completable requests")
        self.assertEqual(
            sft["write_classes"]["fulfil"]["verified_write_without_lookup"], sft["episodes_by_intent"]["fulfil"]
        )
        # "The other 31" and "the 42 that held off" are the refusal-required
        # episodes whose first reply held no update_order call.
        for label, counts in (("sft", sft), ("grpo", grpo)):
            with self.subTest(label=label):
                later = counts["later_write_after_failed_authenticate"]["by_intent"]["refuse"]
                first = counts["first_reply_update_order"]["by_intent"]["refuse"]["episodes"]
                self.assertEqual(later["waited"], counts["episodes_by_intent"]["refuse"] - first)
                # Every unauthenticated write came either from a first reply or after waiting.
                self.assertEqual(
                    first + later["wrote_after_failed_authenticate"],
                    counts["write_classes"]["refuse"]["unauthenticated_write"],
                )
        base_later = base["later_write_after_failed_authenticate"]["by_intent"]["refuse"]
        self.assertEqual(base_later["wrote_after_failed_authenticate"], 0)
        # "calls get_order in 2 of 600 episodes" is the lookup rate times the episodes.
        lookups = grpo["lookup_rate"] * grpo["episodes"]
        self.assertAlmostEqual(lookups, round(lookups))
        self.assertGreater(grpo["lookup_rate"], 0.0, "almost, not exactly, the same as SFT")
        # "Most of this failure is horizon collapse": most episodes write in the first reply.
        for counts in (sft, grpo):
            self.assertGreater(2 * counts["first_reply_update_order"]["episodes"], counts["episodes"])

    def test_the_strict_pass1_breakdown(self) -> None:
        # Every strict pass is a correct attempt, and the per-intent counts add up to the table.
        for label in ORDER_LABELS:
            counts = strict_counts(label)
            with self.subTest(label=label):
                total = sum(c["clean"] for c in counts.values())
                self.assertEqual(total, round(mode_comparison(label, "clean_correct", 1)["multi_turn_pass_power_k"] * 600))
                for c in counts.values():
                    self.assertLessEqual(c["clean"], c["correct"])
        base = strict_counts("base")
        summary = load(MULTI_TURN[0])
        # The refusals that miss strict pass^1 are as many as the episodes with no tool call.
        no_call = round((1 - summary["diagnostics"]["called_any_tool_rate"]) * summary["episodes"])
        self.assertEqual(base["refuse"]["correct"] - base["refuse"]["clean"], no_call)
        self.assertEqual(summary["multi_turn"]["by_intent"]["refuse"]["write_classes"]["no_write"], 296)
        sft = strict_counts("sft")
        self.assertEqual(sft["fulfil"]["clean"], 0)
        self.assertEqual(sft["refuse"]["clean"], sft["refuse"]["correct"])

    @unittest.skipUnless((RESULTS / "episodes-order-support-multiturn-base-3cacc39.jsonl").exists(),
                         "the multi-turn episode logs are on the v1.1.0 release")
    def test_the_rejected_refusals_ask_for_the_postal_code(self) -> None:
        log = RESULTS / "episodes-order-support-multiturn-base-3cacc39.jsonl"
        rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
        rejected = [r for r in rows if r["intent"] == "refuse" and r["correct"] and not r["clean_correct"]]
        self.assertEqual(len(rejected), 83)
        for r in rejected:
            self.assertEqual(r["template_id"], "address_change_no_postcode")
            self.assertEqual(r["executed_calls"], 0)
            self.assertFalse(r["gate_violation"])
            self.assertRegex(r["completions"][-1].lower(), "postcode|postal code")
        skipped = [r for r in rows if r["intent"] == "fulfil" and r["correct"] and not r["clean_correct"]]
        self.assertEqual({r["write_class"] for r in skipped}, {"verified_write_without_lookup"})

    def test_every_sft_example_is_one_reply_with_one_tool_call(self) -> None:
        stats = load("sft-dataset-54218c4.json")["selection_stats"]
        self.assertEqual(stats["completion_shapes"], {"bare_tool_call": stats["selected"]})

    def test_multi_turn_token_counts(self) -> None:
        tokens = token_counts()["order_support_multiturn"]
        self.assertEqual(sorted(tokens), sorted(ORDER_LABELS))
        for label, name in zip(ORDER_LABELS, MULTI_TURN):
            entry = tokens[label]
            with self.subTest(label=label):
                self.assertEqual(entry["summary"], name)
                self.assertEqual(entry["max_new_tokens"], load(name)["rollout"]["max_new_tokens"])
                self.assertEqual(entry["generated"]["source"], "rebuilt_from_logs")
                self.assertEqual(entry["prompt"]["source"], "recorded")
                self.assertEqual(entry["generated"]["turns_at_cap"], 0, "no turn reached the cap")
                self.assertEqual(entry["episodes"], load(name)["episodes"])

    def test_the_example_task_is_in_the_test_split(self) -> None:
        test = json.loads(GSM8K_SPLITS.read_text(encoding="utf-8"))["splits"]["test"]
        ids = {entry["task_id"] if isinstance(entry, dict) else entry for entry in test}
        self.assertIn("gsm8k:test:230", ids)


class ExampleTests(unittest.TestCase):
    """The R1 retry example, replayed through the sandbox."""

    FIRST = "12 * 20000 + 10% * 12 * 20000 + 1000 * 12"
    SECOND = FIRST.replace("10%", "0.10")

    def assert_printed(self, text: str) -> None:
        self.assertTrue(text in README, f"not in the README: {text}")

    def test_the_first_attempt_fails_with_the_printed_feedback(self) -> None:
        call = json.dumps({"name": "calculator", "arguments": {"expression": self.FIRST}})
        self.assert_printed(f"Attempt: {call}")
        trace = build_gsm8k_registry().execute(
            parse_tool_calls(f"<tool_call>\n{call}\n</tool_call>"), {}, gate_engine=GateEngine.from_mapping({})
        )
        self.assert_printed(f"Feedback: {_failure_observation(trace)} Retry:")

    def test_the_retry_computes_the_gold_answer(self) -> None:
        call = json.dumps({"name": "calculator", "arguments": {"expression": self.SECOND}})
        self.assert_printed(f"Retry: {call}")
        value = evaluate_expression(self.SECOND)
        self.assertEqual(value, 276000)
        self.assert_printed(f"Sandbox: {value} gold answer 276000: correct")


class LinkTests(unittest.TestCase):
    """Every relative link in the published Markdown must resolve."""

    LINK = re.compile(r"\]\(([^)\s]+)\)")

    def test_relative_links_resolve(self) -> None:
        for document in (README_PATH, RESULTS_README_PATH):
            for target in self.LINK.findall(document.read_text(encoding="utf-8")):
                if re.match(r"[a-z]+://", target):
                    continue
                path = (document.parent / target.split("#")[0]).resolve()
                with self.subTest(document=document.name, target=target):
                    self.assertTrue(path.exists(), f"{target} does not exist")

    def test_the_release_links(self) -> None:
        for document in (README_PATH, RESULTS_README_PATH):
            text = document.read_text(encoding="utf-8")
            self.assertIn(RELEASE_URL, text)
            self.assertIn(MULTI_TURN_RELEASE_URL, text)
            self.assertNotIn("not yet published", text)


class WordingTests(unittest.TestCase):
    BANNED = (
        "not recorded", "phase_a", "phase_b", "Phase A", "Phase B",
        "dearer", "fluke", "maths", "judgement", "behaviour", "analyse",
        "robust", "comprehensive", "seamless", "leverage", "utilize", "crucial",
        "pivotal", "delve", "moreover", "furthermore", "additionally",
        "rigorous", "testament", "it is worth noting", "in conclusion",
        "sheds light", "underscores", "landscape", "paradigm", "synergy",
        "holistic", "meticulous", "bulletproof", "most likely", "\u2014",
    )
    # Claims earlier versions had to withdraw.
    WITHDRAWN = (
        "a third as much",
        "barely moves",
        "most of the untrained gap",
        "Reinforcement learning added nothing",
        "never once calls",
        "never calls the lookup tool",
        "worse than the one that does nothing",
        "three tools it has never seen",
        "Three of the four reward terms never varied",
        "did not forget anything",
        "blind to losses",
        "unseen tools",
        "would not have caught",
        "We call",
    )
    # Plain-word phrasings the README replaced with standard terms.
    RETIRED = (
        "Three steps came after test results existed",
        "trained again on the pinned base",
        "that wrote the training data",
        "ran to the right answer",
        "a turn is one reply",
        "right number written in prose",
        "ran before the evaluator counted tokens",
        "It never calls `get_order`, so its strict pass^1 is 0.003",
    )

    def check(self, text: str) -> None:
        lowered = text.lower()
        for phrase in self.BANNED + self.WITHDRAWN + self.RETIRED:
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase.lower(), lowered)

    def test_readme(self) -> None:
        self.check(README)

    def test_postcode_only_as_the_field_name(self) -> None:
        prose = re.sub(r"`[^`\n]+`", " ", FENCE.sub(" ", README_TEXT))
        self.assertNotIn("postcode", prose.lower())

    def test_results_readme_prose(self) -> None:
        # File and script names are the real ones, so code spans are skipped.
        text = RESULTS_README_PATH.read_text(encoding="utf-8")
        text = FENCE.sub(" ", text)
        self.check(collapse(re.sub(r"`[^`\n]+`", " ", text)))


if __name__ == "__main__":
    unittest.main()
