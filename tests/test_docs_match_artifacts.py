"""The README's numbers must be the numbers in the frozen artifacts.

Every table row the README prints is rebuilt here from the files in results/,
including the one token-counts artifact, and looked up in the README with its
whitespace collapsed. Numbers in the prose are checked in context: each must
appear close to a phrase that says what it is. The last number test lists
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
GSM8K_SPLITS = PROJECT_ROOT / "configs" / "splits" / "phase_a_gsm8k.json"
ORDER_SPLITS = PROJECT_ROOT / "configs" / "splits" / "phase_b_orders.json"
# GPU runs whose platform record names the Python version.
GPU_RUNS = SFT_RUNS + tuple(g[2] for g in GRPO) + TRANSFER + MMLU_ARMS
RELEASE_URL = "https://github.com/DEZ-byte/agent-reliability/releases/tag/v1.0.0"


# --- loading ---------------------------------------------------------------


@functools.cache
def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


@functools.cache
def token_counts() -> dict:
    found = sorted(RESULTS.glob("token-counts-*.json"))
    if len(found) != 1:
        raise AssertionError(
            f"expected exactly one results/token-counts-*.json, found {len(found)}; "
            "run `python scripts/analyze.py backfill-tokens` on a clean tree"
        )
    payload = json.loads(found[0].read_text(encoding="utf-8"))
    if payload.get("kind") != "token_counts":
        raise AssertionError(f"{found[0].name} is not a token_counts artifact")
    return payload


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


def transfer_rows() -> list[str]:
    arms = [load(name) for name in TRANSFER]
    gates = load(TRANSFER_GATES)["arms"]
    fulfil, refuse = (gates["sft"]["episodes_by_intent"][intent] for intent in ("fulfil", "refuse"))
    wrote = [
        sum(n for kind, n in gates[label]["write_classes"]["refuse"].items() if kind != "no_write")
        for label in ("base", "sft", "grpo")
    ]
    return [
        row("pass^1", *(f3(a["metrics"]["pass^1"]) for a in arms)),
        row("pass^4", *(f3(a["metrics"]["pass^4"]) for a in arms)),
        row(
            f"Completable requests completed ({fulfil} attempts)",
            *(f3(a["by_intent"]["fulfil"]["correct_rate"]) for a in arms),
        ),
        row(
            f"Refusal-required requests refused ({refuse} attempts)",
            *(f3(a["by_intent"]["refuse"]["correct_rate"]) for a in arms),
        ),
        row("Refusal-required attempts that wrote anyway", *(str(n) for n in wrote)),
    ]


def all_rows() -> list[str]:
    return gsm8k_rows() + seed_rows() + paired_rows() + failure_rows() + grpo_rows() + mmlu_rows() + transfer_rows()


def claims() -> list[tuple[str, str]]:
    """(number as printed, a phrase it must appear near), section by section."""

    out: list[tuple[str, str]] = []

    # Key findings: SFT against the scaffolded 8B and against the teacher.
    vs_llama = [comparison(name, "R1", 4) for name in H1]
    out += [(signed(c["difference"]), "excludes zero") for c in vs_llama]
    run3 = vs_llama[2]
    out += [(interval(run3["difference_ci95"]), "run 3"), (pvalue(run3["p_permutation_two_sided"]), "permutation p")]
    rows = {entry["row"]: entry for entry in token_counts()["rows"]}
    llama_cost = rows["Llama-3.1-8B + retry"]["rungs"]["R1"]["cost"]
    ratios = [rows[f"Qwen3-1.7B, SFT run {run}"]["rungs"]["R1"]["cost"] / llama_cost for run in (1, 2, 3)]
    out += [(f"{percent(min(ratios))} to {percent(max(ratios))}", "of one 8B attempt")]
    teacher = rung(BASELINE, QWEN_TEACHER, "R1")["metrics"]["pass^4"]
    sft4 = [rung(name, QWEN_SMALL, "R1")["metrics"]["pass^4"] for name in SFT_TESTS]
    out += [(f"{f3(teacher)} against {f3(min(sft4))} to {f3(max(sft4))}", "three SFT runs")]

    # Key findings: GRPO.
    for rate, paired, run, _ in GRPO:
        out.append((f"{signed(comparison(paired, 'R0', 1)['difference'])} at", rate))
    late = comparison(GRPO[1][1], "R0", 1)
    out.append((f"p = {p3(late['p_sign_test_exact'])}", "sign test"))
    out.append((f"p = {p3(late['p_permutation_two_sided'])}", "permutation"))
    out.append((interval(late["difference_ci95"]), "exploratory"))
    shares = [load(run)["group_health"]["zero_variance_fraction"] for _, _, run, _ in GRPO]
    out.append((f"{percent(min(shares))} to {percent(max(shares))}", "8-rollout groups"))
    out.append((str(load(GRPO[0][2])["resolved"]["num_generations"]), "-rollout groups"))

    # Key findings: transfer.
    base, sft, _ = (load(name)["by_intent"] for name in TRANSFER)
    out += [
        (f3(sft["fulfil"]["correct_rate"]), "completable requests"),
        (f3(base["fulfil"]["correct_rate"]), "for the untrained model"),
        (f"from {f3(base['refuse']['correct_rate'])} to {f3(sft['refuse']['correct_rate'])}", "falls"),
    ]
    refuse = load(TRANSFER_GATES)["arms"]["sft"]["write_classes"]["refuse"]
    out.append((f"{sum(refuse.values()) - refuse['no_write']} of {sum(refuse.values())}", "wrote to the order"))
    halted = [templates("base")[t] for t in ("address_change_verified", "address_change_wrong_postcode")]
    lone = sum(t["call_sequences"].get("authenticate", 0) for t in halted)
    out.append((f"{lone} of {sum(t['episodes'] for t in halted)}", "authenticate"))
    silent = templates("base")["address_change_no_postcode"]
    no_call = f"{silent['call_sequences']['(none)']} of {silent['episodes']}"
    out.append((no_call, "no tool call"))

    # Figure captions.
    out.append((str(failures("base-1.7b")["episodes"]), "out of"))

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
        (f"min(tokens + 1, {token_counts()['max_new_tokens']})", "re-tokenized"),
        (f"all {len(token_counts()['validation']['qwen'])} recorded rung totals", "exactly"),
    ]

    # GSM8K results.
    llama = rung(COMPARATOR, LLAMA, "R1")
    out += [
        (f"{gsm8k_r1['tasks']} test tasks, {gsm8k_r1['runs_per_task']} attempts each", "R1"),
        (f"{f3(llama['metrics']['pass@4'])} of tasks at least once", "Llama-3.1-8B"),
        (f"only {f3(llama['metrics']['pass^4'])} on all four", "Llama-3.1-8B"),
        (f"{params_b(LLAMA) / params_b(QWEN_SMALL):.1f} times as many parameters", "raw tokens"),
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
        (f"{run['train']['global_step']} steps", "two per step"),
        (f"{run['resolved']['num_generations']} completions per prompt", "two per step"),
    ]
    spread = [load(r)["group_health"]["mean_component_std"] for _, _, r, _ in GRPO]
    out.append((f"{f3(spread[0]['accuracy'])} and {f3(spread[1]['accuracy'])}", "standard deviation"))
    others = max(s[term] for s in spread for term in ("format", "gate", "efficiency"))
    out.append((f"at most {f3(others)}", "any other term"))

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
        (interval(everything["difference_ci95"]), "95% interval"),
        (f"about {round(100 * max(-everything['difference_ci95'][0], everything['difference_ci95'][1]))} points",
         "cannot be"),
        (f"{len(unreadable)} questions", "no readable answer"),
        (f"{len(cut)} of those", "cut off"),
        (f"{readable['questions']} questions", "both named"),
        (f"{signed(readable['difference'])} {interval(readable['difference_ci95'])}", "both named"),
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
        (f"{no_postcode} give no postal code", "refused"),
        (f"{wrong_postcode} give the wrong one", "refused"),
    ]
    sft_fulfil = load(TRANSFER_GATES)["arms"]["sft"]["write_classes"]["fulfil"]
    completed = sft_fulfil["clean_write"] + sft_fulfil["verified_write_without_lookup"]
    out.append((f"{sft_fulfil['verified_write_without_lookup']} of SFT run 3's {completed}", "get_order"))
    out.append((f"{lone} of {sum(t['episodes'] for t in halted)} episodes", "postal code"))
    out.append((f"{silent['call_sequences']['(none)']} of its {silent['episodes']} replies", "no tool call"))
    none = {arm: templates(arm)["address_change_no_postcode"] for arm in ("sft", "grpo")}
    out.append((f"{none['sft']['wrote']} of {none['sft']['episodes']}", "no postal code"))
    out.append((f"GRPO in {none['grpo']['wrote']}", "no postal code"))

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

    def test_transfer_table(self) -> None:
        self.check(transfer_rows())


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

    def test_no_multi_turn_result_exists(self) -> None:
        self.assertEqual(list(RESULTS.glob("order-support-multiturn-*.json")), [], "the README says it has not run")

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

    def test_the_release_link(self) -> None:
        for document in (README_PATH, RESULTS_README_PATH):
            self.assertIn(RELEASE_URL, document.read_text(encoding="utf-8"))


class WordingTests(unittest.TestCase):
    BANNED = (
        "not recorded", "phase_a", "phase_b", "Phase A", "Phase B",
        "dearer", "fluke", "maths", "judgement", "behaviour", "analyse",
        "robust", "comprehensive", "seamless", "leverage", "utilize", "crucial",
        "pivotal", "delve", "moreover", "furthermore", "additionally",
        "rigorous", "testament", "it is worth noting", "in conclusion",
        "sheds light", "underscores", "landscape", "paradigm", "synergy",
        "holistic", "meticulous", "bulletproof", "\u2014",
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
    )

    def check(self, text: str) -> None:
        lowered = text.lower()
        for phrase in self.BANNED + self.WITHDRAWN:
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
