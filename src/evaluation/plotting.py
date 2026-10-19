"""The data behind the two README figures, and the code that draws them.

Every number a figure shows is read from a file in results/ or from the
token-counts artifact. The `*_data` functions read and cross-check those files
and return plain dicts, so they run without matplotlib. The `render_*`
functions import matplotlib when called and only draw what the dicts hold.

The file names below name frozen inputs. Nothing in this module holds a
result value.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final


class PlotError(RuntimeError):
    """An input is missing, ambiguous, or disagrees with another input."""


QWEN_SMALL: Final = "Qwen/Qwen3-1.7B"
QWEN_TEACHER: Final = "Qwen/Qwen3-4B"
LLAMA: Final = "meta-llama/Llama-3.1-8B-Instruct"

# One color per model family, used the same way in every panel.
COLORS: Final = {
    "untrained": "#94a3b8",
    "llama": "#f59e0b",
    "sft": "#2563eb",
    "grpo": "#0d9488",
    "teacher": "#475569",
}

TOKEN_COUNTS_GLOB: Final = "token-counts-*.json"
RUNG: Final = "R1"
METRICS: Final = ("pass@4", "pass^1", "pass^4")

# Every run in the reliability and cost figure: which summary holds it, and
# for the SFT runs, which paired comparison against Llama it has.
RUNS: Final = (
    {
        "key": "untrained",
        "label": "Qwen3-1.7B untrained",
        "family": "untrained",
        "summary": "baseline-phase_a-3cc174f.json",
        "model": QWEN_SMALL,
    },
    {
        "key": "llama",
        "label": "Llama-3.1-8B + retry",
        "family": "llama",
        "summary": "comparator-8b-2889b6d.json",
        "model": LLAMA,
    },
    {
        "key": "sft1",
        "label": "SFT run 1",
        "family": "sft",
        "summary": "sft-test-qwen3-1.7b-c16530e.json",
        "model": QWEN_SMALL,
        "vs_llama": "h1-comparison-2889b6d.json",
    },
    {
        "key": "sft2",
        "label": "SFT run 2",
        "family": "sft",
        "summary": "sft-test-qwen3-1.7b-seed20260823-67488ea.json",
        "model": QWEN_SMALL,
        "vs_llama": "h1-comparison-seed20260823-2889b6d.json",
    },
    {
        "key": "sft3",
        "label": "SFT run 3",
        "family": "sft",
        "summary": "sft-test-qwen3-1.7b-seed20260824-51f16bf.json",
        "model": QWEN_SMALL,
        "vs_llama": "h1-comparison-seed20260824-2889b6d.json",
    },
    {
        "key": "grpo-1e-6",
        "label": "SFT + GRPO, lr 1e-6",
        "family": "grpo",
        "summary": "grpo-test-qwen3-1.7b-f6138ee.json",
        "model": QWEN_SMALL,
    },
    {
        "key": "grpo-1e-5",
        "label": "SFT + GRPO, lr 1e-5",
        "family": "grpo",
        "summary": "grpo-test-lr1e5-ce5f2ca.json",
        "model": QWEN_SMALL,
    },
    {
        "key": "teacher",
        "label": "Qwen3-4B teacher",
        "family": "teacher",
        "summary": "baseline-phase_a-3cc174f.json",
        "model": QWEN_TEACHER,
    },
)

# The bar groups of the capability vs. reliability panel, in drawing order.
GROUPS: Final = (
    {"key": "untrained", "label": "Qwen3-1.7B\nuntrained", "runs": ("untrained",)},
    {"key": "llama", "label": "Llama-3.1-8B\n+ retry", "runs": ("llama",)},
    {"key": "sft", "label": "Qwen3-1.7B\nSFT (LoRA)\n3 seeds", "runs": ("sft1", "sft2", "sft3")},
    {
        "key": "grpo",
        "label": "Qwen3-1.7B\nSFT + GRPO\n2 learning rates",
        "runs": ("grpo-1e-6", "grpo-1e-5"),
    },
    {"key": "teacher", "label": "Qwen3-4B\nteacher", "runs": ("teacher",)},
)

# The runs whose pass@4 to pass^4 drop is called out.
DROP_RUNS: Final = ("llama", "sft3")

FAILURE_MODES: Final = "failure-modes-2f70608.json"
FAILURE_PARTS: Final = (
    ("no_call_or_unparsed", "no call or unparsable call"),
    ("call_failed_to_run", "call failed to run"),
    ("wrong_value", "call ran, wrong value"),
)
# Labels and order for the arms the failure file is known to hold. Any other
# arm is drawn after these, under its own key.
FAILURE_ARMS: Final = {
    "base-1.7b": ("Qwen3-1.7B untrained", "untrained"),
    "sft-run1": ("SFT run 1", "sft"),
    "sft-run2": ("SFT run 2", "sft"),
    "sft-run3": ("SFT run 3", "sft"),
    "grpo-1e-6": ("SFT + GRPO, lr 1e-6", "grpo"),
    "grpo-1e-5": ("SFT + GRPO, lr 1e-5", "grpo"),
    "llama-8b": ("Llama-3.1-8B + retry", "llama"),
    "teacher-4b": ("Qwen3-4B teacher", "teacher"),
}

# The order-support runs, and the GSM8K summary whose adapter each one must
# have loaded. The untrained run loads no adapter.
ORDER_SUPPORT: Final = (
    {
        "key": "untrained",
        "label": "Qwen3-1.7B untrained",
        "family": "untrained",
        "file": "phase_b-base-10f9d97.json",
        "same_adapter_as": None,
    },
    {
        "key": "sft3",
        "label": "SFT run 3",
        "family": "sft",
        "file": "phase_b-sft-aa59330.json",
        "same_adapter_as": "sft-test-qwen3-1.7b-seed20260824-51f16bf.json",
    },
    {
        "key": "grpo-1e-6",
        "label": "SFT + GRPO, lr 1e-6",
        "family": "grpo",
        "file": "phase_b-grpo-42e347c.json",
        "same_adapter_as": "grpo-test-qwen3-1.7b-f6138ee.json",
    },
)

GRPO_RUNS: Final = (
    "grpo-run-qwen3-1.7b-a498a7b.json",
    "grpo-run-qwen3-1.7b-lr1e5-3d7e90f.json",
)


# -- reading -----------------------------------------------------------------


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PlotError(f"missing input {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def find_token_counts(results_dir: Path) -> Path:
    """The one token-counts artifact in results/. Zero or several is an error."""

    found = sorted(results_dir.glob(TOKEN_COUNTS_GLOB))
    if len(found) != 1:
        names = ", ".join(path.name for path in found) or "none"
        raise PlotError(
            f"expected exactly one {TOKEN_COUNTS_GLOB} in {results_dir.name}/, found {names}; "
            "pass --token-counts"
        )
    return found[0]


def summary_rung(summary: Mapping[str, Any], model: str, rung: str, name: str) -> dict[str, Any]:
    """One model's results at one rung, from an evaluation summary."""

    results = [r for r in summary.get("results", []) if r["candidate"]["id"] == model]
    if summary.get("executed") is not True or len(results) != 1:
        raise PlotError(f"{name} holds no single executed result for {model}")
    rungs = results[0].get("rungs", {})
    if rung not in rungs:
        raise PlotError(f"{name} has no {rung} result for {model}")
    return rungs[rung]


def token_rung(
    token_counts: Mapping[str, Any], summary: str, model: str, rung: str
) -> dict[str, Any]:
    """The token-counts row for one summary and model, at one rung."""

    rows = [
        row for row in token_counts.get("rows", [])
        if row.get("summary") == summary and row.get("model") == model
    ]
    if len(rows) != 1:
        raise PlotError(f"the token counts hold {len(rows)} rows for {model} in {summary}")
    entry = rows[0]["rungs"].get(rung)
    if entry is None:
        raise PlotError(f"the token counts have no {rung} entry for {model} in {summary}")
    return {**entry, "unique_params_b": rows[0]["unique_params_b"]}


def _same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)


# -- figure 1: reliability and cost ------------------------------------------


def reliability_cost_data(results_dir: Path, token_counts_path: Path) -> dict[str, Any]:
    """Pass rates, intervals, costs and the SFT vs. Llama comparison, at R1."""

    token_counts = read_json(token_counts_path)
    if token_counts.get("kind") != "token_counts":
        raise PlotError(f"{token_counts_path.name} is not a token-counts artifact")
    summaries: dict[str, dict[str, Any]] = {}

    recorded = {item["file"]: item["sha256"] for item in token_counts.get("inputs", [])}

    def summary(name: str) -> dict[str, Any]:
        if name not in summaries:
            path = results_dir / name
            data = read_json(path)
            # The token counts must have been built from these exact bytes.
            if name not in recorded:
                raise PlotError(f"{token_counts_path.name} does not list {name} as an input")
            if hashlib.sha256(path.read_bytes()).hexdigest() != recorded[name]:
                raise PlotError(
                    f"{name} differs from the bytes {token_counts_path.name} was built from"
                )
            summaries[name] = data
        return summaries[name]

    runs: list[dict[str, Any]] = []
    for spec in RUNS:
        stored = summary_rung(summary(spec["summary"]), spec["model"], RUNG, spec["summary"])
        metrics = stored["metrics"]
        tokens = token_rung(token_counts, spec["summary"], spec["model"], RUNG)
        if tokens["episodes"] != stored["episodes"]:
            raise PlotError(
                f"{spec['label']}: the token counts cover {tokens['episodes']} episodes, "
                f"the summary {stored['episodes']}"
            )
        runs.append({
            "key": spec["key"],
            "label": spec["label"],
            "family": spec["family"],
            "color": COLORS[spec["family"]],
            "summary": spec["summary"],
            "tasks": stored["tasks"],
            "runs_per_task": stored["runs_per_task"],
            **{name: metrics[name] for name in METRICS},
            # Only pass^1 and pass^4 carry a stored interval.
            "ci95": {
                name: list(metrics[f"{name}_ci95"])
                for name in METRICS
                if f"{name}_ci95" in metrics
            },
            "tokens_per_attempt": tokens["tokens_per_attempt"],
            "unique_params_b": tokens["unique_params_b"],
            "cost": tokens["cost"],
            "token_source": tokens["source"],
        })
    by_key = {run["key"]: run for run in runs}

    shapes = {(run["tasks"], run["runs_per_task"]) for run in runs}
    if len(shapes) != 1:
        raise PlotError(f"the runs differ in tasks or attempts per task: {sorted(shapes)}")
    tasks, attempts = shapes.pop()

    groups: list[dict[str, Any]] = []
    for spec in GROUPS:
        members = [by_key[key] for key in spec["runs"]]
        metrics: dict[str, Any] = {}
        for name in METRICS:
            values = [run[name] for run in members]
            entry: dict[str, Any] = {"value": sum(values) / len(values), "points": values}
            if len(members) > 1:
                entry.update(low=min(values), high=max(values), whisker="range")
            elif name in members[0]["ci95"]:
                low, high = members[0]["ci95"][name]
                entry.update(low=low, high=high, whisker="ci95")
            else:
                entry.update(low=None, high=None, whisker=None)
            metrics[name] = entry
        family = members[0]["family"]
        groups.append({
            "key": spec["key"],
            "label": spec["label"],
            "color": COLORS[family],
            "runs": list(spec["runs"]),
            "metrics": metrics,
        })

    drops = [
        {
            "run": key,
            "label": by_key[key]["label"],
            "group": next(g["key"] for g in GROUPS if key in g["runs"]),
            "pass@4": by_key[key]["pass@4"],
            "pass^4": by_key[key]["pass^4"],
        }
        for key in DROP_RUNS
    ]

    # The SFT vs. Llama gains come from the paired comparisons, which must
    # agree with the summaries they were computed from.
    llama = by_key["llama"]
    gains: list[dict[str, Any]] = []
    for spec in RUNS:
        if "vs_llama" not in spec:
            continue
        paired = read_json(results_dir / spec["vs_llama"])
        rows = [c for c in paired["comparisons"] if c["rung"] == RUNG and c["k"] == attempts]
        if len(rows) != 1:
            raise PlotError(f"{spec['vs_llama']} has no single {RUNG} k={attempts} comparison")
        row, run = rows[0], by_key[spec["key"]]
        if not (_same(row["baseline_pass_power_k"], llama["pass^4"])
                and _same(row["treatment_pass_power_k"], run["pass^4"])):
            raise PlotError(
                f"{spec['vs_llama']} does not match the summaries of {run['label']} and Llama"
            )
        gains.append({
            "run": run["key"],
            "label": run["label"],
            "file": spec["vs_llama"],
            "difference": row["difference"],
            "difference_ci95": list(row["difference_ci95"]),
            "cost_ratio": llama["cost"] / run["cost"],
        })

    return {
        "rung": RUNG,
        "tasks": tasks,
        "attempts": attempts,
        "token_counts": token_counts_path.name,
        "runs": runs,
        "groups": groups,
        "drops": drops,
        "sft_vs_llama": {
            "gains": gains,
            "gain_min": min(g["difference"] for g in gains),
            "gain_max": max(g["difference"] for g in gains),
            "cost_ratio_min": min(g["cost_ratio"] for g in gains),
            "cost_ratio_max": max(g["cost_ratio"] for g in gains),
        },
    }


# -- figure 2: failures and transfer -----------------------------------------


def failure_modes_data(results_dir: Path, rung: str = "R0") -> dict[str, Any]:
    """Failed attempts per model, split by how they failed."""

    data = read_json(results_dir / FAILURE_MODES)
    arms = data["arms"]
    order = [key for key in FAILURE_ARMS if key in arms]
    order += [key for key in arms if key not in FAILURE_ARMS]
    rows: list[dict[str, Any]] = []
    for key in order:
        stored = arms[key]["rungs"][rung]
        failures = stored["failures"]
        parts = {name: failures[name] for name, _ in FAILURE_PARTS}
        if sum(parts.values()) != failures["total"]:
            raise PlotError(f"{key}: the failure parts do not add up to the total")
        if failures["total"] + stored["correct"] != stored["episodes"]:
            raise PlotError(f"{key}: failures plus correct is not the episode count")
        label, family = FAILURE_ARMS.get(key, (key, None))
        rows.append({
            "arm": key,
            "label": label,
            "family": family,
            "candidate": arms[key]["candidate"],
            "episodes": stored["episodes"],
            "correct": stored["correct"],
            **parts,
            "total": failures["total"],
        })
    episodes = {row["episodes"] for row in rows}
    if len(episodes) != 1:
        raise PlotError(f"the arms have different episode counts: {sorted(episodes)}")
    return {
        "file": FAILURE_MODES,
        "rung": rung,
        "episodes": episodes.pop(),
        "parts": [{"key": name, "label": label} for name, label in FAILURE_PARTS],
        "rows": rows,
    }


def order_support_data(results_dir: Path) -> dict[str, Any]:
    """Completion and correct-refusal rates on the order-support test split."""

    rows: list[dict[str, Any]] = []
    for spec in ORDER_SUPPORT:
        data = read_json(results_dir / spec["file"])
        if data.get("executed") is not True or data.get("model", {}).get("id") != QWEN_SMALL:
            raise PlotError(f"{spec['file']} is not an executed {QWEN_SMALL} run")
        reference = spec["same_adapter_as"]
        if reference is None:
            if data.get("adapter") is not None:
                raise PlotError(f"{spec['file']} loaded an adapter; the untrained run must not")
        else:
            expected = read_json(results_dir / reference)["adapter"]["weights_sha256"]
            if data.get("weights_sha256") != expected:
                raise PlotError(f"{spec['file']} did not load the adapter that {reference} tested")
        fulfil, refuse = data["by_intent"]["fulfil"], data["by_intent"]["refuse"]
        rows.append({
            "key": spec["key"],
            "label": spec["label"],
            "color": COLORS[spec["family"]],
            "file": spec["file"],
            "completion_rate": fulfil["correct_rate"],
            "completion_episodes": fulfil["episodes"],
            "refusal_rate": refuse["correct_rate"],
            "refusal_episodes": refuse["episodes"],
            "refusal_engaged_rate": refuse["engaged_rate"],
            "attempted_write_rate": data["diagnostics"]["attempted_write_rate"],
            "episodes": data["episodes"],
        })
    return {"rows": rows}


def _learning_rate_label(value: float) -> str:
    mantissa, exponent = f"{value:e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"lr {mantissa}e{int(exponent)}"


def grpo_groups_data(results_dir: Path) -> dict[str, Any]:
    """The share of GRPO rollout groups whose rewards were all identical."""

    rows: list[dict[str, Any]] = []
    for name in GRPO_RUNS:
        data = read_json(results_dir / name)
        health = data["group_health"]
        groups, fraction = health["groups"], health["zero_variance_fraction"]
        count = fraction * groups
        rows.append({
            "file": name,
            "learning_rate": data["learning_rate"]["used"],
            "label": _learning_rate_label(data["learning_rate"]["used"]),
            "rollouts_per_group": data["resolved"]["num_generations"],
            "groups": groups,
            "zero_variance_fraction": fraction,
            # The file stores a fraction; the count is shown only when it is whole.
            "zero_variance_groups": round(count) if abs(count - round(count)) < 1e-9 else None,
        })
    sizes = {row["rollouts_per_group"] for row in rows}
    if len(sizes) != 1:
        raise PlotError(f"the GRPO runs used different group sizes: {sorted(sizes)}")
    return {"rollouts_per_group": sizes.pop(), "rows": rows}


def transfer_data(results_dir: Path) -> dict[str, Any]:
    return {
        "failure_modes": failure_modes_data(results_dir),
        "order_support": order_support_data(results_dir),
        "grpo_groups": grpo_groups_data(results_dir),
    }


# -- rendering ---------------------------------------------------------------

INK: Final = "#111827"
MUTED: Final = "#6b7280"
GRID: Final = "#9ca3af"
# Light to dark: pass@4, pass^1, pass^4 share the model's hue.
METRIC_ALPHA: Final = {"pass@4": 0.32, "pass^1": 0.62, "pass^4": 1.0}
METRIC_LABEL: Final = {
    "pass@4": "pass@4: solved in at least 1 of 4",
    "pass^1": "pass^1: one attempt",
    "pass^4": "pass^4: solved in all 4",
}
# How far each failure got, darkest first. One hue, kept apart from the model colors.
FAILURE_COLORS: Final = ("#3f2fa3", "#7c6ee0", "#c9c2f2")
SAVE: Final = {"dpi": 300, "bbox_inches": "tight", "metadata": {"Software": None}}


# Fixed fonts and sizes, so two runs give the same bytes. DejaVu Sans ships
# with matplotlib, so every machine has it.
STYLE: Final = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 9,
    "axes.labelcolor": INK,
    "axes.edgecolor": "#4b5563",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _grid(ax, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)


def _panel_letter(ax, letter: str) -> None:
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="bottom", ha="right", color=INK)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", **SAVE)


def render_reliability_cost(data: Mapping[str, Any], path: Path) -> None:
    plt = _pyplot()
    with plt.rc_context(STYLE):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"wspace": 0.22})
        try:
            _draw_capability_panel(ax_a, data)
            _draw_cost_panel(ax_b, data)
            _save(fig, path)
        finally:
            plt.close(fig)


def _draw_capability_panel(ax, data: Mapping[str, Any]) -> None:
    width = 0.26
    offsets = {name: (i - 1) * width for i, name in enumerate(METRICS)}
    centers = {}
    for x, group in enumerate(data["groups"]):
        centers[group["key"]] = x
        dashed = group["key"] == "teacher"
        for name in METRICS:
            entry = group["metrics"][name]
            bx = x + offsets[name]
            ax.bar(bx, entry["value"], width * 0.9, color=group["color"],
                   alpha=METRIC_ALPHA[name], linewidth=0, zorder=2)
            ax.bar(bx, entry["value"], width * 0.9, fill=False, edgecolor=group["color"],
                   linewidth=1.0, linestyle="--" if dashed else "-", zorder=3)
            if entry["whisker"] is not None:
                below, above = entry["value"] - entry["low"], entry["high"] - entry["value"]
                ax.errorbar(bx, entry["value"], yerr=[[below], [above]], fmt="none", ecolor=INK,
                            elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=4)
            if len(entry["points"]) > 1:
                ax.scatter([bx] * len(entry["points"]), entry["points"], s=11, color="white",
                           edgecolor=INK, linewidth=0.7, zorder=5)

    by_key = {run["key"]: run for run in data["runs"]}
    groups = {group["key"]: group for group in data["groups"]}
    for drop in data["drops"]:
        x = centers[drop["group"]]
        top = max(_top(m) for m in groups[drop["group"]]["metrics"].values())
        who = "" if by_key[drop["run"]]["family"] == "llama" else f"{drop['label']}:\n"
        text = (f"{who}pass@4 {drop['pass@4']:.3f}\n"
                f"\u2192 pass^4 {drop['pass^4']:.3f}\n"
                f"(\u2212{drop['pass@4'] - drop['pass^4']:.3f})")
        ax.text(x, top + 0.03, text, ha="center", va="bottom", fontsize=7.8, color=INK,
                linespacing=1.25)

    ax.set_xticks(range(len(data["groups"])))
    ax.set_xticklabels([g["label"] for g in data["groups"]], fontsize=8.2)
    tops = [_top(m) for g in data["groups"] for m in g["metrics"].values()]
    ax.set_ylim(0, math.ceil((max(tops) + 0.25) * 10) / 10)
    ax.set_xlim(-0.6, len(data["groups"]) - 0.4)
    ax.set_ylabel("share of tasks")
    _grid(ax)
    ax.set_title(f"Capability vs. reliability on GSM8K tool use\n"
                 f"({data['rung']}, {data['tasks']} tasks \u00d7 {data['attempts']} attempts)",
                 pad=8)
    _panel_letter(ax, "A")

    from matplotlib.patches import Patch

    handles = [Patch(facecolor=MUTED, alpha=METRIC_ALPHA[name], edgecolor=MUTED,
                     label=METRIC_LABEL[name]) for name in METRICS]
    ax.legend(handles=handles, loc="upper left", ncol=1, handlelength=1.4,
              bbox_to_anchor=(0.0, 1.0))
    ax.text(0.0, -0.21,
            "Bars: mean over runs; white dots: the individual runs. Whiskers: min-max range over "
            "runs\n(SFT, SFT + GRPO), or for a single run the 95% interval stored in its summary "
            "(pass^1, pass^4).\nNo interval is stored for pass@4, so a single run's pass@4 bar "
            "has no whisker.",
            transform=ax.transAxes, fontsize=7.3, color=MUTED, va="top", ha="left",
            linespacing=1.35)


def _top(metric: Mapping[str, Any]) -> float:
    return metric["high"] if metric["high"] is not None else metric["value"]


def _spread(values: list[float], gap: float) -> list[float]:
    """Push sorted label positions apart by at least `gap`, keeping their mean."""

    out = list(values)
    for _ in range(50):
        moved = False
        for i in range(1, len(out)):
            if out[i] - out[i - 1] < gap:
                shift = (gap - (out[i] - out[i - 1])) / 2
                out[i - 1] -= shift
                out[i] += shift
                moved = True
        if not moved:
            break
    return out


def _draw_cost_panel(ax, data: Mapping[str, Any]) -> None:
    runs = data["runs"]
    teacher = next(run for run in runs if run["family"] == "teacher")

    costs = [r["cost"] for r in runs]
    scores = [r["pass^4"] for r in runs]
    x_max = math.ceil(max(costs) * 1.12 / 20) * 20
    y_low = math.floor((min(scores) - 0.06) * 20) / 20
    y_high = math.ceil((max(scores) + 0.07) * 20) / 20
    ax.set_xlim(0, x_max)
    ax.set_ylim(y_low, y_high)

    def name(run: Mapping[str, Any]) -> str:
        mark = "\u2020" if run["token_source"] != "recorded" else ""
        return f"{run['label']}{mark}"

    # The teacher is drawn as a dashed reference line, labeled on the line.
    ax.axhline(teacher["pass^4"], color=teacher["color"], linestyle=(0, (5, 3)), linewidth=1.0,
               zorder=1)
    highest = " (highest)" if teacher["pass^4"] == max(scores) else ""
    ax.text(teacher["cost"] + x_max * 0.02, teacher["pass^4"] + (y_high - y_low) * 0.012,
            f"{name(teacher)}, pass^4 {teacher['pass^4']:.3f}{highest}", ha="left",
            va="bottom", fontsize=7.8, color=INK)

    for run in runs:
        ax.scatter(run["cost"], run["pass^4"], s=46, color=run["color"], edgecolor="white",
                   linewidth=1.2, zorder=4)

    # Runs that sit close together on cost get a label column to their right.
    cluster = sorted((r for r in runs if r["cost"] < min(costs) + x_max * 0.06),
                     key=lambda r: r["pass^4"])
    lifted = _spread([r["pass^4"] for r in cluster], (y_high - y_low) * 0.07)
    label_x = max(r["cost"] for r in cluster) + x_max * 0.045
    for run, y in zip(cluster, lifted):
        ax.annotate(f"{name(run)}  {run['pass^4']:.3f}", xy=(run["cost"], run["pass^4"]),
                    xytext=(label_x, y), va="center", ha="left", fontsize=7.8, color=INK,
                    arrowprops={"arrowstyle": "-", "color": GRID, "linewidth": 0.6,
                                "shrinkA": 0, "shrinkB": 3})
    for run in runs:
        if run in cluster or run is teacher:
            continue
        right = run["cost"] < x_max * 0.75
        ax.annotate(f"{name(run)}  {run['pass^4']:.3f}", xy=(run["cost"], run["pass^4"]),
                    xytext=(7 if right else -7, -11 if right else 11), textcoords="offset points",
                    ha="left" if right else "right", va="center", fontsize=7.8, color=INK)

    comparison = data["sft_vs_llama"]
    gains = ", ".join(f"{g['difference']:+.3f}" for g in comparison["gains"])
    text = (f"SFT runs 1-3 vs. Llama-3.1-8B + retry\n"
            f"({data['rung']}, k={data['attempts']}, paired per run):\n"
            f"pass^4 gain: {gains}\n"
            f"SFT cost per attempt: {100 / comparison['cost_ratio_max']:.0f}% to "
            f"{100 / comparison['cost_ratio_min']:.0f}%\nof Llama's")
    ax.text(0.985, 0.5, text, transform=ax.transAxes, ha="right", va="top", fontsize=7.8,
            color=INK, linespacing=1.4, multialignment="left",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "#d1d5db",
                  "linewidth": 0.8})

    ax.set_xlabel(f"{data['rung']} cost per attempt: generated tokens \u00d7 unique parameters (B)")
    ax.set_ylabel(f"{data['rung']} pass^4 (solved in all {data['attempts']} attempts)")
    _grid(ax, "y")
    ax.set_title("Reliability vs. inference cost", pad=8)
    _panel_letter(ax, "B")
    ax.text(0.0, -0.21,
            "\u2020 Token counts rebuilt by re-tokenizing the logged completions; the other runs "
            "were counted\nat generation time. Cost share: each SFT run's cost divided by "
            "Llama's cost.",
            transform=ax.transAxes, fontsize=7.3, color=MUTED, va="top", ha="left",
            linespacing=1.35)


def render_transfer(data: Mapping[str, Any], path: Path) -> None:
    plt = _pyplot()
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                                 gridspec_kw={"width_ratios": [1.45, 1.1, 0.75], "wspace": 0.3})
        try:
            _draw_failure_panel(axes[0], data["failure_modes"])
            _draw_order_support_panel(axes[1], data["order_support"])
            _draw_grpo_panel(axes[2], data["grpo_groups"])
            _save(fig, path)
        finally:
            plt.close(fig)


def _percent(value: float) -> str:
    """A share as a whole percentage, matching the README; counts are drawn beside it."""

    return f"{value * 100:.0f}%"


def _draw_failure_panel(ax, data: Mapping[str, Any]) -> None:
    rows = data["rows"]
    episodes = data["episodes"]
    y = list(range(len(rows)))[::-1]
    left = [0] * len(rows)
    for part, color in zip(data["parts"], FAILURE_COLORS):
        values = [row[part["key"]] for row in rows]
        ax.barh(y, values, left=left, height=0.62, color=color, edgecolor="white",
                linewidth=1.0, label=part["label"], zorder=2)
        left = [a + b for a, b in zip(left, values)]
    for yi, row in zip(y, rows):
        breakdown = " + ".join(str(row[part["key"]]) for part in data["parts"])
        ax.text(row["total"] + episodes * 0.012, yi, f"{row['total']}", va="center", ha="left",
                fontsize=8.2, fontweight="bold", color=INK)
        ax.text(row["total"] + episodes * 0.075, yi, f"= {breakdown}", va="center", ha="left",
                fontsize=7.6, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([row["label"] for row in rows])
    ax.set_xlim(0, episodes)
    ax.set_ylim(-0.6, len(rows) - 0.4 + 1.3)
    ax.set_xlabel(f"failed attempts, out of {episodes}")
    _grid(ax, "x")
    ax.set_title(f"Failure modes, GSM8K ({data['rung']}, {episodes} attempts per model)", pad=8)
    _panel_letter(ax, "A")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), handlelength=1.2,
              columnspacing=1.0)


def _engaged(row: Mapping[str, Any]) -> str:
    """The engaged share of refusal attempts, with the count when it is a whole number."""

    rate, episodes = row["refusal_engaged_rate"], row["refusal_episodes"]
    share = f"{100 * rate:.1f}%"
    count = rate * episodes
    if abs(count - round(count)) < 1e-6:
        return f"{share} of them ({round(count)} of {episodes})"
    return f"{share} of them"


def _draw_order_support_panel(ax, data: Mapping[str, Any]) -> None:
    rows = data["rows"]
    width = 0.26
    for i, row in enumerate(rows):
        offset = (i - (len(rows) - 1) / 2) * width
        for x, value in ((0, row["completion_rate"]), (1, row["refusal_rate"])):
            ax.bar(x + offset, value, width * 0.9, color=row["color"], linewidth=0, zorder=2,
                   label=row["label"] if x == 0 else None)
            ax.text(x + offset, value + 0.015, f"{value:.3f}", ha="center", va="bottom",
                    fontsize=7.6, color=INK)
    completable = {row["completion_episodes"] for row in rows}
    refusals = {row["refusal_episodes"] for row in rows}
    if len(completable) != 1 or len(refusals) != 1:
        raise PlotError("the order-support runs scored different numbers of attempts")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([
        f"completable requests:\ncompleted\n({completable.pop()} attempts)",
        f"refusal-required requests:\ncorrectly refused\n({refusals.pop()} attempts)",
    ])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("rate")
    _grid(ax)
    ax.set_title("Order-support transfer (single-turn)", pad=8)
    _panel_letter(ax, "B")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), handlelength=0.9,
              columnspacing=0.7, handletextpad=0.4, fontsize=7.6)
    # A refusal is graded on the order being unchanged, so a run that never
    # writes scores 1.000 on refusals whatever else it did. Say so on the panel.
    notes = [
        f"The {row['label']} run wrote in none of its {row['episodes']} attempts, so\n"
        f"every refusal-required attempt counts as refused. A non-write tool call\n"
        f"ran without error in {_engaged(row)}."
        for row in rows
        if row["attempted_write_rate"] == 0
    ]
    if notes:
        ax.text(0.0, -0.25, "\n".join(notes), transform=ax.transAxes, fontsize=7.3, color=MUTED,
                va="top", ha="left", linespacing=1.35)


def _draw_grpo_panel(ax, data: Mapping[str, Any]) -> None:
    rows = data["rows"]
    color = COLORS["grpo"]
    x = list(range(len(rows)))
    zero = [row["zero_variance_fraction"] for row in rows]
    rest = [1 - value for value in zero]
    size = data["rollouts_per_group"]
    ax.bar(x, zero, 0.58, color=color, alpha=0.35, edgecolor="white", linewidth=1.0, zorder=2,
           label=f"all {size} rewards identical (no gradient)")
    ax.bar(x, rest, 0.58, bottom=zero, color=color, edgecolor="white", linewidth=1.0, zorder=2,
           label="rewards differ")
    for xi, row in zip(x, rows):
        share = row["zero_variance_fraction"]
        count = row["zero_variance_groups"]
        detail = f"\n{count} of {row['groups']}" if count is not None else ""
        ax.text(xi, share / 2, f"{_percent(share)}{detail}", ha="center", va="center",
                fontsize=8, color=INK)
        ax.text(xi, share + (1 - share) / 2, _percent(1 - share), ha="center", va="center",
                fontsize=8, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"SFT + GRPO\n{row['label']}" for row in rows])
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel(f"share of {size}-rollout groups")
    _grid(ax)
    ax.set_title("GRPO groups with zero reward variance", pad=8)
    _panel_letter(ax, "C")
    ax.legend(loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.0), handlelength=1.0)
