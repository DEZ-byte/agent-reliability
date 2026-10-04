"""The numbers in README and FINDINGS must be the numbers in the frozen artifacts.

Every figure below is recomputed from a committed artifact and looked up in the
documents as it is printed there. A document edited to a nicer number, or an
artifact whose number the document never caught up with, fails here.

Figures that come only from local logs or model configs are marked with a dagger
in the documents and are not checked here, with two exceptions: the parameter
counts used for the cost proxy are pinned below, so the arithmetic that uses them
is still checked.

The last test lists phrases that ERRATA withdrew. They must not come back.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
# Whitespace is collapsed so a figure wrapped across two lines still matches.
README = " ".join((PROJECT_ROOT / "README.md").read_text(encoding="utf-8").split())
FINDINGS = " ".join((PROJECT_ROOT / "FINDINGS.md").read_text(encoding="utf-8").split())

MINUS = "−"  # the typographic minus the tables use

# Model-config parameter counts, marked with a dagger in the README.
QWEN3_1_7B_PARAMS_B = 2.03
LLAMA_3_1_8B_PARAMS_B = 8.03

SFT_TESTS = (
    "sft-test-qwen3-1.7b-c16530e.json",
    "sft-test-qwen3-1.7b-seed20260823-67488ea.json",
    "sft-test-qwen3-1.7b-seed20260824-51f16bf.json",
)
GRPO_TESTS = ("grpo-test-qwen3-1.7b-f6138ee.json", "grpo-test-lr1e5-ce5f2ca.json")
H1 = (
    "h1-comparison-2889b6d.json",
    "h1-comparison-seed20260823-2889b6d.json",
    "h1-comparison-seed20260824-2889b6d.json",
)
TEACHER = tuple(f"sft-vs-teacher-run{run}-d692d43.json" for run in (1, 2, 3))


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def rung(name: str, candidate: str, which: str) -> dict:
    for result in load(name)["results"]:
        if result["candidate"]["id"] == candidate:
            return result["rungs"][which]
    raise KeyError(candidate)


def f3(value: float) -> str:
    return f"{value:.3f}"


def signed(value: float, minus: str = MINUS) -> str:
    return ("+" if value >= 0 else minus) + f"{abs(value):.3f}"


def span(values: list[float]) -> str:
    return f"{f3(min(values))} – {f3(max(values))}"


def comparison(name: str, which: str, k: int) -> dict:
    for entry in load(name)["comparisons"]:
        if entry["rung"] == which and entry["k"] == k:
            return entry
    raise KeyError((name, which, k))


class HeadlineTests(unittest.TestCase):
    """The result table, every arm at R1."""

    def assertInBoth(self, text: str) -> None:
        self.assertIn(text, README)
        self.assertIn(text, FINDINGS)

    def test_untrained_1_7b_row(self) -> None:
        r1 = rung("baseline-phase_a-3cc174f.json", "Qwen/Qwen3-1.7B", "R1")["metrics"]
        self.assertIn(f"| R1 | {f3(r1['pass^1'])} | {f3(r1['pass^4'])} |", README)

    def test_teacher_row(self) -> None:
        r1 = rung("baseline-phase_a-3cc174f.json", "Qwen/Qwen3-4B", "R1")["metrics"]
        self.assertIn(f"| R1 | {f3(r1['pass^1'])} | {f3(r1['pass^4'])} |", README)
        self.assertIn(f"| `pass^1` | 0.415 | 0.517 – 0.553 | {f3(r1['pass^1'])} |", FINDINGS)

    def test_8b_row_and_cost(self) -> None:
        r1 = rung("comparator-8b-2889b6d.json", "meta-llama/Llama-3.1-8B-Instruct", "R1")
        cost = round(r1["generated_tokens_per_episode"] * LLAMA_3_1_8B_PARAMS_B)
        self.assertIn(
            f"| R1 | {f3(r1['metrics']['pass^1'])} | {f3(r1['metrics']['pass^4'])} | {cost} |",
            README,
        )

    def test_fine_tuned_and_grpo_ranges(self) -> None:
        for names in (SFT_TESTS, GRPO_TESTS):
            r1 = [rung(name, "Qwen/Qwen3-1.7B", "R1") for name in names]
            self.assertIn(span([r["metrics"]["pass^1"] for r in r1]), README)
            self.assertIn(span([r["metrics"]["pass^4"] for r in r1]), README)
            costs = [
                round(r["generated_tokens_per_episode"] * QWEN3_1_7B_PARAMS_B)
                for r in r1
                if r.get("generated_tokens_per_episode") is not None
            ]
            self.assertIn(f"{min(costs)} – {max(costs)}", README)

    def test_the_cost_ratio_is_about_31_percent(self) -> None:
        big = rung("comparator-8b-2889b6d.json", "meta-llama/Llama-3.1-8B-Instruct", "R1")
        big_cost = big["generated_tokens_per_episode"] * LLAMA_3_1_8B_PARAMS_B
        for name in SFT_TESTS:
            tokens = rung(name, "Qwen/Qwen3-1.7B", "R1").get("generated_tokens_per_episode")
            if tokens is None:
                continue
            ratio = tokens * QWEN3_1_7B_PARAMS_B / big_cost
            self.assertTrue(0.30 <= ratio <= 0.32, (name, ratio))
        self.assertIn("about 31% of the 8B's parameter-weighted", README)


class PairedComparisonTests(unittest.TestCase):
    def test_against_the_8b(self) -> None:
        for run, name in enumerate(H1, start=1):
            one, four = comparison(name, "R1", 1), comparison(name, "R1", 4)
            row = (
                f"| Run {run} | {signed(one['difference'])} | "
                f"{f3(one['difference_ci95'][0])} – {f3(one['difference_ci95'][1])} | "
                f"{signed(four['difference'])} | "
                f"{f3(four['difference_ci95'][0])} – {f3(four['difference_ci95'][1])} |"
            )
            self.assertIn(row, README)

    def test_against_the_teacher(self) -> None:
        for run, name in enumerate(TEACHER, start=1):
            one, four = comparison(name, "R1", 1), comparison(name, "R1", 4)
            row = (
                f"| Run {run} | {signed(one['difference'])} | "
                f"{signed(one['difference_ci95'][0])} to {signed(one['difference_ci95'][1])} | "
                f"{signed(four['difference'])} | "
                f"{signed(four['difference_ci95'][0])} to {signed(four['difference_ci95'][1])} |"
            )
            self.assertIn(row, README)
            self.assertIn(row, FINDINGS)
            self.assertLess(four["difference_ci95"][1], 0.0, "the pass^4 gap must exclude zero")


class GrpoTests(unittest.TestCase):
    def test_pass1_change_and_weight_shift_come_from_the_same_checkpoints(self) -> None:
        shifts = {
            Path(entry["after"]).name + "@" + Path(entry["after"]).parent.name: entry
            for entry in load("weight-change-b23567a.json")["comparisons"]
        }
        rows = (
            ("1e-6", "grpo-vs-sft-c364562.json", "checkpoint-200@qwen3-1.7b-grpo-pinned"),
            ("1e-5", "grpo-lr1e5-vs-sft-91a2de9.json", "checkpoint-300@qwen3-1.7b-grpo-lr1e5-pinned"),
        )
        for rate, name, adapter in rows:
            one = comparison(name, "R0", 1)
            shift = shifts[adapter]
            row = (
                f"| {rate} | R0 | {signed(one['difference'])} | "
                f"{signed(one['difference_ci95'][0])} – {f3(one['difference_ci95'][1])} | "
                f"{100 * shift['adapter_parameter_relative_change']:.2f}% | "
                f"{100 * shift['effective_delta_relative_change']:.1f}% |"
            )
            self.assertIn(row, README)

    def test_dead_problems_and_reward_spread(self) -> None:
        low = load("grpo-run-qwen3-1.7b-a498a7b.json")["group_health"]
        high = load("grpo-run-qwen3-1.7b-lr1e5-3d7e90f.json")["group_health"]
        self.assertEqual(low["groups"], 800, "one logged group per problem, two per step")
        self.assertEqual(high["groups"], 800, "one logged group per problem, two per step")
        text = (
            f"{round(100 * low['zero_variance_fraction'])}% of problems at 1e-6, "
            f"{round(100 * high['zero_variance_fraction'])}% at 1e-5"
        )
        self.assertIn(text, README)
        spread = low["mean_component_std"]
        self.assertIn(
            f"Accuracy {f3(spread['accuracy'])}, format {f3(spread['format'])}, "
            f"efficiency {f3(spread['efficiency'])}, gate {f3(spread['gate'])}",
            README,
        )


class CapabilityTests(unittest.TestCase):
    def test_section_4_at_r0(self) -> None:
        base = rung("baseline-phase_a-3cc174f.json", "Qwen/Qwen3-1.7B", "R0")["metrics"]
        sft = [rung(name, "Qwen/Qwen3-1.7B", "R0")["metrics"] for name in SFT_TESTS]
        self.assertIn(f"| {f3(base['pass@4'])} | {span([m['pass@4'] for m in sft])} |", README)
        self.assertIn(f"| {f3(base['pass^4'])} | {span([m['pass^4'] for m in sft])} |", README)
        at4 = [m["pass@4"] - base["pass@4"] for m in sft]
        power4 = [m["pass^4"] - base["pass^4"] for m in sft]
        self.assertIn(f"+{min(at4):.2f} to +{max(at4):.2f}", README)
        self.assertIn(f"+{min(power4):.2f} to +{max(power4):.2f}", README)


class MmluTests(unittest.TestCase):
    def test_paired_comparison(self) -> None:
        payload = load("utility-comparison-26ce399.json")
        entries = {(c["treatment"], c["baseline"], c["scope"]): c for c in payload["comparisons"]}
        sft = entries[("sft", "base", "all_questions")]
        readable = entries[("sft", "base", "readable_in_both")]
        low, high = sft["difference_ci95"]
        difference = ("+" if sft["difference"] >= 0 else MINUS) + f"{abs(sft['difference']):.4f}"
        self.assertIn(f"Paired difference {difference}", README)
        self.assertIn(f"95% interval {MINUS}{abs(low):.3f} to +{high:.3f}", README)
        self.assertIn(
            f"{sft['improved']} questions improved, {sft['regressed']} got worse", README
        )
        self.assertIn(f"({sft['baseline_correct']}/400)", README)
        self.assertIn(f"({sft['treatment_correct']}/400)", README)
        self.assertIn(f"With {sft['improved'] + sft['regressed']} of 400 questions", README)
        self.assertIn(f"Restricted to the {readable['questions']} questions", README)
        r_low, r_high = readable["difference_ci95"]
        self.assertIn(
            f"{MINUS}{abs(readable['difference']):.3f} (95% interval "
            f"{MINUS}{abs(r_low):.3f} to +{r_high:.3f})",
            README,
        )

    def test_truncation_rates(self) -> None:
        names = {"base": "001023e", "sft": "bb6764e", "grpo": "76a1e3f"}
        rates = {
            arm: load(f"utility-{arm}-{commit}.json")["summary"]["truncated_rate"]
            for arm, commit in names.items()
        }
        self.assertIn(
            f"{100 * rates['sft']:.1f}% of answers hit the 320-token limit, against "
            f"{100 * rates['base']:.2f}% untrained and {100 * rates['grpo']:.2f}% after GRPO",
            README,
        )


class PhaseBTests(unittest.TestCase):
    def setUp(self) -> None:
        names = {"base": "10f9d97", "sft": "aa59330", "grpo": "42e347c"}
        self.summaries = {arm: load(f"phase_b-{arm}-{commit}.json") for arm, commit in names.items()}
        self.gates = load("phase_b-gates-8d5825b.json")["arms"]

    def row(self, label: str, values: list[str], bold: bool = False) -> str:
        cells = [f"**{v}**" if bold else v for v in values]
        return f"| {label} | " + " | ".join(cells) + " |"

    def test_summary_rows(self) -> None:
        s = self.summaries
        self.assertIn(
            self.row("`pass^1`", [f3(s[a]["metrics"]["pass^1"]) for a in s]), README
        )
        self.assertIn(
            self.row(
                "Completes a legitimate request (outcome)",
                [f3(s[a]["by_intent"]["fulfil"]["correct_rate"]) for a in s],
                bold=True,
            ),
            README,
        )
        self.assertIn(
            self.row(
                "Correctly refuses an unverified one",
                [f3(s[a]["by_intent"]["refuse"]["correct_rate"]) for a in s],
                bold=True,
            ),
            README,
        )

    def test_gate_split_rows(self) -> None:
        g = self.gates
        unauth = [str(g[a]["write_classes"]["refuse"]["unauthenticated_write"]) for a in g]
        lookup = [str(g[a]["write_classes"]["fulfil"]["verified_write_without_lookup"]) for a in g]
        calls = [str(g[a]["episodes_that_called_the_lookup_tool"]) for a in g]
        self.assertIn(self.row("Writes for an unverified customer (of 296 refusal episodes)", unauth), README)
        self.assertIn(self.row("Verified writes that skipped the lookup (of 304 legitimate episodes)", lookup), README)
        self.assertIn(self.row("Episodes that called the lookup tool (of 600)", calls), README)
        self.assertEqual(g["sft"]["episodes_by_intent"], {"fulfil": 304, "refuse": 296})

    def test_reward_rows(self) -> None:
        g = self.gates
        scored = [signed(g[a]["mean_reward"]) for a in g]
        rescored = [signed(g[a]["mean_reward_if_only_authentication_gated"]) for a in g]
        self.assertIn(f"| Mean reward, as scored | {scored[0]} | **{scored[1]}** | **{scored[2]}** |", README)
        self.assertIn(self.row("Mean reward, if skipping the lookup were not penalised", rescored), README)
        for arm, summary in self.summaries.items():
            self.assertAlmostEqual(g[arm]["mean_reward"], summary["diagnostics"]["mean_reward"])

    def test_the_prose_counts(self) -> None:
        sft = self.gates["sft"]
        self.assertIn(f"{sft['write_classes']['refuse']['unauthenticated_write']} of 296", README)
        self.assertIn(
            f"{sft['write_classes']['fulfil']['verified_write_without_lookup']} of its "
            f"{sft['gate_violations']} violations",
            README,
        )


class WithdrawnClaimsTests(unittest.TestCase):
    """ERRATA withdrew these. They must not reappear in either document."""

    WITHDRAWN = (
        "a third as much",
        "barely moves",
        "most of the untrained gap",
        "Reinforcement learning added nothing",
        "never once calls",
        "never calls the lookup tool",
        "`authenticated: false`",
        "worse than the one that does nothing",
        "worse in this environment than the one that does nothing",
        "three tools it has never seen",
        "three tools it had never seen",
        "Three of the four reward terms never varied",
        "16% against 40%",
        "less than half as often",
        "did not forget anything",
        "a thousand calculator trajectories",
        "blind to losses",
        "Nothing left to reach",
        "identical dev peak",
        "Two nulls",
        "Null, twice",
        "The rate went down instead",
        "within two points of SFT",
    )

    def test_no_withdrawn_phrase_survives(self) -> None:
        for phrase in self.WITHDRAWN:
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, README)
                self.assertNotIn(phrase, FINDINGS)


if __name__ == "__main__":
    unittest.main()
