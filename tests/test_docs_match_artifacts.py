"""Every number in the README must be the number in a frozen artifact.

Each figure below is recomputed from a committed file in results/ and looked up
in the README as it is printed there. A README edited to a nicer number, or an
artifact the README never caught up with, fails here. The two parameter counts
used for the cost column come from the model configs and are pinned below.

The last tests check that every relative link resolves, and that phrases an
earlier version had to withdraw do not come back.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
README_PATH = PROJECT_ROOT / "README.md"
# Whitespace is collapsed so a figure wrapped across two lines still matches.
README = " ".join(README_PATH.read_text(encoding="utf-8").split())

MINUS = "−"  # the typographic minus the README uses

# From the model configs, printed in the README beside the cost column.
QWEN3_1_7B_PARAMS_B = 2.03
LLAMA_3_1_8B_PARAMS_B = 8.03

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
GRPO_TESTS = ("grpo-test-qwen3-1.7b-f6138ee.json", "grpo-test-lr1e5-ce5f2ca.json")
H1 = (
    "h1-comparison-2889b6d.json",
    "h1-comparison-seed20260823-2889b6d.json",
    "h1-comparison-seed20260824-2889b6d.json",
)
TEACHER = tuple(f"sft-vs-teacher-run{run}-d692d43.json" for run in (1, 2, 3))
GRPO_VS_SFT = (
    ("1e-6", "grpo-vs-sft-c364562.json", "grpo-run-qwen3-1.7b-a498a7b.json"),
    ("1e-5", "grpo-lr1e5-vs-sft-91a2de9.json", "grpo-run-qwen3-1.7b-lr1e5-3d7e90f.json"),
)


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def rung(name: str, candidate: str, which: str) -> dict:
    for result in load(name)["results"]:
        if result["candidate"]["id"] == candidate:
            return result["rungs"][which]
    raise KeyError(candidate)


def comparison(name: str, which: str, k: int) -> dict:
    for entry in load(name)["comparisons"]:
        if entry["rung"] == which and entry["k"] == k:
            return entry
    raise KeyError((name, which, k))


def f3(value: float) -> str:
    return f"{value:.3f}"


def signed(value: float) -> str:
    return ("+" if value >= 0 else MINUS) + f"{abs(value):.3f}"


def span(values: list[float]) -> str:
    return f"{f3(min(values))} – {f3(max(values))}"


def row(*cells: str) -> str:
    return "| " + " | ".join(cells) + " |"


def bold(text: str) -> str:
    return f"**{text}**"


def cost(rungs: list[dict], params: float) -> list[int]:
    return [
        round(r["generated_tokens_per_episode"] * params)
        for r in rungs
        if r.get("generated_tokens_per_episode") is not None
    ]


class ResultTableTests(unittest.TestCase):
    """The result table: every model at R1."""

    def test_untrained_and_teacher_rows(self) -> None:
        for label, candidate in (
            ("Qwen3-1.7B, untrained", QWEN_SMALL),
            ("Qwen3-4B, untrained (the teacher)", QWEN_TEACHER),
        ):
            r1 = rung(BASELINE, candidate, "R1")
            self.assertNotIn("generated_tokens_per_episode", r1, "cost is printed as not recorded")
            metrics = r1["metrics"]
            self.assertIn(
                row(label, f3(metrics["pass^1"]), f3(metrics["pass^4"]), "not recorded"), README
            )

    def test_8b_row(self) -> None:
        r1 = rung(COMPARATOR, LLAMA, "R1")
        (big,) = cost([r1], LLAMA_3_1_8B_PARAMS_B)
        metrics = r1["metrics"]
        self.assertIn(
            row("Llama-3.1-8B + retry", f3(metrics["pass^1"]), f3(metrics["pass^4"]), str(big)),
            README,
        )

    def test_sft_and_grpo_rows(self) -> None:
        for label, names, emphasise in (
            ("Qwen3-1.7B, SFT", SFT_TESTS, True),
            ("Qwen3-1.7B, SFT + GRPO", GRPO_TESTS, False),
        ):
            r1 = [rung(name, QWEN_SMALL, "R1") for name in names]
            costs = cost(r1, QWEN3_1_7B_PARAMS_B)
            cells = [
                label,
                span([r["metrics"]["pass^1"] for r in r1]),
                span([r["metrics"]["pass^4"] for r in r1]),
                f"{min(costs)} – {max(costs)}",
            ]
            if emphasise:
                cells = [bold(cell) for cell in cells]
            self.assertIn(row(*cells), README)

    def test_cost_ratio_and_raw_token_caveat(self) -> None:
        big = rung(COMPARATOR, LLAMA, "R1")["generated_tokens_per_episode"]
        measured = 0
        for name in SFT_TESTS:
            tokens = rung(name, QWEN_SMALL, "R1").get("generated_tokens_per_episode")
            if tokens is None:
                continue
            measured += 1
            ratio = tokens * QWEN3_1_7B_PARAMS_B / (big * LLAMA_3_1_8B_PARAMS_B)
            self.assertTrue(0.30 <= ratio <= 0.32, (name, ratio))
            self.assertLess(big, tokens, "the README says the 8B writes fewer tokens")
        self.assertEqual(measured, 2, "the README says only run 1 lacks token counts")
        self.assertIn("about 31% of the 8B's cost", README)
        self.assertIn("2.03 for Qwen3-1.7B and 8.03 for Llama-3.1-8B", README)


class EightBTests(unittest.TestCase):
    def test_paired_rows(self) -> None:
        for run, name in enumerate(H1, start=1):
            one, four = comparison(name, "R1", 1), comparison(name, "R1", 4)
            self.assertIn(
                row(
                    f"Run {run}",
                    signed(one["difference"]),
                    span(one["difference_ci95"]),
                    signed(four["difference"]),
                    span(four["difference_ci95"]),
                ),
                README,
            )
            self.assertGreater(one["difference_ci95"][0], 0.0)
            self.assertGreater(four["difference_ci95"][0], 0.0)


class TeacherTests(unittest.TestCase):
    def test_gaps(self) -> None:
        one = [comparison(name, "R1", 1) for name in TEACHER]
        four = [comparison(name, "R1", 4) for name in TEACHER]
        gaps4 = [entry["difference"] for entry in four]
        gaps1 = [entry["difference"] for entry in one]
        self.assertIn(f"`pass^4` gap is {signed(min(gaps4))} to {signed(max(gaps4))}", README)
        self.assertIn(f"`pass^1` gap is smaller, {signed(min(gaps1))} to {signed(max(gaps1))}", README)
        for entry in four:
            self.assertLess(entry["difference_ci95"][1], 0.0, "every pass^4 interval excludes zero")
        low, high = one[2]["difference_ci95"]
        self.assertTrue(low < 0.0 < high, "run 3's pass^1 interval includes zero")


class CapabilityTests(unittest.TestCase):
    """Single attempts, R0."""

    def test_rows(self) -> None:
        base = rung(BASELINE, QWEN_SMALL, "R0")["metrics"]
        sft = [rung(name, QWEN_SMALL, "R0")["metrics"] for name in SFT_TESTS]

        def gain(metric: str) -> str:
            gains = [m[metric] - base[metric] for m in sft]
            return f"+{min(gains):.2f} to +{max(gains):.2f}"

        self.assertIn(
            row(
                "Solved at least once in 4 (`pass@4`)",
                f3(base["pass@4"]),
                span([m["pass@4"] for m in sft]),
                gain("pass@4"),
            ),
            README,
        )
        self.assertIn(
            row(
                "Solved all 4 times (`pass^4`)",
                f3(base["pass^4"]),
                span([m["pass^4"] for m in sft]),
                gain("pass^4"),
            ),
            README,
        )
        bands = [m["pass@4"] - m["pass^4"] for m in sft]
        base_band = base["pass@4"] - base["pass^4"]
        self.assertGreater(min(bands), base_band)
        self.assertIn(
            row("Solved only sometimes", f3(base_band), span(bands), "wider"), README
        )
        at4 = [m["pass@4"] - base["pass@4"] for m in sft]
        power4 = [m["pass^4"] - base["pass^4"] for m in sft]
        self.assertIn(f"training gained up to {round(100 * max(at4))} points", README)
        self.assertIn(
            f"it gained {round(100 * min(power4))} to {round(100 * max(power4))}", README
        )


class GrpoTests(unittest.TestCase):
    def test_rows(self) -> None:
        for rate, name, run in GRPO_VS_SFT:
            one = comparison(name, "R0", 1)
            health = load(run)["group_health"]
            self.assertEqual(health["groups"], 800, "one logged group per problem, two per step")
            self.assertEqual(load(run)["train"]["global_step"], 400)
            self.assertEqual(load(run)["resolved"]["num_generations"], 8)
            low, high = one["difference_ci95"]
            self.assertIn(
                row(
                    rate,
                    signed(one["difference"]),
                    f"{signed(low)} to {signed(high)}",
                    f"{round(100 * health['zero_variance_fraction'])}%",
                ),
                README,
            )

    def test_the_caveats_on_the_higher_rate(self) -> None:
        name = GRPO_VS_SFT[1][1]
        one, four = comparison(name, "R0", 1), comparison(name, "R0", 4)
        self.assertGreater(one["difference_ci95"][0], 0.0)
        self.assertIn(f"(p = {one['p_sign_test_exact']:.2f})", README)
        self.assertGreater(one["p_sign_test_exact"], 0.05)
        low, high = four["difference_ci95"]
        self.assertTrue(low < 0.0 < high)
        self.assertIn(f"interval ({signed(low)} to {signed(high)}) includes zero", README)


class MmluTests(unittest.TestCase):
    def test_paired_comparison(self) -> None:
        payload = load("utility-comparison-26ce399.json")
        entries = {(c["treatment"], c["baseline"], c["scope"]): c for c in payload["comparisons"]}
        sft = entries[("sft", "base", "all_questions")]
        self.assertEqual(sft["questions"], 400)
        self.assertIn(
            f"{100 * sft['baseline_correct'] / 400:.2f}% untrained ({sft['baseline_correct']}/400)",
            README,
        )
        self.assertIn(
            f"{100 * sft['treatment_correct'] / 400:.1f}% after SFT ({sft['treatment_correct']}/400)",
            README,
        )
        low, high = sft["difference_ci95"]
        difference = ("+" if sft["difference"] >= 0 else MINUS) + f"{abs(sft['difference']):.4f}"
        self.assertIn(
            f"Paired difference {difference}, 95% interval {signed(low)} to {signed(high)}", README
        )
        self.assertTrue(0.04 < max(-low, high) <= 0.05, "printed as about 5 points")

    def test_no_checkpoint_called_a_tool(self) -> None:
        for name in ("utility-base-001023e.json", "utility-sft-bb6764e.json", "utility-grpo-76a1e3f.json"):
            summary = load(name)["summary"]
            self.assertEqual(summary["questions"], 400)
            self.assertEqual(summary["tool_call_rate"], 0.0, name)


class TransferTests(unittest.TestCase):
    def setUp(self) -> None:
        names = {"base": "10f9d97", "sft": "aa59330", "grpo": "42e347c"}
        self.summaries = {arm: load(f"phase_b-{arm}-{commit}.json") for arm, commit in names.items()}
        self.gates = load("phase_b-gates-8d5825b.json")["arms"]

    def test_rows(self) -> None:
        s = self.summaries
        self.assertIn(row("`pass^1`", *(f3(s[a]["metrics"]["pass^1"]) for a in s)), README)
        self.assertIn(
            row(
                "Completes a legitimate request",
                *(f3(s[a]["by_intent"]["fulfil"]["correct_rate"]) for a in s),
            ),
            README,
        )
        self.assertIn(
            row(
                "Correctly refuses an unverified one",
                *(f3(s[a]["by_intent"]["refuse"]["correct_rate"]) for a in s),
            ),
            README,
        )

    def test_the_checkpoints_are_the_ones_named(self) -> None:
        self.assertIn("qwen3-1.7b-sft-seed20260824/checkpoint-86", self.summaries["sft"]["adapter"])
        self.assertIn("qwen3-1.7b-grpo-pinned/checkpoint-200", self.summaries["grpo"]["adapter"])

    def test_the_prose_counts(self) -> None:
        sft = self.gates["sft"]
        self.assertEqual(sft["episodes_by_intent"], {"fulfil": 304, "refuse": 296})
        self.assertIn(
            f"{sft['write_classes']['refuse']['unauthenticated_write']} of 296 refusal episodes",
            README,
        )
        calls = [self.gates[a]["episodes_that_called_the_lookup_tool"] for a in self.gates]
        self.assertLess(max(calls), 0.02 * 600, "printed as almost never")


class HowItWorksTests(unittest.TestCase):
    def test_split_sizes(self) -> None:
        splits = json.loads(
            (PROJECT_ROOT / "configs" / "splits" / "phase_a_gsm8k.json").read_text(encoding="utf-8")
        )["splits"]
        sizes = {name: len(ids) for name, ids in splits.items()}
        self.assertEqual(sizes, {"train": 1000, "dev": 100, "test": 150})
        self.assertIn("1,000 train, 100 dev and 150 test tasks", README)

    def test_sft_data(self) -> None:
        stats = load("sft-dataset-54218c4.json")["selection_stats"]
        self.assertIn(f"wrote {stats['candidates']:,} attempts", README)
        self.assertIn(f"{stats['usable']:,} executed to the right answer", README)
        self.assertEqual(stats["selected"], stats["distinct_tasks"])
        self.assertIn(f"One per task was kept: {stats['selected']} examples", README)

    def test_identical_answers_limit(self) -> None:
        degeneracy = load("sft-comparison-qwen3-1.7b-89a7fbb.json")["sampling_degeneracy"]["R1"]
        baseline = degeneracy["baseline"]
        self.assertIn(
            f"four identical answers on {baseline['tasks_with_identical_completions']} "
            f"of {baseline['tasks']} test tasks",
            README,
        )


class LinkTests(unittest.TestCase):
    """Every relative link in the published Markdown must resolve."""

    LINK = re.compile(r"\]\(([^)\s]+)\)")

    def test_relative_links_resolve(self) -> None:
        for document in (README_PATH, RESULTS / "README.md"):
            for target in self.LINK.findall(document.read_text(encoding="utf-8")):
                if re.match(r"[a-z]+://", target):
                    continue
                path = (document.parent / target.split("#")[0]).resolve()
                with self.subTest(document=document.name, target=target):
                    self.assertTrue(path.exists(), f"{target} does not exist")


class WithdrawnClaimsTests(unittest.TestCase):
    """Earlier versions had to withdraw these. They must not reappear."""

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

    def test_every_number_has_an_artifact(self) -> None:
        """The dagger used to mark numbers with no committed artifact. None remain."""

        self.assertNotIn("†", README)


if __name__ == "__main__":
    unittest.main()
