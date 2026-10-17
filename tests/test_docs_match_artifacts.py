"""The README's result numbers must be the numbers in the frozen artifacts.

Each figure below is recomputed from a committed file in results/ and looked up
in the README as it is printed there, together with the sentence that states
it. A README edited to a nicer number, or an artifact the README never caught
up with, fails here. The two parameter counts used for the cost column come
from the model configs, which are not in the repository, so they are pinned
below.

The last tests check that every relative link resolves, and that phrases an
earlier version had to withdraw do not come back.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from env.phase_a import calculator_tool_schema  # noqa: E402
from env.phase_b import phase_b_tool_schemas  # noqa: E402

RESULTS = PROJECT_ROOT / "results"
README_PATH = PROJECT_ROOT / "README.md"
RESULTS_README_PATH = RESULTS / "README.md"


def _collapse(path: Path) -> str:
    # Whitespace is collapsed so a figure wrapped across two lines still matches.
    return " ".join(path.read_text(encoding="utf-8").split())


README = _collapse(README_PATH)

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
FAILURE_MODES = "failure-modes-2f70608.json"
TEMPLATES = "phase_b-templates-2f70608.json"
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


def posix(path: str) -> str:
    return path.replace("\\", "/")


def cost(rungs: list[dict], params: float) -> list[int]:
    return [
        round(r["generated_tokens_per_episode"] * params)
        for r in rungs
        if r.get("generated_tokens_per_episode") is not None
    ]


class ResultTableTests(unittest.TestCase):
    """The result table: every model at R1."""

    def test_the_setup_line(self) -> None:
        for name, candidate in (
            [(BASELINE, QWEN_SMALL), (BASELINE, QWEN_TEACHER), (COMPARATOR, LLAMA)]
            + [(name, QWEN_SMALL) for name in SFT_TESTS + GRPO_TESTS]
        ):
            r1 = rung(name, candidate, "R1")
            self.assertEqual((r1["tasks"], r1["runs_per_task"]), (150, 4), name)
        self.assertIn("150 held-out test tasks, 4 attempts per task.", README)

    def test_what_the_ranges_span(self) -> None:
        self.assertEqual(len(SFT_TESTS), 3)
        self.assertIn("The SFT range spans three training runs", README)
        run3 = posix(load(SFT_TESTS[2])["adapter"]["path"])
        for _, _, run in GRPO_VS_SFT:
            self.assertEqual(posix(load(run)["init_from_adapter"]), run3)
        self.assertIn("spans two learning rates, one run each, both from SFT run 3", README)

    def test_the_teacher_has_the_best_reliability(self) -> None:
        teacher = rung(BASELINE, QWEN_TEACHER, "R1")["metrics"]
        others = [rung(COMPARATOR, LLAMA, "R1")["metrics"]] + [
            rung(name, QWEN_SMALL, "R1")["metrics"] for name in SFT_TESTS + GRPO_TESTS
        ]
        for metric in ("pass^1", "pass^4"):
            self.assertGreater(teacher[metric], max(m[metric] for m in others), metric)
        self.assertIn("teacher still has the best `pass^1` and `pass^4`", README)

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
        self.assertNotIn("generated_tokens_per_episode", rung(SFT_TESTS[0], QWEN_SMALL, "R1"))
        self.assertIn("SFT run 1 did not record token counts", README)
        for name in SFT_TESTS:
            tokens = rung(name, QWEN_SMALL, "R1").get("generated_tokens_per_episode")
            if tokens is None:
                continue
            measured += 1
            ratio = tokens * QWEN3_1_7B_PARAMS_B / (big * LLAMA_3_1_8B_PARAMS_B)
            self.assertTrue(0.30 <= ratio <= 0.32, (name, ratio))
            self.assertLess(big, tokens, "the README says the 8B writes fewer tokens")
        self.assertEqual(measured, 2)
        self.assertIn("8B is cheaper, because it writes fewer tokens per attempt", README)
        self.assertIn("about 31% of the 8B's cost", README)
        self.assertIn("2.03 for Qwen3-1.7B and 8.03 for Llama-3.1-8B", README)


class RetryTests(unittest.TestCase):
    """The note under the result table: why R0 and R1 barely differ after training."""

    def test_how_often_the_retry_fired(self) -> None:
        def fired(name: str) -> int:
            r1 = rung(name, QWEN_SMALL, "R1")
            return r1["model_decisions"] - r1["episodes"]

        trained = [fired(name) for name in SFT_TESTS + GRPO_TESTS]
        self.assertIn(
            f"After training it fired in {min(trained)} to {max(trained)} of 600 attempts, "
            f"against {fired(BASELINE)} for the untrained model.",
            README,
        )

    def test_r0_and_r1_after_training(self) -> None:
        gaps = []
        for name in SFT_TESTS + GRPO_TESTS:
            r0, r1 = (rung(name, QWEN_SMALL, which)["metrics"] for which in ("R0", "R1"))
            self.assertEqual(r0["pass^4"], r1["pass^4"], name)
            gaps.append(r1["pass^1"] - r0["pass^1"])
        self.assertGreaterEqual(min(gaps), 0.0)
        self.assertEqual(f"{max(gaps):.3f}", "0.005")
        self.assertIn("differ by at most 0.005 in `pass^1`, and not at all in `pass^4`", README)


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
        self.assertIn("SFT beats the scaffolded 8B.", README)
        self.assertIn("Every interval excludes zero.", README)


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
        self.assertIn("and every interval excludes zero", README)
        low, high = one[2]["difference_ci95"]
        self.assertTrue(low < 0.0 < high)
        self.assertIn("run 3's interval includes zero", README)


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


class FailureModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.arms = load(FAILURE_MODES)["arms"]
        self.r0 = {label: arm["rungs"]["R0"] for label, arm in self.arms.items()}
        self.sft = [self.r0[f"sft-run{run}"] for run in (1, 2, 3)]

    def test_the_artifact_read_the_files_this_page_cites(self) -> None:
        cited = {
            "base-1.7b": BASELINE,
            "sft-run1": SFT_TESTS[0],
            "sft-run2": SFT_TESTS[1],
            "sft-run3": SFT_TESTS[2],
            "llama-8b": COMPARATOR,
        }
        for label, name in cited.items():
            self.assertEqual(self.arms[label]["summary"], f"results/{name}")
            digest = hashlib.sha256((RESULTS / name).read_bytes()).hexdigest()
            self.assertEqual(self.arms[label]["summary_sha256"], digest)

    def test_rows(self) -> None:
        def cells(field: str) -> list[str]:
            if field == "total":
                values = [arm["failures"]["total"] for arm in self.sft]
                ends = (self.r0["base-1.7b"]["failures"]["total"], self.r0["llama-8b"]["failures"]["total"])
            else:
                values = [arm["failures"][field] for arm in self.sft]
                ends = (self.r0["base-1.7b"]["failures"][field], self.r0["llama-8b"]["failures"][field])
            return [str(ends[0]), f"{min(values)} – {max(values)}", str(ends[1])]

        for label, field in (
            ("No call, or a call that did not parse", "no_call_or_unparsed"),
            ("Call parsed but failed to run", "call_failed_to_run"),
            ("Call ran, wrong number", "wrong_value"),
            ("Total", "total"),
        ):
            with self.subTest(row=label):
                self.assertIn(row(label, *cells(field)), README)
        for arm in self.r0.values():
            self.assertEqual(arm["episodes"], 600)

    def test_the_shares_and_the_retry(self) -> None:
        shares = [round(100 * arm["failures"]["wrong_value"] / arm["failures"]["total"]) for arm in self.sft]
        self.assertIn(f"After it, {min(shares)}% to {max(shares)}% of failures are a call", README)
        base = self.r0["base-1.7b"]["failures"]
        self.assertIn(f"before training ({round(100 * base['wrong_value'] / base['total'])}%)", README)
        rescued = [self.arms[f"sft-run{run}"]["rungs"]["R1"]["retry"]["rescued"] for run in (1, 2, 3)]
        retry = self.arms["base-1.7b"]["rungs"]["R1"]["retry"]
        self.assertIn(
            f"it rescued {min(rescued)} to {max(rescued)} attempts per run, against "
            f"{retry['rescued']} of {retry['fired']} for the untrained model",
            README,
        )


class GrpoTests(unittest.TestCase):
    def test_the_setup(self) -> None:
        for _, _, run in GRPO_VS_SFT:
            self.assertEqual(load(run)["train"]["global_step"], 400)
            self.assertEqual(load(run)["resolved"]["num_generations"], 8)
        self.assertIn("400 steps from SFT run 3", README)
        self.assertIn("all 8 of its attempts score the same", README)
        picked = [posix(load(name)["adapter"]["path"]) for name in GRPO_TESTS]
        self.assertTrue(picked[0].endswith("grpo-pinned/checkpoint-200"), picked[0])
        self.assertTrue(picked[1].endswith("grpo-lr1e5-pinned/checkpoint-300"), picked[1])
        self.assertIn("picked step 200 at 1e-6 and step 300 at 1e-5", README)

    def test_rows(self) -> None:
        for rate, name, run in GRPO_VS_SFT:
            one = comparison(name, "R0", 1)
            health = load(run)["group_health"]
            self.assertEqual(health["groups"], 800, "one logged group per problem, two per step")
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
        self.assertIn(f"the sign test does not agree (p = {one['p_sign_test_exact']:.2f})", README)
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
        self.assertIn("On 400 MMLU questions with no tool offered", README)
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
        self.assertTrue(0.04 < max(-low, high) <= 0.05)
        self.assertIn("cannot rule out a change of up to about 5 points", README)

    def test_the_readable_questions_caveat(self) -> None:
        payload = load("utility-comparison-26ce399.json")
        (readable,) = [
            c
            for c in payload["comparisons"]
            if (c["treatment"], c["baseline"], c["scope"]) == ("sft", "base", "readable_in_both")
        ]
        low, high = readable["difference_ci95"]
        self.assertTrue(low < 0.0 < high)
        self.assertTrue(-0.08 < low < -0.07, "printed as almost 8 points down")
        self.assertIn(
            f"On the {readable['questions']} questions where both models named an answer, "
            f"the difference is {signed(readable['difference'])} "
            f"(95% interval {signed(low)} to {signed(high)})",
            README,
        )

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

    def test_the_setup(self) -> None:
        for summary in self.summaries.values():
            self.assertEqual((summary["tasks"], summary["runs_per_task"]), (150, 4))
            self.assertEqual(summary["gate_mode"], "audit")
        self.assertIn("There are 150 requests, run 4 times each.", README)
        self.assertIn("The grader runs in audit mode", README)
        self.assertIn("| | Untrained 1.7B | SFT run 3 | GRPO 1e-6 |", README)

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

    def test_three_tools_none_of_them_trained_on(self) -> None:
        tools = phase_b_tool_schemas()
        self.assertEqual(len(tools), 3)
        self.assertNotIn(calculator_tool_schema(), tools)
        self.assertIn("three tools that were not in training", README)

    def test_the_checkpoints_are_the_ones_named(self) -> None:
        self.assertIn("qwen3-1.7b-sft-seed20260824/checkpoint-86", self.summaries["sft"]["adapter"])
        self.assertIn("qwen3-1.7b-grpo-pinned/checkpoint-200", self.summaries["grpo"]["adapter"])

    def test_the_prose_counts(self) -> None:
        sft = self.gates["sft"]
        self.assertEqual(sft["episodes_by_intent"], {"fulfil": 304, "refuse": 296})
        fulfil = sft["write_classes"]["fulfil"]
        completed = fulfil["clean_write"] + fulfil["verified_write_without_lookup"]
        rate = self.summaries["sft"]["by_intent"]["fulfil"]["correct_rate"]
        self.assertEqual(completed, round(rate * 304), "every completion is a write")
        self.assertIn(
            f"{fulfil['verified_write_without_lookup']} of its {completed} completions "
            "skipped the order lookup",
            README,
        )

    def test_the_template_split(self) -> None:
        split = load(TEMPLATES)
        self.assertEqual(split["gates_breakdown"]["artifact"], "results/phase_b-gates-8d5825b.json")
        arms = {label: arm["by_template"] for label, arm in split["arms"].items()}
        for label, summary in self.summaries.items():
            self.assertEqual(split["arms"][label]["totals"]["correct"], round(summary["metrics"]["pass^1"] * 600))

        clean = "authenticate>get_order>update_order"
        verified = {label: by["address_change_verified"] for label, by in arms.items()}
        self.assertEqual(
            [verified[a]["call_sequences"].get(clean, 0) for a in ("sft", "grpo")],
            [self.gates[a]["write_classes"]["fulfil"]["clean_write"] for a in ("sft", "grpo")],
        )
        self.assertIn(
            f"{verified['sft']['call_sequences'][clean]} SFT and "
            f"{verified['grpo']['call_sequences'][clean]} GRPO completions did",
            README,
        )

        base = arms["base"]
        with_postcode = [base["address_change_verified"], base["address_change_wrong_postcode"]]
        episodes = sum(t["episodes"] for t in with_postcode)
        lone = sum(t["call_sequences"].get("authenticate", 0) for t in with_postcode)
        self.assertEqual(lone, episodes)
        self.assertTrue(all(t["single_call"] == t["episodes"] for t in with_postcode))
        self.assertIn(f"it made one authenticate call and stopped: {lone} of {episodes} episodes", README)

        none = {label: by["address_change_no_postcode"] for label, by in arms.items()}
        self.assertEqual(none["sft"]["episodes"], 148)
        self.assertIn(
            f"There SFT still wrote in {none['sft']['wrote']} of {none['sft']['episodes']} "
            f"episodes, and GRPO in {none['grpo']['wrote']}.",
            README,
        )


class HowItWorksTests(unittest.TestCase):
    def test_split_sizes(self) -> None:
        splits = json.loads(
            (PROJECT_ROOT / "configs" / "splits" / "phase_a_gsm8k.json").read_text(encoding="utf-8")
        )["splits"]
        sizes = {name: len(ids) for name, ids in splits.items()}
        self.assertEqual(sizes, {"train": 1000, "dev": 100, "test": 150})
        self.assertIn("1,000 train, 100 dev and 150 test tasks", README)

    def test_sft_data(self) -> None:
        counts = load("sft-candidates-a2dbe77.json")["result"]["counts"]
        stats = load("sft-dataset-54218c4.json")["selection_stats"]
        self.assertEqual(counts["episodes"], stats["candidates"])
        self.assertEqual(counts["graded_correct"] - counts["laundered"], stats["usable"])
        self.assertIn(f"wrote {stats['candidates']:,} attempts", README)
        self.assertIn(f"{counts['graded_correct']:,} executed to the right answer", README)
        self.assertIn(f"{counts['laundered']} of those only restated the answer", README)
        self.assertEqual(stats["selected"], stats["distinct_tasks"])
        self.assertIn(f"One per task was kept: {stats['selected']} examples", README)

    def test_identical_answers_limit(self) -> None:
        untrained = load("sft-comparison-qwen3-1.7b-89a7fbb.json")["sampling_degeneracy"]["R1"]["baseline"]
        teacher_side = load(TEACHER[0])
        self.assertIn("Qwen3-4B", teacher_side["baseline_label"])
        teacher = teacher_side["sampling_degeneracy"]["R1"]["baseline"]
        self.assertIn(
            f"On {untrained['tasks_with_identical_completions']} of {untrained['tasks']} test tasks "
            "the untrained 1.7B wrote exactly the same text in all four attempts",
            README,
        )
        self.assertIn(f"4B teacher did the same on {teacher['tasks_with_identical_completions']}.", README)

    def test_the_decoding_limit(self) -> None:
        for name in (BASELINE, COMPARATOR) + SFT_TESTS + GRPO_TESTS:
            decoding = load(name)["decoding"]
            self.assertEqual((decoding["temperature"], decoding["top_p"]), (0.7, 0.95), name)
            self.assertNotIn("top_k", decoding, "the README says top-k was not recorded")
        self.assertIn("sampled at temperature 0.7 and top-p 0.95", README)
        self.assertIn("The results do not record this difference", README)

    def test_the_mmlu_truncation_limit(self) -> None:
        payload = load("utility-comparison-26ce399.json")
        unreadable = [
            question
            for question in payload["rows"]
            if any(question["arms"][arm]["extracted"] is None for arm in ("base", "sft"))
        ]
        cut_off = [
            question
            for question in unreadable
            if any(
                question["arms"][arm]["extracted"] is None and question["arms"][arm]["truncated"]
                for arm in ("base", "sft")
            )
        ]
        self.assertIn(
            f"{len(unreadable)} questions have no readable answer from the untrained or the "
            f"fine-tuned model, and {len(cut_off)} of those were cut off at the limit",
            README,
        )
        rates = {
            arm: load(f"utility-{arm}-{commit}.json")["summary"]["truncated_rate"]
            for arm, commit in (("base", "001023e"), ("sft", "bb6764e"))
        }
        self.assertIn(
            f"({100 * rates['base']:.2f}% of untrained answers, {100 * rates['sft']:.1f}% after SFT)",
            README,
        )
        self.assertEqual(load("utility-base-001023e.json")["decoding"]["max_new_tokens"], 320)
        self.assertIn("320-token answer limit", README)


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
        documents = (README, _collapse(RESULTS_README_PATH))
        for phrase in self.WITHDRAWN + ("unseen tools",):
            for document in documents:
                with self.subTest(phrase=phrase):
                    self.assertNotIn(phrase, document)

    def test_every_number_has_an_artifact(self) -> None:
        """The dagger used to mark numbers with no committed artifact. None remain."""

        self.assertNotIn("†", README)


if __name__ == "__main__":
    unittest.main()
