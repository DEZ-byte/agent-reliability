"""The README figures read their numbers from the frozen results.

The data functions are checked against plain json loads of the same files, so
a figure cannot show a number the results do not hold. The token-counts input
is a small fixture written here, never a file in results/. The data side needs
no matplotlib; only the rendering tests do, and they skip without it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from commands import plots  # noqa: E402
from evaluation import plotting  # noqa: E402
from evaluation.plotting import PlotError  # noqa: E402

RESULTS = PROJECT_ROOT / "results"
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def stored_r1(spec: dict) -> dict:
    """A run's R1 block, read straight from its summary."""

    summary = load(spec["summary"])
    [result] = [r for r in summary["results"] if r["candidate"]["id"] == spec["model"]]
    return result["rungs"]["R1"]


def token_counts_fixture(costs: dict[str, float] | None = None) -> dict:
    """A token-counts artifact with one row per plotted run and made-up costs."""

    rows = []
    for i, spec in enumerate(plotting.RUNS):
        cost = (costs or {}).get(spec["key"], 10.0 * (i + 1))
        rows.append({
            "row": spec["label"],
            "model": spec["model"],
            "summary": spec["summary"],
            "unique_params_b": 1.0,
            "rungs": {
                "R1": {
                    "tokens_per_attempt": cost,
                    "tokens_total": int(cost * 600),
                    "episodes": stored_r1(spec)["episodes"],
                    "decisions": stored_r1(spec)["episodes"],
                    "source": "recorded" if i % 2 else "rebuilt_from_logs",
                    "cost": cost,
                }
            },
        })
    # The real artifact records the hash of every summary it read.
    inputs = [
        {"file": name, "sha256": hashlib.sha256((RESULTS / name).read_bytes()).hexdigest()}
        for name in sorted({spec["summary"] for spec in plotting.RUNS})
    ]
    return {"schema_version": 1, "kind": "token_counts", "inputs": inputs, "rows": rows}


def copy_figure_one_inputs(root: Path) -> None:
    """Copy every results file figure 1 reads into `root`."""

    for spec in plotting.RUNS:
        shutil.copy(RESULTS / spec["summary"], root / spec["summary"])
        if "vs_llama" in spec:
            shutil.copy(RESULTS / spec["vs_llama"], root / spec["vs_llama"])


def multi_turn_fixture(spec: dict, index: int) -> dict:
    """A closed-loop summary with made-up rates, paired with a real single-turn run.

    The rates differ from every single-turn rate, so reading the wrong file shows.
    """

    single = load(spec["file"])
    completion, refusal = 0.25 + 0.1 * index, 0.6 - 0.1 * index
    by_intent = {
        "fulfil": {"episodes": single["by_intent"]["fulfil"]["episodes"],
                   "correct_rate": completion},
        "refuse": {"episodes": single["by_intent"]["refuse"]["episodes"],
                   "correct_rate": refusal},
    }
    return {
        "kind": plotting.MULTI_TURN_KIND,
        "label": spec["multi_turn"],
        "executed": True,
        "model": dict(single["model"]),
        "adapter": single["adapter"],
        "weights_sha256": single["weights_sha256"],
        "max_turns": 4,
        "tasks": 150,
        "runs_per_task": 4,
        "episodes": single["episodes"],
        "by_intent": by_intent,
        "multi_turn": {
            "clean_metrics": {"pass^1": 0.05 + 0.07 * index, "pass^4": 0.01 * index},
            "by_intent": json.loads(json.dumps(by_intent)),
            "mean_turns_used": 1.5 + 0.25 * index,
        },
    }


def multi_turn_name(spec: dict) -> str:
    return plotting.MULTI_TURN_GLOB.format(label=spec["multi_turn"]).replace("*", "abc1234")


def write_order_support_inputs(root: Path) -> dict[str, dict]:
    """Copy the single-turn inputs into `root` and write a multi-turn fixture per model."""

    fixtures = {}
    for index, spec in enumerate(plotting.ORDER_SUPPORT):
        shutil.copy(RESULTS / spec["file"], root / spec["file"])
        if spec["same_adapter_as"]:
            shutil.copy(RESULTS / spec["same_adapter_as"], root / spec["same_adapter_as"])
        fixtures[spec["key"]] = multi_turn_fixture(spec, index)
        (root / multi_turn_name(spec)).write_text(json.dumps(fixtures[spec["key"]]),
                                                  encoding="utf-8")
    return fixtures


class _TempDir(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write_json(self, name: str, payload: dict) -> Path:
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path


class ReliabilityCostDataTests(_TempDir):
    def setUp(self) -> None:
        super().setUp()
        self.token_counts = self.write_json("token-counts-test.json", token_counts_fixture())
        self.data = plotting.reliability_cost_data(RESULTS, self.token_counts)
        self.runs = {run["key"]: run for run in self.data["runs"]}
        self.groups = {group["key"]: group for group in self.data["groups"]}
        self.specs = {spec["key"]: spec for spec in plotting.RUNS}

    def test_every_run_shows_its_summary_metrics(self) -> None:
        self.assertEqual(set(self.runs), set(self.specs))
        for key, run in self.runs.items():
            metrics = stored_r1(self.specs[key])["metrics"]
            with self.subTest(run=key):
                for name in ("pass@4", "pass^1", "pass^4"):
                    self.assertEqual(run[name], metrics[name])
                self.assertEqual(run["ci95"]["pass^4"], metrics["pass^4_ci95"])
                self.assertNotIn("pass@4", run["ci95"])

    def test_the_title_counts_come_from_the_summaries(self) -> None:
        r1 = stored_r1(self.specs["llama"])
        self.assertEqual(self.data["tasks"], r1["tasks"])
        self.assertEqual(self.data["attempts"], r1["runs_per_task"])

    def test_multi_run_groups_show_the_mean_and_the_range(self) -> None:
        values = [stored_r1(self.specs[k])["metrics"]["pass^4"] for k in ("sft1", "sft2", "sft3")]
        entry = self.groups["sft"]["metrics"]["pass^4"]
        self.assertAlmostEqual(entry["value"], sum(values) / 3, places=12)
        self.assertEqual((entry["low"], entry["high"]), (min(values), max(values)))
        self.assertEqual(entry["whisker"], "range")
        self.assertEqual(sorted(entry["points"]), sorted(values))

    def test_single_runs_use_the_stored_interval_or_none(self) -> None:
        metrics = stored_r1(self.specs["llama"])["metrics"]
        llama = self.groups["llama"]["metrics"]
        self.assertEqual([llama["pass^1"]["low"], llama["pass^1"]["high"]], metrics["pass^1_ci95"])
        self.assertEqual(llama["pass^1"]["whisker"], "ci95")
        self.assertIsNone(llama["pass@4"]["whisker"])
        self.assertIsNone(llama["pass@4"]["low"])

    def test_the_called_out_drops_match_the_summaries(self) -> None:
        drops = {drop["run"]: drop for drop in self.data["drops"]}
        self.assertEqual(set(drops), {"llama", "sft3"})
        llama = load("comparator-8b-2889b6d.json")["results"][0]["rungs"]["R1"]["metrics"]
        self.assertEqual(drops["llama"]["pass@4"], llama["pass@4"])
        self.assertEqual(drops["llama"]["pass^4"], llama["pass^4"])
        sft3 = stored_r1(self.specs["sft3"])["metrics"]
        self.assertEqual((drops["sft3"]["pass@4"], drops["sft3"]["pass^4"]),
                         (sft3["pass@4"], sft3["pass^4"]))

    def test_the_gains_come_from_the_paired_comparisons(self) -> None:
        comparison = self.data["sft_vs_llama"]
        expected = []
        for key in ("sft1", "sft2", "sft3"):
            paired = load(self.specs[key]["vs_llama"])
            [row] = [c for c in paired["comparisons"] if c["rung"] == "R1" and c["k"] == 4]
            expected.append(row["difference"])
        self.assertEqual([g["difference"] for g in comparison["gains"]], expected)
        self.assertEqual(comparison["gain_min"], min(expected))
        self.assertEqual(comparison["gain_max"], max(expected))

    def test_cost_and_cost_ratio_come_from_the_token_counts(self) -> None:
        fixture = token_counts_fixture()
        costs = {
            spec["key"]: row["rungs"]["R1"]["cost"]
            for spec, row in zip(plotting.RUNS, fixture["rows"])
        }
        for key, run in self.runs.items():
            self.assertEqual(run["cost"], costs[key])
        ratios = [costs["llama"] / costs[k] for k in ("sft1", "sft2", "sft3")]
        comparison = self.data["sft_vs_llama"]
        self.assertEqual([g["cost_ratio"] for g in comparison["gains"]], ratios)
        self.assertEqual(comparison["cost_ratio_min"], min(ratios))
        self.assertEqual(comparison["cost_ratio_max"], max(ratios))

    def test_token_counts_for_other_episodes_are_refused(self) -> None:
        fixture = token_counts_fixture()
        fixture["rows"][0]["rungs"]["R1"]["episodes"] -= 1
        path = self.write_json("token-counts-short.json", fixture)
        with self.assertRaises(PlotError):
            plotting.reliability_cost_data(RESULTS, path)

    def test_a_missing_token_row_is_refused(self) -> None:
        fixture = token_counts_fixture()
        del fixture["rows"][-1]
        path = self.write_json("token-counts-missing.json", fixture)
        with self.assertRaisesRegex(PlotError, "0 rows"):
            plotting.reliability_cost_data(RESULTS, path)

    def test_a_file_of_another_kind_is_refused(self) -> None:
        path = self.write_json("token-counts-wrong.json", {"kind": "something_else"})
        with self.assertRaises(PlotError):
            plotting.reliability_cost_data(RESULTS, path)

    def test_token_counts_built_from_other_summary_bytes_are_refused(self) -> None:
        fixture = token_counts_fixture()
        fixture["inputs"][0]["sha256"] = "0" * 64
        path = self.write_json("token-counts-stale.json", fixture)
        with self.assertRaisesRegex(PlotError, "differs from the bytes"):
            plotting.reliability_cost_data(RESULTS, path)

    def test_token_counts_that_do_not_list_a_summary_are_refused(self) -> None:
        fixture = token_counts_fixture()
        del fixture["inputs"][0]
        path = self.write_json("token-counts-unlisted.json", fixture)
        with self.assertRaisesRegex(PlotError, "does not list"):
            plotting.reliability_cost_data(RESULTS, path)

    def test_a_paired_comparison_that_disagrees_with_the_summaries_is_refused(self) -> None:
        inputs = self.root / "results"
        inputs.mkdir()
        copy_figure_one_inputs(inputs)
        plotting.reliability_cost_data(inputs, self.token_counts)
        name = self.specs["sft1"]["vs_llama"]
        changed = load(name)
        for row in changed["comparisons"]:
            if row["rung"] == "R1" and row["k"] == 4:
                row["treatment_pass_power_k"] += 0.01
        (inputs / name).write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(PlotError, "does not match the summaries"):
            plotting.reliability_cost_data(inputs, self.token_counts)


class FindTokenCountsTests(_TempDir):
    def test_exactly_one_artifact_is_required(self) -> None:
        with self.assertRaisesRegex(PlotError, "found none"):
            plotting.find_token_counts(self.root)
        one = self.write_json("token-counts-aaaaaaa.json", {})
        self.assertEqual(plotting.find_token_counts(self.root), one)
        self.write_json("token-counts-bbbbbbb.json", {})
        with self.assertRaisesRegex(PlotError, "exactly one"):
            plotting.find_token_counts(self.root)


class TransferDataTests(_TempDir):
    def test_failure_modes_match_the_file_for_every_arm(self) -> None:
        data = plotting.failure_modes_data(RESULTS)
        stored = load(plotting.FAILURE_MODES)["arms"]
        self.assertEqual({row["arm"] for row in data["rows"]}, set(stored))
        for row in data["rows"]:
            r0 = stored[row["arm"]]["rungs"]["R0"]
            with self.subTest(arm=row["arm"]):
                for part in ("no_call_or_unparsed", "call_failed_to_run", "wrong_value", "total"):
                    self.assertEqual(row[part], r0["failures"][part])
                self.assertEqual(row["episodes"], r0["episodes"])
        self.assertEqual(data["episodes"], stored["base-1.7b"]["rungs"]["R0"]["episodes"])

    def test_an_unknown_arm_is_drawn_last_under_its_key(self) -> None:
        payload = load(plotting.FAILURE_MODES)
        payload["arms"]["new-arm"] = payload["arms"]["sft-run1"]
        self.write_json(plotting.FAILURE_MODES, payload)
        rows = plotting.failure_modes_data(self.root)["rows"]
        self.assertEqual(rows[-1]["arm"], "new-arm")
        self.assertEqual(rows[-1]["label"], "new-arm")

    def test_failure_parts_that_do_not_add_up_are_refused(self) -> None:
        payload = load(plotting.FAILURE_MODES)
        payload["arms"]["sft-run1"]["rungs"]["R0"]["failures"]["wrong_value"] += 1
        self.write_json(plotting.FAILURE_MODES, payload)
        with self.assertRaisesRegex(PlotError, "add up"):
            plotting.failure_modes_data(self.root)

    def test_order_support_rates_match_by_intent(self) -> None:
        rows = plotting.order_support_data(RESULTS)["rows"]
        self.assertEqual([row["file"] for row in rows],
                         [spec["file"] for spec in plotting.ORDER_SUPPORT])
        for row in rows:
            by_intent = load(row["file"])["by_intent"]
            with self.subTest(file=row["file"]):
                self.assertEqual(row["completion_rate"], by_intent["fulfil"]["correct_rate"])
                self.assertEqual(row["refusal_rate"], by_intent["refuse"]["correct_rate"])
                self.assertEqual(row["completion_episodes"], by_intent["fulfil"]["episodes"])
                self.assertEqual(row["refusal_engaged_rate"], by_intent["refuse"]["engaged_rate"])
                self.assertEqual(row["attempted_write_rate"],
                                 load(row["file"])["diagnostics"]["attempted_write_rate"])

    def test_only_the_run_that_never_wrote_gets_the_refusal_note(self) -> None:
        rows = plotting.order_support_data(RESULTS)["rows"]
        never_wrote = [row["key"] for row in rows if row["attempted_write_rate"] == 0]
        self.assertEqual(never_wrote, [plotting.ORDER_SUPPORT[0]["key"]])

    def test_an_order_support_run_with_another_adapter_is_refused(self) -> None:
        for spec in plotting.ORDER_SUPPORT:
            shutil.copy(RESULTS / spec["file"], self.root / spec["file"])
            if spec["same_adapter_as"]:
                shutil.copy(RESULTS / spec["same_adapter_as"], self.root / spec["same_adapter_as"])
        plotting.order_support_data(self.root)
        changed = load("phase_b-grpo-42e347c.json")
        changed["weights_sha256"] = "0" * 64
        self.write_json("phase_b-grpo-42e347c.json", changed)
        with self.assertRaisesRegex(PlotError, "did not load the adapter"):
            plotting.order_support_data(self.root)

    def test_an_untrained_order_support_run_that_loaded_an_adapter_is_refused(self) -> None:
        for spec in plotting.ORDER_SUPPORT:
            shutil.copy(RESULTS / spec["file"], self.root / spec["file"])
            if spec["same_adapter_as"]:
                shutil.copy(RESULTS / spec["same_adapter_as"], self.root / spec["same_adapter_as"])
        untrained = plotting.ORDER_SUPPORT[0]
        self.assertIsNone(untrained["same_adapter_as"])
        changed = load(untrained["file"])
        changed["adapter"] = "checkpoints/some-adapter"
        self.write_json(untrained["file"], changed)
        with self.assertRaisesRegex(PlotError, "loaded an adapter"):
            plotting.order_support_data(self.root)

    def test_multi_turn_rates_come_from_the_multi_turn_summaries(self) -> None:
        fixtures = write_order_support_inputs(self.root)
        rows = plotting.multi_turn_order_support_data(self.root)["rows"]
        self.assertEqual([row["key"] for row in rows],
                         [spec["key"] for spec in plotting.ORDER_SUPPORT])
        for spec, row in zip(plotting.ORDER_SUPPORT, rows):
            loop = fixtures[spec["key"]]["multi_turn"]
            with self.subTest(run=spec["key"]):
                self.assertEqual(row["file"], multi_turn_name(spec))
                fulfil, refuse = loop["by_intent"]["fulfil"], loop["by_intent"]["refuse"]
                self.assertEqual(row["completion_rate"], fulfil["correct_rate"])
                self.assertEqual(row["refusal_rate"], refuse["correct_rate"])
                self.assertEqual(row["strict_pass1"], loop["clean_metrics"]["pass^1"])
                self.assertEqual(row["mean_turns_used"], loop["mean_turns_used"])
                self.assertEqual((row["tasks"], row["runs_per_task"], row["max_turns"]),
                                 (150, 4, 4))

    def test_the_single_turn_markers_use_the_single_turn_rates(self) -> None:
        write_order_support_inputs(self.root)
        rows = plotting.multi_turn_order_support_data(self.root)["rows"]
        for spec, row in zip(plotting.ORDER_SUPPORT, rows):
            stored = load(spec["file"])
            with self.subTest(run=spec["key"]):
                self.assertEqual(row["single_turn"]["file"], spec["file"])
                self.assertEqual(row["single_turn"]["completion_rate"],
                                 stored["by_intent"]["fulfil"]["correct_rate"])
                self.assertEqual(row["single_turn"]["refusal_rate"],
                                 stored["by_intent"]["refuse"]["correct_rate"])
                self.assertEqual(row["single_turn"]["attempted_write_rate"],
                                 stored["diagnostics"]["attempted_write_rate"])
                self.assertNotEqual(row["single_turn"]["completion_rate"], row["completion_rate"])

    def test_exactly_one_multi_turn_summary_per_model_is_required(self) -> None:
        write_order_support_inputs(self.root)
        spec = plotting.ORDER_SUPPORT[1]
        extra = self.root / multi_turn_name(spec).replace("abc1234", "def5678")
        shutil.copy(self.root / multi_turn_name(spec), extra)
        with self.assertRaisesRegex(PlotError, "exactly one"):
            plotting.multi_turn_order_support_data(self.root)
        extra.unlink()
        (self.root / multi_turn_name(spec)).unlink()
        with self.assertRaisesRegex(PlotError, "found none"):
            plotting.multi_turn_order_support_data(self.root)

    def test_a_multi_turn_run_with_other_weights_is_refused(self) -> None:
        fixtures = write_order_support_inputs(self.root)
        spec = plotting.ORDER_SUPPORT[2]
        changed = fixtures[spec["key"]]
        changed["weights_sha256"] = "0" * 64
        self.write_json(multi_turn_name(spec), changed)
        with self.assertRaisesRegex(PlotError, "same weights"):
            plotting.multi_turn_order_support_data(self.root)

    def test_an_untrained_multi_turn_run_on_another_base_revision_is_refused(self) -> None:
        fixtures = write_order_support_inputs(self.root)
        spec = plotting.ORDER_SUPPORT[0]
        self.assertIsNone(spec["same_adapter_as"])
        changed = fixtures[spec["key"]]
        self.assertIsNone(changed["weights_sha256"])
        changed["model"]["revision"] = "0" * 40
        self.write_json(multi_turn_name(spec), changed)
        with self.assertRaisesRegex(PlotError, "same weights"):
            plotting.multi_turn_order_support_data(self.root)

    def test_a_summary_that_is_not_a_multi_turn_run_is_refused(self) -> None:
        fixtures = write_order_support_inputs(self.root)
        spec = plotting.ORDER_SUPPORT[1]
        for field, value in (("kind", "transfer_eval"), ("executed", False),
                             ("model", {"id": "Qwen/Qwen3-8B"})):
            with self.subTest(field=field):
                self.write_json(multi_turn_name(spec), {**fixtures[spec["key"]], field: value})
                with self.assertRaisesRegex(PlotError, "is not an executed multi-turn"):
                    plotting.multi_turn_order_support_data(self.root)

    def test_multi_turn_runs_that_scored_other_attempts_are_refused(self) -> None:
        fixtures = write_order_support_inputs(self.root)
        # The same shift in every run, so only the single-turn pairing can catch it.
        for spec in plotting.ORDER_SUPPORT:
            changed = fixtures[spec["key"]]
            changed["multi_turn"]["by_intent"]["fulfil"]["episodes"] -= 4
            self.write_json(multi_turn_name(spec), changed)
        with self.assertRaisesRegex(PlotError, "scored different attempts"):
            plotting.multi_turn_order_support_data(self.root)

    def test_the_turn_cap_comes_from_the_summaries(self) -> None:
        fixtures = write_order_support_inputs(self.root)
        for spec in plotting.ORDER_SUPPORT:
            self.write_json(multi_turn_name(spec), {**fixtures[spec["key"]], "max_turns": 3})
        rows = plotting.multi_turn_order_support_data(self.root)["rows"]
        self.assertEqual([row["max_turns"] for row in rows], [3, 3, 3])

    def test_disagreeing_multi_turn_correct_rates_are_refused(self) -> None:
        fixtures = write_order_support_inputs(self.root)
        spec = plotting.ORDER_SUPPORT[0]
        changed = fixtures[spec["key"]]
        changed["by_intent"]["refuse"]["correct_rate"] += 0.01
        self.write_json(multi_turn_name(spec), changed)
        with self.assertRaisesRegex(PlotError, "refuse correct rates disagree"):
            plotting.multi_turn_order_support_data(self.root)

    def test_grpo_group_shares_match_group_health(self) -> None:
        data = plotting.grpo_groups_data(RESULTS)
        self.assertEqual(len(data["rows"]), len(plotting.GRPO_RUNS))
        for row in data["rows"]:
            stored = load(row["file"])
            health = stored["group_health"]
            with self.subTest(file=row["file"]):
                self.assertEqual(row["zero_variance_fraction"], health["zero_variance_fraction"])
                self.assertEqual(row["groups"], health["groups"])
                self.assertEqual(row["zero_variance_groups"],
                                 round(health["zero_variance_fraction"] * health["groups"]))
                self.assertEqual(row["learning_rate"], stored["learning_rate"]["used"])
                self.assertEqual(row["rollouts_per_group"], stored["resolved"]["num_generations"])

    def test_grpo_runs_with_different_group_sizes_are_refused(self) -> None:
        for name in plotting.GRPO_RUNS:
            shutil.copy(RESULTS / name, self.root / name)
        plotting.grpo_groups_data(self.root)
        changed = load(plotting.GRPO_RUNS[0])
        changed["resolved"]["num_generations"] += 1
        self.write_json(plotting.GRPO_RUNS[0], changed)
        with self.assertRaisesRegex(PlotError, "different group sizes"):
            plotting.grpo_groups_data(self.root)

    def test_learning_rate_labels(self) -> None:
        self.assertEqual(plotting._learning_rate_label(1e-06), "lr 1e-6")
        self.assertEqual(plotting._learning_rate_label(1e-05), "lr 1e-5")
        self.assertEqual(plotting._learning_rate_label(2.5e-06), "lr 2.5e-6")

    def test_turn_labels_drop_trailing_zeros(self) -> None:
        self.assertEqual(plotting._turns(2.26), "2.26")
        self.assertEqual(plotting._turns(1.095), "1.095")
        self.assertEqual(plotting._turns(1.1233333333333333), "1.123")
        self.assertEqual(plotting._turns(2.0), "2")

    def test_the_data_side_never_imports_matplotlib(self) -> None:
        path = self.write_json("token-counts-test.json", token_counts_fixture())
        probe = (
            "import json, sys\n"
            f"sys.path.insert(0, {str(PROJECT_ROOT / 'src')!r})\n"
            "from pathlib import Path\n"
            "from evaluation import plotting\n"
            f"results = Path({str(RESULTS)!r})\n"
            f"plotting.reliability_cost_data(results, Path({str(path)!r}))\n"
            "plotting.transfer_data(results)\n"
            "print(json.dumps(sorted(m for m in sys.modules if m.startswith('matplotlib'))))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [])


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is not installed")
class DrawnValueTests(_TempDir):
    """Each panel draws the fields its labels name, read back from the artists."""

    def setUp(self) -> None:
        super().setUp()
        self.plt = plotting._pyplot()
        self.fig, self.ax = self.plt.subplots()

    def tearDown(self) -> None:
        self.plt.close(self.fig)
        super().tearDown()

    def draw(self, panel, data) -> None:
        with self.plt.rc_context(plotting.STYLE):
            panel(self.ax, data)

    def test_the_cost_panel_puts_cost_on_x_and_pass4_on_y(self) -> None:
        path = self.write_json("token-counts-test.json", token_counts_fixture())
        data = plotting.reliability_cost_data(RESULTS, path)
        # Keep tokens apart from cost, so drawing the wrong field shows.
        for run in data["runs"]:
            run["tokens_per_attempt"] = run["cost"] / 2
        self.draw(plotting._draw_cost_panel, data)
        drawn = [tuple(c.get_offsets()[0]) for c in self.ax.collections]
        self.assertEqual(drawn, [(run["cost"], run["pass^4"]) for run in data["runs"]])

    def test_the_capability_panel_bars_are_the_metric_values(self) -> None:
        path = self.write_json("token-counts-test.json", token_counts_fixture())
        data = plotting.reliability_cost_data(RESULTS, path)
        self.draw(plotting._draw_capability_panel, data)
        # Each metric is drawn twice: a filled bar, then its outline.
        heights = [patch.get_height() for patch in self.ax.patches]
        expected = [
            group["metrics"][name]["value"]
            for group in data["groups"]
            for name in plotting.METRICS
            for _ in range(2)
        ]
        self.assertEqual(heights, expected)

    def test_the_order_support_panel_draws_multi_turn_bars_and_single_turn_markers(self) -> None:
        write_order_support_inputs(self.root)
        data = plotting.multi_turn_order_support_data(self.root)
        self.draw(plotting._draw_order_support_panel, data)
        bars = [(round(p.get_x() + p.get_width() / 2), p.get_height()) for p in self.ax.patches]
        expected = [
            pair
            for row in data["rows"]
            for pair in ((0, row["completion_rate"]), (1, row["refusal_rate"]),
                         (2, row["strict_pass1"]))
        ]
        self.assertEqual(bars, expected)
        markers = [[float(y) for _, y in c.get_offsets()] for c in self.ax.collections]
        self.assertEqual(markers, [
            [row["single_turn"]["completion_rate"], row["single_turn"]["refusal_rate"]]
            for row in data["rows"]
        ])
        labels = [text.get_text() for text in self.ax.texts]
        for row in data["rows"]:
            for name in ("completion_rate", "refusal_rate", "strict_pass1"):
                self.assertIn(f"{row[name]:.3f}", labels)
        legend = [text.get_text() for text in self.ax.get_legend().get_texts()]
        self.assertEqual(legend[:-1], [
            f"{row['label']}, {plotting._turns(row['mean_turns_used'])} turns"
            for row in data["rows"]
        ])
        self.assertEqual(self.ax.get_title(loc="left"),
                         "Order-support transfer, multi-turn closed loop")

    def test_the_order_support_note_names_the_recorded_turn_cap(self) -> None:
        fixtures = write_order_support_inputs(self.root)
        for spec in plotting.ORDER_SUPPORT:
            self.write_json(multi_turn_name(spec), {**fixtures[spec["key"]], "max_turns": 3})
        data = plotting.multi_turn_order_support_data(self.root)
        self.draw(plotting._draw_order_support_panel, data)
        notes = " ".join(text.get_text() for text in self.ax.texts)
        self.assertIn("up to 3 turns per episode", notes)
        self.assertNotIn("up to 4 turns", notes)
        self.assertIn("its mean turns per episode", notes)
        self.assertNotIn("replies per episode", notes)

    def test_the_failure_panel_stacks_the_failure_parts(self) -> None:
        data = plotting.failure_modes_data(RESULTS)
        self.draw(plotting._draw_failure_panel, data)
        widths = [patch.get_width() for patch in self.ax.patches]
        expected = [row[part["key"]] for part in data["parts"] for row in data["rows"]]
        self.assertEqual(widths, expected)

    def test_the_grpo_panel_splits_each_run_at_its_zero_variance_share(self) -> None:
        data = plotting.grpo_groups_data(RESULTS)
        self.draw(plotting._draw_grpo_panel, data)
        heights = [patch.get_height() for patch in self.ax.patches]
        shares = [row["zero_variance_fraction"] for row in data["rows"]]
        self.assertEqual(heights, shares + [1 - share for share in shares])


class CommandTests(_TempDir):
    def test_a_missing_token_counts_file_is_a_refusal(self) -> None:
        with mock.patch("builtins.print"):
            with self.assertRaises(plots.REFUSALS):
                plots.main(["--token-counts", str(self.root / "absent.json"),
                            "--output-dir", str(self.root / "out")])
        self.assertFalse((self.root / "out").exists())

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is not installed")
    def test_the_command_writes_both_figures_byte_stable(self) -> None:
        path = self.write_json("token-counts-test.json", token_counts_fixture())
        outputs = []
        for name in ("first", "second"):
            out = self.root / name
            with mock.patch("builtins.print"):
                code = plots.main(["--token-counts", str(path), "--output-dir", str(out)])
            self.assertEqual(code, 0)
            outputs.append(out)
        for figure in (plots.RELIABILITY_COST, plots.TRANSFER):
            with self.subTest(figure=figure):
                first = (outputs[0] / figure).read_bytes()
                self.assertTrue(first.startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertEqual(first, (outputs[1] / figure).read_bytes())
        self.assertEqual(sorted(p.name for p in outputs[0].iterdir()),
                         sorted((plots.RELIABILITY_COST, plots.TRANSFER)))


if __name__ == "__main__":
    unittest.main()
