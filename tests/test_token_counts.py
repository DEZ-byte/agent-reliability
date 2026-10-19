"""Token counts rebuilt from episode logs, and the unique parameter counts.

Three README rows have no recorded token counts, so the backfill command
rebuilds them by re-tokenizing the logged completions. These tests pin the
rule, the per-rung sums, the refusal when the rule does not reproduce a
recorded total, the parameter count from safetensors headers, and the shape of
the artifact. A whitespace tokenizer stands in for the real ones, so no model
library is needed.
"""

from __future__ import annotations

import json
import re
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from commands import backfill_tokens  # noqa: E402
from evaluation.provenance import DirtyWorktreeError  # noqa: E402
from evaluation.token_counts import (  # noqa: E402
    UNIQUE_PARAMS_B,
    RungCount,
    TokenCountError,
    check_rounded,
    cost,
    decision_tokens,
    parameter_counts,
    rebuild_rung,
    recorded_rung,
    validate,
)

SMALL = "Qwen/Qwen3-1.7B"
TEACHER = "Qwen/Qwen3-4B"
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
HEAD = "c0ffee" * 6 + "c0ff"
LIMIT = 256


def encode(text: str) -> list[str]:
    return text.split()


def row(candidate: str, rung: str, task: int, run: int, *completions: str) -> dict:
    return {
        "candidate": candidate,
        "task_id": f"gsm8k:test:{task}",
        "run_index": run,
        "rung": rung,
        "completions": list(completions),
        "counters": {"policy_model_decision_count": len(completions)},
    }


def episodes(candidate: str) -> list[dict]:
    """Two tasks of two runs at R0, and the same at R1 with one retry."""

    rows = []
    for rung in ("R0", "R1"):
        rows.append(row(candidate, rung, 1, 0, "one two three"))
        rows.append(row(candidate, rung, 1, 1, "one"))
        rows.append(row(candidate, rung, 2, 0, "one two"))
        if rung == "R1":
            rows.append(row(candidate, rung, 2, 1, "plain prose", "a b c d e"))
        else:
            rows.append(row(candidate, rung, 2, 1, "plain prose"))
    return rows


def rung_summary(rows: list[dict], candidate: str, rung: str, *, record: bool) -> dict:
    count = rebuild_rung(rows, candidate=candidate, rung=rung, encode=encode, max_new_tokens=LIMIT)
    stored = {"episodes": count.episodes, "model_decisions": count.decisions}
    if record:
        stored["generated_tokens_total"] = count.tokens_total
        stored["generated_tokens_per_episode"] = count.tokens_per_attempt
    return stored


def summary(rows: list[dict], candidates: list[str], *, record: bool) -> dict:
    return {
        "executed": True,
        "decoding": {"max_new_tokens": LIMIT},
        "results": [
            {
                "candidate": {"id": candidate},
                "rungs": {
                    rung: rung_summary(rows, candidate, rung, record=record)
                    for rung in ("R0", "R1")
                },
            }
            for candidate in candidates
        ],
    }


def write_safetensors(path: Path, shapes: dict[str, list[int]]) -> None:
    header = {name: {"dtype": "BF16", "shape": shape, "data_offsets": [0, 0]}
              for name, shape in shapes.items()}
    header["__metadata__"] = {"format": "pt"}
    data = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(data)) + data)


# -- the pure functions ------------------------------------------------------


class RuleTests(unittest.TestCase):
    def test_the_dropped_eos_is_added_back(self) -> None:
        self.assertEqual(decision_tokens("one two three", encode, LIMIT), 4)
        self.assertEqual(decision_tokens("", encode, LIMIT), 1)

    def test_a_reply_that_hit_the_limit_is_capped(self) -> None:
        self.assertEqual(decision_tokens(" ".join(["x"] * 255), encode, LIMIT), 256)
        self.assertEqual(decision_tokens(" ".join(["x"] * 256), encode, LIMIT), 256)
        self.assertEqual(decision_tokens(" ".join(["x"] * 400), encode, LIMIT), 256)

    def test_cost_is_tokens_times_billions(self) -> None:
        self.assertAlmostEqual(cost(36.0, 1.72), 61.92)


class AggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = episodes(SMALL) + episodes(TEACHER)

    def _rebuild(self, rung: str, rows: list[dict] | None = None) -> RungCount:
        return rebuild_rung(
            self.rows if rows is None else rows,
            candidate=SMALL,
            rung=rung,
            encode=encode,
            max_new_tokens=LIMIT,
        )

    def test_r0_sums_one_decision_per_episode(self) -> None:
        # 4 + 2 + 3 + 3, and the other candidate's rows are left out.
        self.assertEqual(self._rebuild("R0"), RungCount(episodes=4, decisions=4, tokens_total=12))

    def test_r1_counts_the_retry_as_its_own_decision(self) -> None:
        count = self._rebuild("R1")
        self.assertEqual(count, RungCount(episodes=4, decisions=5, tokens_total=18))
        self.assertEqual(count.tokens_per_attempt, 4.5)

    def test_completions_that_disagree_with_the_counters_are_refused(self) -> None:
        broken = episodes(SMALL)
        broken[0]["counters"]["policy_model_decision_count"] = 2
        with self.assertRaisesRegex(TokenCountError, "1 completions but 2 logged"):
            self._rebuild("R0", broken)

    def test_a_candidate_with_no_rows_is_refused(self) -> None:
        with self.assertRaisesRegex(TokenCountError, "no R0 rows"):
            self._rebuild("R0", episodes(TEACHER))

    def test_recorded_counts_are_read_and_checked(self) -> None:
        self.assertIsNone(recorded_rung({"episodes": 4, "model_decisions": 4}))
        stored = {"episodes": 4, "model_decisions": 5, "generated_tokens_total": 18,
                  "generated_tokens_per_episode": 4.5}
        self.assertEqual(recorded_rung(stored), RungCount(4, 5, 18))
        with self.assertRaises(TokenCountError):
            recorded_rung({**stored, "generated_tokens_per_episode": 4.0})


class ValidationTests(unittest.TestCase):
    def test_matching_totals_pass(self) -> None:
        validate([{"label": "a R0", "recorded": 10, "rebuilt": 10}])

    def test_one_wrong_total_refuses_and_names_it(self) -> None:
        checks = [
            {"label": "a R0", "recorded": 10, "rebuilt": 10},
            {"label": "a R1", "recorded": 12, "rebuilt": 11},
        ]
        with self.assertRaisesRegex(TokenCountError, r"1 of 2 .*a R1: recorded 12, rebuilt 11"):
            validate(checks)

    def test_nothing_to_validate_against_refuses(self) -> None:
        with self.assertRaises(TokenCountError):
            validate([])


class ParameterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _snapshot(self, tied: bool) -> Path:
        (self.root / "config.json").write_text(
            json.dumps({"tie_word_embeddings": tied}), encoding="utf-8"
        )
        write_safetensors(self.root / "a.safetensors", {"model.embed_tokens.weight": [10, 4]})
        write_safetensors(
            self.root / "b.safetensors", {"model.norm.weight": [4], "lm_head.weight": [10, 4]}
        )
        return self.root

    def test_a_tied_head_stored_as_a_copy_is_counted_once(self) -> None:
        counts = parameter_counts(self._snapshot(tied=True))
        self.assertEqual((counts["stored"], counts["unique"]), (84, 44))
        self.assertTrue(counts["head_stored_separately"])

    def test_an_untied_head_is_its_own_parameters(self) -> None:
        counts = parameter_counts(self._snapshot(tied=False))
        self.assertEqual((counts["stored"], counts["unique"]), (84, 84))

    def test_the_rounded_constants_must_match_the_exact_count(self) -> None:
        check_rounded(SMALL, 1_720_574_976)
        with self.assertRaisesRegex(TokenCountError, "does not round"):
            check_rounded(SMALL, 2_031_739_904)
        self.assertEqual(set(UNIQUE_PARAMS_B), {SMALL, TEACHER, LLAMA})


# -- the command -------------------------------------------------------------


class CommandTests(unittest.TestCase):
    """One rebuilt row, two recorded Qwen rows to validate on, and Llama."""

    ROWS = (
        {"row": "small, untrained", "model": SMALL, "summary": "base.json",
         "episodes": "episodes-base.jsonl", "source": backfill_tokens.REBUILT},
        {"row": "llama", "model": LLAMA, "summary": "llama.json",
         "episodes": "episodes-llama.jsonl", "source": backfill_tokens.RECORDED},
        {"row": "small, SFT run 2", "seed": 7, "model": SMALL, "summary": "sft.json",
         "episodes": "episodes-sft.jsonl", "source": backfill_tokens.RECORDED},
        {"row": "teacher", "model": TEACHER, "summary": "base.json",
         "episodes": "episodes-base.jsonl", "source": backfill_tokens.REBUILT},
    )

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.results = self.root / "results"
        self.results.mkdir()
        self._write("base", episodes(SMALL) + episodes(TEACHER), [SMALL, TEACHER], record=False)
        self._write("sft", episodes(SMALL), [SMALL], record=True)
        self._write("llama", episodes(LLAMA), [LLAMA], record=True)
        # Llama's real counts differ from the rule; the command keeps them.
        llama = json.loads((self.results / "llama.json").read_text(encoding="utf-8"))
        for stored in llama["results"][0]["rungs"].values():
            stored["generated_tokens_total"] += 4
            stored["generated_tokens_per_episode"] = stored["generated_tokens_total"] / 4
        (self.results / "llama.json").write_text(json.dumps(llama), encoding="utf-8")

        self.encoders: list[str] = []
        self.guard = mock.Mock(return_value=HEAD)
        patches = [
            mock.patch.object(backfill_tokens, "ROWS", self.ROWS),
            mock.patch.object(backfill_tokens, "RESULTS_DIR", self.results),
            mock.patch.object(backfill_tokens, "require_clean_worktree", self.guard),
            mock.patch.object(backfill_tokens, "load_encoder", self._load_encoder),
            mock.patch.object(backfill_tokens, "snapshot_dir", return_value=None),
            mock.patch("builtins.print"),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, name: str, rows: list[dict], candidates: list[str], *, record: bool) -> None:
        (self.results / f"episodes-{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
        (self.results / f"{name}.json").write_text(
            json.dumps(summary(rows, candidates, record=record)), encoding="utf-8"
        )

    def _load_encoder(self, model: str, revision: str):
        self.encoders.append(model)
        return encode

    def _artifact(self) -> dict:
        backfill_tokens.main([])
        return json.loads((self.results / f"token-counts-{HEAD[:7]}.json").read_text("utf-8"))

    def test_the_guard_runs_once_with_the_project_root(self) -> None:
        artifact = self._artifact()
        self.guard.assert_called_once_with(backfill_tokens.PROJECT_ROOT)
        self.assertEqual(artifact["source_commit"], HEAD)

    def test_the_artifact_has_every_row_with_both_rungs(self) -> None:
        artifact = self._artifact()
        self.assertEqual(artifact["schema_version"], 1)
        self.assertEqual(artifact["kind"], "token_counts")
        for key in ("created_at_utc", "platform", "method", "parameters", "validation", "inputs"):
            self.assertIn(key, artifact)
        self.assertEqual([r["row"] for r in artifact["rows"]], [s["row"] for s in self.ROWS])

        by_row = {r["row"]: r for r in artifact["rows"]}
        r1 = by_row["small, untrained"]["rungs"]["R1"]
        self.assertEqual(
            {k: r1[k] for k in ("tokens_total", "episodes", "decisions", "source")},
            {"tokens_total": 18, "episodes": 4, "decisions": 5, "source": "rebuilt_from_logs"},
        )
        self.assertEqual(r1["tokens_per_attempt"], 4.5)
        self.assertAlmostEqual(r1["cost"], 4.5 * UNIQUE_PARAMS_B[SMALL])
        self.assertEqual(by_row["small, untrained"]["summary"], "base.json")
        self.assertEqual(by_row["small, SFT run 2"]["seed"], 7)

        # Llama keeps its recorded totals, and the rule's gap is shown beside them.
        llama = by_row["llama"]["rungs"]
        self.assertEqual((llama["R0"]["tokens_total"], llama["R0"]["source"]), (16, "recorded"))
        gaps = [check["difference"] for check in artifact["validation"]["rule_on_llama"]]
        self.assertEqual(gaps, [-4, -4])

        checks = artifact["validation"]["qwen"]
        labels = [c["label"] for c in checks]
        self.assertEqual(labels, ["small, SFT run 2 R0", "small, SFT run 2 R1"])
        self.assertTrue(all(c["match"] for c in checks))
        self.assertEqual(
            {entry["file"] for entry in artifact["inputs"]},
            {"base.json", "sft.json", "llama.json",
             "episodes-base.jsonl", "episodes-sft.jsonl", "episodes-llama.jsonl"},
        )
        self.assertFalse(artifact["parameters"]["models"][SMALL]["checked_against_checkpoint"])
        self.assertEqual(sorted(set(self.encoders)), sorted({SMALL, TEACHER, LLAMA}))

    def test_the_artifact_holds_no_absolute_path(self) -> None:
        text = json.dumps(self._artifact())
        self.assertNotIn(self._tmp.name.replace("\\", "\\\\"), text)
        self.assertNotIn(str(PROJECT_ROOT).replace("\\", "\\\\"), text)
        self.assertIsNone(re.search(r"[A-Za-z]:\\\\|\"/(home|Users|tmp)/", text))

    def test_a_rule_that_misses_a_recorded_total_writes_nothing(self) -> None:
        sft = json.loads((self.results / "sft.json").read_text(encoding="utf-8"))
        stored = sft["results"][0]["rungs"]["R1"]
        stored["generated_tokens_total"] += 1
        stored["generated_tokens_per_episode"] = stored["generated_tokens_total"] / 4
        (self.results / "sft.json").write_text(json.dumps(sft), encoding="utf-8")
        missed = "small, SFT run 2 R1: recorded 19, rebuilt 18"
        with self.assertRaisesRegex(TokenCountError, missed):
            backfill_tokens.main([])
        self.assertEqual(list(self.results.glob("token-counts-*")), [])

    def test_a_rebuilt_row_whose_log_misses_an_episode_writes_nothing(self) -> None:
        # The recorded rows still validate; only the rebuilt row's log is short.
        rows = [
            r for r in episodes(SMALL)
            if not (r["rung"] == "R1" and r["task_id"] == "gsm8k:test:2" and r["run_index"] == 1)
        ] + episodes(TEACHER)
        (self.results / "episodes-base.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
        with self.assertRaisesRegex(TokenCountError, "small, untrained R1: the log gives 3 episodes"):
            backfill_tokens.main([])
        self.assertEqual(list(self.results.glob("token-counts-*")), [])

    def test_a_rebuilt_row_whose_summary_already_counts_tokens_is_refused(self) -> None:
        # A row marked for rebuilding must not silently replace recorded counts.
        base = json.loads((self.results / "base.json").read_text(encoding="utf-8"))
        stored = base["results"][0]["rungs"]["R0"]
        stored["generated_tokens_total"] = 12
        stored["generated_tokens_per_episode"] = 3.0
        (self.results / "base.json").write_text(json.dumps(base), encoding="utf-8")
        with self.assertRaisesRegex(
            backfill_tokens.BackfillError, "base.json already records R0 token counts"
        ):
            backfill_tokens.main([])
        self.assertEqual(list(self.results.glob("token-counts-*")), [])

    def test_a_summary_with_another_token_limit_is_refused(self) -> None:
        with self.assertRaisesRegex(TokenCountError, "max_new_tokens=256, not 128"):
            backfill_tokens.main(["--max-new-tokens", "128"])

    def test_a_dirty_tree_stops_before_any_tokenizer_loads(self) -> None:
        self.guard.side_effect = DirtyWorktreeError("dirty")
        with self.assertRaises(DirtyWorktreeError):
            backfill_tokens.main([])
        self.assertEqual(self.encoders, [])
        self.assertEqual(list(self.results.glob("token-counts-*")), [])

    def test_a_checked_snapshot_records_exact_counts(self) -> None:
        snapshot = self.root / "snapshot"
        snapshot.mkdir()
        (snapshot / "config.json").write_text('{"tie_word_embeddings": true}', encoding="utf-8")
        write_safetensors(snapshot / "m.safetensors", {"w": [10]})
        with mock.patch.object(backfill_tokens, "snapshot_dir", return_value=snapshot), \
                mock.patch.object(backfill_tokens, "check_rounded") as checked:
            artifact = self._artifact()
        self.assertEqual(checked.call_count, 3)
        entry = artifact["parameters"]["models"][SMALL]
        self.assertEqual((entry["unique"], entry["stored"]), (10, 10))
        self.assertTrue(entry["checked_against_checkpoint"])
        self.assertEqual(entry["rounded_billions"], 1.72)


class RealRowsTests(unittest.TestCase):
    def test_the_rows_name_each_readme_model_once(self) -> None:
        rows = backfill_tokens.ROWS
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({r["row"] for r in rows}), 8)
        rebuilt = [r["row"] for r in rows if r["source"] == backfill_tokens.REBUILT]
        self.assertEqual(
            rebuilt,
            ["Qwen3-1.7B, untrained", "Qwen3-1.7B, SFT run 1", "Qwen3-4B, untrained (the teacher)"],
        )
        validated = [r for r in rows if r["source"] == backfill_tokens.RECORDED
                     and r["model"] != LLAMA]
        self.assertEqual(len(validated), 4)
        self.assertEqual(sorted(r["seed"] for r in rows if "seed" in r),
                         [20260820, 20260823, 20260824])


if __name__ == "__main__":
    unittest.main()
