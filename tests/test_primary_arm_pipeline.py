"""The unattended pipeline must resume, skip, and never re-run a finished test.

What is worth a test here. Batched teacher generation must produce the same
episodes in the same order as the sequential path and must refuse to pair a
completion with a prompt it was not generated for. Checkpoint selection must
reuse a score only from an executed artifact that carries the metric and was
scored on the weights now on disk. The stage runner must skip an executed
artifact, treat a plan-only or errored artifact as not done (the runners write
a failed model load as result data with exit 0), keep smoke output apart from
real output, and name outputs the way `results/` expects.

None of this loads a model. The GPU paths were exercised by a local smoke run
of the whole chain on the 4060 with the 1.7B as teacher and student.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts import generate_sft_trajectories as gen  # noqa: E402
from scripts import run_primary_arm_pipeline as pipeline  # noqa: E402
from scripts import select_checkpoint as selector  # noqa: E402

BASE_ARGS = ["--student", "Qwen/Qwen3-4B", "--seed", "7", "--skip-comparator", "--teacher-deviation", "D-079"]


def _fake_popen(side_effect=None, returncode=0, lines=("boom line\n",)):
    """A Popen stand-in that streams `lines`, runs `side_effect`, then exits."""

    def factory(command, cwd, stdout, stderr, text, errors):
        if side_effect is not None:
            side_effect()
        fake = mock.Mock()
        fake.stdout = iter(lines)
        fake.wait = mock.Mock(return_value=None)
        fake.returncode = returncode
        return fake

    return factory


class BatchingTests(unittest.TestCase):
    def test_batches_preserve_order_and_cover_every_item(self) -> None:
        items = list(range(10))
        chunks = gen.batches(items, 4)
        self.assertEqual(chunks, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]])
        self.assertEqual([x for chunk in chunks for x in chunk], items)

    def test_batch_size_below_one_is_refused(self) -> None:
        with self.assertRaises(gen.GenerationError):
            gen.batches([1, 2], 0)

    def test_precomputed_policy_answers_once_and_only_its_own_prompt(self) -> None:
        expected = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
        policy = gen.precomputed_policy(expected, "completion")
        self.assertEqual(policy(list(expected)), "completion")
        with self.assertRaises(gen.GenerationError):
            policy(list(expected))  # a second decision is not what was generated
        other = gen.precomputed_policy(expected, "completion")
        with self.assertRaises(gen.GenerationError):
            other([{"role": "user", "content": "different prompt"}])

    def test_first_decision_messages_match_run_episode(self) -> None:
        from env.phase_a import PhaseATask

        task = PhaseATask(task_id="t", template_id="t", question="2+2?", gold_answer=4.0, source="test")
        messages = gen._first_decision_messages(task)
        self.assertEqual(messages[0]["content"], gen.SYSTEM_PROMPT)
        self.assertEqual(messages[1]["content"], gen.USER_PROMPT.format(question="2+2?"))

    def test_batched_generation_refuses_a_rung_with_more_than_one_decision(self) -> None:
        config = {"generation": {"rung": "R1", "seed_base": 1}, "retention": {"min_question_match_ratio": 0.0}}
        with self.assertRaises(gen.GenerationError):
            gen.generate(model={"id": "x", "revision": "y"}, tasks=[], config=config, rows_out=None, batch_size=2)


class SelectionResumeTests(unittest.TestCase):
    def _write(self, directory: Path, payload: dict) -> Path:
        path = directory / "select-checkpoint-25.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _good(self, **extra) -> dict:
        return {
            "executed": True,
            "results": [{"rungs": {"R0": {"metrics": {"pass^1": 0.4725}, "no_arithmetic_rate": 0.005}}}],
            **extra,
        }

    def test_missing_or_plan_only_or_partial_artifacts_are_not_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.assertIsNone(selector._reusable_score(directory / "absent.json", rung="R0", metric="pass^1"))
            plan_only = self._write(directory, {**self._good(), "executed": False})
            self.assertIsNone(selector._reusable_score(plan_only, rung="R0", metric="pass^1"))
            partial = self._write(directory, {"executed": True, "results": [{"rungs": {"R0": {"metrics": {}}}}]})
            self.assertIsNone(selector._reusable_score(partial, rung="R0", metric="pass^1"))
            errored = self._write(directory, {"executed": True, "results": [{"error": "boom"}]})
            self.assertIsNone(selector._reusable_score(errored, rung="R0", metric="pass^1"))
            partial.write_text("{not json", encoding="utf-8")
            self.assertIsNone(selector._reusable_score(partial, rung="R0", metric="pass^1"))

    def test_an_executed_artifact_with_the_metric_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), self._good())
            self.assertEqual(
                selector._reusable_score(path, rung="R0", metric="pass^1"),
                {"score": 0.4725, "no_arithmetic_rate": 0.005},
            )

    def test_a_score_is_reused_only_for_the_weights_now_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint-25"
            checkpoint.mkdir()
            weights = checkpoint / "adapter_model.safetensors"
            weights.write_bytes(b"weights-v1")
            digest = hashlib.sha256(b"weights-v1").hexdigest()
            path = self._write(Path(tmp), self._good(adapter={"weights_sha256": digest}))
            self.assertIsNotNone(selector._reusable_score(path, rung="R0", metric="pass^1", checkpoint=checkpoint))
            weights.write_bytes(b"weights-v2")  # retrained in place under a new commit
            self.assertIsNone(selector._reusable_score(path, rung="R0", metric="pass^1", checkpoint=checkpoint))
            weights.unlink()
            self.assertIsNone(selector._reusable_score(path, rung="R0", metric="pass^1", checkpoint=checkpoint))


class StageRunnerTests(unittest.TestCase):
    def _pipeline(self, run_dir: Path, *extra: str):
        return pipeline.Pipeline(pipeline.parse_args(["--run-dir", str(run_dir), *BASE_ARGS, *extra]))

    def test_executed_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.json"
            self.assertFalse(pipeline.executed(path))
            path.write_text(json.dumps({"executed": False}), encoding="utf-8")
            self.assertFalse(pipeline.executed(path))
            path.write_text("{broken", encoding="utf-8")
            self.assertFalse(pipeline.executed(path))
            path.write_text(json.dumps({"executed": True, "results": [{"candidate": "x", "error": "401 gated"}]}), encoding="utf-8")
            self.assertFalse(pipeline.executed(path), "a recorded failure is not a finished stage")
            path.write_text(json.dumps({"executed": True, "results": [{"candidate": "x", "rungs": {}}]}), encoding="utf-8")
            self.assertTrue(pipeline.executed(path))
            path.write_text(json.dumps({"comparisons": []}), encoding="utf-8")
            self.assertTrue(pipeline.executed(path), "artifacts without the field count once they exist")

    def test_an_executed_artifact_skips_its_stage_and_a_failure_stops_the_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._pipeline(Path(tmp))
            done = run.results / "done.json"
            done.write_text(json.dumps({"executed": True}), encoding="utf-8")
            with mock.patch.object(pipeline.subprocess, "Popen") as fake:
                run._run("stage", ["python", "x.py"], done)
                fake.assert_not_called()

            with mock.patch.object(pipeline.subprocess, "Popen", side_effect=_fake_popen()),                     mock.patch.object(pipeline, "_git", return_value=""):
                with self.assertRaises(pipeline.PipelineError):
                    run._run("stage", ["python", "x.py"], run.results / "never.json")
            records = [json.loads(line) for line in run.log_path.read_text().splitlines()]
            self.assertEqual([r["status"] for r in records], ["skipped", "failed"])
            self.assertEqual(records[1]["output_tail"], ["boom line"])
            self.assertTrue((run.run_dir / "logs" / "stage.log").is_file())

    def test_an_errored_artifact_is_set_aside_so_the_stage_runs_again(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._pipeline(Path(tmp))
            target = run.results / "comparator-8b-abc1234.json"

            def writes_error():
                target.write_text(json.dumps({"executed": True, "results": [{"candidate": "llama", "error": "gated"}]}), encoding="utf-8")

            with mock.patch.object(pipeline.subprocess, "Popen", side_effect=_fake_popen(writes_error)),                     mock.patch.object(pipeline, "_git", return_value=""):
                with self.assertRaises(pipeline.PipelineError):
                    run._run("comparator", ["python", "x.py"], target)
            self.assertFalse(target.exists(), "the errored artifact must not block the next attempt")
            self.assertEqual(len(list(run.results.glob("comparator-8b-abc1234.failed-*.json"))), 1)
            self.assertFalse(pipeline.executed(target))

    def test_smoke_flags_move_every_output_into_a_smoke_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            real = self._pipeline(Path(tmp))
            smoke = self._pipeline(Path(tmp), "--limit", "2", "--max-steps", "2")
            self.assertEqual(real.run_dir, Path(tmp).resolve())
            self.assertEqual(smoke.run_dir, Path(tmp).resolve() / "smoke-limit2-steps2")
            self.assertTrue(smoke.smoke and not real.smoke)
            self.assertNotEqual(real.results, smoke.results)

    def test_teacher_deviation_is_required_and_must_be_a_decision_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                pipeline.parse_args(["--run-dir", tmp, "--skip-comparator"])
            with self.assertRaises(SystemExit):
                pipeline.parse_args(["--run-dir", tmp, "--skip-comparator", "--teacher-deviation", "pending"])
            for ok in ("D-079", "none"):
                self.assertEqual(pipeline.parse_args(["--run-dir", tmp, "--skip-comparator", "--teacher-deviation", ok]).teacher_deviation, ok)

    def test_registered_teacher_uses_its_pinned_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._pipeline(Path(tmp), "--teacher", "Qwen/Qwen3-4B", "--teacher-revision", "deadbeef")
            self.assertEqual(run.teacher_revision_source, "registry")
            self.assertNotEqual(run.teacher_revision, "deadbeef")
            unregistered = self._pipeline(Path(tmp), "--teacher", "Qwen/Qwen3-14B", "--teacher-revision", "abc")
            self.assertEqual((unregistered.teacher_revision_source, unregistered.teacher_revision), ("explicit", "abc"))

    def test_outputs_are_named_for_results_and_live_outside_the_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._pipeline(Path(tmp), "--dry-run")
            commands: list[tuple] = []
            with mock.patch("builtins.print", side_effect=lambda *a, **k: commands.append(a)):
                run.run()
            planned = " ".join(str(part) for line in commands for part in line)
            self.assertIn(f"sft-run-qwen3-4b-seed7-{run.commit}.json", planned)
            self.assertIn(f"sft-test-qwen3-4b-seed7-{run.commit}.json", planned)
            self.assertIn(f"sft-comparison-qwen3-4b-seed7-{run.commit}.json", planned)
            self.assertIn(f"qwen3-4b-sft-seed7-{run.commit}", planned, "checkpoint dirs carry the commit")
            self.assertNotIn(str(PROJECT_ROOT / "results"), planned, "nothing may be written into the repository")
            self.assertIn("--resume", planned)
            self.assertIn("--teacher-deviation D-079", planned)

    def test_reuse_needs_the_matching_test_summary_and_the_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reuse = Path(tmp) / "reuse"
            reuse.mkdir()
            rows = [{"candidate": "Qwen/Qwen3-1.7B"}, {"candidate": "Qwen/Qwen3-4B"}]
            for stem in ("phase_a-abc1234", "phase_a-dev-abc1234"):
                (reuse / f"episodes-{stem}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            run = self._pipeline(Path(tmp) / "run", "--reuse-dir", str(reuse))
            self.assertIsNone(run._find_reusable("episodes-phase_a-*.jsonl", "Qwen/Qwen3-4B"), "episodes without their summary are not reusable")
            # Summary artifacts record the registry entry, episode rows the bare id.
            summary = {"executed": True, "eval_config": "eval.yaml", "results": [{"candidate": {"role": "primary_small", "id": "Qwen/Qwen3-4B", "revision": "x"}, "rungs": {}}, {"candidate": "Qwen/Qwen3-1.7B", "rungs": {}}]}
            (reuse / "phase_a-dev-abc1234.json").write_text(json.dumps({**summary, "eval_config": "eval_dev.yaml"}), encoding="utf-8")
            self.assertIsNone(run._find_reusable("episodes-phase_a-*.jsonl", "Qwen/Qwen3-4B"), "a dev-split artifact never stands in for test")
            # The M1 baseline pair on disk is episodes-phase_a-X.jsonl + baseline-phase_a-X.json.
            (reuse / "baseline-phase_a-abc1234.json").write_text(json.dumps(summary), encoding="utf-8")
            found = run._find_reusable("episodes-phase_a-*.jsonl", "Qwen/Qwen3-4B")
            self.assertEqual(found["episodes"].name, "episodes-phase_a-abc1234.jsonl")
            self.assertEqual(found["summary"].name, "baseline-phase_a-abc1234.json")
            self.assertIsNone(run._find_reusable("episodes-phase_a-*.jsonl", "meta-llama/Llama-3.1-8B-Instruct"))

    def test_markdown_flags_a_smoke_run_and_a_reused_row(self) -> None:
        summary = {
            "source_commit": "abcdef0123",
            "teacher": {"id": "Qwen/Qwen3-14B", "revision": "40c0698", "revision_source": "explicit", "deviation": "D-078"},
            "students": ["Qwen/Qwen3-4B"],
            "seeds": [1],
            "smoke": True,
            "limit": 2,
            "max_steps": 2,
            "arms": [
                {"arm": "x", "rung": "R0", "pass^1": 0.5, "pass^1_ci95": [0.4, 0.6], "pass^4": 0.3, "pass^4_ci95": None, "pass@4": 0.6, "tokens_per_episode": 30.0, "no_arithmetic_rate": 0.01, "source": "reused baseline-phase_a-3cc174f.json"},
                {"arm": "y", "rung": "—", "missing": "/tmp/comparator-8b-abc.json"},
            ],
            "paired_contrasts": [],
        }
        text = pipeline.render_markdown(summary)
        self.assertIn("SMOKE RUN", text)
        self.assertIn("reused baseline-phase_a-3cc174f.json", text)
        self.assertIn("missing: comparator-8b-abc.json", text)
        self.assertIn("no-arithmetic rate", text)


if __name__ == "__main__":
    unittest.main()
