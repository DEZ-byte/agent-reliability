"""Teacher generation batches safely, and checkpoint selection resumes safely.

Batched teacher generation must produce the same episodes in the same order as
the sequential path and must refuse to pair a completion with a prompt it was
not generated for. Checkpoint selection must reuse a score only from an
executed artifact that carries the metric and was scored on the weights now on
disk.

None of this loads a model.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts import generate_sft_trajectories as gen  # noqa: E402
from scripts import select_checkpoint as selector  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
