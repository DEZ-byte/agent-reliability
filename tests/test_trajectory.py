from __future__ import annotations

import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from evaluation.trajectory import (
    TRAJECTORY_SCHEMA_VERSION,
    TrajectoryJSONLError,
    TrajectoryRecord,
    read_trajectory_jsonl,
    write_trajectory_jsonl,
)


def make_record(*, run_idx: int = 0) -> TrajectoryRecord:
    return TrajectoryRecord(
        schema_version=TRAJECTORY_SCHEMA_VERSION,
        task_id="gsm8k:test:42",
        run_idx=run_idx,
        prompt={"messages": [{"role": "user", "content": "What is 6 × 7?"}]},
        raw_completion="<tool_call>calculator(6 * 7)</tool_call>",
        parsed_tool_calls=[{"name": "calculator", "arguments": {"expr": "6 * 7"}}],
        sandbox_trace=[{"stdout": "42", "exit_code": 0}],
        gate_events=[{"gate": "tool_required", "passed": True}],
        ground_truth={"answer": 42},
        reward_breakdown={"accuracy": 1.0, "tool_required_penalty": 0.0},
    )


class TrajectoryRecordTests(unittest.TestCase):
    def test_jsonl_round_trip_is_lossless(self) -> None:
        original = [make_record(run_idx=0), make_record(run_idx=1)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectories.jsonl"

            written = write_trajectory_jsonl(original, path)
            restored = read_trajectory_jsonl(path)

        self.assertEqual(written, 2)
        self.assertEqual(restored, original)
        self.assertEqual(restored[0].schema_version, TRAJECTORY_SCHEMA_VERSION)

    def test_jsonl_contains_version_and_every_required_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.jsonl"
            write_trajectory_jsonl([make_record()], path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(
            set(payload),
            {
                "schema_version",
                "task_id",
                "run_idx",
                "prompt",
                "raw_completion",
                "parsed_tool_calls",
                "sandbox_trace",
                "gate_events",
                "ground_truth",
                "reward_breakdown",
            },
        )

    def test_non_json_payload_is_rejected(self) -> None:
        payload = make_record().model_dump()
        payload["sandbox_trace"] = ("tuple",)

        with self.assertRaises(ValidationError):
            TrajectoryRecord.model_validate(payload)

    def test_post_construction_nested_non_json_mutation_is_rejected(self) -> None:
        record = make_record()
        self.assertIsInstance(record.parsed_tool_calls, list)
        record.parsed_tool_calls.append(("tuple",))  # type: ignore[union-attr,arg-type]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated.jsonl"
            with self.assertRaises(ValidationError):
                write_trajectory_jsonl([record], path)
            self.assertFalse(path.exists())

    def test_schema_version_is_required(self) -> None:
        payload = make_record().model_dump()
        del payload["schema_version"]

        with self.assertRaises(ValidationError):
            TrajectoryRecord.model_validate(payload)

    def test_missing_and_unsupported_versions_are_invalid_jsonl(self) -> None:
        valid_payload = make_record().model_dump(mode="json")
        invalid_payloads = []

        missing_version = dict(valid_payload)
        del missing_version["schema_version"]
        invalid_payloads.append(missing_version)

        unsupported_version = dict(valid_payload)
        unsupported_version["schema_version"] = 2
        invalid_payloads.append(unsupported_version)

        for payload in invalid_payloads:
            with self.subTest(schema_version=payload.get("schema_version", "missing")):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "invalid-version.jsonl"
                    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(TrajectoryJSONLError, r"line 1"):
                        read_trajectory_jsonl(path)

    def test_malformed_jsonl_reports_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "malformed.jsonl"
            valid = make_record().model_dump_json()
            path.write_text(f"{valid}\n{{not-json}}\n", encoding="utf-8")

            with self.assertRaisesRegex(TrajectoryJSONLError, r"line 2"):
                read_trajectory_jsonl(path)

    def test_schema_invalid_jsonl_reports_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-record.jsonl"
            path.write_text('{"schema_version":1,"task_id":"missing-fields"}\n')

            with self.assertRaisesRegex(TrajectoryJSONLError, r"line 1"):
                read_trajectory_jsonl(path)

    def test_blank_jsonl_line_is_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blank-line.jsonl"
            path.write_text("\n", encoding="utf-8")

            with self.assertRaisesRegex(TrajectoryJSONLError, r"line 1"):
                read_trajectory_jsonl(path)

    def test_unencodable_record_leaves_an_existing_file_intact(self) -> None:
        """Regression: the writer must not truncate results it cannot replace."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectories.jsonl"
            write_trajectory_jsonl([make_record(run_idx=0)], path)
            before = path.read_bytes()

            doomed = make_record(run_idx=1).model_dump(mode="python")
            doomed["raw_completion"] = "lone surrogate " + chr(0xD800)

            # Rejected at validation, before the destination is opened. The
            # writer's own encode guard remains as defence in depth.
            with self.assertRaises(ValueError):
                write_trajectory_jsonl([doomed], path)

            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(read_trajectory_jsonl(path), [make_record(run_idx=0)])
            leftovers = [p.name for p in Path(directory).iterdir() if ".tmp" in p.name]
            self.assertEqual(leftovers, [])

    def test_invalid_record_leaves_an_existing_file_intact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectories.jsonl"
            write_trajectory_jsonl([make_record(run_idx=0)], path)
            before = path.read_bytes()

            with self.assertRaises(ValidationError):
                write_trajectory_jsonl(
                    [{"schema_version": 1, "task_id": "incomplete"}], path
                )

            self.assertEqual(path.read_bytes(), before)

    def test_write_leaves_no_temporary_file_behind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectories.jsonl"
            write_trajectory_jsonl([make_record(run_idx=0)], path)

            self.assertEqual([p.name for p in Path(directory).iterdir()], [path.name])

    def test_concurrent_writers_use_distinct_temporary_files(self) -> None:
        """Concurrent replacements may race, but neither writer may collide."""

        first = [make_record(run_idx=1)]
        second = [make_record(run_idx=2)]
        barrier = threading.Barrier(2)
        sources: list[Path] = []
        sources_lock = threading.Lock()
        real_named_temporary_file = tempfile.NamedTemporaryFile

        def synchronized_temporary_file(*args: object, **kwargs: object) -> object:
            handle = real_named_temporary_file(*args, **kwargs)
            with sources_lock:
                sources.append(Path(handle.name))
            barrier.wait(timeout=5)
            return handle

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectories.jsonl"
            with patch(
                "evaluation.trajectory.tempfile.NamedTemporaryFile",
                side_effect=synchronized_temporary_file,
            ):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [
                        executor.submit(write_trajectory_jsonl, records, path)
                        for records in (first, second)
                    ]
                    self.assertEqual(
                        [future.result(timeout=10) for future in futures],
                        [1, 1],
                    )

            self.assertEqual(len(set(sources)), 2)
            self.assertIn(read_trajectory_jsonl(path), (first, second))
            self.assertEqual([p.name for p in Path(directory).iterdir()], [path.name])


if __name__ == "__main__":
    unittest.main()
