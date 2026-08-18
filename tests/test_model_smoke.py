from __future__ import annotations

import builtins
import hashlib
from contextlib import nullcontext
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from scripts import smoke_models as smoke


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "model_smoke.json"
QWEN_NATIVE_TRAINING_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message.role == \"user\" %}USER"
    "{%- elif message.role == \"assistant\" %}ASSISTANT"
    "{%- elif message.role == \"tool\" %}TOOL"
    "{%- endif %}"
    "{%- endfor %}"
)


class ModelSmokeConfigTests(unittest.TestCase):
    def test_checked_in_config_is_valid_and_versioned(self) -> None:
        config = smoke.load_config(CONFIG_PATH)

        self.assertEqual(config.schema_version, smoke.CONFIG_SCHEMA_VERSION)
        self.assertEqual(
            [candidate.name for candidate in config.candidates],
            [
                "qwen2.5-3b-instruct",
                "qwen3-4b",
                "qwen2.5-1.5b-instruct",
                "qwen3-1.7b",
            ],
        )
        self.assertEqual(
            [candidate.role for candidate in config.candidates],
            ["primary_small", "primary_small", "scale_check", "scale_check"],
        )
        self.assertTrue(
            all(len(candidate.revision) == 40 for candidate in config.candidates)
        )
        self.assertNotIn("main", {candidate.revision for candidate in config.candidates})
        self.assertEqual(config.probe.warmup_runs, 2)
        self.assertEqual(config.probe.target_cuda_device_index, 0)
        self.assertEqual(
            {tool.mutative for tool in config.probe.tools}, {False, True}
        )
        self.assertEqual(config.lane.identity, "phase-a-windows-unsloth-trl024")
        self.assertEqual(config.lane.lock_path, "requirements-smoke.lock")
        self.assertEqual(
            config.lane.expected_lock_sha256,
            smoke._sha256_file(PROJECT_ROOT / config.lane.lock_path),
        )
        self.assertFalse(config.lane.m6_environment_factory_in_scope)
        self.assertEqual(
            config.lane.probe_implementation,
            {
                "P0": "implemented",
                "P1": "implemented",
                "P2": "implemented",
                "P3": "implemented",
                "P4": "implemented",
                "P5": "implemented",
                "P6": "implemented",
            },
        )
        # Resolved by D-048. A resolved gate must carry all three of scope,
        # decision, and bundles; the model rejects any partial resolution.
        self.assertEqual(config.release_gate.status, "resolved")
        self.assertEqual(config.release_gate.decision_record, "D-048")
        self.assertEqual(config.release_gate.eligible_bundles, ["qwen3"])
        self.assertTrue(config.release_gate.intended_release_scope)
        self.assertEqual(
            config.release_gate.expected_registry_sha256,
            smoke._sha256_file(PROJECT_ROOT / config.release_gate.registry_path),
        )
        self.assertEqual(
            config.release_gate.intended_release_scope,
            "public-portfolio-permissive",
        )
        self.assertTrue(config.release_gate.selection_allowed)

    def test_lane_probe_contract_is_strict(self) -> None:
        original = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        mutations = (
            ("missing-p0", lambda payload: payload["lane"]["probe_implementation"].pop("P0")),
            (
                "not-implemented-p6",
                lambda payload: payload["lane"]["probe_implementation"].update(
                    {"P6": "not_implemented"}
                ),
            ),
            (
                "m6-in-scope",
                lambda payload: payload["lane"].update(
                    {"m6_environment_factory_in_scope": True}
                ),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                payload = json.loads(json.dumps(original))
                mutate(payload)
                with self.assertRaises(ValidationError):
                    smoke.SmokeConfig.model_validate(payload)

    def test_release_gate_requires_an_explicit_resolved_decision(self) -> None:
        original = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

        # A pending gate may not name scope, decision, or bundles.
        invalid_pending = json.loads(json.dumps(original))
        invalid_pending["release_gate"].update(
            status="pending",
            intended_release_scope=None,
            decision_record=None,
            eligible_bundles=["qwen3"],
        )
        with self.assertRaises(ValidationError):
            smoke.SmokeConfig.model_validate(invalid_pending)

        # A resolved gate may not omit any of the three.
        for field in ("intended_release_scope", "decision_record"):
            partial = json.loads(json.dumps(original))
            partial["release_gate"][field] = None
            with self.assertRaises(ValidationError):
                smoke.SmokeConfig.model_validate(partial)
        no_bundles = json.loads(json.dumps(original))
        no_bundles["release_gate"]["eligible_bundles"] = []
        with self.assertRaises(ValidationError):
            smoke.SmokeConfig.model_validate(no_bundles)

        resolved = json.loads(json.dumps(original))
        resolved["release_gate"].update(
            {
                "status": "resolved",
                "intended_release_scope": "public research artifacts",
                "decision_record": "D-039",
                "eligible_bundles": ["qwen3"],
            }
        )
        parsed = smoke.SmokeConfig.model_validate(resolved)
        self.assertTrue(parsed.release_gate.selection_allowed)

    def test_invalid_model_id_is_rejected(self) -> None:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        payload["candidates"][0]["model_id"] = "../local-model"

        with self.assertRaises(ValidationError):
            smoke.SmokeConfig.model_validate(payload)

    def test_mutable_revision_and_invalid_role_are_rejected(self) -> None:
        original = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        mutations = (("revision", "main"), ("role", "other"))
        for field, value in mutations:
            with self.subTest(field=field):
                payload = json.loads(json.dumps(original))
                payload["candidates"][0][field] = value
                with self.assertRaises(ValidationError):
                    smoke.SmokeConfig.model_validate(payload)

    def test_invalid_or_missing_schema_version_fails(self) -> None:
        original = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        for version in (None, 2):
            with self.subTest(version=version):
                payload = dict(original)
                if version is None:
                    payload.pop("schema_version")
                else:
                    payload["schema_version"] = version
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "invalid.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(smoke.SmokeConfigError):
                        smoke.load_config(path)

    def test_config_lists_content_and_generation_work_are_bounded(self) -> None:
        original = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

        too_many_candidates = json.loads(json.dumps(original))
        too_many_candidates["candidates"].append(
            {
                **too_many_candidates["candidates"][0],
                "name": "fifth-candidate",
                "revision": "0" * 40,
            }
        )

        oversized_message = json.loads(json.dumps(original))
        oversized_message["probe"]["messages"][0]["content"] = (
            "x" * (smoke.MAX_MESSAGE_CHARS + 1)
        )

        too_many_cases = json.loads(json.dumps(original))
        too_many_cases["probe"]["generation_cases"] = []
        for index in range(smoke.MAX_GENERATION_CASES + 1):
            case = json.loads(
                json.dumps(original["probe"]["generation_cases"][index % 2])
            )
            case["name"] = f"bounded-case-{index}"
            too_many_cases["probe"]["generation_cases"].append(case)

        too_much_generation = json.loads(json.dumps(original))
        third_case = json.loads(
            json.dumps(original["probe"]["generation_cases"][0])
        )
        third_case["name"] = "third-generation-case"
        too_much_generation["probe"]["generation_cases"].append(third_case)
        too_much_generation["probe"]["warmup_runs"] = 10
        too_much_generation["probe"]["timed_runs"] = 20

        oversized_tool = json.loads(json.dumps(original))
        oversized_tool["probe"]["tools"][0]["function"]["parameters"][
            "properties"
        ]["expression"]["description"] = "x" * (smoke.MAX_TOOL_SCHEMA_CHARS + 1)

        for name, payload in (
            ("candidate count", too_many_candidates),
            ("message content", oversized_message),
            ("generation case count", too_many_cases),
            ("total generation work", too_much_generation),
            ("tool schema content", oversized_tool),
        ):
            with self.subTest(name=name), self.assertRaises(ValidationError):
                smoke.SmokeConfig.model_validate(payload)

    def test_config_json_rejects_duplicate_keys_nonfinite_numbers_and_size(self) -> None:
        raw = CONFIG_PATH.read_text(encoding="utf-8")
        invalid_payloads = {
            "duplicate": raw.replace(
                '"schema_version": 1,',
                '"schema_version": 1,\n  "schema_version": 1,',
                1,
            ),
            "nan": raw.replace('"seed": 17', '"seed": NaN', 1),
            "infinity": raw.replace('"top_p": 1.0', '"top_p": Infinity', 1),
            "oversized": " " * (smoke.MAX_CONFIG_BYTES + 1),
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, invalid in invalid_payloads.items():
                with self.subTest(name=name):
                    path = Path(directory) / f"{name}.json"
                    path.write_text(invalid, encoding="utf-8")
                    with self.assertRaises(smoke.SmokeConfigError):
                        smoke.load_config(path)


class ModelSmokeDryRunTests(unittest.TestCase):
    def test_default_mode_avoids_optional_imports_and_model_execution(self) -> None:
        optional_roots = set(smoke.OPTIONAL_LIBRARIES)
        real_import = builtins.__import__

        def guarded_import(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".", maxsplit=1)[0] in optional_roots:
                raise AssertionError(f"dry-run imported optional library {name}")
            return real_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dry-run.json"
            with (
                patch("builtins.__import__", side_effect=guarded_import),
                patch.object(smoke.shutil, "which", return_value=None),
                patch.object(
                    smoke,
                    "_execute_candidate",
                    side_effect=AssertionError("dry-run attempted model access"),
                ),
            ):
                exit_code = smoke.main(
                    ["--config", str(CONFIG_PATH), "--output", str(output)]
                )

            result = smoke.read_result(output)

        self.assertEqual(exit_code, 0)
        self.assertTrue(result.options.dry_run)
        self.assertFalse(result.options.allow_download)
        self.assertFalse(result.options.run_load)
        self.assertEqual(len(result.candidates), 4)
        self.assertTrue(all(len(candidate.probes) == 5 for candidate in result.candidates))
        self.assertTrue(
            all(
                probe.status == "planned"
                for candidate in result.candidates
                for probe in candidate.probes
            )
        )
        self.assertTrue(
            all(candidate.p6.status == "planned" for candidate in result.candidates)
        )
        self.assertTrue(
            all(candidate.environment_compatible is None for candidate in result.candidates)
        )
        self.assertFalse(result.selection_eligible)
        self.assertEqual(result.lane.identity, "phase-a-windows-unsloth-trl024")
        self.assertTrue(result.lane.lock_matches_expected)
        self.assertEqual(result.release_gate.status, "resolved")
        self.assertTrue(result.release_gate.selection_allowed)
        # A resolved gate still cannot make an unmeasured plan selectable.
        self.assertFalse(result.selection_eligible)
        self.assertTrue(
            all(
                probe.metrics == {}
                for candidate in result.candidates
                for probe in candidate.probes
            )
        )

    def test_candidate_filter_preserves_requested_order(self) -> None:
        config = smoke.load_config(CONFIG_PATH)
        config_bytes = CONFIG_PATH.read_bytes()
        with patch.object(smoke, "collect_hardware_facts") as hardware:
            hardware.return_value = smoke.HardwareFacts(
                python_version="3.11.0",
                python_implementation="CPython",
                executable="python",
                platform="test",
                system="test",
                release="test",
                machine="test",
                processor="test",
                logical_cpu_count=1,
                cuda=smoke.CudaFacts(
                    status="unavailable",
                    cuda_visible_devices=None,
                    nvidia_smi_path=None,
                    devices=[],
                ),
            )
            result = smoke.build_result(
                config,
                config_path=CONFIG_PATH,
                config_bytes=config_bytes,
                command=["python", "scripts/smoke_models.py"],
                run_load=False,
                allow_download=False,
                selected_names=["qwen3-1.7b", "qwen2.5-1.5b-instruct"],
            )

        self.assertEqual(
            [candidate.name for candidate in result.candidates],
            ["qwen3-1.7b", "qwen2.5-1.5b-instruct"],
        )


class ModelSmokeProbeTests(unittest.TestCase):
    def test_generation_quality_is_ranked_without_failing_compatibility(self) -> None:
        class FakeVector:
            shape = (1, 2)

            def to(self, device: object) -> "FakeVector":
                return self

        class FakeTokenSlice:
            def detach(self) -> "FakeTokenSlice":
                return self

            def cpu(self) -> "FakeTokenSlice":
                return self

            @staticmethod
            def tolist() -> list[int]:
                return [7]

        class FakeGenerated:
            def __getitem__(self, key: object) -> FakeTokenSlice:
                return FakeTokenSlice()

        class FakeModel:
            @staticmethod
            def eval() -> None:
                return None

            @staticmethod
            def generate(**kwargs: object) -> FakeGenerated:
                return FakeGenerated()

        class FakeTokenizer:
            eos_token_id = 0

            @staticmethod
            def apply_chat_template(*args: object, **kwargs: object) -> str:
                return "rendered prompt"

            @staticmethod
            def decode(*args: object, **kwargs: object) -> str:
                return "I would use a tool, but this is not a tool call."

            def __call__(self, *args: object, **kwargs: object) -> dict[str, FakeVector]:
                return {"input_ids": FakeVector()}

        fake_torch = SimpleNamespace(
            device=lambda value: value,
            manual_seed=lambda seed: None,
            inference_mode=nullcontext,
            cuda=SimpleNamespace(
                manual_seed_all=lambda seed: None,
                synchronize=lambda device_index: None,
            ),
        )
        probe = smoke.load_config(CONFIG_PATH).probe
        plan = {
            item.name: item.plan for item in smoke.probe_plans(probe)
        }["deterministic_generation"]

        result = smoke._run_generation_probe(
            torch=fake_torch,
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            probe=probe,
            plan=plan,
        )

        self.assertEqual(result.status, "passed")
        self.assertTrue(all(result.metrics["compatibility_checks"].values()))
        self.assertFalse(
            result.metrics["quality_observations"][
                "every_output_has_exactly_one_expected_dispatchable_call"
            ]
        )
        self.assertEqual(result.metrics["ranking_metrics"]["zero_tool_call_rate"], 1.0)
        self.assertFalse(
            result.metrics["tool_call_quality_gates_environment_compatibility"]
        )

    def test_environment_compatibility_requires_every_p1_through_p5_probe(self) -> None:
        def probe_result(name: str, status: str) -> smoke.ProbeResult:
            return smoke.ProbeResult.model_validate(
                {"name": name, "status": status, "plan": {}, "metrics": {}}
            )

        candidate = smoke._candidate_result(
            name="candidate",
            bundle="qwen3",
            role="primary_small",
            model_id="owner/model",
            requested_revision="0" * 40,
            resolved_revision="0" * 40,
            probes=[
                probe_result("tool_chat_template", "passed"),
                probe_result("four_bit_load", "passed"),
                probe_result("deterministic_generation", "failed"),
                probe_result("training_stack_imports", "passed"),
                probe_result("training_template_masking", "passed"),
            ],
        )
        self.assertFalse(candidate.environment_compatible)
        self.assertFalse(candidate.selection_eligible)

        passing_probes = [
            probe_result(probe.name, "passed") for probe in candidate.probes
        ]
        technically_eligible = smoke._candidate_result(
            name="candidate",
            bundle="qwen3",
            role="primary_small",
            model_id="owner/model",
            requested_revision="0" * 40,
            resolved_revision="0" * 40,
            probes=passing_probes,
            p6=smoke.MinimalTrainingResult(
                status="passed",
                executed=True,
                passed=True,
                plan=smoke._minimal_training_plan(),
                metrics={"assistant_only_loss": 1.0},
            ),
        )
        self.assertTrue(technically_eligible.selection_eligible)

        failed_p3 = technically_eligible.model_dump(mode="python")
        failed_p3["probes"][1]["status"] = "failed"
        failed_p3["environment_compatible"] = False
        failed_p3["selection_eligible"] = False
        self.assertFalse(
            smoke.CandidateResult.model_validate(failed_p3).environment_compatible
        )

    def test_p6_reuses_exact_p5_batch_and_builds_assistant_only_labels(self) -> None:
        p5 = smoke.ProbeResult(
            name="training_template_masking",
            status="passed",
            plan={},
            metrics={
                "training_input_ids": [10, 11, 12, 13, 14],
                "assistant_token_mask": [0, 0, 1, 1, 0],
            },
        )

        input_ids, assistant_mask, labels = smoke._p5_training_batch(p5)

        self.assertEqual(input_ids, [10, 11, 12, 13, 14])
        self.assertEqual(assistant_mask, [0, 0, 1, 1, 0])
        self.assertEqual(labels, [-100, -100, 12, 13, -100])
        self.assertEqual(input_ids, p5.metrics["training_input_ids"])
        self.assertEqual(assistant_mask, p5.metrics["assistant_token_mask"])

        invalid_batches = (
            ([1], [1]),
            ([1, 2], [0, 0]),
            ([1, 2], [1]),
            ([1, 2], [0, 2]),
            (list(range(smoke.MAX_P6_TOKENS + 1)), [0] * smoke.MAX_P6_TOKENS + [1]),
        )
        for ids, mask in invalid_batches:
            with self.subTest(length=len(ids), mask=mask[:3]):
                with self.assertRaises(ValueError):
                    smoke._assistant_only_labels(ids, mask)

    def test_unsloth_loader_binds_exact_revision_and_training_tokenizer(self) -> None:
        revision = "a" * 40
        quantization_config = object()
        compute_dtype = object()
        calls: list[dict[str, object]] = []

        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "snapshots" / revision
            snapshot.mkdir(parents=True)
            tokenizer_config = snapshot / "tokenizer_config.json"
            tokenizer_config.write_text("{}", encoding="utf-8")

            class FastLanguageModel:
                @staticmethod
                def from_pretrained(**kwargs: object) -> tuple[object, object]:
                    calls.append(kwargs)
                    tokenizer = SimpleNamespace()
                    model = SimpleNamespace(_saved_temp_tokenizer=tokenizer)
                    return model, tokenizer

            unsloth = SimpleNamespace(FastLanguageModel=FastLanguageModel)
            hub = SimpleNamespace(
                hf_hub_download=lambda **kwargs: str(tokenizer_config)
            )

            def imported(name: str) -> object:
                return unsloth if name == "unsloth" else hub

            with patch.object(smoke.importlib, "import_module", side_effect=imported):
                model, tokenizer, evidence = smoke._load_unsloth_four_bit_model(
                    model_id="owner/model",
                    revision=revision,
                    quantization_config=quantization_config,
                    compute_dtype=compute_dtype,
                    target_cuda_device_index=2,
                    seed=17,
                )

        self.assertIs(model._saved_temp_tokenizer, tokenizer)
        self.assertEqual(evidence["source"], "huggingface_hub_local_snapshot")
        self.assertEqual(len(calls), 1)
        kwargs = calls[0]
        self.assertEqual(kwargs["model_name"], "owner/model")
        self.assertEqual(kwargs["revision"], revision)
        self.assertEqual(kwargs["device_map"], {"": 2})
        self.assertEqual(kwargs["max_seq_length"], smoke.MAX_P6_TOKENS)
        self.assertIs(kwargs["quantization_config"], quantization_config)
        self.assertIs(kwargs["dtype"], compute_dtype)
        self.assertTrue(kwargs["load_in_4bit"])
        self.assertTrue(kwargs["use_exact_model_name"])
        self.assertFalse(kwargs["fast_inference"])

        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "snapshots" / revision
            snapshot.mkdir(parents=True)
            tokenizer_config = snapshot / "tokenizer_config.json"
            tokenizer_config.write_text("{}", encoding="utf-8")

            class MissingAttachmentFastLanguageModel:
                @staticmethod
                def from_pretrained(**kwargs: object) -> tuple[object, object]:
                    return SimpleNamespace(), SimpleNamespace()

            broken_unsloth = SimpleNamespace(
                FastLanguageModel=MissingAttachmentFastLanguageModel
            )
            hub = SimpleNamespace(
                hf_hub_download=lambda **kwargs: str(tokenizer_config)
            )

            def imported(name: str) -> object:
                return broken_unsloth if name == "unsloth" else hub

            with patch.object(
                smoke.importlib, "import_module", side_effect=imported
            ):
                with self.assertRaisesRegex(ValueError, "attach its tokenizer"):
                    smoke._load_unsloth_four_bit_model(
                        model_id="owner/model",
                        revision=revision,
                        quantization_config=quantization_config,
                        compute_dtype=compute_dtype,
                        target_cuda_device_index=2,
                        seed=17,
                    )

    def test_immutable_cache_evidence_requires_exact_snapshot(self) -> None:
        revision = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            exact = Path(temporary) / "snapshots" / revision / "config.json"
            exact.parent.mkdir(parents=True)
            exact.write_text("{}", encoding="utf-8")
            calls: list[dict[str, object]] = []
            hub = SimpleNamespace(
                hf_hub_download=lambda **kwargs: (
                    calls.append(kwargs) or str(exact)
                )
            )
            with patch.object(smoke.importlib, "import_module", return_value=hub):
                evidence = smoke._immutable_cache_revision_evidence(
                    model_id="owner/model",
                    revision=revision,
                    filename="config.json",
                    artifact=SimpleNamespace(),
                )

            self.assertTrue(evidence["snapshot_revision_matches_requested"])
            self.assertFalse(evidence["artifact_revision_exposed"])
            self.assertNotIn(temporary, json.dumps(evidence))
            self.assertEqual(
                calls,
                [{
                    "repo_id": "owner/model",
                    "filename": "config.json",
                    "revision": revision,
                    "local_files_only": True,
                }],
            )

            wrong = Path(temporary) / "snapshots" / ("b" * 40) / "config.json"
            wrong.parent.mkdir(parents=True)
            wrong.write_text("{}", encoding="utf-8")
            wrong_hub = SimpleNamespace(
                hf_hub_download=lambda **kwargs: str(wrong)
            )
            with patch.object(
                smoke.importlib, "import_module", return_value=wrong_hub
            ):
                with self.assertRaisesRegex(ValueError, "snapshot revision mismatch"):
                    smoke._immutable_cache_revision_evidence(
                        model_id="owner/model",
                        revision=revision,
                        filename="config.json",
                        artifact=SimpleNamespace(),
                    )

            exact_hub = SimpleNamespace(
                hf_hub_download=lambda **kwargs: str(exact)
            )
            with patch.object(
                smoke.importlib, "import_module", return_value=exact_hub
            ):
                with self.assertRaisesRegex(ValueError, "artifact revision mismatch"):
                    smoke._immutable_cache_revision_evidence(
                        model_id="owner/model",
                        revision=revision,
                        filename="config.json",
                        artifact=SimpleNamespace(_commit_hash="b" * 40),
                    )

    def test_p6_prerequisites_fail_closed(self) -> None:
        def result(name: str, status: str) -> smoke.ProbeResult:
            return smoke.ProbeResult.model_validate(
                {"name": name, "status": status, "plan": {}, "metrics": {}}
            )

        passing = [
            result("four_bit_load", "passed"),
            result("training_stack_imports", "passed"),
            result("training_template_masking", "passed"),
        ]
        self.assertIsNone(smoke._p6_prerequisite_error(passing))

        passing[1] = result("training_stack_imports", "unavailable")
        error = smoke._p6_prerequisite_error(passing)
        self.assertIsNotNone(error)
        self.assertIn("training-stack imports=unavailable", error or "")
        skipped = smoke._skipped_minimal_training(error or "missing")
        self.assertEqual(skipped.status, "skipped")
        self.assertFalse(skipped.executed)
        self.assertFalse(skipped.passed)

    def test_reference_adapter_context_is_executed_and_restored(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.enabled = True
                self.events: list[str] = []

            def get_model_status(self) -> SimpleNamespace:
                return SimpleNamespace(enabled=self.enabled)

            def disable_adapter(self) -> object:
                model = self

                class Context:
                    def __enter__(self) -> None:
                        model.events.append("enter")
                        model.enabled = False

                    def __exit__(self, *args: object) -> None:
                        model.enabled = True
                        model.events.append("exit")

                return Context()

        model = FakeModel()

        def operation() -> str:
            model.events.append("operation")
            self.assertFalse(model.enabled)
            return "reference-logps"

        value, checks = smoke._run_with_adapters_disabled(model, operation)

        self.assertEqual(value, "reference-logps")
        self.assertTrue(all(checks.values()))
        self.assertTrue(model.enabled)
        self.assertEqual(model.events, ["enter", "operation", "exit"])

    def test_p6_result_status_and_metrics_are_fail_closed(self) -> None:
        passed = smoke.MinimalTrainingResult(
            status="passed",
            executed=True,
            passed=True,
            plan=smoke._minimal_training_plan(),
            metrics={"assistant_only_loss": 1.0},
        )
        self.assertTrue(passed.passed)

        invalid_payloads = (
            {"status": "passed", "executed": False, "passed": True},
            {"status": "failed", "executed": True, "passed": False},
            {"status": "skipped", "executed": False, "passed": False},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValidationError):
                    smoke.MinimalTrainingResult.model_validate(payload)

    def test_model_placement_requires_one_exact_cuda_device(self) -> None:
        class FakeModel:
            def __init__(self, device_map: dict[str, object], devices: list[str]) -> None:
                self.hf_device_map = device_map
                self._parameters = [
                    SimpleNamespace(device=device, dtype="torch.float16")
                    for device in devices
                ]

            def parameters(self) -> list[SimpleNamespace]:
                return self._parameters

        cases = (
            (FakeModel({"": 0}, ["cuda:0"]), True, False),
            (FakeModel({}, ["cuda:0"]), True, False),
            (FakeModel({}, ["cpu"]), False, True),
            (FakeModel({}, ["meta"]), False, True),
            (FakeModel({}, ["cuda:0", "cpu"]), False, True),
            (FakeModel({"layer": "cpu"}, ["cuda:0", "cpu"]), False, True),
            (FakeModel({"layer": "disk"}, ["cuda:0"]), False, True),
            (
                FakeModel({"layer.0": 0, "layer.1": 1}, ["cuda:0", "cuda:1"]),
                False,
                False,
            ),
            (FakeModel({"": 0}, ["cuda:1"]), False, False),
        )
        for model, expected_pass, expected_offload in cases:
            with self.subTest(device_map=model.hf_device_map):
                metrics, passed = smoke._inspect_model_placement(
                    model, target_cuda_device_index=0
                )
                self.assertEqual(passed, expected_pass)
                self.assertEqual(metrics["offload_detected"], expected_offload)
                self.assertIn("actual_parameter_dtype", metrics)

        empty_metrics, empty_passed = smoke._inspect_model_placement(
            FakeModel({}, ["cuda:0"]), target_cuda_device_index=0
        )
        self.assertTrue(empty_passed)
        self.assertTrue(empty_metrics["device_map_attribute_present"])
        self.assertTrue(empty_metrics["device_map_format_valid"])
        self.assertFalse(empty_metrics["device_map_has_entries"])
        self.assertEqual(empty_metrics["device_map_entry_count"], 0)
        self.assertIsNone(empty_metrics["every_device_map_entry_on_target"])
        self.assertTrue(
            empty_metrics["device_map_does_not_contradict_parameters"]
        )

        mapless_model = SimpleNamespace(
            parameters=lambda: [
                SimpleNamespace(device="cuda:0", dtype="torch.float16")
            ]
        )
        mapless_metrics, mapless_passed = smoke._inspect_model_placement(
            mapless_model, target_cuda_device_index=0
        )
        self.assertTrue(mapless_passed)
        self.assertFalse(mapless_metrics["device_map_attribute_present"])
        self.assertFalse(mapless_metrics["device_map_has_entries"])
        self.assertIsNone(mapless_metrics["every_device_map_entry_on_target"])

        malformed_model = SimpleNamespace(
            hf_device_map="cuda:0",
            parameters=lambda: [
                SimpleNamespace(device="cuda:0", dtype="torch.float16")
            ],
        )
        malformed_metrics, malformed_passed = smoke._inspect_model_placement(
            malformed_model, target_cuda_device_index=0
        )
        self.assertFalse(malformed_passed)
        self.assertFalse(malformed_metrics["device_map_format_valid"])

    def test_effective_nf4_quantization_requires_runtime_and_class_evidence(self) -> None:
        class Params4bit:
            __module__ = "bitsandbytes.nn.modules"

            def __init__(self) -> None:
                self.device = "cuda:0"
                self.dtype = "torch.float16"

        class Linear4bit:
            __module__ = "bitsandbytes.nn.modules"

        class FakeModel:
            def __init__(
                self,
                *,
                loaded: bool = True,
                quantization: str = "nf4",
                double_quant: bool = True,
                include_class_evidence: bool = True,
            ) -> None:
                self.is_loaded_in_4bit = loaded
                self.quantization_config = SimpleNamespace(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=quantization,
                    bnb_4bit_use_double_quant=double_quant,
                    bnb_4bit_compute_dtype="torch.float16",
                )
                self._parameters = [Params4bit()] if include_class_evidence else [
                    SimpleNamespace(device="cuda:0", dtype="torch.float16")
                ]
                self._modules = [Linear4bit()] if include_class_evidence else [self]

            def parameters(self) -> list[object]:
                return self._parameters

            def modules(self) -> list[object]:
                return self._modules

        cases = (
            (FakeModel(), True),
            (FakeModel(loaded=False), False),
            (FakeModel(quantization="fp4"), False),
            (FakeModel(double_quant=False), False),
            (FakeModel(include_class_evidence=False), False),
        )
        for model, expected_pass in cases:
            with self.subTest(model=model):
                metrics, passed = smoke._inspect_effective_quantization(
                    model,
                    expected_quantization="nf4",
                    expected_double_quant=True,
                    expected_compute_dtype="torch.float16",
                )
                self.assertEqual(passed, expected_pass)
                self.assertIn("effective_quantization_checks", metrics)

    def test_strict_tool_scoring_uses_normalized_parser_and_registered_schema(self) -> None:
        parser_module = importlib.import_module("agent.parser")
        probe = smoke.load_config(CONFIG_PATH).probe
        valid = (
            '<tool_call>{"name":"calculator","arguments":'
            '{"expression":"17 * 23"}}</tool_call>'
        )
        wrong_schema = (
            '<tool_call>{"name":"calculator","arguments":'
            '{"expression":391}}</tool_call>'
        )
        extra_valid_call = valid + (
            '<tool_call>{"name":"update_order","arguments":'
            '{"order_id":"ORD-17","status":"shipped"}}</tool_call>'
        )

        valid_score = smoke._score_tool_output(
            valid,
            expected_tool="calculator",
            probe=probe,
            parser_module=parser_module,
        )
        wrong_score = smoke._score_tool_output(
            wrong_schema,
            expected_tool="calculator",
            probe=probe,
            parser_module=parser_module,
        )
        prose_score = smoke._score_tool_output(
            "I would use the calculator.",
            expected_tool="calculator",
            probe=probe,
            parser_module=parser_module,
        )
        extra_score = smoke._score_tool_output(
            extra_valid_call,
            expected_tool="calculator",
            probe=probe,
            parser_module=parser_module,
        )

        self.assertTrue(valid_score["strict_json_parse_success"])
        self.assertTrue(valid_score["registered_schema_valid_output"])
        self.assertTrue(valid_score["expected_tool_dispatchable"])
        self.assertTrue(valid_score["exactly_one_expected_dispatchable_call"])
        self.assertFalse(valid_score["handler_or_gate_executed"])
        self.assertFalse(valid_score["zero_tool_call"])
        self.assertTrue(wrong_score["strict_json_parse_success"])
        self.assertFalse(wrong_score["registered_schema_valid_output"])
        self.assertFalse(wrong_score["has_dispatchable_call"])
        self.assertTrue(prose_score["zero_tool_call"])
        self.assertFalse(prose_score["has_dispatchable_call"])
        self.assertEqual(extra_score["registered_schema_valid_call_count"], 2)
        self.assertFalse(extra_score["exactly_one_expected_dispatchable_call"])
        self.assertFalse(extra_score["handler_or_gate_executed"])

    def test_decoded_output_retention_is_bounded_and_auditable(self) -> None:
        bounded = smoke._bounded_text("abcdef", 4)

        self.assertEqual(bounded["text"], "abcd")
        self.assertTrue(bounded["truncated"])
        self.assertEqual(bounded["character_count"], 6)

    def test_template_token_and_error_text_limits_fail_closed(self) -> None:
        probe = smoke.load_config(CONFIG_PATH).probe

        class OversizedPrefixTokenizer:
            def apply_chat_template(self, *args: object, **kwargs: object) -> object:
                return list(range(smoke.MAX_TEMPLATE_TOKENS + 1))

        diagnostic = smoke._prefix_diagnostic(
            tokenizer=OversizedPrefixTokenizer(),
            template="template",
            probe=probe,
            tools=smoke._tool_payloads(probe),
        )
        self.assertEqual(diagnostic["status"], "error")
        self.assertIn("token limit", diagnostic["error"])

        bounded_error = smoke._error_text(
            ValueError("x" * (smoke.MAX_ERROR_CHARS * 2))
        )
        self.assertEqual(len(bounded_error), smoke.MAX_ERROR_CHARS)
        self.assertTrue(bounded_error.endswith("...[truncated]"))

    def test_training_prompt_persistence_limit_fails_before_metrics(self) -> None:
        probe = smoke.load_config(CONFIG_PATH).probe

        class OversizedRenderTokenizer:
            def apply_chat_template(self, *args: object, **kwargs: object) -> object:
                if kwargs.get("tokenize"):
                    return [1]
                return "x" * (smoke.MAX_PERSISTED_PROMPT_CHARS + 1)

        plan = {
            item.name: item.plan for item in smoke.probe_plans(probe)
        }["training_template_masking"]
        result = smoke._run_training_template_probe(
            tokenizer=OversizedRenderTokenizer(),
            native_template=QWEN_NATIVE_TRAINING_TEMPLATE,
            probe=probe,
            plan=plan,
            applied_demotions=(),
            m6_in_scope=False,
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metrics, {})
        self.assertIn("character limit", result.error or "")

    def test_import_success_alone_cannot_pass_training_mask(self) -> None:
        class MasklessTokenizer:
            def apply_chat_template(self, *args: object, **kwargs: object) -> object:
                if kwargs.get("return_dict"):
                    return {"input_ids": [1, 2, 3]}
                if kwargs.get("tokenize"):
                    return [1, 2]
                return (
                    "MASK_SYSTEM_7A31 MASK_USER_4B92 "
                    "MASK_ASSISTANT_8C53 MASK_TOOL_RESULT_2D74"
                )

        probe = smoke.load_config(CONFIG_PATH).probe
        plan = {
            item.name: item.plan for item in smoke.probe_plans(probe)
        }["training_template_masking"]
        result = smoke._run_training_template_probe(
            tokenizer=MasklessTokenizer(),
            native_template=QWEN_NATIVE_TRAINING_TEMPLATE,
            probe=probe,
            plan=plan,
            applied_demotions=(),
            m6_in_scope=False,
        )

        self.assertEqual(result.status, "failed")
        self.assertIn("assistant mask", result.error or "")

    def test_training_mask_requires_complete_generation_spans(self) -> None:
        probe = smoke.load_config(CONFIG_PATH).probe

        class MaskingTokenizer:
            def __init__(self, *, complete_mask: bool) -> None:
                self.complete_mask = complete_mask

            @staticmethod
            def _message_text(message: dict[str, object]) -> str:
                return json.dumps(message, sort_keys=True, separators=(",", ":"))

            def _render(
                self,
                messages: list[dict[str, object]],
                template: object,
            ) -> tuple[str, tuple[int, int] | None]:
                instrumented = (
                    isinstance(template, str)
                    and smoke.GENERATION_START_MARKER in template
                )
                parts: list[str] = []
                assistant_span: tuple[int, int] | None = None
                cursor = 0
                for message in messages:
                    text = self._message_text(message)
                    if message["role"] == "assistant":
                        assistant_span = (cursor, cursor + len(text))
                        if instrumented:
                            text = (
                                smoke.GENERATION_START_MARKER
                                + text
                                + smoke.GENERATION_END_MARKER
                            )
                    part = text + "\n"
                    parts.append(part)
                    cursor += len(part)
                return "".join(parts), assistant_span

            def apply_chat_template(
                self, messages: list[dict[str, object]], **kwargs: object
            ) -> object:
                rendered, assistant_span = self._render(
                    messages, kwargs.get("chat_template")
                )
                if not kwargs.get("tokenize"):
                    return rendered
                token_ids = [ord(character) for character in rendered]
                if kwargs.get("return_dict"):
                    assert assistant_span is not None
                    start, stop = assistant_span
                    if not self.complete_mask:
                        assistant = probe.mask_sentinels["assistant"]
                        start = rendered.index(assistant)
                        stop = start + len(assistant)
                    mask = [int(start <= index < stop) for index in range(len(rendered))]
                    return {"input_ids": token_ids, "assistant_masks": mask}
                return token_ids

            def __call__(self, rendered: str, **kwargs: object) -> dict[str, object]:
                return {
                    "input_ids": [ord(character) for character in rendered],
                    "offset_mapping": [
                        (index, index + 1) for index in range(len(rendered))
                    ],
                }

        plan = {
            item.name: item.plan for item in smoke.probe_plans(probe)
        }["training_template_masking"]
        results = []
        for complete_mask in (True, False):
            results.append(
                smoke._run_training_template_probe(
                    tokenizer=MaskingTokenizer(complete_mask=complete_mask),
                    native_template=QWEN_NATIVE_TRAINING_TEMPLATE,
                    probe=probe,
                    plan=plan,
                    applied_demotions=(),
                    m6_in_scope=False,
                )
            )

        correct, incomplete = results
        self.assertEqual(correct.status, "passed")
        self.assertTrue(
            correct.metrics["checks"]["prefix_preserved_after_tool_observation"]
        )
        self.assertTrue(
            correct.metrics["checks"][
                "assistant_mask_exactly_matches_generation_spans"
            ]
        )
        self.assertEqual(
            correct.metrics["assistant_token_mask"],
            correct.metrics["expected_assistant_token_mask"],
        )
        self.assertNotEqual(
            correct.metrics["native_chat_template_sha256"],
            correct.metrics["training_chat_template_sha256"],
        )
        self.assertEqual(
            correct.metrics["training_template_source"],
            "project_owned_qwen_assistant_branch_generation_wrapper",
        )
        self.assertTrue(
            correct.metrics["checks"]["project_template_render_matches_native"]
        )
        self.assertTrue(
            correct.metrics["checks"]["project_template_token_ids_match_native"]
        )
        self.assertEqual(incomplete.status, "failed")
        self.assertFalse(
            incomplete.metrics["checks"][
                "assistant_mask_exactly_matches_generation_spans"
            ]
        )

    def test_qwen_training_template_patch_is_exact_and_rejects_ambiguity(self) -> None:
        patched = smoke._build_qwen_training_template(
            QWEN_NATIVE_TRAINING_TEMPLATE
        )
        self.assertEqual(patched.count("{% generation %}"), 1)
        self.assertEqual(patched.count("{% endgeneration %}"), 1)
        assistant = patched.index('{%- elif message.role == "assistant" %}')
        start = patched.index("{% generation %}")
        stop = patched.index("{% endgeneration %}")
        tool = patched.index('{%- elif message.role == "tool" %}')
        self.assertLess(assistant, start)
        self.assertLess(start, stop)
        self.assertLess(stop, tool)

        ambiguous = QWEN_NATIVE_TRAINING_TEMPLATE.replace(
            '{%- elif message.role == "tool" %}',
            '{%- elif message.role == "assistant" %}SECOND'
            '{%- elif message.role == "tool" %}',
        )
        with self.assertRaisesRegex(ValueError, "exactly one assistant branch"):
            smoke._build_qwen_training_template(ambiguous)

        already_patched = QWEN_NATIVE_TRAINING_TEMPLATE.replace(
            "ASSISTANT", "{% generation %}ASSISTANT{% endgeneration %}"
        )
        with self.assertRaisesRegex(ValueError, "already contains generation tags"):
            smoke._build_qwen_training_template(already_patched)

    def test_generation_end_marker_follows_indented_qwen_block_tag(self) -> None:
        patched = smoke._build_qwen_training_template(
            QWEN_NATIVE_TRAINING_TEMPLATE.replace(
                "ASSISTANT", "ASSISTANT\n    "
            )
        )
        instrumented = smoke._instrument_generation_blocks(patched)
        end_tag = instrumented.index("{% endgeneration %}")
        end_marker = instrumented.index(smoke.GENERATION_END_MARKER)
        tool_branch = instrumented.index(
            '{%- elif message.role == "tool" %}'
        )
        self.assertLess(end_tag, end_marker)
        self.assertLess(end_marker, tool_branch)

    def test_plans_separate_true_gates_preflight_and_ranking(self) -> None:
        plans = {
            item.name: item.plan
            for item in smoke.probe_plans(smoke.load_config(CONFIG_PATH).probe)
        }

        self.assertIn(
            "per-device peak CUDA reserved bytes",
            plans["four_bit_load"]["measurements"],
        )
        self.assertIn("hard_gate", plans["four_bit_load"])
        self.assertIn("preflight", plans["training_stack_imports"])
        self.assertNotIn("hard_gate", plans["training_stack_imports"])
        self.assertIn("ranking_observations", plans["deterministic_generation"])
        self.assertTrue(
            any(
                "native chat-template SHA-256" in item
                for item in plans["tool_chat_template"]["record"]
            )
        )
        self.assertEqual(
            plans["training_template_masking"]["note"],
            "imports alone cannot pass this probe",
        )


class ModelSmokePersistenceTests(unittest.TestCase):
    def test_result_json_rejects_duplicate_keys_and_nonfinite_numbers(self) -> None:
        invalid_payloads = {
            "duplicate": '{"schema_version":1,"schema_version":1}',
            "nan": '{"schema_version":NaN}',
            "infinity": '{"schema_version":Infinity}',
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, payload in invalid_payloads.items():
                with self.subTest(name=name):
                    path = Path(directory) / f"{name}.json"
                    path.write_text(payload, encoding="utf-8")
                    with self.assertRaises(smoke.SmokeResultError):
                        smoke.read_result(path)

    def test_result_schema_and_atomic_write_round_trip(self) -> None:
        config = smoke.load_config(CONFIG_PATH)
        with (
            patch.object(smoke, "collect_hardware_facts") as hardware,
            patch.object(
                smoke,
                "collect_library_versions",
                return_value={name: None for name in smoke.OPTIONAL_LIBRARIES},
            ),
        ):
            hardware.return_value = smoke.HardwareFacts(
                python_version="3.11.0",
                python_implementation="CPython",
                executable="python",
                platform="test-platform",
                system="test-system",
                release="test-release",
                machine="test-machine",
                processor="test-processor",
                logical_cpu_count=2,
                cuda=smoke.CudaFacts(
                    status="unavailable",
                    cuda_visible_devices=None,
                    nvidia_smi_path=None,
                    devices=[],
                ),
            )
            result = smoke.build_result(
                config,
                config_path=CONFIG_PATH,
                config_bytes=CONFIG_PATH.read_bytes(),
                command=["python", "scripts/smoke_models.py"],
                run_load=False,
                allow_download=False,
                selected_names=[],
            )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "result.json"
            output.parent.mkdir()
            output.write_text("old contents", encoding="utf-8")

            smoke.write_result_atomic(result, output)
            restored = smoke.read_result(output)
            leftovers = list(output.parent.glob(f".{output.name}.*.tmp"))

        self.assertEqual(restored, result)
        self.assertEqual(restored.schema_version, smoke.RESULT_SCHEMA_VERSION)
        self.assertEqual(
            restored.source_identity.smoke_config_sha256,
            restored.config_sha256,
        )
        self.assertEqual(
            restored.source_identity.smoke_script_sha256,
            smoke._sha256_file(Path(smoke.__file__).resolve()),
        )
        self.assertEqual(len(restored.source_identity.git_commit_sha), 40)
        self.assertEqual(
            restored.lane.actual_lock_sha256,
            smoke._sha256_file(PROJECT_ROOT / "requirements-smoke.lock"),
        )
        self.assertTrue(restored.lane.lock_matches_expected)
        self.assertFalse(restored.selection_eligible)
        self.assertTrue(
            all(candidate.p6.status == "planned" for candidate in restored.candidates)
        )
        self.assertEqual(leftovers, [])

    def test_selection_eligibility_requires_executed_passed_p6(self) -> None:
        config = smoke.load_config(CONFIG_PATH)
        result = smoke.build_result(
            config,
            config_path=CONFIG_PATH,
            config_bytes=CONFIG_PATH.read_bytes(),
            command=[],
            run_load=False,
            allow_download=False,
            selected_names=["qwen3-1.7b"],
        )
        candidate = result.candidates[0].model_dump(mode="python")
        candidate["selection_eligible"] = True

        with self.assertRaises(ValidationError):
            smoke.CandidateResult.model_validate(candidate)

        missing_p6 = result.candidates[0].model_dump(mode="python")
        missing_p6.pop("p6")
        with self.assertRaises(ValidationError):
            smoke.CandidateResult.model_validate(missing_p6)

        inconsistent_p6 = result.candidates[0].p6.model_dump(mode="python")
        inconsistent_p6.update({"status": "passed", "executed": False, "passed": True})
        with self.assertRaises(ValidationError):
            smoke.MinimalTrainingResult.model_validate(inconsistent_p6)

    def test_top_level_selection_requires_four_candidates_and_release_gate(self) -> None:
        config = smoke.load_config(CONFIG_PATH)
        full_result = smoke.build_result(
            config,
            config_path=CONFIG_PATH,
            config_bytes=CONFIG_PATH.read_bytes(),
            command=[],
            run_load=False,
            allow_download=False,
            selected_names=[],
        )

        technical_candidates = []
        for candidate in full_result.candidates:
            payload = candidate.model_dump(mode="python")
            for probe in payload["probes"]:
                probe["status"] = "passed"
            payload["p6"] = smoke.MinimalTrainingResult(
                status="passed",
                executed=True,
                passed=True,
                plan=smoke._minimal_training_plan(),
                metrics={"assistant_only_loss": 1.0},
            ).model_dump(mode="python")
            payload["environment_compatible"] = True
            payload["selection_eligible"] = True
            technical_candidates.append(
                smoke.CandidateResult.model_validate(payload)
            )

        # The repository config is resolved (D-048), so the pending half of
        # this test supplies its own pending gate rather than depending on the
        # committed state.
        pending_payload = full_result.model_dump(mode="python")
        pending_payload["release_gate"].update(
            {
                "status": "pending",
                "intended_release_scope": None,
                "decision_record": None,
                "eligible_bundles": [],
            }
        )
        pending_payload["candidates"] = [
            candidate.model_dump(mode="python")
            for candidate in technical_candidates
        ]
        pending_payload["candidates_with_demoted_gate_failures"] = sorted(
            candidate.name
            for candidate in technical_candidates
            if candidate.demoted_gate_failures
        )
        pending_payload["selection_eligible"] = False
        pending = smoke.SmokeResult.model_validate(pending_payload)
        self.assertFalse(pending.selection_eligible)
        pending_payload["selection_eligible"] = True
        with self.assertRaises(ValidationError):
            smoke.SmokeResult.model_validate(pending_payload)

        resolved_payload = full_result.model_dump(mode="python")
        resolved_payload["release_gate"].update(
            {
                "status": "resolved",
                "intended_release_scope": "public research artifacts",
                "decision_record": "D-039",
                "eligible_bundles": ["qwen3"],
            }
        )
        resolved_payload["candidates"] = [
            candidate.model_dump(mode="python")
            for candidate in technical_candidates
        ]
        resolved_payload["selection_eligible"] = True
        resolved = smoke.SmokeResult.model_validate(resolved_payload)
        self.assertTrue(resolved.selection_eligible)

        one_candidate_payload = full_result.model_dump(mode="python")
        one_candidate_payload["release_gate"].update(
            {
                "status": "resolved",
                "intended_release_scope": "public research artifacts",
                "decision_record": "D-039",
                "eligible_bundles": ["qwen3"],
            }
        )
        one_candidate_payload["candidates"] = [
            technical_candidates[0].model_dump(mode="python")
        ]
        one_candidate_payload["selection_eligible"] = False
        one_candidate = smoke.SmokeResult.model_validate(one_candidate_payload)
        self.assertFalse(one_candidate.selection_eligible)
        one_candidate_payload["selection_eligible"] = True
        with self.assertRaises(ValidationError):
            smoke.SmokeResult.model_validate(one_candidate_payload)

    def test_mutated_result_is_rejected_before_replacing_destination(self) -> None:
        config = smoke.load_config(CONFIG_PATH)
        with (
            patch.object(smoke, "collect_hardware_facts") as hardware,
            patch.object(
                smoke,
                "collect_library_versions",
                return_value={name: None for name in smoke.OPTIONAL_LIBRARIES},
            ),
        ):
            hardware.return_value = smoke.HardwareFacts(
                python_version="3.11.0",
                python_implementation="CPython",
                executable="python",
                platform="test",
                system="test",
                release="test",
                machine="test",
                processor="test",
                logical_cpu_count=1,
                cuda=smoke.CudaFacts(
                    status="unavailable",
                    cuda_visible_devices=None,
                    nvidia_smi_path=None,
                    devices=[],
                ),
            )
            result = smoke.build_result(
                config,
                config_path=CONFIG_PATH,
                config_bytes=CONFIG_PATH.read_bytes(),
                command=[],
                run_load=False,
                allow_download=False,
                selected_names=[],
            )
        result.candidates[0].probes[0].metrics["bad"] = ("not", "json")  # type: ignore[assignment]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_text("sentinel", encoding="utf-8")
            with self.assertRaises(ValidationError):
                smoke.write_result_atomic(result, output)
            result.candidates[0].probes[0].metrics["bad"] = float("nan")
            with self.assertRaises(smoke.SmokeResultError):
                smoke.write_result_atomic(result, output)
            contents = output.read_text(encoding="utf-8")

        self.assertEqual(contents, "sentinel")


class ModelSmokeSafetyTests(unittest.TestCase):
    def test_resolved_release_gate_is_bound_to_registry_and_decision(self) -> None:
        config = smoke.load_config(CONFIG_PATH)
        registry = json.loads(
            (PROJECT_ROOT / config.release_gate.registry_path).read_text(
                encoding="utf-8"
            )
        )
        smoke_ids = {candidate.model_id: candidate for candidate in config.candidates}
        for entries in registry["roles"].values():
            for entry in entries:
                candidate = smoke_ids.get(entry["id"])
                if candidate is None:
                    continue
                entry["release_eligibility"] = (
                    "eligible" if candidate.bundle == "qwen3" else "ineligible"
                )
                entry["release_decision"] = "D-039"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_path = root / "configs" / "model_candidates.json"
            registry_path.parent.mkdir()
            registry_bytes = json.dumps(
                registry, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            registry_path.write_bytes(registry_bytes)
            (root / "DECISIONS.md").write_text(
                "### D-039 — Resolve model release scope\n"
                "Release scope: `public research artifacts`\n"
                "Release-eligible bundles: `qwen3`\n",
                encoding="utf-8",
            )
            gate = smoke.ReleaseSelectionGate.model_validate(
                {
                    "registry_path": "configs/model_candidates.json",
                    "expected_registry_sha256": smoke.hashlib.sha256(
                        registry_bytes
                    ).hexdigest(),
                    "status": "resolved",
                    "intended_release_scope": "public research artifacts",
                    "decision_record": "D-039",
                    "eligible_bundles": ["qwen3"],
                }
            )
            with patch.object(smoke, "PROJECT_ROOT", root):
                actual = smoke._validate_release_registry(gate, config.candidates)
                self.assertEqual(actual, gate.expected_registry_sha256)

                mismatched = gate.model_copy(
                    update={"eligible_bundles": ["qwen2.5"]}
                )
                with self.assertRaisesRegex(
                    smoke.SmokeConfigError, "do not match"
                ):
                    smoke._validate_release_registry(mismatched, config.candidates)

    def test_measured_run_rejects_dirty_worktree_before_model_access(self) -> None:
        config = smoke.load_config(CONFIG_PATH)
        with (
            patch.object(
                smoke,
                "_git_worktree_changes",
                return_value=[" M src/agent/parser.py", "?? untracked.py"],
            ),
            patch.object(
                smoke,
                "_collect_source_identity",
                side_effect=AssertionError("source collection must not start"),
            ),
            patch.object(
                smoke,
                "_execute_candidate",
                side_effect=AssertionError("model execution must not start"),
            ),
            self.assertRaisesRegex(smoke.SmokeConfigError, "clean Git worktree"),
        ):
            smoke.build_result(
                config,
                config_path=CONFIG_PATH,
                config_bytes=CONFIG_PATH.read_bytes(),
                command=[],
                run_load=True,
                allow_download=True,
                selected_names=[],
            )

    def test_offline_plan_does_not_require_clean_worktree(self) -> None:
        config = smoke.load_config(CONFIG_PATH)
        with patch.object(
            smoke,
            "_git_worktree_changes",
            side_effect=AssertionError("offline plan must not inspect worktree state"),
        ):
            result = smoke.build_result(
                config,
                config_path=CONFIG_PATH,
                config_bytes=CONFIG_PATH.read_bytes(),
                command=[],
                run_load=False,
                allow_download=False,
                selected_names=["qwen3-1.7b"],
            )

        self.assertTrue(result.options.dry_run)
        self.assertEqual(result.candidates[0].p6.status, "planned")

    def test_measured_run_rejects_missing_or_mismatched_lane_lock(self) -> None:
        original = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        mismatched_payload = json.loads(json.dumps(original))
        mismatched_payload["lane"]["expected_lock_sha256"] = "0" * 64
        mismatched = smoke.SmokeConfig.model_validate(mismatched_payload)

        with (
            patch.object(
                smoke,
                "_execute_candidate",
                side_effect=AssertionError("model execution must not start"),
            ),
            self.assertRaisesRegex(smoke.SmokeConfigError, "does not match"),
        ):
            smoke.build_result(
                mismatched,
                config_path=CONFIG_PATH,
                config_bytes=json.dumps(mismatched_payload).encode("utf-8"),
                command=[],
                run_load=True,
                allow_download=True,
                selected_names=[],
            )

        config = smoke.load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.lock"
            with (
                patch.object(smoke, "_resolve_lane_lock_path", return_value=missing),
                patch.object(
                    smoke,
                    "_execute_candidate",
                    side_effect=AssertionError("model execution must not start"),
                ),
                self.assertRaisesRegex(smoke.SmokeConfigError, "requires the lane lock"),
            ):
                smoke.build_result(
                    config,
                    config_path=CONFIG_PATH,
                    config_bytes=CONFIG_PATH.read_bytes(),
                    command=[],
                    run_load=True,
                    allow_download=True,
                    selected_names=[],
                )

    def test_unsafe_flag_combinations_fail_without_output(self) -> None:
        combinations = (["--run-load"], ["--allow-download"])
        for flags in combinations:
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "must-not-exist.json"
                with self.assertRaises(SystemExit) as raised:
                    smoke.main(
                        [
                            "--config",
                            str(CONFIG_PATH),
                            "--output",
                            str(output),
                            *flags,
                        ]
                    )
                self.assertEqual(raised.exception.code, 2)
                self.assertFalse(output.exists())

    def test_output_cannot_overwrite_config(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            smoke.main(["--config", str(CONFIG_PATH), "--output", str(CONFIG_PATH)])

        self.assertEqual(raised.exception.code, 2)


class GateDemotionTests(unittest.TestCase):
    """D-046 scoped the P5 prefix check. These pin that it stayed scoped."""

    def _demotion(self, **overrides):
        payload = dict(
            probe="training_template_masking",
            check="prefix_preserved_after_tool_observation",
            demoted_to="recorded_phase_a_diagnostic",
            scope_lane_identity="phase-a-windows-unsloth-trl024",
            scope_stage="blueprint_7_1_stage_1_single_turn",
            valid_for_training_stages=["stage_1_single_turn"],
            invalid_for_training_stages=[
                "stage_2_scripted_multi_turn",
                "stage_3_tau2_multi_turn",
                "m6_environment_factory",
            ],
            still_hard_gate_when="multi_turn_or_m6_in_scope",
            timing="post_hoc_after_measurement",
            decision_record="D-046",
            demoted_on="2026-08-18",
            motive="x" * 60,
            rationale="y" * 220,
            validity_precondition="z" * 60,
            rearm_conditions=["rerun under M6 with the gate enforced"],
        )
        payload.update(overrides)
        return smoke.GateDemotion(**payload)

    def test_demotion_never_applies_when_multi_turn_is_in_scope(self) -> None:
        demotion = self._demotion()
        self.assertEqual(
            smoke._applied_gate_demotions(
                [demotion], m6_environment_factory_in_scope=False
            ),
            (demotion,),
        )
        self.assertEqual(
            smoke._applied_gate_demotions(
                [demotion], m6_environment_factory_in_scope=True
            ),
            (),
        )

    def test_demotable_set_equals_the_multi_turn_rearm_set(self) -> None:
        self.assertEqual(
            smoke.P5_DEMOTABLE_CHECK_IDS, smoke.MULTI_TURN_HARD_GATE_CHECK_IDS
        )
        self.assertTrue(smoke.P5_DEMOTABLE_CHECK_IDS <= set(smoke.P5_CHECK_IDS))

    def test_post_hoc_demotion_requires_a_motive_and_full_rationale(self) -> None:
        with self.assertRaises(ValidationError):
            self._demotion(rationale="too short")
        with self.assertRaises(ValidationError):
            self._demotion(motive="   ")

    def test_declared_demotion_must_match_its_decision_record(self) -> None:
        digests = smoke._validate_gate_demotions([self._demotion()])
        self.assertEqual(
            set(digests), {"P5:prefix_preserved_after_tool_observation"}
        )
        for digest in digests.values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
        with self.assertRaises(smoke.SmokeConfigError):
            smoke._validate_gate_demotions([self._demotion(demoted_on="2020-01-01")])

    def test_demoted_pass_must_announce_that_it_is_not_multi_turn_evidence(self) -> None:
        with self.assertRaises(ValidationError):
            smoke.ProbeResult(
                name="training_template_masking",
                status="passed_with_demoted_gates",
                plan={},
                metrics={},
                error=None,
            )
        with self.assertRaises(ValidationError):
            smoke.ProbeResult(
                name="training_template_masking",
                status="passed_with_demoted_gates",
                plan={},
                metrics={},
                error="quietly fine",
            )
        with self.assertRaises(ValidationError):
            smoke.ProbeResult(
                name="training_template_masking",
                status="passed",
                plan={},
                metrics={},
                error="a clean pass may not carry an error",
            )

    def test_committed_artifacts_are_internally_consistent(self) -> None:
        """History is not rewritten and every artifact states its own regime."""

        paths = sorted((smoke.PROJECT_ROOT / "results").glob("model_smoke-*.json"))
        self.assertGreaterEqual(len(paths), 5)
        pre, post = 0, 0
        for path in paths:
            with self.subTest(artifact=path.name):
                result = smoke.read_result(path)
                declared = bool(result.lane.gate_demotions)
                self.assertEqual(
                    set(result.lane.gate_demotion_decision_sha256),
                    {item.gate_id for item in result.lane.gate_demotions},
                )
                demoted_probe_names = {
                    candidate.name
                    for candidate in result.candidates
                    for probe in candidate.probes
                    if probe.status == "passed_with_demoted_gates"
                }
                if not declared:
                    # Pre-D-046 evidence: the stronger rule applied, and the
                    # recorded failure is never reinterpreted as a pass.
                    pre += 1
                    self.assertEqual(demoted_probe_names, set())
                    self.assertEqual(result.candidates_with_demoted_gate_failures, [])
                    self.assertFalse(result.post_hoc_gate_demotion_present)
                    for candidate in result.candidates:
                        self.assertEqual(candidate.demoted_gate_failures, [])
                else:
                    post += 1
                    self.assertEqual(
                        sorted(demoted_probe_names),
                        result.candidates_with_demoted_gate_failures,
                    )
                    for candidate in result.candidates:
                        if candidate.name not in demoted_probe_names:
                            continue
                        self.assertTrue(candidate.demoted_gate_failures)
                        for probe in candidate.probes:
                            if probe.status != "passed_with_demoted_gates":
                                continue
                            self.assertFalse(
                                probe.metrics["passed_under_preregistered_p5_rule"]
                            )
        self.assertGreater(pre, 0, "expected retained pre-D-046 artifacts")
        self.assertGreater(post, 0, "expected post-D-046 artifacts")

    def test_pre_d046_qwen3_artifacts_still_record_a_hard_failure(self) -> None:
        """The demotion must never rewrite the evidence it was based on."""

        found = 0
        for path in sorted((smoke.PROJECT_ROOT / "results").glob("model_smoke-qwen3-*.json")):
            result = smoke.read_result(path)
            if result.lane.gate_demotions:
                continue
            for candidate in result.candidates:
                for probe in candidate.probes:
                    if probe.name != "training_template_masking":
                        continue
                    if probe.status == "unavailable":
                        continue
                    with self.subTest(artifact=path.name):
                        self.assertEqual(probe.status, "failed")
                    found += 1
        self.assertGreater(found, 0)

    def test_gate_demotions_do_not_reach_the_library(self) -> None:
        """The smoke demotion must never leak into the reward or eval path."""

        banned = (
            "gate_demotions",
            "demoted_gate_failures",
            "passed_with_demoted_gates",
            "prefix_preserved_after_tool_observation",
            "smoke_models",
        )
        for path in sorted((smoke.PROJECT_ROOT / "src").rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for token in banned:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token, text)


class ChatTemplateControlTests(unittest.TestCase):
    """The Qwen3-vs-Qwen2.5 parity control must be measured, not assumed."""

    class _Tok:
        def __init__(self, reads: bool) -> None:
            self.reads = reads

        def apply_chat_template(self, messages, **kwargs):
            if self.reads and kwargs.get("enable_thinking"):
                return "RENDER<think>"
            return "RENDER"

    def _probe(self):
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["probe"]
        return smoke.ProbeConfig.model_validate(payload)

    def test_a_template_that_reads_the_control_is_recorded_as_honored(self) -> None:
        result = smoke._chat_template_kwargs_honored(
            tokenizer=self._Tok(reads=True), messages=[], tools=[], probe=self._probe()
        )
        self.assertIs(result["enable_thinking"], True)

    def test_a_template_that_ignores_the_control_is_recorded_as_not_honored(self) -> None:
        """Not a failure: Qwen2.5 has no thinking mode. It must still be visible."""

        result = smoke._chat_template_kwargs_honored(
            tokenizer=self._Tok(reads=False), messages=[], tools=[], probe=self._probe()
        )
        self.assertIs(result["enable_thinking"], False)

    def test_template_errors_become_evidence_rather_than_crashes(self) -> None:
        class Boom:
            def apply_chat_template(self, messages, **kwargs):
                raise RuntimeError("template exploded")

        result = smoke._chat_template_kwargs_honored(
            tokenizer=Boom(), messages=[], tools=[], probe=self._probe()
        )
        self.assertIsInstance(result["enable_thinking"], str)
        self.assertIn("error", result["enable_thinking"])


class ArtifactImmutabilityTests(unittest.TestCase):
    """Measurement records are permanent. They may be added to, never edited."""

    MANIFEST = smoke.PROJECT_ROOT / "results" / "artifact_manifest.json"

    def _manifest(self) -> dict:
        return json.loads(self.MANIFEST.read_text(encoding="utf-8"))

    def _artifacts(self) -> list:
        return sorted((smoke.PROJECT_ROOT / "results").glob("model_smoke-*.json"))

    def test_every_committed_artifact_matches_its_frozen_hash(self) -> None:
        recorded = self._manifest()["artifacts"]
        for path in self._artifacts():
            with self.subTest(artifact=path.name):
                self.assertIn(
                    path.name,
                    recorded,
                    "a result artifact is not listed in results/artifact_manifest.json",
                )
                raw = path.read_bytes()
                self.assertEqual(
                    hashlib.sha256(raw).hexdigest(),
                    recorded[path.name]["sha256"],
                    "a committed measurement record was modified after the fact",
                )
                self.assertEqual(len(raw), recorded[path.name]["bytes"])

    def test_no_listed_artifact_has_been_deleted(self) -> None:
        present = {path.name for path in self._artifacts()}
        for name in self._manifest()["artifacts"]:
            with self.subTest(artifact=name):
                self.assertIn(name, present, "a recorded measurement was removed")

    def test_manifest_agrees_with_each_artifact_about_its_evidence_regime(self) -> None:
        """The manifest cannot claim a demotion the artifact does not declare."""

        recorded = self._manifest()["artifacts"]
        for path in self._artifacts():
            with self.subTest(artifact=path.name):
                result = smoke.read_result(path)
                self.assertEqual(
                    bool(result.lane.gate_demotions),
                    recorded[path.name]["declares_gate_demotion"],
                )
                self.assertEqual(
                    result.config_sha256, recorded[path.name]["config_sha256"]
                )

    def test_both_evidence_regimes_are_retained(self) -> None:
        """Pre-D-046 failures and post-D-046 runs must both survive."""

        regimes = {
            entry["declares_gate_demotion"]
            for entry in self._manifest()["artifacts"].values()
        }
        self.assertEqual(regimes, {False, True})


if __name__ == "__main__":
    unittest.main()
