from __future__ import annotations

import builtins
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

        helper_module = SimpleNamespace(
            get_training_chat_template=lambda tokenizer: (
                "head{% generation %}body{% endgeneration %}tail"
            )
        )
        plan = {
            item.name: item.plan for item in smoke.probe_plans(probe)
        }["training_template_masking"]
        with patch.object(smoke.importlib, "import_module", return_value=helper_module):
            result = smoke._run_training_template_probe(
                tokenizer=OversizedRenderTokenizer(),
                native_template="native",
                probe=probe,
                plan=plan,
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

        helper_module = SimpleNamespace(
            get_training_chat_template=lambda tokenizer: "patched-template"
        )
        probe = smoke.load_config(CONFIG_PATH).probe
        plan = {
            item.name: item.plan for item in smoke.probe_plans(probe)
        }["training_template_masking"]
        with patch.object(smoke.importlib, "import_module", return_value=helper_module):
            result = smoke._run_training_template_probe(
                tokenizer=MasklessTokenizer(),
                native_template="native-template",
                probe=probe,
                plan=plan,
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

        helper_module = SimpleNamespace(
            get_training_chat_template=lambda tokenizer: (
                "head{% generation %}body{% endgeneration %}tail"
            )
        )
        plan = {
            item.name: item.plan for item in smoke.probe_plans(probe)
        }["training_template_masking"]
        results = []
        with patch.object(smoke.importlib, "import_module", return_value=helper_module):
            for complete_mask in (True, False):
                results.append(
                    smoke._run_training_template_probe(
                        tokenizer=MaskingTokenizer(complete_mask=complete_mask),
                        native_template="native-template",
                        probe=probe,
                        plan=plan,
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
        self.assertEqual(incomplete.status, "failed")
        self.assertFalse(
            incomplete.metrics["checks"][
                "assistant_mask_exactly_matches_generation_spans"
            ]
        )

    def test_plans_include_reserved_memory_hashes_and_training_hard_gates(self) -> None:
        plans = {
            item.name: item.plan
            for item in smoke.probe_plans(smoke.load_config(CONFIG_PATH).probe)
        }

        self.assertIn(
            "per-device peak CUDA reserved bytes",
            plans["four_bit_load"]["measurements"],
        )
        self.assertIn("hard_gate", plans["four_bit_load"])
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
        self.assertEqual(leftovers, [])

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


if __name__ == "__main__":
    unittest.main()
