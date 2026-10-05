from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "configs" / "model_candidates.json"

EXPECTED_ROLES = {
    "primary_small",
    "scale_check",
    "cross_family_check",
    "scaffolded_comparator",
    "user_simulator",
    "function_calling_dataset",
    "phase_a_task_dataset",
}
ALLOWED_ACCESS = {"public", "manual_gate", "automatic_gate"}
ALLOWED_SELECTION_STATUSES = {"pending", "selected", "rejected"}
ALLOWED_RELEASE_ELIGIBILITY = {"pending", "eligible", "ineligible"}
ALLOWED_LICENSE_IDS = {
    "apache-2.0",
    "cc-by-4.0",
    "llama3.1",
    "llama3.2",
    "mit",
    "other",
}
ALLOWED_LICENSE_NAMES = {"qwen-research"}
IMMUTABLE_REVISION = re.compile(r"[0-9a-f]{40}\Z")
RELEASE_DECISION = re.compile(r"D-[0-9]{3}\Z")


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{path} must contain a JSON object")
    return payload


class ModelCandidateRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = _load_json_object(REGISTRY_PATH)

    def _entries(self) -> list[tuple[str, dict[str, Any]]]:
        roles = self.registry["roles"]
        entries: list[tuple[str, dict[str, Any]]] = []
        for role, candidates in roles.items():
            self.assertIsInstance(candidates, list, msg=f"{role} must be a list")
            self.assertGreater(len(candidates), 0, msg=f"{role} must not be empty")
            for candidate in candidates:
                self.assertIsInstance(candidate, dict, msg=f"invalid entry in {role}")
                entries.append((role, candidate))
        return entries

    def test_schema_version_and_exact_role_set(self) -> None:
        self.assertIs(type(self.registry.get("schema_version")), int)
        self.assertEqual(self.registry["schema_version"], 1)
        self.assertIsInstance(self.registry.get("roles"), dict)
        self.assertEqual(set(self.registry["roles"]), EXPECTED_ROLES)

    def test_ids_are_unique_across_roles(self) -> None:
        seen: dict[str, str] = {}
        for role, candidate in self._entries():
            model_id = candidate.get("id")
            self.assertIsInstance(model_id, str, msg=f"invalid ID in {role}")
            self.assertTrue(model_id, msg=f"empty ID in {role}")
            self.assertNotIn(
                model_id,
                seen,
                msg=f"{model_id} appears in both {seen.get(model_id)} and {role}",
            )
            seen[model_id] = role

    def test_revisions_are_full_lowercase_immutable_shas(self) -> None:
        for role, candidate in self._entries():
            with self.subTest(role=role, model_id=candidate.get("id")):
                revision = candidate.get("revision")
                self.assertIsInstance(revision, str)
                self.assertNotIn(revision, {"main", "master"})
                self.assertIsNotNone(
                    IMMUTABLE_REVISION.fullmatch(revision),
                    msg=f"{revision!r} is not a full lowercase 40-hex SHA",
                )

    def test_controlled_vocabulary_and_pending_stage(self) -> None:
        for role, candidate in self._entries():
            with self.subTest(role=role, model_id=candidate.get("id")):
                access = candidate.get("access")
                status = candidate.get("selection_status")
                release_eligibility = candidate.get("release_eligibility")
                release_decision = candidate.get("release_decision")
                license_id = candidate.get("license_id")
                license_name = candidate.get("license_name")

                self.assertIsInstance(access, str)
                self.assertIn(access, ALLOWED_ACCESS)
                self.assertIsInstance(status, str)
                self.assertIn(status, ALLOWED_SELECTION_STATUSES)
                # Selection and release eligibility resolve together, by a
                # dated decision (D-048). Neither may move without the other.
                self.assertEqual(
                    status == "pending",
                    candidate.get("release_eligibility") == "pending",
                    "selection_status and release_eligibility must resolve together",
                )
                if status == "selected":
                    self.assertEqual(candidate.get("release_eligibility"), "eligible")
                if status == "rejected":
                    self.assertEqual(
                        candidate.get("release_eligibility"), "ineligible"
                    )
                self.assertIsInstance(release_eligibility, str)
                self.assertIn(
                    release_eligibility, ALLOWED_RELEASE_ELIGIBILITY
                )
                if release_eligibility == "pending":
                    self.assertIsNone(release_decision)
                else:
                    self.assertIsInstance(release_decision, str)
                    self.assertIsNotNone(
                        RELEASE_DECISION.fullmatch(release_decision)
                    )
                self.assertEqual(
                    release_decision is None,
                    release_eligibility == "pending",
                )
                self.assertIsInstance(license_id, str)
                self.assertIn(license_id, ALLOWED_LICENSE_IDS)

                if license_name is not None:
                    self.assertIsInstance(license_name, str)
                    self.assertIn(license_name, ALLOWED_LICENSE_NAMES)
                if license_id == "other":
                    self.assertEqual(license_name, "qwen-research")
                else:
                    self.assertIsNone(license_name)
                self.assertEqual(
                    license_name == "qwen-research",
                    license_id == "other",
                )


if __name__ == "__main__":
    unittest.main()
