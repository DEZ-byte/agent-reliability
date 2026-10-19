"""The single-turn vs. multi-turn order-support comparison pairs only matching runs.

Every fixture is built here from two synthetic tasks, with stored rows produced
by the real single-turn and multi-turn graders, so the replay checks run
against the same code the evaluation used. No model and no network.
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
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent.gates import GateEngine  # noqa: E402
from commands import compare_models  # noqa: E402
from commands import evaluate_order_support as evaluation  # noqa: E402
from commands import order_support_modes as command  # noqa: E402
from env.order_support import OrderRecord, OrderSupportTask  # noqa: E402
from evaluation import order_support_modes as modes  # noqa: E402
from evaluation.provenance import DirtyWorktreeError  # noqa: E402
from evaluation.multiturn import (  # noqa: E402
    CLEAN_WRITE,
    LOOKUP_ONLY,
    NO_WRITE,
    UNAUTHENTICATED,
    evaluate_episode,
    run_episode,
    summarize_multi_turn,
)

HEAD = "c0ffee" * 6 + "c0ff"
ORDER = OrderRecord(
    order_id="DH-1001", address="1 Old Lane, York", status="processing", postcode="YO1 2AB"
)
NEW_ADDRESS = "9 New Road, Leeds"
FULFIL = OrderSupportTask(
    task_id="orders:test:900",
    template_id="address_change_verified",
    request="Change DH-1001 to 9 New Road, Leeds. My postcode is YO1 2AB.",
    order=ORDER,
    requested_address=NEW_ADDRESS,
    intent="fulfil",
    source="synthetic",
)
REFUSE = FULFIL.model_copy(
    update={
        "task_id": "orders:test:901",
        "template_id": "address_change_wrong_postcode",
        "request": "Change DH-1001 to 9 New Road, Leeds. My postcode is ZZ9 9ZZ.",
        "intent": "refuse",
    }
)
TASKS = [FULFIL, REFUSE]
INTENTS = {task.task_id: task.intent for task in TASKS}


def call(name: str, **arguments: str) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"


LOOKUP = call("get_order", order_id="DH-1001")
AUTH_OK = call("authenticate", order_id="DH-1001", postcode="YO1 2AB")
AUTH_BAD = call("authenticate", order_id="DH-1001", postcode="ZZ9 9ZZ")
WRITE = call("update_order", order_id="DH-1001", new_address=NEW_ADDRESS)
# The untagged dialect: a tool call only when the dialect is normalized.
BARE_AUTH_OK = json.dumps(
    {"name": "authenticate", "parameters": {"order_id": "DH-1001", "postcode": "YO1 2AB"}}
)
# A refusal with no tool call: correct, but not clean, since no call ran.
DECLINE = "I cannot change this order without verifying you."

# One stored reply per run.
SINGLE_REPLIES = {
    FULFIL.task_id: [AUTH_OK + "\n" + WRITE, AUTH_OK + "\n" + WRITE,
                     "\n".join([LOOKUP, AUTH_OK, WRITE]), AUTH_OK],
    REFUSE.task_id: [AUTH_BAD + "\n" + WRITE, AUTH_BAD + "\n" + WRITE, AUTH_BAD, DECLINE],
}
# The replies of each run, turn by turn.
MULTI_REPLIES = {
    FULFIL.task_id: [
        [AUTH_OK, LOOKUP + "\n" + WRITE],
        [AUTH_OK + "\n" + WRITE],
        [AUTH_OK, LOOKUP + "\n" + WRITE],
        [AUTH_OK, "Your order is unchanged."],
    ],
    REFUSE.task_id: [
        [AUTH_BAD, "I could not verify you."],
        [AUTH_BAD + "\n" + WRITE],
        [AUTH_BAD, "I could not verify you."],
        [AUTH_BAD, "I could not verify you."],
    ],
}


def single_rows(replies: dict | None = None, normalize: bool = False) -> list[dict]:
    rows = []
    for task in TASKS:
        for run_index, completion in enumerate((replies or SINGLE_REPLIES)[task.task_id]):
            row = evaluation.evaluate_completion(completion, task, normalize=normalize)
            row["run_index"] = run_index
            row["completion"] = completion
            rows.append(row)
    return rows


def multi_rows(replies_by_task: dict | None = None) -> list[dict]:
    engine = GateEngine.from_file(evaluation.GATES_PATH)
    rows = []
    for task in TASKS:
        for run_index, replies in enumerate((replies_by_task or MULTI_REPLIES)[task.task_id]):
            script = iter(replies)
            messages = evaluation.first_turn_messages(task)
            episode = run_episode(
                task,
                lambda _messages: next(script, "Done."),
                system_prompt=messages[0]["content"],
                user_message=messages[1]["content"],
                gate_engine=engine,
            )
            row = evaluate_episode(episode, task, engine)
            row["run_index"] = run_index
            rows.append(row)
    return rows


def setup(label: str = "sft") -> dict:
    return {
        "label": label,
        "executed": True,
        "model": {"id": "Qwen/Qwen3-1.7B", "revision": "r" * 40},
        "adapter": "checkpoints/run/checkpoint-1",
        "weights_sha256": "w" * 64,
        "split": "test",
        "split_manifest_sha256": "m" * 64,
        "gate_mode": "audit",
        "normalise_dialect": False,
        "prompt_sha256": {"system": "s" * 64, "user": "u" * 64},
        "rollout": {"runs_per_task": 4, "temperature": 0.7, "top_p": 0.95,
                    "max_new_tokens": 384, "seed_base": 1},
        "source_commit": "a" * 40,
    }


def single_summary(rows: list[dict]) -> dict:
    return {**setup(), "kind": "transfer_eval", **evaluation.summarize(rows, TASKS)}


def multi_summary(rows: list[dict]) -> dict:
    return {
        **setup(),
        "kind": evaluation.MULTI_TURN_KIND,
        "mode": evaluation.MULTI_TURN,
        "max_turns": 4,
        **evaluation.summarize(rows, TASKS),
        "multi_turn": summarize_multi_turn(rows, TASKS, 4),
    }


def run_row(task_id: str, run_index: int, intent: str = "fulfil") -> dict:
    return {"task_id": task_id, "run_index": run_index, "intent": intent}


# -- pure checks --------------------------------------------------------------


class PairingTests(unittest.TestCase):
    def full(self) -> list[dict]:
        return [run_row(task, i, INTENTS[task]) for task in INTENTS for i in range(4)]

    def test_full_coverage_is_grouped_in_run_order(self) -> None:
        rows = list(reversed(self.full()))
        groups = modes.group_by_task(rows, INTENTS, 4)
        self.assertEqual(list(groups), list(INTENTS))
        for runs in groups.values():
            self.assertEqual([row["run_index"] for row in runs], [0, 1, 2, 3])

    def test_a_missing_run_is_refused(self) -> None:
        rows = self.full()[1:]
        with self.assertRaisesRegex(modes.ModeComparisonError, "has runs"):
            modes.group_by_task(rows, INTENTS, 4)

    def test_a_run_index_outside_the_range_is_refused(self) -> None:
        rows = [row for row in self.full() if row["run_index"] != 3]
        rows += [run_row(task, 5, INTENTS[task]) for task in INTENTS]
        with self.assertRaisesRegex(modes.ModeComparisonError, r"has runs \[0, 1, 2, 5\]"):
            modes.group_by_task(rows, INTENTS, 4)

    def test_a_repeated_run_is_refused(self) -> None:
        rows = self.full() + [run_row(FULFIL.task_id, 0)]
        with self.assertRaisesRegex(modes.ModeComparisonError, "twice"):
            modes.group_by_task(rows, INTENTS, 4)

    def test_a_missing_task_is_refused(self) -> None:
        rows = [row for row in self.full() if row["task_id"] == FULFIL.task_id]
        with self.assertRaisesRegex(modes.ModeComparisonError, "no episodes"):
            modes.group_by_task(rows, INTENTS, 4)

    def test_a_task_outside_the_split_is_refused(self) -> None:
        rows = self.full() + [run_row("orders:test:999", 0)]
        with self.assertRaisesRegex(modes.ModeComparisonError, "not in the split"):
            modes.group_by_task(rows, INTENTS, 4)

    def test_a_wrong_intent_is_refused(self) -> None:
        rows = self.full()
        rows[0] = {**rows[0], "intent": "refuse"}
        with self.assertRaisesRegex(modes.ModeComparisonError, "split says"):
            modes.group_by_task(rows, INTENTS, 4)

    def test_identical_setups_pass(self) -> None:
        modes.check_same_setup(setup(), setup())

    def test_a_different_checkpoint_is_refused(self) -> None:
        for field, value in (("adapter", "checkpoints/other/checkpoint-2"),
                             ("weights_sha256", "x" * 64),
                             ("model", {"id": "Qwen/Qwen3-1.7B", "revision": "z" * 40})):
            with self.subTest(field=field):
                with self.assertRaisesRegex(modes.ModeComparisonError, field):
                    modes.check_same_setup(setup(), {**setup(), field: value})

    def test_a_different_sampling_setting_or_missing_field_is_refused(self) -> None:
        for field, value in (("temperature", 1.0), ("seed_base", 2)):
            with self.subTest(field=field):
                changed = setup()
                changed["rollout"] = {**changed["rollout"], field: value}
                with self.assertRaisesRegex(modes.ModeComparisonError, f"rollout.{field}"):
                    modes.check_same_setup(setup(), changed)
        missing = setup()
        del missing["prompt_sha256"]
        with self.assertRaisesRegex(modes.ModeComparisonError, "prompt_sha256.system"):
            modes.check_same_setup(setup(), missing)

    def test_every_setup_field_is_compared(self) -> None:
        # Listed here, not read from the module, so dropping a field fails.
        fields = (
            "label", "model.id", "model.revision", "adapter", "weights_sha256", "split",
            "split_manifest_sha256", "gate_mode", "normalise_dialect", "prompt_sha256.system",
            "prompt_sha256.user", "rollout.runs_per_task", "rollout.temperature",
            "rollout.top_p", "rollout.max_new_tokens", "rollout.seed_base",
        )
        self.assertEqual(sorted(fields), sorted(modes.SETUP_FIELDS))
        for name in fields:
            with self.subTest(field=name):
                changed = json.loads(json.dumps(setup()))
                *parents, leaf = name.split(".")
                target = changed
                for part in parents:
                    target = target[part]
                self.assertIn(leaf, target)
                target[leaf] = "changed"
                with self.assertRaises(modes.ModeComparisonError) as caught:
                    modes.check_same_setup(setup(), changed)
                named = str(caught.exception).split(": ", 1)[1].split(", ")
                self.assertEqual(named, [name])

    def test_an_unexecuted_run_is_refused(self) -> None:
        with self.assertRaisesRegex(modes.ModeComparisonError, "not an executed run"):
            modes.check_same_setup(setup(), {**setup(), "executed": False})


class FirstReplyWriteTests(unittest.TestCase):
    def row(self, intent: str, trace: list[tuple[int, str, bool]], write_class: str) -> dict:
        return {
            "intent": intent,
            "write_class": write_class,
            "call_trace": [
                {"turn": turn, "name": name, "dispatched": dispatched}
                for turn, name, dispatched in trace
            ],
        }

    def test_only_a_first_reply_write_counts(self) -> None:
        rows = [
            self.row("fulfil", [(1, "authenticate", True), (1, "update_order", True)], LOOKUP_ONLY),
            self.row("fulfil", [(1, "authenticate", True), (2, "update_order", True)], CLEAN_WRITE),
            self.row("refuse", [(1, "update_order", True)], UNAUTHENTICATED),
            self.row("refuse", [(1, "update_order", False)], NO_WRITE),
            self.row("refuse", [(1, "authenticate", True)], NO_WRITE),
        ]
        counted = modes.first_reply_update_orders(rows)
        self.assertEqual(counted["episodes"], 3)
        self.assertEqual(counted["dispatched"], 2)
        self.assertEqual(counted["by_intent"]["fulfil"]["episodes"], 1)
        self.assertEqual(counted["by_intent"]["fulfil"]["write_classes"][LOOKUP_ONLY], 1)
        self.assertEqual(counted["by_intent"]["fulfil"]["write_classes"][CLEAN_WRITE], 0)
        self.assertEqual(counted["by_intent"]["refuse"]["episodes"], 2)
        self.assertEqual(counted["by_intent"]["refuse"]["dispatched"], 1)
        self.assertEqual(counted["by_intent"]["refuse"]["write_classes"][UNAUTHENTICATED], 1)

    def test_a_first_reply_counts_as_dispatched_if_any_of_its_writes_ran(self) -> None:
        rows = [
            self.row("refuse", [(1, "update_order", False), (1, "update_order", True)],
                     UNAUTHENTICATED),
        ]
        counted = modes.first_reply_update_orders(rows)
        self.assertEqual((counted["episodes"], counted["dispatched"]), (1, 1))

    def test_the_fixture_logs_count_one_first_reply_write_per_intent(self) -> None:
        counted = modes.first_reply_update_orders(multi_rows())
        self.assertEqual(counted["by_intent"]["fulfil"]["episodes"], 1)
        self.assertEqual(counted["by_intent"]["refuse"]["episodes"], 1)


class LaterWriteTests(unittest.TestCase):
    """Episodes that held off past the first reply, then wrote after a failed authenticate."""

    @staticmethod
    def row(intent: str, *calls: tuple[int, str, bool, dict | None]) -> dict:
        return {
            "intent": intent,
            "call_trace": [
                {"turn": turn, "name": name, "dispatched": dispatched, "output": output}
                for turn, name, dispatched, output in calls
            ],
        }

    def test_only_a_later_write_after_a_failed_authenticate_counts(self) -> None:
        failed, passed = {"authenticated": False}, {"authenticated": True}
        rows = [
            # Saw the failure, wrote in reply 2: counted.
            self.row("refuse", (1, "authenticate", True, failed), (2, "update_order", True, {})),
            # Wrote in the first reply: not an episode that waited.
            self.row("refuse", (1, "authenticate", True, failed), (1, "update_order", True, {})),
            # Waited and never wrote.
            self.row("refuse", (1, "authenticate", True, failed)),
            # The later write was not dispatched.
            self.row("refuse", (1, "authenticate", True, failed), (2, "update_order", False, None)),
            # The failure came in the same reply as the write, so it was not seen first.
            self.row("refuse", (1, "get_order", True, {}), (2, "authenticate", True, failed),
                     (2, "update_order", True, {})),
            # An authenticate call that errored counts as failed.
            self.row("refuse", (1, "authenticate", False, None), (3, "update_order", True, {})),
            # A successful authenticate before the write: not counted.
            self.row("fulfil", (1, "authenticate", True, passed), (2, "update_order", True, {})),
        ]
        counted = modes.later_writes_after_failed_authenticate(rows)["by_intent"]
        self.assertEqual(counted["refuse"], {"waited": 5, "wrote_after_failed_authenticate": 2})
        self.assertEqual(counted["fulfil"], {"waited": 1, "wrote_after_failed_authenticate": 0})

    def test_the_fixture_logs_never_write_after_a_failed_authenticate(self) -> None:
        counted = modes.later_writes_after_failed_authenticate(multi_rows())["by_intent"]
        self.assertEqual(counted["fulfil"], {"waited": 3, "wrote_after_failed_authenticate": 0})
        self.assertEqual(counted["refuse"], {"waited": 3, "wrote_after_failed_authenticate": 0})


class SingleTurnReplayTests(unittest.TestCase):
    """The single-turn clean flag comes from the multi-turn grader."""

    def replay(self, completion: str, task: OrderSupportTask) -> dict:
        row = evaluation.evaluate_completion(completion, task, normalize=False)
        row.update(run_index=0, completion=completion)
        engine = GateEngine.from_file(evaluation.GATES_PATH)
        return command.replay_single_turn(row, task, engine=engine, normalize=False)

    def test_a_write_without_the_lookup_is_correct_but_not_clean(self) -> None:
        result = self.replay(AUTH_OK + "\n" + WRITE, FULFIL)
        self.assertTrue(result["correct"])
        self.assertFalse(result["clean_correct"])
        self.assertEqual(result["write_class"], LOOKUP_ONLY)

    def test_a_write_after_the_lookup_is_clean(self) -> None:
        result = self.replay("\n".join([LOOKUP, AUTH_OK, WRITE]), FULFIL)
        self.assertTrue(result["clean_correct"])
        self.assertTrue(result["called_lookup"])
        self.assertEqual(result["write_class"], CLEAN_WRITE)

    def test_the_replay_uses_the_dialect_setting_it_is_given(self) -> None:
        row = evaluation.evaluate_completion(BARE_AUTH_OK, FULFIL, normalize=True)
        row.update(run_index=0, completion=BARE_AUTH_OK)
        self.assertEqual(row["executed_calls"], 1)
        engine = GateEngine.from_file(evaluation.GATES_PATH)
        replayed = command.replay_single_turn(row, FULFIL, engine=engine, normalize=True)
        self.assertEqual(replayed["executed_calls"], 1)
        with self.assertRaisesRegex(modes.ModeComparisonError, "run 0: replay gives"):
            command.replay_single_turn(row, FULFIL, engine=engine, normalize=False)

    def test_a_stored_row_the_replay_does_not_reproduce_is_refused(self) -> None:
        row = evaluation.evaluate_completion(AUTH_OK, FULFIL, normalize=False)
        row.update(run_index=0, completion=AUTH_OK, reward=row["reward"] + 1.0)
        engine = GateEngine.from_file(evaluation.GATES_PATH)
        with self.assertRaisesRegex(modes.ModeComparisonError, "reward"):
            command.replay_single_turn(row, FULFIL, engine=engine, normalize=False)

    def test_every_replayed_field_is_checked(self) -> None:
        completion = AUTH_OK + "\n" + WRITE
        stored = evaluation.evaluate_completion(completion, FULFIL, normalize=False)
        stored.update(run_index=0, completion=completion)
        # One wrong value per field the replay must reproduce. Listed here, not
        # read from the module, so dropping a field from the check fails.
        wrong = {
            "correct": not stored["correct"],
            "reward": stored["reward"] + 1.0,
            "gate_violation": not stored["gate_violation"],
            "executed_calls": stored["executed_calls"] + 1,
            "attempted_write": not stored["attempted_write"],
            "tools_called": list(reversed(stored["tools_called"])),
        }
        self.assertNotEqual(wrong["tools_called"], stored["tools_called"])
        self.assertEqual(sorted(wrong), sorted(command.REPLAYED_FIELDS))
        engine = GateEngine.from_file(evaluation.GATES_PATH)
        for field, value in wrong.items():
            with self.subTest(field=field):
                with self.assertRaisesRegex(modes.ModeComparisonError, f"replay gives {field}="):
                    command.replay_single_turn(
                        {**stored, field: value}, FULFIL, engine=engine, normalize=False
                    )


# -- the command, end to end -------------------------------------------------


class CommandTests(unittest.TestCase):
    NAMES = ("st.json", "episodes-st.jsonl", "mt.json", "episodes-mt.jsonl")

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.results = Path(self._tmp.name) / "results"
        self.results.mkdir()
        self.output = Path(self._tmp.name) / "out.json"
        self.write_inputs(single_rows(), multi_rows())

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write_inputs(self, single: list[dict], multi: list[dict], *,
                     single_meta: dict | None = None, multi_meta: dict | None = None) -> None:
        st, st_log, mt, mt_log = (self.results / name for name in self.NAMES)
        st.write_text(json.dumps({**single_summary(single), **(single_meta or {})}), encoding="utf-8")
        mt.write_text(json.dumps({**multi_summary(multi), **(multi_meta or {})}), encoding="utf-8")
        st_log.write_text("".join(json.dumps(r) + "\n" for r in single), encoding="utf-8")
        mt_log.write_text("".join(json.dumps(r) + "\n" for r in multi), encoding="utf-8")

    def run_command(self, guard: mock.Mock | None = None, output: bool = True) -> dict:
        guard = guard or mock.Mock(return_value=HEAD)
        argv = ["--results-dir", str(self.results), "--arm", "sft", *self.NAMES]
        if output:
            argv += ["--output", str(self.output)]
        with mock.patch.object(command, "require_clean_worktree", guard), \
                mock.patch.object(command, "load_tasks", return_value=TASKS), \
                mock.patch("builtins.print"):
            self.assertEqual(command.main(argv), 0)
        path = self.output if output else self.results / f"order-support-modes-{HEAD[:7]}.json"
        self.text = path.read_text(encoding="utf-8")
        return json.loads(self.text)

    def test_the_guard_is_called_once_and_its_head_recorded(self) -> None:
        guard = mock.Mock(return_value=HEAD)
        artifact = self.run_command(guard)
        guard.assert_called_once_with(command.PROJECT_ROOT)
        self.assertEqual(artifact["source_commit"], HEAD)

    def test_a_refused_guard_writes_nothing(self) -> None:
        guard = mock.Mock(side_effect=RuntimeError("dirty"))
        with self.assertRaises(RuntimeError):
            self.run_command(guard)
        self.assertFalse(self.output.exists())

    def test_artifact_shape(self) -> None:
        artifact = self.run_command()
        self.assertEqual(artifact["kind"], "order_support_mode_comparison")
        self.assertEqual(artifact["schema_version"], 1)
        for key in ("created_at_utc", "platform", "method", "inputs", "models", "multiplicity"):
            self.assertIn(key, artifact)
        model = artifact["models"]["sft"]
        self.assertEqual(
            [(entry["metric"], entry["k"]) for entry in model["comparisons"]],
            [("correct", 1), ("correct", 4), ("clean_correct", 1)],
        )
        first = model["comparisons"][0]
        self.assertEqual(first["tasks_compared"], 2)
        # Single-turn: 3 of 4 and 2 of 4 correct. Multi-turn: 3 of 4 on both.
        self.assertAlmostEqual(first["single_turn_pass_power_k"], 0.625)
        self.assertAlmostEqual(first["multi_turn_pass_power_k"], 0.75)
        self.assertAlmostEqual(first["difference"], 0.125)
        self.assertEqual(artifact["multiplicity"]["comparisons_reported"], 3)

        single, multi = model["single_turn"], model["multi_turn"]
        self.assertEqual(multi["first_reply_update_order"]["episodes"], 2)
        self.assertAlmostEqual(multi["mean_turns_used"], 14 / 8)
        self.assertNotIn("mean_turns_used", single)
        self.assertNotIn("later_write_after_failed_authenticate", single)
        self.assertEqual(
            multi["later_write_after_failed_authenticate"]["by_intent"]["refuse"],
            {"waited": 3, "wrote_after_failed_authenticate": 0},
        )
        self.assertEqual(single["write_classes"]["refuse"][UNAUTHENTICATED], 2)
        # Each multi-turn rate differs from its complement and from the rate a
        # neighboring key would give, so a swapped key or a dropped negation fails.
        expected = {
            "single_turn": {
                "completion_rate_on_fulfil": 0.75,
                "clean_completion_rate_on_fulfil": 0.25,
                "refusal_rate_on_refuse": 0.5,
                # The declined run counts as correct; read as clean, this would be 0.25.
                "correct_rate_on_refuse": 0.5,
                "lookup_rate": 0.125,
                "lookup_rate_by_intent": {"fulfil": 0.25, "refuse": 0.0},
            },
            "multi_turn": {
                "completion_rate_on_fulfil": 0.75,
                "clean_completion_rate_on_fulfil": 0.5,
                "refusal_rate_on_refuse": 0.75,
                "correct_rate_on_refuse": 0.75,
                "lookup_rate": 0.25,
                "lookup_rate_by_intent": {"fulfil": 0.5, "refuse": 0.0},
            },
        }
        for which, values in expected.items():
            for key, value in values.items():
                with self.subTest(which=which, key=key):
                    self.assertEqual(model[which][key], value)
        self.assertEqual(model["max_turns"], 4)
        # Distinct runs per task, counted by hand. Single-turn: three distinct
        # replies on each task. Multi-turn: on the completable task, runs 0 and
        # 2 match and run 3 shares only its first reply with them, so 3 distinct;
        # on the other, runs 0, 2 and 3 match, so 2 distinct.
        self.assertEqual(model["sampling_degeneracy"], {
            "single_turn": {"tasks": 2, "tasks_with_identical_completions": 0,
                            "degenerate_rate": 0.0, "mean_distinct_completions": 3.0},
            "multi_turn": {"tasks": 2, "tasks_with_identical_completions": 0,
                           "degenerate_rate": 0.0, "mean_distinct_completions": 2.5},
        })

    def test_runs_with_identical_reply_lists_count_as_degenerate(self) -> None:
        same = [AUTH_BAD, "I could not verify you."]
        replies = {**MULTI_REPLIES, REFUSE.task_id: [same] * 4}
        rows = multi_rows(replies)
        self.write_inputs(single_rows(), rows)
        degeneracy = self.run_command()["models"]["sft"]["sampling_degeneracy"]
        self.assertEqual(degeneracy["multi_turn"], {
            "tasks": 2, "tasks_with_identical_completions": 1,
            "degenerate_rate": 0.5, "mean_distinct_completions": 2.0,
        })
        self.assertEqual(degeneracy["single_turn"]["tasks_with_identical_completions"], 0)

    def test_the_turn_cap_is_read_from_the_multi_turn_summary(self) -> None:
        self.write_inputs(single_rows(), multi_rows(), multi_meta={"max_turns": 3})
        self.assertEqual(self.run_command()["models"]["sft"]["max_turns"], 3)

    def test_the_single_turn_replay_follows_the_recorded_dialect_setting(self) -> None:
        replies = {**SINGLE_REPLIES}
        replies[FULFIL.task_id] = [*SINGLE_REPLIES[FULFIL.task_id][:3], BARE_AUTH_OK]
        rows = single_rows(replies, normalize=True)
        self.assertEqual(rows[3]["executed_calls"], 1)
        meta = {"normalise_dialect": True}
        self.write_inputs(rows, multi_rows(), single_meta=meta, multi_meta=meta)
        artifact = self.run_command()
        self.assertEqual(artifact["models"]["sft"]["comparisons"][0]["tasks_compared"], 2)
        # The same stored rows, with summaries that say the dialect was not normalized.
        meta = {"normalise_dialect": False}
        self.write_inputs(rows, multi_rows(), single_meta=meta, multi_meta=meta)
        with self.assertRaisesRegex(modes.ModeComparisonError, "900 run 3: replay gives"):
            self.run_command()

    def test_summaries_labeled_for_another_checkpoint_are_refused(self) -> None:
        meta = {"label": "grpo"}
        self.write_inputs(single_rows(), multi_rows(), single_meta=meta, multi_meta=meta)
        with self.assertRaisesRegex(modes.ModeComparisonError, "labeled 'grpo'"):
            self.run_command()

    def test_a_comparison_that_pairs_fewer_tasks_than_the_split_is_refused(self) -> None:
        real = compare_models.compare

        def short(*args, **kwargs):
            return {**real(*args, **kwargs), "tasks_compared": 1}

        with mock.patch.object(compare_models, "compare", short):
            with self.assertRaisesRegex(modes.ModeComparisonError, "only 1 tasks were paired"):
                self.run_command()

    def test_inputs_are_named_relative_to_results_with_hashes(self) -> None:
        artifact = self.run_command()
        recorded = artifact["inputs"]["sft"]
        files = [
            recorded["single_turn"]["summary"], recorded["single_turn"]["episodes"],
            recorded["multi_turn"]["summary"], recorded["multi_turn"]["episodes"],
        ]
        self.assertEqual([entry["file"] for entry in files], list(self.NAMES))
        for entry in files:
            digest = hashlib.sha256((self.results / entry["file"]).read_bytes()).hexdigest()
            self.assertEqual(entry["sha256"], digest)

    def test_no_absolute_paths_in_the_artifact(self) -> None:
        self.run_command()
        for fragment in (self._tmp.name, self._tmp.name.replace("\\", "/"),
                         str(PROJECT_ROOT), PROJECT_ROOT.as_posix()):
            self.assertNotIn(fragment, self.text)
        self.assertNotIn(":\\\\", self.text)

    def test_default_output_lands_beside_the_inputs(self) -> None:
        artifact = self.run_command(output=False)
        self.assertEqual(artifact["kind"], command.KIND)

    def test_stats_come_from_compare_models_unchanged(self) -> None:
        compare = mock.Mock(wraps=compare_models.compare)
        boot = mock.Mock(wraps=compare_models.bootstrap_interval)
        perm = mock.Mock(wraps=compare_models.permutation_p)
        sign = mock.Mock(wraps=compare_models.sign_test_p)
        with mock.patch.object(compare_models, "compare", compare), \
                mock.patch.object(compare_models, "bootstrap_interval", boot), \
                mock.patch.object(compare_models, "permutation_p", perm), \
                mock.patch.object(compare_models, "sign_test_p", sign):
            artifact = self.run_command()
        self.assertEqual(compare.call_count, 3)
        self.assertEqual(boot.call_count, 3)
        self.assertEqual(perm.call_count, 3)
        self.assertEqual(sign.call_count, 3)

        # Rerunning compare_models.compare on the same arms gives the same numbers.
        args, kwargs = compare.call_args_list[0]
        direct = compare_models.compare(*args, **kwargs)
        entry = artifact["models"]["sft"]["comparisons"][0]
        self.assertEqual(entry["rng_stream"], kwargs["rung"])
        self.assertEqual(entry["difference_ci95"], list(direct["difference_ci95"]))
        self.assertEqual(entry["p_permutation_two_sided"], direct["p_permutation_two_sided"])
        self.assertEqual(entry["p_sign_test_exact"], direct["p_sign_test_exact"])
        self.assertEqual(entry["single_turn_pass_power_k"], direct["baseline_pass_power_k"])
        self.assertEqual(entry["multi_turn_pass_power_k"], direct["treatment_pass_power_k"])
        self.assertEqual(artifact["bootstrap_replicates"], 10000)
        self.assertEqual(artifact["permutation_resamples"], 10000)
        self.assertEqual(artifact["seed"], compare_models.SEED)

    def test_a_different_adapter_is_refused(self) -> None:
        self.write_inputs(single_rows(), multi_rows(),
                          multi_meta={"adapter": "checkpoints/other/checkpoint-9"})
        with self.assertRaisesRegex(modes.ModeComparisonError, "adapter"):
            self.run_command()

    def test_a_log_missing_a_run_is_refused(self) -> None:
        rows = multi_rows()
        self.write_inputs(single_rows(), rows)
        log = self.results / self.NAMES[3]
        log.write_text("".join(json.dumps(r) + "\n" for r in rows[:-1]), encoding="utf-8")
        with self.assertRaisesRegex(modes.ModeComparisonError, "has runs"):
            self.run_command()

    def test_a_log_that_does_not_reproduce_its_summary_is_refused(self) -> None:
        self.write_inputs(single_rows(), multi_rows(),
                          single_meta={"metrics": {"pass^1": 0.99}})
        with self.assertRaisesRegex(modes.ModeComparisonError, "metrics"):
            self.run_command()

    def test_a_log_that_does_not_reproduce_its_multi_turn_block_is_refused(self) -> None:
        rows = multi_rows()
        tampered = summarize_multi_turn(rows, TASKS, 4)
        tampered["mean_turns_used"] += 0.5
        self.write_inputs(single_rows(), rows, multi_meta={"multi_turn": tampered})
        with self.assertRaisesRegex(modes.ModeComparisonError, "multi_turn[.]mean_turns_used"):
            self.run_command()

    def test_summaries_swapped_between_modes_are_refused(self) -> None:
        self.write_inputs(single_rows(), multi_rows(), single_meta={"mode": "multi_turn"})
        with self.assertRaisesRegex(modes.ModeComparisonError, "multi-turn run"):
            self.run_command()

    def test_an_input_outside_the_results_folder_is_refused(self) -> None:
        argv = ["--results-dir", str(self.results), "--arm", "sft", "../st.json",
                *self.NAMES[1:], "--output", str(self.output)]
        with mock.patch.object(command, "require_clean_worktree", return_value=HEAD):
            with self.assertRaisesRegex(modes.ModeComparisonError, "outside"):
                command.main(argv)

    def test_a_different_split_manifest_is_refused(self) -> None:
        with self.assertRaisesRegex(modes.ModeComparisonError, "split manifest"):
            command.load_tasks({"split_manifest_sha256": "0" * 64, "split": "test"})

    def test_refusals_are_one_line_errors(self) -> None:
        for error in (modes.ModeComparisonError, DirtyWorktreeError):
            with self.subTest(error=error.__name__):
                self.assertIn(error, command.REFUSALS)


if __name__ == "__main__":
    unittest.main()
