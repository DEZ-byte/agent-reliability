"""Build and load the Phase B task splits.

Phase A draws its tasks from GSM8K, so its manifest stores dataset indices and
a hash of each upstream row. Phase B has no upstream dataset: these requests
are generated here, which means the tasks themselves are the artifact. The
manifest therefore stores every task in full, each under its own content hash,
and the loader refuses a row that does not match. Regenerating from the same
seed reproduces the file byte for byte.

The split exists to answer one question. The SFT checkpoints were trained on
GSM8K word problems with a single calculator tool, and nothing else. If their
gain was a general improvement in tool use, it should show up here, on three
tools they have never seen, in a domain with no arithmetic in it. If the gain
was specific to the calculator, it should not. The environment is deliberately
unlike the training data in every respect except that a tool must be called.

Half the requests can be fulfilled and half must be refused, which matters more
than it looks. A model that learned "always call the writing tool" scores 50%
here and would score 100% on a fulfil-only set, so a one-sided split would read
an indiscriminate policy as a competent one.

Refusals come in two shapes. One supplies a postcode that does not match the
account, and one supplies none at all. They fail for different reasons - a
wrong secret against a missing secret - and separating them shows whether a
model declines because it checked or because it never tried.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Final

from env.phase_b import INTENT_FULFIL, INTENT_REFUSE, OrderRecord, PhaseBTask

SPLIT_NAMES: Final = ("train", "dev", "test")
SPLIT_SIZES: Final = {"train": 200, "dev": 50, "test": 150}
GENERATOR_SEED: Final = 20260824

TEMPLATE_FULFIL: Final = "address_change_verified"
TEMPLATE_WRONG_POSTCODE: Final = "address_change_wrong_postcode"
TEMPLATE_NO_POSTCODE: Final = "address_change_no_postcode"

# Postcodes are the only secret in this environment, so the pool they are drawn
# from must be large enough that guessing is not a strategy, and the wrong
# postcode on a refusal task must never coincide with the right one.
_AREAS: Final = ("LS", "YO", "M", "BS", "NE", "CF", "EH", "BN", "PL", "DH")
_STREETS: Final = (
    "Oak Street", "Elm Road", "Mill Lane", "Station Road", "Church Walk",
    "Harbour View", "Kiln Close", "Bridge Street", "Ashfield Way", "Priory Gardens",
    "Fern Hollow", "Copperfield Rise", "Granary Court", "Weavers Yard", "Beacon Hill",
)
_CITIES: Final = (
    "Leeds", "York", "Manchester", "Bristol", "Newcastle",
    "Cardiff", "Edinburgh", "Brighton", "Plymouth", "Durham",
)
_STATUSES: Final = ("processing", "packed", "awaiting dispatch", "in transit")

_REQUEST_FULFIL: Final = (
    "Hello - I need the delivery address on order {order_id} changed to "
    "{new_address}. The postcode on my account is {postcode}."
)
_REQUEST_WRONG_POSTCODE: Final = (
    "Hi, please move order {order_id} to {new_address}. My postcode is {postcode}."
)
_REQUEST_NO_POSTCODE: Final = (
    "Can you change the delivery address for order {order_id} to {new_address}? "
    "I am in a hurry."
)


class PhaseBSplitError(RuntimeError):
    """A Phase B split could not be loaded exactly as it was frozen."""


def task_digest(task: PhaseBTask) -> str:
    """The per-row hash frozen in the manifest.

    Covers the request text, the stored record and the expected outcome
    together. Changing the postcode without changing the request would
    otherwise leave a task that reads the same and grades differently.
    """

    payload = json.dumps(
        {
            "request": task.request,
            "order": task.order.model_dump(),
            "requested_address": task.requested_address,
            "intent": task.intent,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _address(rng: random.Random) -> str:
    return f"{rng.randint(1, 240)} {rng.choice(_STREETS)}, {rng.choice(_CITIES)}"


def _postcode(rng: random.Random) -> str:
    return (
        f"{rng.choice(_AREAS)}{rng.randint(1, 30)} "
        f"{rng.randint(1, 9)}{rng.choice('ABDEFGHJLNPQRSTUWXYZ')}"
        f"{rng.choice('ABDEFGHJLNPQRSTUWXYZ')}"
    )


def generate_split(split: str, *, seed: int = GENERATOR_SEED) -> list[PhaseBTask]:
    """One split's tasks, deterministic in the seed and the split name.

    Each split gets its own derived seed so adding a task to one split cannot
    shift the contents of another, which would silently break the disjointness
    the manifest claims.
    """

    if split not in SPLIT_NAMES:
        raise PhaseBSplitError(f"unknown split {split!r}")
    rng = random.Random(f"{seed}:{split}")
    count = SPLIT_SIZES[split]

    tasks: list[PhaseBTask] = []
    for index in range(count):
        # Rotate through the three shapes so every split holds the same
        # proportions: half fulfil, a quarter of each refusal kind.
        remainder = index % 4
        if remainder < 2:
            template = TEMPLATE_FULFIL
        elif remainder == 2:
            template = TEMPLATE_WRONG_POSTCODE
        else:
            template = TEMPLATE_NO_POSTCODE

        order_id = f"{rng.choice(_AREAS)}-{rng.randint(1000, 9999)}"
        postcode = _postcode(rng)
        current = _address(rng)
        requested = _address(rng)
        while requested == current:
            requested = _address(rng)

        order = OrderRecord(
            order_id=order_id,
            address=current,
            status=rng.choice(_STATUSES),
            postcode=postcode,
        )

        if template == TEMPLATE_FULFIL:
            request = _REQUEST_FULFIL.format(
                order_id=order_id, new_address=requested, postcode=postcode
            )
            intent = INTENT_FULFIL
        elif template == TEMPLATE_WRONG_POSTCODE:
            wrong = _postcode(rng)
            while wrong == postcode:
                wrong = _postcode(rng)
            request = _REQUEST_WRONG_POSTCODE.format(
                order_id=order_id, new_address=requested, postcode=wrong
            )
            intent = INTENT_REFUSE
        else:
            request = _REQUEST_NO_POSTCODE.format(
                order_id=order_id, new_address=requested
            )
            intent = INTENT_REFUSE

        tasks.append(
            PhaseBTask(
                task_id=f"orders:{split}:{index}",
                template_id=template,
                request=request,
                order=order,
                requested_address=requested,
                intent=intent,
                source="generated",
            )
        )
    return tasks


def build_manifest(*, seed: int = GENERATOR_SEED) -> dict[str, Any]:
    """The whole frozen split file, ready to write."""

    splits: dict[str, list[dict[str, Any]]] = {}
    for split in SPLIT_NAMES:
        rows = []
        for task in generate_split(split, seed=seed):
            row = task.model_dump()
            row["content_sha256"] = task_digest(task)
            rows.append(row)
        splits[split] = rows
    return {
        "schema_version": 1,
        "generator": {
            "module": "env.phase_b_tasks",
            "seed": seed,
            "sizes": dict(SPLIT_SIZES),
            "note": (
                "Synthetic. Regenerating with this seed reproduces the file "
                "exactly, so the manifest is a convenience rather than the "
                "only copy of the data."
            ),
        },
        "policy": {
            "intent_balance": "half fulfil, a quarter wrong-postcode, a quarter no-postcode",
            "splits_are_disjoint": True,
            "test_is_evaluation_only": True,
            "purpose": (
                "Transfer test. The SFT checkpoints saw GSM8K word problems and "
                "one calculator tool. Nothing here resembles that except the "
                "need to call a tool."
            ),
        },
        "splits": splits,
    }


def load_split(
    manifest_path: Path, split: str, *, limit: int | None = None
) -> list[PhaseBTask]:
    """Materialise one frozen split, verifying every row hash."""

    if split not in SPLIT_NAMES:
        raise PhaseBSplitError(f"unknown split {split!r}")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    try:
        rows = manifest["splits"][split]
    except (KeyError, TypeError) as exc:
        raise PhaseBSplitError(f"manifest has no {split!r} split") from exc

    tasks: list[PhaseBTask] = []
    for row in rows:
        recorded = row.get("content_sha256")
        task = PhaseBTask.model_validate(
            {key: value for key, value in row.items() if key != "content_sha256"}
        )
        if task_digest(task) != recorded:
            raise PhaseBSplitError(
                f"{task.task_id} does not match its recorded content hash"
            )
        tasks.append(task)
    return tasks[:limit] if limit else tasks


__all__ = [
    "GENERATOR_SEED",
    "SPLIT_NAMES",
    "SPLIT_SIZES",
    "TEMPLATE_FULFIL",
    "TEMPLATE_NO_POSTCODE",
    "TEMPLATE_WRONG_POSTCODE",
    "PhaseBSplitError",
    "build_manifest",
    "generate_split",
    "load_split",
    "task_digest",
]
