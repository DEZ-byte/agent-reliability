"""Combinatorial pass metrics computed from shared rollout arrays.

Both metrics are evaluated from the same rectangular task-by-run success
array.  This keeps comparisons paired and prevents separate sampling from
quietly changing the population behind either estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Iterable, TypeAlias


SuccessValue: TypeAlias = bool | int
RunSuccessArray: TypeAlias = Iterable[Iterable[SuccessValue]]


@dataclass(frozen=True, slots=True)
class PassMetrics:
    """Per-task and aggregate pass metrics for a single value of ``k``."""

    k: int
    runs_per_task: int
    pass_power_k_per_task: tuple[float, ...]
    pass_at_k_per_task: tuple[float, ...]
    pass_power_k: float
    pass_at_k: float


def compute_pass_metrics(run_successes: RunSuccessArray, k: int) -> PassMetrics:
    """Compute blueprint ``pass^k`` and ``pass@k`` estimates.

    For a task with ``c`` successes among ``n`` runs, the estimates are::

        pass^k = C(c, k) / C(n, k)
        pass@k = 1 - C(n - c, k) / C(n, k)

    Args:
        run_successes: A non-empty rectangular task-by-run array. Values must
            be booleans or integer ``0``/``1`` values.
        k: The number of draws, with ``1 <= k <= n``.

    Raises:
        ValueError: If the array or ``k`` violates the contract.
    """

    successes = _validate_and_count_successes(run_successes)
    runs_per_task = successes[0][1]

    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError("k must be an integer")
    if not 1 <= k <= runs_per_task:
        raise ValueError(f"k must satisfy 1 <= k <= {runs_per_task}")

    denominator = comb(runs_per_task, k)
    pass_power_values: list[float] = []
    pass_at_values: list[float] = []

    for success_count, _ in successes:
        all_success_combinations = (
            comb(success_count, k) if success_count >= k else 0
        )
        failure_count = runs_per_task - success_count
        all_failure_combinations = (
            comb(failure_count, k) if failure_count >= k else 0
        )

        pass_power_values.append(all_success_combinations / denominator)
        # Taking the integer complement before division is algebraically the
        # blueprint formula and makes pass^1 exactly equal to pass@1.
        pass_at_values.append(
            (denominator - all_failure_combinations) / denominator
        )

    task_count = len(successes)
    return PassMetrics(
        k=k,
        runs_per_task=runs_per_task,
        pass_power_k_per_task=tuple(pass_power_values),
        pass_at_k_per_task=tuple(pass_at_values),
        pass_power_k=sum(pass_power_values) / task_count,
        pass_at_k=sum(pass_at_values) / task_count,
    )


def _validate_and_count_successes(
    run_successes: RunSuccessArray,
) -> tuple[tuple[int, int], ...]:
    try:
        rows = tuple(tuple(row) for row in run_successes)
    except TypeError as exc:
        raise ValueError("run_successes must be an iterable of iterable rows") from exc

    if not rows:
        raise ValueError("run_successes must contain at least one task")

    runs_per_task = len(rows[0])
    if runs_per_task == 0:
        raise ValueError("each task must contain at least one run")

    counts: list[tuple[int, int]] = []
    for task_index, row in enumerate(rows):
        if len(row) != runs_per_task:
            raise ValueError(
                "run_successes must be rectangular: "
                f"task 0 has {runs_per_task} runs but task {task_index} "
                f"has {len(row)}"
            )

        success_count = 0
        for run_index, value in enumerate(row):
            if not isinstance(value, (bool, int)) or value not in (0, 1):
                raise ValueError(
                    "run_successes values must be booleans or integer 0/1; "
                    f"found {value!r} at task {task_index}, run {run_index}"
                )
            success_count += int(value)
        counts.append((success_count, runs_per_task))

    return tuple(counts)
