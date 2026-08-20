"""Decide which graded trajectories are safe to train on.

The grader says whether an episode reached the right number. That is necessary
and not sufficient for training data, because D-062 recorded a way to reach the
right number without computing it: solve the problem in your head and hand the
answer to the calculator. `calculator("391")` scores +1.0 having computed
nothing.

`evaluation.contamination.expression_does_arithmetic` catches the bare form. It
does not catch the decorated form, and decoration is one character of effort:
`391 + 0` and `391 * 1` both parse as arithmetic and both compute nothing. A
filter that misses them lets the training set fill with answer-first
reconstructions, which is precisely the behaviour BLUEPRINT_v2 section 5.4 says
must not be rewarded.

So retention asks a harder question: does this expression look like it was
built from the numbers in the problem, or from the answer? The signals are
deliberately simple and checkable, because a filter nobody can reason about is
a filter nobody will notice has broken.

Thresholds are arguments rather than constants. They are pinned in
`configs/train_config.yaml` after being measured on the dev split, so a
threshold cannot be chosen to make a particular dataset look better.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Final, Iterable

from evaluation.contamination import expression_does_arithmetic

# Numbers as they appear in GSM8K prose: 1,000 and 2.5 and 15 all count.
_NUMBER_PATTERN: Final = re.compile(r"\d[\d,]*(?:\.\d+)?")

# Fewer operands than this cannot be a computation: a lone constant with a sign
# is a restatement. Occurrences are counted rather than distinct values, because
# `2 + 2*2` is genuine work on a question that says "2" once and "twice" once,
# and a distinct-value rule rejects it.
MIN_LITERAL_OCCURRENCES: Final = 2


@dataclass(frozen=True, slots=True)
class LaunderingVerdict:
    """Why one expression was accepted or rejected as genuine computation."""

    laundered: bool
    reason: str | None
    literals: tuple[float, ...]
    question_literals_matched: int
    gold_appears_in_question: bool

    @property
    def retained(self) -> bool:
        return not self.laundered


def numbers_in_text(text: str) -> tuple[float, ...]:
    """Every number written in the problem statement.

    Thousands separators are stripped so `1,000` in prose matches `1000` in an
    expression; a mismatch there would reject correct work as laundering.
    """

    found: list[float] = []
    for match in _NUMBER_PATTERN.finditer(text):
        try:
            found.append(float(match.group(0).replace(",", "")))
        except ValueError:  # pragma: no cover - the pattern cannot produce this
            continue
    return tuple(found)


def numeric_literals(expression: str) -> tuple[float, ...]:
    """Every numeric constant in an expression, with unary minus applied.

    Read from the parsed tree rather than by regex so that `2e3`, nested
    parentheses and negative constants are all counted the way Python will
    evaluate them.
    """

    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        return ()

    literals: list[float] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            operand = node.operand
            if isinstance(operand, ast.Constant) and isinstance(
                operand.value, (int, float)
            ) and not isinstance(operand.value, bool):
                literals.append(-float(operand.value))
        elif isinstance(node, ast.Constant) and isinstance(
            node.value, (int, float)
        ) and not isinstance(node.value, bool):
            literals.append(float(node.value))

    # A negated constant contributes both its own node and the positive one, so
    # drop the positive twin rather than counting the operand twice.
    negatives = [value for value in literals if value < 0]
    for value in negatives:
        if -value in literals:
            literals.remove(-value)
    return tuple(literals)


def _close(left: float, right: float, tolerance: float) -> bool:
    return abs(left - right) <= tolerance


def laundering_verdict(
    *,
    expression: str,
    question: str,
    gold_answer: float,
    tolerance: float = 1e-6,
    min_question_match_ratio: float = 0.5,
) -> LaunderingVerdict:
    """Judge whether an expression computed the answer or restated it.

    Four rejections, in order of how certain they are:

    1. It performs no arithmetic at all. The bare case D-062 already named.
    2. It has fewer than two operands, so nothing was combined.
    3. The gold answer appears as one of its literals. Writing the answer into
       the expression is the decorated form, whatever operator surrounds it.
       Allowed only when that number is also in the problem statement, since
       then it is not evidence of anything.
    4. Too few of its literals appear in the problem. An expression assembled
       from numbers the question never mentions was not derived from the
       question.
    """

    literals = numeric_literals(expression)
    question_numbers = numbers_in_text(question)
    gold_in_question = any(
        _close(value, gold_answer, tolerance) for value in question_numbers
    )
    matched = sum(
        1
        for value in literals
        if any(_close(value, other, tolerance) for other in question_numbers)
    )

    def verdict(laundered: bool, reason: str | None) -> LaunderingVerdict:
        return LaunderingVerdict(
            laundered=laundered,
            reason=reason,
            literals=literals,
            question_literals_matched=matched,
            gold_appears_in_question=gold_in_question,
        )

    if not expression_does_arithmetic(expression):
        return verdict(True, "no_arithmetic")

    if len(literals) < MIN_LITERAL_OCCURRENCES:
        return verdict(True, "single_operand")

    if not gold_in_question and any(
        _close(value, gold_answer, tolerance) for value in literals
    ):
        return verdict(True, "gold_answer_is_a_literal")

    if literals and matched / len(literals) < min_question_match_ratio:
        return verdict(True, "literals_absent_from_question")

    return verdict(False, None)


def rejection_counts(verdicts: Iterable[LaunderingVerdict]) -> dict[str, int]:
    """How many candidates each rule rejected. Reported, never suppressed."""

    counts: dict[str, int] = {}
    for verdict in verdicts:
        key = verdict.reason or "retained"
        counts[key] = counts.get(key, 0) + 1
    return counts


__all__ = [
    "MIN_LITERAL_OCCURRENCES",
    "LaunderingVerdict",
    "laundering_verdict",
    "numbers_in_text",
    "numeric_literals",
    "rejection_counts",
]
