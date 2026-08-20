"""Measuring how much of GSM8K the base models already remember.

BLUEPRINT_v2 section 5.4 requires this before any Phase A baseline is read.
The models under test were trained on public web text, and GSM8K is public web
text. A score that looks like reasoning may be recall.

Two things follow, and only the first is obvious.

Execution-backed accuracy (D-060) already stops *prose* recall from scoring:
writing "the answer is 391" earns nothing. But it does not stop a model from
recalling 391 and then laundering it through the calculator as
``calculator("391")``. That call executes, returns 391, and scores correct
while performing no arithmetic. So this module measures recall directly, and
also classifies whether a tool expression did any work.
"""

from __future__ import annotations

import ast
import re
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

# Answers appear in prose in many shapes: "= 391", "**391**", "391.", "1,234".
_NUMBER_RE: Final = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

_ARITHMETIC_NODES: Final = (
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Compare,
    ast.IfExp,
)


class NoToolProbe(BaseModel):
    """One no-tool attempt at a task, scored against the gold answer.

    ``condition`` matters more than the score. Given room to think, a model can
    solve GSM8K in prose, and a correct answer says nothing about memorisation.
    Only the token-starved condition, where multi-step reasoning does not fit,
    is evidence that the answer was recalled rather than derived.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    task_id: str = Field(min_length=1)
    condition: Literal["unconstrained", "token_starved"]
    gold_answer: float
    completion: str
    extracted_answer: float | None
    correct: bool


def extract_final_number(text: str) -> float | None:
    """Take the last number a completion states.

    Reading prose is normally forbidden here. This function is the deliberate
    exception: the quantity being measured *is* whether the model can state the
    answer without computing it. It must never be used to score a task.
    """

    matches = _NUMBER_RE.findall(text)
    while matches:
        candidate = matches.pop().replace(",", "")
        try:
            return float(candidate)
        except ValueError:
            continue
    return None


def expression_does_arithmetic(expression: str) -> bool:
    """True when an expression computes something rather than restating it.

    ``17 * 23`` computes. ``391`` does not, and neither does ``(391)``. A model
    that recalls the answer and wraps it in a tool call produces the second
    kind, which is how memorisation can survive execution-backed grading.
    """

    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        return False
    return any(isinstance(node, _ARITHMETIC_NODES) for node in ast.walk(tree))


def score_no_tool_attempt(
    *,
    task_id: str,
    condition: Literal["unconstrained", "token_starved"],
    gold_answer: float,
    completion: str,
    tolerance: float,
) -> NoToolProbe:
    """Score one no-tool completion under a named condition."""

    extracted = extract_final_number(completion)
    correct = extracted is not None and abs(extracted - gold_answer) <= tolerance
    return NoToolProbe(
        task_id=task_id,
        condition=condition,
        gold_answer=gold_answer,
        completion=completion,
        extracted_answer=extracted,
        correct=correct,
    )


def correct_rate(probes: list[NoToolProbe]) -> float | None:
    """Fraction answered correctly. Read it together with the condition."""

    if not probes:
        return None
    return sum(1 for probe in probes if probe.correct) / len(probes)
