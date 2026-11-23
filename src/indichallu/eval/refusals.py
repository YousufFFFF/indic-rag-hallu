"""Refusals as their own class (DECISIONS D-028, D-029).

A refusal is **correct** if the answer was not available to the generator: the gold
passage was not among the passages shown, or the question has no answer in it
(``answerable is False``). Refusals are excluded from detection metrics (use
:func:`answered`) and reported here.
"""

from __future__ import annotations

from collections.abc import Sequence

from indichallu.data.schema import Record, ResponseLabel


def answer_available(r: Record) -> bool:
    """Could the generator have answered from what it was shown?"""
    return r.answerable is not False and any(p.is_gold for p in r.retrieved)


def answered(records: Sequence[Record]) -> list[Record]:
    """Records that are scored as answers: neither predicted nor labelled refusals."""
    return [
        r
        for r in records
        if not r.refusal and not (r.labels and r.labels.response is ResponseLabel.REFUSAL)
    ]


def refusal_metrics(records: Sequence[Record]) -> dict[str, float | int | None]:
    """How often the system refused, and whether it refused when it should have."""
    unchecked = [r.id for r in records if r.refusal is None]
    if unchecked:
        raise ValueError(
            f"{len(unchecked)} records were not checked for refusals (refusal is None), "
            f"e.g. {unchecked[:3]}; detect with refusals='separate'"
        )
    refused = [r for r in records if r.refusal]
    should = [r for r in records if not answer_available(r)]
    correct = sum(not answer_available(r) for r in refused)
    return {
        "n": len(records),
        "n_refused": len(refused),
        "n_correct_refusals": correct,
        "n_incorrect_refusals": len(refused) - correct,
        "n_answer_unavailable": len(should),
        "refusal_rate": len(refused) / len(records) if records else None,
        # of the refusals, how many were right to refuse
        "refusal_precision": correct / len(refused) if refused else None,
        # of the answers that weren't available, how many were refused
        "refusal_recall": sum(bool(r.refusal) for r in should) / len(should) if should else None,
    }
