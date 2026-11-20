"""Decision-threshold tuning on the VAL split only (deck slide 17; CLAUDE.md §3 row 9).

Records are scored once (claims with their full claim x passage matrix); every grid point
then re-decides the verdicts from those stored scores with :func:`redecide`, without
re-running the NLI model. The objective is span-level (character) F1 over all error
types; ties go to response-level F1, then to the higher entailment threshold. The tuned
values are recorded in docs/DECISIONS.md and reported.
"""

from __future__ import annotations

from collections.abc import Sequence

from indichallu.data.schema import Record, Split
from indichallu.detection.aggregate import Thresholds, decide
from indichallu.eval.detection import response_level_metrics
from indichallu.eval.spans import span_metrics

DEFAULT_GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))  # 0.05 .. 0.95


def redecide(record: Record, th: Thresholds) -> Record:
    claims = []
    for c in record.claims:
        if not c.passage_scores:
            raise ValueError(f"{record.id}: claims need passage_scores to be re-decided")
        claims.append(decide(c.text, c.start, c.end, c.passage_scores, th).model_dump())
    return Record.model_validate({**record.model_dump(), "claims": claims})


def tune_thresholds(
    records: Sequence[Record],
    entail_grid: Sequence[float] = DEFAULT_GRID,
    contradict_grid: Sequence[float] = DEFAULT_GRID,
) -> tuple[Thresholds, list[dict]]:
    """Best thresholds on validation records, and the score of every grid point."""
    bad = sorted({r.split.value for r in records if r.split is not Split.VAL})
    if bad:
        raise ValueError(f"thresholds are tuned on the val split only; got {bad}")
    if not records:
        raise ValueError("no validation records")
    table = []
    for e in entail_grid:
        for c in contradict_grid:
            th = Thresholds(entail=e, contradict=c)
            decided = [redecide(r, th) for r in records]
            span_f1 = span_metrics(decided)["span_f1"]
            resp_f1 = response_level_metrics(decided)["f1"]
            table.append({"entail": e, "contradict": c, "span_f1": span_f1, "response_f1": resp_f1})
    best = max(table, key=lambda r: (r["span_f1"] or 0.0, r["response_f1"] or 0.0, r["entail"]))
    return Thresholds(entail=best["entail"], contradict=best["contradict"]), table
