"""Span localisation metrics: span-level F1, token IoU, exact-span match (deck slide 21).

All metrics compare *hallucinated* spans on the ORIGINAL answer text (CLAUDE.md §2):

* gold spans: ``record.labels.spans``;
* predicted spans: claims whose verdict is not SUPPORTED, with their predicted ``type``.

Definitions (micro-averaged over all records, so long answers weigh more):

* **span-level F1 (character):** precision/recall over hallucinated character positions;
* **token IoU:** |gold ∩ predicted| / |gold ∪ predicted| over hallucinated *tokens*, where
  a token (from :func:`indichallu.text.sentseg.tokens`) counts as hallucinated if any of
  its characters is;
* **exact-span match:** P/R/F1 over spans, a predicted span matching a gold span only if
  ``start`` and ``end`` are identical.

**Per type** (CLAUDE.md §3 row 11): gold spans of a type vs predicted spans with the
same predicted type, with the same three metrics. Undefined values are ``None``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from indichallu.data.schema import Record, SpanType, Verdict
from indichallu.text.sentseg import tokens

Span = tuple[int, int]


def predicted_spans(record: Record, span_type: SpanType | None = None) -> list[Span]:
    return [
        (c.start, c.end)
        for c in record.claims
        if c.verdict is not Verdict.SUPPORTED and (span_type is None or c.type is span_type)
    ]


def gold_spans(record: Record, span_type: SpanType | None = None) -> list[Span]:
    if record.labels is None:
        raise ValueError(f"{record.id}: no gold labels")
    return [
        (s.start, s.end) for s in record.labels.spans if span_type is None or s.type is span_type
    ]


def _chars(spans: Iterable[Span]) -> set[int]:
    return {i for start, end in spans for i in range(start, end)}


def _token_set(answer: str, chars: set[int]) -> set[int]:
    return {
        k for k, t in enumerate(tokens(answer)) if any(i in chars for i in range(t.start, t.end))
    }


def _prf(tp: int, n_pred: int, n_gold: int) -> tuple[float | None, float | None, float | None]:
    p = tp / n_pred if n_pred else None
    r = tp / n_gold if n_gold else None
    f1 = 2 * tp / (n_pred + n_gold) if n_pred + n_gold else None  # = 2TP / (2TP + FP + FN)
    return p, r, f1


def span_metrics(
    records: Sequence[Record], span_type: SpanType | None = None
) -> dict[str, float | int | None]:
    char_tp = char_pred = char_gold = 0
    tok_inter = tok_union = 0
    exact_tp = exact_pred = exact_gold = 0
    for r in records:
        g, p = gold_spans(r, span_type), predicted_spans(r, span_type)
        gc, pc = _chars(g), _chars(p)
        char_tp, char_pred, char_gold = (
            char_tp + len(gc & pc),
            char_pred + len(pc),
            char_gold + len(gc),
        )
        gt, pt = _token_set(r.answer, gc), _token_set(r.answer, pc)
        tok_inter, tok_union = tok_inter + len(gt & pt), tok_union + len(gt | pt)
        # one-to-one: each gold span can be matched by at most one predicted span with the
        # same offsets; duplicate predictions are counted (and the extras are false positives)
        exact_tp += sum((Counter(g) & Counter(p)).values())
        exact_pred, exact_gold = exact_pred + len(p), exact_gold + len(g)
    cp, cr, cf = _prf(char_tp, char_pred, char_gold)
    ep, er, ef = _prf(exact_tp, exact_pred, exact_gold)
    return {
        "n_records": len(records),
        "gold_spans": exact_gold,
        "predicted_spans": exact_pred,
        "span_precision": cp,
        "span_recall": cr,
        "span_f1": cf,
        "token_iou": tok_inter / tok_union if tok_union else None,
        "exact_precision": ep,
        "exact_recall": er,
        "exact_f1": ef,
    }


def span_metrics_by_type(records: Sequence[Record]) -> dict[str, dict[str, float | int | None]]:
    """All types together (``"ALL"``) and each of the four error types separately."""
    out = {"ALL": span_metrics(records)}
    for t in SpanType:
        out[t.value] = span_metrics(records, t)
    return out
