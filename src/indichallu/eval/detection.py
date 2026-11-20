"""Detection metrics: Precision, Recall, F1, Balanced accuracy, AUROC (deck slide 21).

The positive class is *hallucinated*, the minority class (slide 21: plain accuracy would
look excellent while missing most of them). AUROC uses a continuous hallucination
score, not the thresholded label. Any metric that is undefined on the given data
(no positives, one class only) is ``None``, never a made-up 0 or 1.

Response level (RAGTruth-style):

* gold: ``labels.response == "hallucinated"``;
* predicted label: any claim whose verdict is not SUPPORTED;
* predicted score: the maximum claim hallucination score, where a claim's score is
  ``1 - entail`` if NLI scores are present, else 1.0 for a non-SUPPORTED verdict and
  0.0 for SUPPORTED. A response with no claims scores 0.0.

Claim-level and per-type evaluation work on spans: see :mod:`indichallu.eval.spans`.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import roc_auc_score

from indichallu.data.schema import Claim, Record, ResponseLabel, Verdict


def binary_metrics(
    y_true: Sequence[bool], y_pred: Sequence[bool], y_score: Sequence[float] | None = None
) -> dict[str, float | int | None]:
    """P/R/F1 of the positive class, balanced accuracy and AUROC, plus the counts."""
    if len(y_true) != len(y_pred) or (y_score is not None and len(y_score) != len(y_true)):
        raise ValueError("y_true, y_pred (and y_score) must have the same length")
    tp = sum(t and p for t, p in zip(y_true, y_pred, strict=True))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred, strict=True))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred, strict=True))
    tn = len(y_true) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    # 2TP / (2TP + FP + FN): the harmonic mean of P and R when both are defined; 0 when
    # there are positives (gold or predicted) but no hits; undefined with no positives.
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    tnr = tn / (tn + fp) if tn + fp else None
    balanced = (recall + tnr) / 2 if recall is not None and tnr is not None else None
    auroc = None
    if y_score is not None and 0 < sum(map(bool, y_true)) < len(y_true):
        auroc = float(roc_auc_score([bool(t) for t in y_true], list(y_score)))
    return {
        "n": len(y_true),
        "positives": tp + fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced,
        "auroc": auroc,
    }


def claim_score(claim: Claim) -> float:
    """Continuous hallucination score of one claim (higher = more likely hallucinated)."""
    if claim.scores is not None:
        return 1.0 - claim.scores.entail
    return 0.0 if claim.verdict is Verdict.SUPPORTED else 1.0


def response_prediction(record: Record) -> tuple[bool, float]:
    """(predicted hallucinated?, score) for one answer, from its claims."""
    label = any(c.verdict is not Verdict.SUPPORTED for c in record.claims)
    score = max((claim_score(c) for c in record.claims), default=0.0)
    return label, score


def response_level_metrics(records: Sequence[Record]) -> dict[str, float | int | None]:
    """Response-level detection metrics over labelled records."""
    unlabelled = [r.id for r in records if r.labels is None]
    if unlabelled:
        raise ValueError(f"{len(unlabelled)} records have no gold labels, e.g. {unlabelled[:3]}")
    y_true = [r.labels.response is ResponseLabel.HALLUCINATED for r in records]
    preds = [response_prediction(r) for r in records]
    return binary_metrics(y_true, [p for p, _ in preds], [s for _, s in preds])
