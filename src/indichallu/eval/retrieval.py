"""Retrieval metrics: Recall@k, nDCG@k, MRR (deck slide 21; CLAUDE.md §3 row 15).

Relevance is binary: a retrieved passage is relevant iff ``is_gold``. IndicQA has
exactly one gold passage per question, but the functions accept any number.

* Recall@k = |relevant in top k| / |relevant|
* nDCG@k  = DCG@k / IDCG@k, with DCG = sum(rel_i / log2(i + 1)) over ranks i = 1..k
* MRR     = mean over queries of 1 / (rank of the first relevant passage), 0 if none

Queries with no relevant passage are excluded (and counted), because the metrics are
undefined for them. A query whose retrieved list is shorter than ``k`` while more
results existed would bias Recall@k, so callers must retrieve at least ``k`` passages:
lists shorter than ``k`` raise unless the whole corpus is smaller than ``k``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from indichallu.data.schema import Record


@dataclass(frozen=True)
class Query:
    """One query's ranked results as relevance flags (rank 1 first)."""

    relevant_flags: tuple[bool, ...]
    n_relevant: int  # total relevant passages for the query (retrieved or not)


def query_from_record(record: Record, n_relevant: int = 1) -> Query:
    """IndicQA: each question has exactly one gold passage, retrieved or not."""
    flags = tuple(p.is_gold for p in sorted(record.retrieved, key=lambda p: p.rank))
    if sum(flags) > n_relevant:
        raise ValueError(
            f"{record.id}: {sum(flags)} gold passages retrieved, expected <= {n_relevant}"
        )
    return Query(flags, n_relevant=n_relevant)


def recall_at_k(q: Query, k: int) -> float:
    return sum(q.relevant_flags[:k]) / q.n_relevant


def ndcg_at_k(q: Query, k: int) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, rel in enumerate(q.relevant_flags[:k]) if rel)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(q.n_relevant, k)))
    return dcg / idcg


def reciprocal_rank(q: Query) -> float:
    return next((1 / (i + 1) for i, rel in enumerate(q.relevant_flags) if rel), 0.0)


def retrieval_metrics(
    queries: Sequence[Query], ks: Sequence[int] = (1, 3, 5, 10), corpus_size: int | None = None
) -> dict[str, float | int | None]:
    """Mean Recall@k and nDCG@k for each k, plus MRR, over queries with a relevant passage."""
    for k in ks:
        if k < 1:
            raise ValueError("k must be >= 1")
    usable = [q for q in queries if q.n_relevant > 0]
    needed = max(ks) if corpus_size is None else min(max(ks), corpus_size)
    for q in usable:
        if len(q.relevant_flags) < needed:
            raise ValueError(
                f"a query has {len(q.relevant_flags)} retrieved passages but needs {needed} "
                f"for k={max(ks)}; retrieve more (or pass corpus_size if the corpus is smaller)"
            )
    out: dict[str, float | int | None] = {
        "n_queries": len(usable),
        "n_excluded": len(queries) - len(usable),
    }
    for k in ks:
        out[f"recall@{k}"] = _mean([recall_at_k(q, k) for q in usable])
        out[f"ndcg@{k}"] = _mean([ndcg_at_k(q, k) for q in usable])
    out["mrr"] = _mean([reciprocal_rank(q) for q in usable])
    return out


def retrieval_metrics_from_records(
    records: Sequence[Record], ks: Sequence[int] = (1, 3, 5, 10), corpus_size: int | None = None
) -> dict[str, float | int | None]:
    """IndicQA records: one gold passage per question; ``is_gold`` marks it if retrieved."""
    return retrieval_metrics([query_from_record(r) for r in records], ks, corpus_size)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
