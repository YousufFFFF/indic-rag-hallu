"""Top-k retrieval with scores, ranks and gold flags (deck block diagram: "Top-k Passages").

Retrieval scores are carried forward into every record (CLAUDE.md §1, §3 row 4), so a
retrieval failure can be told apart from a generation failure. Results are the
schema's own :class:`~indichallu.data.schema.RetrievedPassage` objects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from indichallu.data.schema import RetrievedPassage
from indichallu.retrieval.embed import Embedder
from indichallu.retrieval.index import PassageIndex, passages_fingerprint


class Retriever:
    """Embeds questions and searches a :class:`PassageIndex` over ``passages``."""

    def __init__(self, embedder: Embedder, index: PassageIndex, passages: Mapping[str, str]):
        missing = [pid for pid in index.passage_ids if pid not in passages]
        if missing:
            raise ValueError(f"{len(missing)} indexed passages have no text, e.g. {missing[:3]}")
        meta = index.meta
        cfg = embedder.config
        for key, value in (("model_id", cfg.model_id), ("revision", cfg.revision)):
            if key in meta and meta[key] != value:
                raise ValueError(f"index built with {key}={meta[key]!r}, embedder has {value!r}")
        self.embedder, self.index, self.passages = embedder, index, passages

    @classmethod
    def build(
        cls, embedder: Embedder, passage_ids: Sequence[str], texts: Sequence[str]
    ) -> Retriever:
        """Embed a corpus and index it, recording which embedder produced the vectors."""
        ids, texts = list(passage_ids), list(texts)
        cfg = embedder.config
        index = PassageIndex.build(
            embedder.embed_passages(texts),
            ids,
            meta={
                "embedder": cfg.name,
                "model_id": cfg.model_id,
                "revision": cfg.revision,
                "passages_fingerprint": passages_fingerprint(ids, texts),
            },
        )
        return cls(embedder, index, dict(zip(ids, texts, strict=True)))

    def retrieve(
        self,
        questions: Sequence[str],
        k: int,
        gold_passage_ids: Sequence[str | None] | None = None,
    ) -> list[list[RetrievedPassage]]:
        """Top-``k`` passages per question, ranked 1..k, with ``is_gold`` set."""
        golds = list(gold_passage_ids) if gold_passage_ids is not None else [None] * len(questions)
        if len(golds) != len(questions):
            raise ValueError("need one gold passage id (or None) per question")
        hits = self.index.search(self.embedder.embed_queries(list(questions)), k)
        return [
            [
                RetrievedPassage(
                    passage_id=pid,
                    text=self.passages[pid],
                    score=score,
                    rank=rank,
                    is_gold=pid == gold,
                )
                for rank, (pid, score) in enumerate(row, start=1)
            ]
            for row, gold in zip(hits, golds, strict=True)
        ]


def gold_rank(retrieved: Sequence[RetrievedPassage]) -> int | None:
    """Rank of the gold passage in a retrieval result, or None if it was not retrieved."""
    return next((p.rank for p in retrieved if p.is_gold), None)
