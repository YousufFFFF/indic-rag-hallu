"""Top-k retrieval with scores, ranks and gold flags (deck block diagram: "Top-k Passages").

Retrieval scores are carried forward into every record (CLAUDE.md §1, §3 row 4), so a
retrieval failure can be told apart from a generation failure. Results are the
schema's own :class:`~indichallu.data.schema.RetrievedPassage` objects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from types import MappingProxyType

from indichallu.data.schema import RetrievedPassage
from indichallu.retrieval.embed import Embedder
from indichallu.retrieval.index import PassageIndex, passages_fingerprint


class Retriever:
    """Embeds questions and searches a :class:`PassageIndex` over ``passages``."""

    def __init__(self, embedder: Embedder, index: PassageIndex, passages: Mapping[str, str]):
        """Accept only a consistent triple. The index must come from :meth:`build` (so its
        metadata records the full embedder config and fingerprints the passages), the
        embedder's config must match it exactly (model, revision, prefixes, max length,
        normalisation: anything that changes the vectors) and must L2-normalise (inner
        product = cosine), and ``passages`` must be exactly the texts the vectors were
        built from. A read-only snapshot of those texts is kept, so later edits to the
        caller's mapping can't change what retrieval returns."""
        cfg = embedder.config
        if not cfg.normalize:
            raise ValueError(
                f"embedder {cfg.name!r} must normalise vectors: the index ranks by inner "
                "product, which equals cosine only for unit vectors"
            )
        meta = index.meta
        if "embedder_config" not in meta:
            raise ValueError(
                "index metadata lacks 'embedder_config'; build it with Retriever.build"
            )
        if meta["embedder_config"] != asdict(cfg):
            diff = sorted(
                k for k in asdict(cfg) if meta["embedder_config"].get(k) != asdict(cfg)[k]
            )
            raise ValueError(
                f"index was built with a different embedder config (differs in {diff})"
            )
        missing = [pid for pid in index.passage_ids if pid not in passages]
        if missing:
            raise ValueError(f"{len(missing)} indexed passages have no text, e.g. {missing[:3]}")
        texts = [passages[pid] for pid in index.passage_ids]
        if meta.get("passages_fingerprint") != passages_fingerprint(index.passage_ids, texts):
            raise ValueError(
                "passage texts differ from the ones the index was built from "
                "(passages_fingerprint mismatch)"
            )
        self.embedder, self.index = embedder, index
        self.passages = MappingProxyType(dict(zip(index.passage_ids, texts, strict=True)))

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
                "embedder_config": asdict(cfg),  # everything that affects the vectors
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
