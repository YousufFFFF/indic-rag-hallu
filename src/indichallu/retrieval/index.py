"""Vector index over a passage corpus (deck block diagram: "Vector Index (Indic corpus)").

An exact FAISS inner-product index (``IndexFlatIP``). The embedder L2-normalises, so
inner product = cosine. The corpus is small (a few hundred passages per language), so
there is no need for approximate search.

Search results are made deterministic: hits are re-sorted by (score descending, corpus
position), so equal scores always come back in the same order.

A saved index is a directory holding versioned subdirectories ``v-*/`` (each with
``index.faiss``, git-ignored by ``*.faiss``; ``ids.json``; ``meta.json``) and a
``CURRENT`` file naming the active one. Saving writes a complete new version, then
switches ``CURRENT`` with one atomic file replace, so a crash at any point leaves the
previous index loadable and readers never see a half-written or missing index. The
metadata records the embedder that produced the vectors and a fingerprint of the
passages, and :meth:`PassageIndex.load` checks it all fits together.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np

FORMAT_VERSION = 1


def passages_fingerprint(passage_ids: list[str], texts: list[str]) -> str:
    """SHA-256 over the (id, text) pairs in corpus order."""
    h = hashlib.sha256()
    for pid, text in zip(passage_ids, texts, strict=True):
        for part in (pid, text):
            data = part.encode("utf-8")
            h.update(len(data).to_bytes(8, "little"))
            h.update(data)
    return h.hexdigest()


@dataclass
class PassageIndex:
    index: faiss.Index
    passage_ids: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.index.ntotal != len(self.passage_ids):
            raise ValueError(
                f"index has {self.index.ntotal} vectors but {len(self.passage_ids)} ids"
            )
        if len(set(self.passage_ids)) != len(self.passage_ids):
            raise ValueError("passage ids must be unique")
        self._position = {pid: i for i, pid in enumerate(self.passage_ids)}

    @property
    def dim(self) -> int:
        return self.index.d

    def __len__(self) -> int:
        return len(self.passage_ids)

    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        passage_ids: list[str],
        meta: dict[str, Any] | None = None,
    ) -> PassageIndex:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(passage_ids):
            raise ValueError(f"need one vector per passage, got {vectors.shape}")
        if not np.isfinite(vectors).all():
            raise ValueError("vectors must be finite")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        meta = {**(meta or {}), "dim": int(vectors.shape[1]), "size": len(passage_ids)}
        return cls(index, list(passage_ids), meta)

    def search(self, queries: np.ndarray, k: int) -> list[list[tuple[str, float]]]:
        """Top-``k`` (passage_id, score) per query, best first, ties by corpus position."""
        if k < 1:
            raise ValueError("k must be >= 1")
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError(f"queries must have shape (m, {self.dim}), got {queries.shape}")
        if not np.isfinite(queries).all():
            raise ValueError("queries must be finite")
        n = len(self)
        k = min(k, n)
        if k == 0 or len(queries) == 0:
            return [[] for _ in range(len(queries))]
        # Ask for extra candidates so ties at the k-th score are visible; FAISS's own
        # choice among tied candidates at its cut-off is not specified.
        k2 = min(n, max(2 * k, k + 10))
        scores, idx = self.index.search(queries, k2)
        out = []
        for q, row_scores, row_idx in zip(queries, scores, idx, strict=True):
            hits = [(int(i), float(s)) for i, s in zip(row_idx, row_scores, strict=True) if i >= 0]
            hits.sort(key=lambda h: (-h[1], h[0]))
            if k2 < n and len(hits) == k2 and hits[k - 1][1] == hits[-1][1]:
                hits = self._exact(q)  # a tie reaches the candidate cut-off: score all
            out.append([(self.passage_ids[i], s) for i, s in hits[:k]])
        return out

    def _exact(self, query: np.ndarray) -> list[tuple[int, float]]:
        """Score every passage for one query; (position, score), best first, ties by position."""
        vectors = self.index.reconstruct_n(0, len(self))
        scores = vectors @ query
        order = np.lexsort((np.arange(len(scores)), -scores))
        return [(int(i), float(scores[i])) for i in order]

    # --- persistence -------------------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """Write the index to ``directory`` (replaced atomically if it exists)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        current = directory / "CURRENT"
        previous = current.read_text("utf-8").strip() if current.exists() else None
        version = Path(tempfile.mkdtemp(dir=directory, prefix="v-"))
        try:
            faiss.write_index(self.index, str(version / "index.faiss"))
            (version / "ids.json").write_text(
                json.dumps(self.passage_ids, ensure_ascii=False), "utf-8"
            )
            meta = {**self.meta, "format_version": FORMAT_VERSION}
            (version / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True), "utf-8"
            )
            pointer = directory / f".CURRENT.{version.name}.tmp"
            pointer.write_text(version.name, "utf-8")
            os.replace(pointer, current)  # the atomic switch
        except BaseException:
            shutil.rmtree(version, ignore_errors=True)
            for leftover in directory.glob(f".CURRENT.{version.name}.tmp"):
                leftover.unlink(missing_ok=True)
            raise
        # Keep the new version and the one before it (a reader may be mid-load on it).
        for old in directory.glob("v-*"):
            if old.is_dir() and old.name not in (version.name, previous):
                shutil.rmtree(old, ignore_errors=True)
        return directory

    @classmethod
    def load(cls, directory: str | Path, expect: dict[str, Any] | None = None) -> PassageIndex:
        """Load a saved index; ``expect`` lists metadata that must match exactly
        (e.g. the embedder's model id and revision, or the passages fingerprint)."""
        directory = Path(directory)
        try:
            version = (directory / "CURRENT").read_text("utf-8").strip()
            if not version.startswith("v-") or "/" in version or "\\" in version:
                raise ValueError(f"bad CURRENT pointer {version!r}")
            files = directory / version
            index = faiss.read_index(str(files / "index.faiss"))
            ids = json.loads((files / "ids.json").read_text("utf-8"))
            meta = json.loads((files / "meta.json").read_text("utf-8"))
        except (OSError, RuntimeError, ValueError) as e:
            raise ValueError(f"{directory}: not a readable saved index ({e})") from e
        if meta.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{directory}: unsupported index format {meta.get('format_version')}")
        if index.d != meta.get("dim") or index.ntotal != meta.get("size"):
            raise ValueError(f"{directory}: index does not match its metadata")
        for key, value in (expect or {}).items():
            if meta.get(key) != value:
                raise ValueError(
                    f"{directory}: meta[{key!r}] is {meta.get(key)!r}, expected {value!r}"
                )
        meta.pop("format_version")
        return cls(index, ids, meta)
