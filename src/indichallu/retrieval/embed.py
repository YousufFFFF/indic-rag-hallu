"""Text embedding for retrieval (deck block diagram: "Multilingual Embedder").

* :class:`EmbedderConfig` comes from ``configs/retrieval/embedders.yaml``: model id, a
  pinned revision, the query/passage prefixes the model card asks for, output dimension
  and whether to L2-normalise.
* :class:`Embedder` adds the right prefix, NFC-normalises the text (embedding models see
  the original script, not the IndicNLP matching form), calls an :class:`Encoder`, and
  checks the output shape and dtype.
* The encoder is pluggable: :class:`SentenceTransformerEncoder` for real models,
  anything with the same ``encode`` method for tests.
* Results can be cached on disk, keyed by a fingerprint of the model, revision, prefix,
  settings and the exact texts, so a corpus is embedded once and any change re-embeds.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
EMBEDDERS_CONFIG = REPO_ROOT / "configs" / "retrieval" / "embedders.yaml"

Kind = Literal["query", "passage"]


@dataclass(frozen=True)
class EmbedderConfig:
    name: str
    model_id: str
    revision: str
    dim: int
    query_prefix: str = ""
    passage_prefix: str = ""
    normalize: bool = True
    max_seq_length: int = 512

    def prefix(self, kind: Kind) -> str:
        if kind == "query":
            return self.query_prefix
        if kind == "passage":
            return self.passage_prefix
        raise ValueError(f"kind must be 'query' or 'passage', got {kind!r}")


def load_embedder_config(name: str, path: str | Path = EMBEDDERS_CONFIG) -> EmbedderConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["embedders"]
    if name not in raw:
        raise KeyError(f"unknown embedder {name!r}; configured: {sorted(raw)}")
    cfg = EmbedderConfig(name=name, **raw[name])
    if len(cfg.revision) != 40 or any(c not in "0123456789abcdef" for c in cfg.revision):
        raise ValueError(f"{name}: revision must be a full 40-char commit hash")
    return cfg


class Encoder(Protocol):
    def encode(self, texts: list[str], batch_size: int) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """A sentence-transformers model at a pinned revision (loaded lazily)."""

    def __init__(self, config: EmbedderConfig, device: str | None = None):
        self.config = config
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.config.model_id, revision=self.config.revision, device=self.device
            )
            self._model.max_seq_length = self.config.max_seq_length
            self._model.eval()
        return self._model

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        return self._load().encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # done in Embedder, identically for every encoder
            show_progress_bar=False,
        )


class Embedder:
    def __init__(
        self,
        config: EmbedderConfig,
        encoder: Encoder | None = None,
        cache_dir: str | Path | None = None,
        batch_size: int = 32,
    ):
        self.config = config
        self.encoder = encoder or SentenceTransformerEncoder(config)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.batch_size = batch_size

    def _inputs(self, texts: list[str], kind: Kind) -> list[str]:
        prefix = self.config.prefix(kind)
        return [prefix + unicodedata.normalize("NFC", t) for t in texts]

    def fingerprint(self, texts: list[str], kind: Kind) -> str:
        h = hashlib.sha256()
        h.update(json.dumps(asdict(self.config), sort_keys=True).encode())
        h.update(kind.encode())
        for t in self._inputs(texts, kind):
            h.update(len(t.encode()).to_bytes(8, "little"))  # unambiguous concatenation
            h.update(t.encode())
        return h.hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / self.config.name / f"{key}.npy"

    def embed(self, texts: list[str], kind: Kind) -> np.ndarray:
        """Embed ``texts`` as queries or passages: float32 array of shape (n, dim)."""
        texts = list(texts)
        key = self.fingerprint(texts, kind)
        path = self._cache_path(key)
        if path is not None and path.exists():
            cached = np.load(path)
            if cached.shape == (len(texts), self.config.dim) and cached.dtype == np.float32:
                return cached
        if texts:
            vectors = np.asarray(self.encoder.encode(self._inputs(texts, kind), self.batch_size))
        else:
            vectors = np.zeros((0, self.config.dim))
        vectors = vectors.astype(np.float32, copy=False)
        if vectors.shape != (len(texts), self.config.dim):
            raise ValueError(
                f"{self.config.name}: encoder returned shape {vectors.shape}, "
                f"expected {(len(texts), self.config.dim)}"
            )
        if not np.isfinite(vectors).all():
            raise ValueError(f"{self.config.name}: encoder returned non-finite values")
        if self.config.normalize and len(texts):
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if (norms == 0).any():
                raise ValueError(f"{self.config.name}: zero vector cannot be normalised")
            vectors = vectors / norms
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npy.part")
            with os.fdopen(fd, "wb") as f:
                np.save(f, vectors)
            os.replace(tmp, path)
        return vectors

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts, "query")

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts, "passage")
