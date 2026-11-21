"""Cached generation under a fixed decoding config (CLAUDE.md §2, §3 row 6).

Every generation is cached in an append-only JSONL file keyed by a SHA-256 over the
backend, model, model identity (Ollama digest / HF revision), decoding config and the
exact prompt. Re-running never recomputes an answer, and an interrupted run resumes
where it stopped. A torn last line (crash mid-write) is ignored on load.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from indichallu.generation.backends import (
    Backend,
    Generation,
    OllamaBackend,
    OpenAICompatibleBackend,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATION_CONFIGS = REPO_ROOT / "configs" / "generation"
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "interim" / "generations"  # git-ignored


@dataclass(frozen=True)
class GeneratorConfig:
    backend: str
    model: str
    decoding: dict[str, Any]
    prompt_template: str
    context_passages: int
    base_url_env: str = "OLLAMA_BASE_URL"
    ollama_digest: str | None = None
    hf_source: str | None = None
    hf_revision: str | None = None
    licence: str | None = None
    # Seconds one request may take. Not part of identity(): it never changes an answer.
    # On the laptop CPU a 3-passage Hindi prompt takes 10+ minutes (D-023).
    request_timeout_s: float = 600.0

    def identity(self) -> dict[str, Any]:
        """Everything that determines an answer apart from the prompt."""
        return {
            "backend": self.backend,
            "model": self.model,
            "ollama_digest": self.ollama_digest,
            "hf_revision": self.hf_revision,
            "decoding": self.decoding,
        }

    def record_info(self) -> dict[str, Any]:
        """The ``generator`` field of a record (schema GeneratorInfo)."""
        return {
            "model": self.model,
            "revision": self.ollama_digest or self.hf_revision or "unpinned",
            "decoding": self.decoding,
            "backend": self.backend,
        }


def load_generator_config(name: str = "m0") -> GeneratorConfig:
    raw = yaml.safe_load((GENERATION_CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))
    cfg = GeneratorConfig(**raw)
    for key in ("temperature", "max_new_tokens"):
        if key not in cfg.decoding:
            raise ValueError(f"generation config {name!r}: decoding needs {key!r}")
    if cfg.ollama_digest is None and cfg.hf_revision is None:
        raise ValueError(f"generation config {name!r}: pin ollama_digest or hf_revision")
    return cfg


def make_backend(cfg: GeneratorConfig) -> Backend:
    base_url = os.environ.get(cfg.base_url_env, "http://127.0.0.1:11434/v1")
    if cfg.backend == "ollama":
        return OllamaBackend(
            cfg.model, base_url, expected_digest=cfg.ollama_digest, timeout=cfg.request_timeout_s
        )
    if cfg.backend == "openai_compatible":
        return OpenAICompatibleBackend(cfg.model, base_url)
    raise ValueError(f"unknown backend {cfg.backend!r}")


def cache_key(identity: dict[str, Any], prompt: str) -> str:
    payload = json.dumps(
        {"identity": identity, "prompt": prompt}, sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class GenerationCache:
    path: Path
    _entries: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.path.exists():
            # bytes, decoded line by line: a crash can cut a line mid-character, and that
            # must cost only that line, not the whole cache
            for raw_line in self.path.read_bytes().splitlines():
                try:
                    entry = json.loads(raw_line.decode("utf-8"))
                    self._entries[entry["key"]] = entry
                except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
                    continue  # torn or foreign line: ignore, it will be regenerated

    def get(self, key: str) -> dict[str, Any] | None:
        return self._entries.get(key)

    def put(self, key: str, entry: dict[str, Any]) -> None:
        entry = {"key": key, **entry}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._entries[key] = entry

    def __len__(self) -> int:
        return len(self._entries)


class Generator:
    def __init__(
        self, cfg: GeneratorConfig, backend: Backend | None = None, cache_dir: Path | None = None
    ):
        self.cfg = cfg
        self.backend = backend or make_backend(cfg)
        safe_model = cfg.model.replace("/", "__").replace(":", "_")
        self.cache = GenerationCache(Path(cache_dir or DEFAULT_CACHE_DIR) / f"{safe_model}.jsonl")

    def _trusted(self, entry: dict[str, Any]) -> bool:
        """An Ollama entry is reused only if it records that its weights were verified
        against the pinned digest when it was generated (Codex review on #23: entries
        written before the digest check existed must not be replayed as pinned output)."""
        if self.cfg.backend != "ollama" or self.cfg.ollama_digest is None:
            return True
        return entry.get("verified_digest") == self.cfg.ollama_digest

    def generate(self, prompt: str) -> tuple[Generation, bool]:
        """(generation, from_cache)."""
        key = cache_key(self.cfg.identity(), prompt)
        hit = self.cache.get(key)
        if hit is not None and self._trusted(hit):
            return Generation(**hit["generation"]), True
        gen = self.backend.generate(prompt, self.cfg.decoding)
        self.cache.put(
            key,
            {
                "identity": self.cfg.identity(),
                "prompt": prompt,
                "generation": asdict(gen),
                "verified_digest": getattr(self.backend, "verified_digest", None),
            },
        )
        return gen, False
