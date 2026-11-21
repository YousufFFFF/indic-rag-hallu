"""Generation backends behind one interface (CLAUDE.md §3 row 6).

* :class:`OllamaBackend`: native Ollama ``/api/chat`` (supports ``think: false`` for
  Qwen3). The M0 generator runs here, on the laptop GPU.
* :class:`OpenAICompatibleBackend`: any OpenAI-compatible server (e.g. vLLM on the T4).
* :class:`MockBackend`: deterministic, for tests.

A 4-bit Hugging Face Transformers backend for the 7-10B generators (and logit access for
baseline B4) is added in M1, when those run on the T4.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


@dataclass(frozen=True)
class Generation:
    text: str  # the answer, with any <think>...</think> block removed
    raw: str  # exactly what the backend returned
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    seconds: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Backend(Protocol):
    name: str

    def generate(self, prompt: str, decoding: dict[str, Any]) -> Generation: ...


def clean_answer(raw: str) -> str:
    """Drop a reasoning block if one slipped through, and surrounding whitespace."""
    return THINK_BLOCK.sub("", raw).strip()


class OllamaBackend:
    name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str,
        timeout: float = 600.0,
        expected_digest: str | None = None,
    ):
        # accept either .../v1 (OpenAI-compatible root) or the server root
        self.model = model
        self.root = base_url.rstrip("/").removesuffix("/v1")
        self.timeout = timeout
        self.expected_digest = expected_digest
        self._verified = expected_digest is None

    def installed_digest(self) -> str | None:
        """The digest the server's tag currently resolves to (None if not installed)."""
        with urllib.request.urlopen(f"{self.root}/api/tags", timeout=30) as resp:
            models = json.loads(resp.read().decode("utf-8")).get("models", [])
        names = {self.model, self.model if ":" in self.model else f"{self.model}:latest"}
        return next((m.get("digest") for m in models if m.get("name") in names), None)

    def verify(self) -> None:
        """Refuse to generate unless the tag resolves to the pinned weights: a tag is
        mutable (``qwen3:4b`` moved to the Thinking model), a digest is not."""
        if self._verified:
            return
        found = self.installed_digest()
        if found != self.expected_digest:
            raise RuntimeError(
                f"Ollama model {self.model!r} resolves to digest {found!r}, but the config "
                f"pins {self.expected_digest!r}; pull the pinned model or update the config"
            )
        self._verified = True

    @property
    def verified_digest(self) -> str | None:
        """The pinned digest once :meth:`verify` has confirmed it, else None."""
        return self.expected_digest if self._verified else None

    def request_body(self, prompt: str, decoding: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": bool(decoding.get("think", False)),
            "options": {
                "temperature": decoding["temperature"],
                "top_p": decoding.get("top_p", 1.0),
                "seed": decoding.get("seed", 0),
                "num_predict": decoding["max_new_tokens"],
                "num_ctx": decoding.get("num_ctx", 8192),
            },
        }

    def generate(self, prompt: str, decoding: dict[str, Any]) -> Generation:
        self.verify()  # once per backend, before the first request
        body = json.dumps(self.request_body(prompt, decoding)).encode("utf-8")
        request = urllib.request.Request(
            f"{self.root}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        seconds = time.perf_counter() - start
        raw = data["message"]["content"]
        return Generation(
            text=clean_answer(raw),
            raw=raw,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            seconds=seconds,
            extra={"done_reason": data.get("done_reason")},
        )


class OpenAICompatibleBackend:
    name = "openai_compatible"

    def __init__(self, model: str, base_url: str, api_key: str = "none"):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(self, prompt: str, decoding: dict[str, Any]) -> Generation:
        start = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=decoding["temperature"],
            top_p=decoding.get("top_p", 1.0),
            seed=decoding.get("seed", 0),
            max_tokens=decoding["max_new_tokens"],
        )
        raw = resp.choices[0].message.content or ""
        usage = resp.usage
        return Generation(
            text=clean_answer(raw),
            raw=raw,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            seconds=time.perf_counter() - start,
            extra={"finish_reason": resp.choices[0].finish_reason},
        )


class MockBackend:
    """Returns ``answer_fn(prompt)``; records every call (for tests)."""

    name = "mock"

    def __init__(self, answer_fn=lambda prompt: "उत्तर।"):
        self.answer_fn = answer_fn
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def generate(self, prompt: str, decoding: dict[str, Any]) -> Generation:
        self.calls.append((prompt, dict(decoding)))
        raw = self.answer_fn(prompt)
        return Generation(text=clean_answer(raw), raw=raw)
