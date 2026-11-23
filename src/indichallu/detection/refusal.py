"""Refusal detection (DECISIONS D-029): answers that decline to answer are a class of
their own, not claims (a refusal sentence is not a checkable fact about the evidence).

Patterns come from ``configs/refusals.yaml``. Detection is per sentence of the ORIGINAL
answer, so callers can drop refusal sentences from a partly refusing answer and keep the
offsets of everything else.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import yaml

from indichallu.text.sentseg import Segment, sentences

REFUSALS = Path(__file__).resolve().parents[3] / "configs" / "refusals.yaml"


@lru_cache(maxsize=4)
def _patterns(path: str = str(REFUSALS)) -> dict[str, tuple[re.Pattern[str], ...]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {lang: tuple(re.compile(p) for p in pats) for lang, pats in raw.items()}


def has_patterns(lang: str) -> bool:
    return bool(_patterns().get(lang))


def is_refusal_sentence(text: str, lang: str) -> bool:
    clean = " ".join(unicodedata.normalize("NFC", text).split())
    return any(p.search(clean) for p in _patterns().get(lang, ()))


def refusal_sentences(answer: str, lang: str) -> list[Segment]:
    """The sentences of ``answer`` (original offsets) that are refusals."""
    if not has_patterns(lang):
        return []
    return [s for s in sentences(answer, lang) if is_refusal_sentence(s.text, lang)]


def inside(seg: Segment, spans: list[Segment]) -> bool:
    return any(s.start <= seg.start and seg.end <= s.end for s in spans)
