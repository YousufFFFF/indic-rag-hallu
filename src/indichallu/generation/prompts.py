"""Prompt assembly: question + retrieved context in the SOURCE language (deck slide 15).

One template per language lives in ``generation/prompts/<lang>.txt`` with ``{passages}``
and ``{question}`` placeholders. Templates are reviewed by a native speaker before use
(CLAUDE.md §3 row 5). The Hindi template was approved by the maintainer on 2026-11-20 (DECISIONS D-023).

Passages are numbered ``[1] ... [k]`` in retrieval-rank order, so an answer can be
traced back to the passages the model actually saw.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from indichallu.data.schema import RetrievedPassage

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_FIELD = re.compile(r"{(passages|question)}")


@lru_cache(maxsize=16)
def load_template(lang: str) -> str:
    path = PROMPTS_DIR / f"{lang}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"no prompt template for {lang!r} ({path})")
    template = path.read_text(encoding="utf-8")
    for field in ("{passages}", "{question}"):
        if template.count(field) != 1:
            raise ValueError(f"{path}: template must contain {field} exactly once")
    return template


def format_passages(passages: Sequence[RetrievedPassage]) -> str:
    ordered = sorted(passages, key=lambda p: p.rank)
    return "\n\n".join(f"[{i}] {p.text.strip()}" for i, p in enumerate(ordered, start=1))


def build_prompt(lang: str, question: str, passages: Sequence[RetrievedPassage]) -> str:
    """The full prompt: the language's template filled with numbered passages + question."""
    if not passages:
        raise ValueError("a RAG prompt needs at least one retrieved passage")
    template = load_template(lang)
    values = {"passages": format_passages(passages), "question": question.strip()}
    # One pass over the TEMPLATE only: inserted text is never rescanned, so a passage that
    # happens to contain "{question}" (or any braces) is kept verbatim.
    return _FIELD.sub(lambda m: values[m.group(1)], template)
