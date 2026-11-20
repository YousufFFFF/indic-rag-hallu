"""Aggregation and span attribution -> SUPPORTED / UNSUPPORTED / CONTRADICTED (slides 16-17).

For each claim, over its scores against every retrieved passage:

* **SUPPORTED** if the highest entailment >= ``entail`` threshold; the evidence is that
  passage;
* else **CONTRADICTED** if the highest contradiction >= ``contradict`` threshold; the
  evidence is the contradicting passage;
* else **UNSUPPORTED**; ``best_passage_id`` points at the closest (most entailing)
  passage, for reference.

Thresholds come from config (tuned on VAL only, never test). The predicted error type of
a non-supported claim is a simple M0 heuristic (:func:`guess_type`). The typed
verifiers (C2) replace it in M1.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from indichallu.data.schema import Claim, NLIScores, PassageScores, SpanType, Verdict
from indichallu.text.numerals import to_ascii_digits

HINDI_MONTHS = (
    "जनवरी",
    "फरवरी",
    "फ़रवरी",
    "मार्च",
    "अप्रैल",
    "मई",
    "जून",
    "जुलाई",
    "अगस्त",
    "सितंबर",
    "सितम्बर",
    "अक्टूबर",
    "अक्तूबर",
    "नवंबर",
    "नवम्बर",
    "दिसंबर",
    "दिसम्बर",
)
TEMPORAL_WORDS = HINDI_MONTHS + ("सदी", "शताब्दी", "ईस्वी", "ई.पू.", "सन्", "सन")
YEAR = re.compile(r"(?<!\d)(1[0-9]{3}|20[0-9]{2})(?!\d)")
DIGIT = re.compile(r"\d")


@dataclass(frozen=True)
class Thresholds:
    entail: float = 0.5
    contradict: float = 0.5

    def __post_init__(self) -> None:
        for name in ("entail", "contradict"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} threshold must be in [0, 1]")


def guess_type(text: str) -> SpanType | None:
    """M0 heuristic: years, months and era words -> TEMPORAL; other digits -> NUMERICAL;
    otherwise unknown (ENTITY/LOCATION need NER: typed verifiers in M1)."""
    ascii_text = to_ascii_digits(text).text
    if YEAR.search(ascii_text) or any(w in text for w in TEMPORAL_WORDS):
        return SpanType.TEMPORAL
    if DIGIT.search(ascii_text):
        return SpanType.NUMERICAL
    return None


def decide(
    text: str, start: int, end: int, scores: Sequence[PassageScores], th: Thresholds
) -> Claim:
    if not scores:
        raise ValueError("a claim needs scores against at least one passage")
    by_entail = max(scores, key=lambda s: s.entail)
    by_contra = max(scores, key=lambda s: s.contradict)
    if by_entail.entail >= th.entail:
        verdict, best = Verdict.SUPPORTED, by_entail
    elif by_contra.contradict >= th.contradict:
        verdict, best = Verdict.CONTRADICTED, by_contra
    else:
        verdict, best = Verdict.UNSUPPORTED, by_entail
    return Claim(
        text=text,
        start=start,
        end=end,
        verdict=verdict,
        best_passage_id=best.passage_id,
        scores=NLIScores(entail=best.entail, neutral=best.neutral, contradict=best.contradict),
        passage_scores=list(scores),
        type=None if verdict is Verdict.SUPPORTED else guess_type(text),
    )
