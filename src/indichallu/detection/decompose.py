"""Claim decomposition: answer -> atomic, independently checkable claims (deck slide 17).

M0 uses a rule-based splitter (DECISIONS D-024); an LLM-based option comes in M1:

1. sentences of the ORIGINAL answer (:func:`indichallu.text.sentseg.sentences`);
2. each sentence is split at coordinating conjunctions (Hindi: और, तथा, एवं, लेकिन,
   परंतु/परन्तु, किंतु/किन्तु, जबकि) and at ``;``, but only where BOTH sides keep at
   least ``MIN_TOKENS`` tokens, so a noun phrase like "राम और श्याम" stays whole;
3. each piece is trimmed of whitespace, sentence-final punctuation and a leading
   demonstrative (यह / वह / ये / वे) when enough of the claim remains.

Every claim is a :class:`Segment` of the original answer, so its offsets go straight into
the record (``claim.text == answer[start:end]``). Languages without a rule set fall back
to one claim per sentence.
"""

from __future__ import annotations

import re

from indichallu.text.sentseg import Segment, sentences, tokens

MIN_TOKENS = 3
TRAILING = "।॥.?!;:,"

CONJUNCTIONS = {
    "hi": ("और", "तथा", "एवं", "लेकिन", "परंतु", "परन्तु", "किंतु", "किन्तु", "जबकि"),
}
LEADING_DEMONSTRATIVES = {"hi": ("यह", "वह", "ये", "वे")}


def _trim(text: str, start: int, end: int, lang: str) -> tuple[int, int]:
    while start < end and (text[start].isspace() or text[start] in TRAILING):
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in TRAILING):
        end -= 1
    toks = tokens(text, start, end)
    lead = LEADING_DEMONSTRATIVES.get(lang, ())
    if len(toks) > MIN_TOKENS and toks[0].text in lead:
        start = toks[1].start
    return start, end


def _split_points(text: str, start: int, end: int, lang: str) -> list[tuple[int, int]]:
    """(cut_start, cut_end) of separators inside [start, end) that may split a claim."""
    words = CONJUNCTIONS.get(lang)
    if not words:
        return []
    pattern = re.compile(r"\s+(?:" + "|".join(map(re.escape, words)) + r")\s+|\s*;\s*")
    return [(m.start() + start, m.end() + start) for m in pattern.finditer(text[start:end])]


def decompose(answer: str, lang: str) -> list[Segment]:
    """Claims of ``answer`` as spans of the original answer text."""
    claims: list[Segment] = []
    for sent in sentences(answer, lang):
        pieces = [(sent.start, sent.end)]
        for cut_start, cut_end in _split_points(answer, sent.start, sent.end, lang):
            head_start, _ = pieces[-1]
            left, right = (head_start, cut_start), (cut_end, sent.end)
            if all(len(tokens(answer, a, b)) >= MIN_TOKENS for a, b in (left, right)):
                pieces[-1] = left
                pieces.append(right)
        for a, b in pieces:
            s, e = _trim(answer, a, b, lang)
            if s < e:
                claims.append(Segment(s, e, answer[s:e]))
    return claims
