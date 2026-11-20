"""Sentence segmentation and tokenisation with offsets into the ORIGINAL text.

* :func:`sentences` runs the normalisation pipeline (so e.g. a ``|`` typed for a danda
  still ends a sentence), finds sentence boundaries in the normalised text, and maps each
  sentence back to the original through the offset map. It uses IndicNLP's delimiter
  rules and abbreviation list but its own boundary logic, because IndicNLP's
  ``sentence_split`` can silently DROP text in its abbreviation-merging pass (e.g. 13 of
  19 characters lost on one input). Sentences here always partition the text.
* :func:`tokens` tokenises the original text directly by Unicode category. Python's
  ``\\w`` would split Indic words at vowel signs and viramas (combining marks), so a word
  is a run of letters, marks, digits and ZWJ/ZWNJ. A number keeps internal ``,``/``.``
  between digits (``1,00,000.50``), and every other non-space character is its own token.

All spans are half-open ``[start, end)`` code-point offsets into the text passed in
(DECISIONS D-014).

>>> [s.text for s in sentences("योजना राज्य की है। आयु २१ वर्ष होनी चाहिए|", "hi")]
['योजना राज्य की है।', 'आयु २१ वर्ष होनी चाहिए|']
>>> [t.text for t in tokens("आयु 1,00,000.50 रु।")]
['आयु', '1,00,000.50', 'रु', '।']
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from indicnlp import langinfo
from indicnlp.tokenize.sentence_tokenize import (
    CONTAINS_DANDA,
    DELIM_PAT_DANDA,
    DELIM_PAT_NO_DANDA,
    is_acronym_abbvr,
)

from indichallu.text.normalize import normalize

ZWNJ, ZWJ = chr(0x200C), chr(0x200D)
# Characters that stay attached to the end of a sentence after its delimiter.
CLOSERS = frozenset("\"')]}" + chr(0x2019) + chr(0x201D) + chr(0x00BB))


@dataclass(frozen=True)
class Segment:
    """A span ``[start, end)`` of the original text and the text it covers."""

    start: int
    end: int
    text: str


# --- sentences -------------------------------------------------------------------------


def _delimiters(text: str, lang: str):
    """IndicNLP's 'auto' choice: for danda languages (hi, bn, pa) whose text contains a
    danda, only ``? ! । ॥`` end sentences; otherwise ``.`` does too."""
    if langinfo.is_danda_delim(lang) and CONTAINS_DANDA.search(text):
        return DELIM_PAT_DANDA
    return DELIM_PAT_NO_DANDA


def _word_before(text: str, p: int) -> str:
    i = p
    while i > 0 and _is_word_char(text[i - 1]):
        i -= 1
    return text[i:p]


def _is_initial(word: str) -> bool:
    """A single bare letter (``क``, ``A``): an initial. Deliberately excludes letters
    with vowel signs, since one-syllable words like ``है`` or ``में`` commonly end
    sentences; abbreviations such as ``डॉ`` come from IndicNLP's list instead."""
    return len(word) == 1 and unicodedata.category(word)[0] == "L"


def _period_ends_sentence(text: str, p: int, lang: str) -> bool:
    """A ``.`` is not a sentence end after a digit (``21.``, ``3.5``), after an
    abbreviation in IndicNLP's list (``डॉ``, ``श्री``, ...), or after a single bare
    letter (initials)."""
    if p > 0 and text[p - 1].isnumeric():
        return False
    word = _word_before(text, p)
    if not word:
        return True
    return not (_is_initial(word) or is_acronym_abbvr(word, lang))


def sentence_cuts(text: str, lang: str) -> list[int]:
    """Positions in ``text`` right after each sentence end (after its delimiter run)."""
    delim = _delimiters(text, lang)
    cuts = []
    p = 0
    while True:
        m = delim.search(text, p)
        if m is None:
            break
        p = m.start()
        if text[p] == "." and not _period_ends_sentence(text, p, lang):
            p += 1
            continue
        end = p + 1  # keep a run of delimiters and closing quotes/brackets together
        while end < len(text) and (delim.match(text, end) or text[end] in CLOSERS):
            end += 1
        cuts.append(end)
        p = end
    return cuts


def _trimmed(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def sentence_spans(text: str, lang: str) -> list[tuple[int, int]]:
    """Sentence spans of ``text`` itself (no normalisation). Lossless: every non-space
    character lies in exactly one span; spans are whitespace-trimmed."""
    bounds = [0, *sentence_cuts(text, lang), len(text)]
    spans = [_trimmed(text, a, b) for a, b in zip(bounds, bounds[1:], strict=False)]
    return [(s, e) for s, e in spans if s < e]


def sentences(text: str, lang: str, **normalize_options: object) -> list[Segment]:
    """Sentences of ``text`` as spans of the ORIGINAL ``text``.

    Sentence ends are found on the normalised text (``normalize_options`` are passed to
    :func:`indichallu.text.normalize.normalize`). The CUT POSITIONS are mapped back and
    the original is split there, so the original is partitioned exactly: even characters
    normalisation deletes (e.g. a ZWNJ at a sentence edge) land inside a sentence. A piece
    holding only such deleted characters is merged into a neighbour, not returned alone.
    Text with no content after normalisation has no sentences.
    """
    norm = normalize(text, lang, **normalize_options)
    bounds = [0]
    for cut in sentence_cuts(norm.text, lang):
        bounds.append(max(bounds[-1], norm.offsets.to_source(cut, cut)[0]))
    bounds.append(len(text))

    spans: list[list[int]] = []
    pending_start: int | None = None  # leading piece with no surviving content
    for a, b in zip(bounds, bounds[1:], strict=False):
        s, e = _trimmed(text, a, b)
        if s >= e:
            continue
        ts, te = norm.offsets.to_target(s, e)
        if not norm.text[ts:te].strip():  # only characters normalisation deletes
            if spans:
                spans[-1][1] = e
            elif pending_start is None:
                pending_start = s
            continue
        if pending_start is not None:
            s, pending_start = pending_start, None
        spans.append([s, e])
    return [Segment(s, e, text[s:e]) for s, e in spans]


# --- tokens ----------------------------------------------------------------------------


def _is_word_char(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "LMN" or ch in (ZWNJ, ZWJ)


def _is_digit(ch: str) -> bool:
    return unicodedata.category(ch) == "Nd"


def tokens(text: str, start: int = 0, end: int | None = None) -> list[Segment]:
    """Tokens of ``text[start:end]``, with offsets into ``text`` (not the slice)."""
    end = len(text) if end is None else end
    if not 0 <= start <= end <= len(text):
        raise ValueError(f"window [{start},{end}) out of range for length {len(text)}")
    out: list[Segment] = []
    i = start
    while i < end:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        j = i + 1
        if _is_word_char(ch):
            while j < end:
                c = text[j]
                if _is_word_char(c):
                    j += 1
                elif (
                    c in ",." and _is_digit(text[j - 1]) and j + 1 < end and _is_digit(text[j + 1])
                ):
                    j += 2  # separator inside a number: 1,00,000.50
                else:
                    break
        out.append(Segment(i, j, text[i:j]))
        i = j
    return out
