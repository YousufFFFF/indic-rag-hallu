"""Unicode + Indic script normalisation that keeps an offset map to the original text.

Pipeline (docs/DECISIONS.md D-012), each step returning an :class:`OffsetMap`:

1. **NFC** (:func:`nfc`): exact Unicode NFC, with an alignment built by normalising the
   text in independent chunks split at NFC-safe boundaries.
2. **IndicNLP** (:func:`indicnlp_normalize`): the language's IndicNLP normaliser
   (nukta forms decomposed, ZWJ/ZWNJ removed, ``|`` -> danda, ``:`` after a letter ->
   visarga, ...), aligned to its input with a character diff because the library
   returns no offsets.

:func:`normalize` runs both and composes the maps, so a span found on the normalised
text maps straight back to the original. Apply the same pipeline to everything that is
compared (claims and passages); the output is a matching form, not guaranteed NFC
(IndicNLP decomposes e.g. U+0929 ``ऩ``, which NFC keeps composed).

>>> r = normalize("क‌ख|", "hi")    # ZWNJ removed, "|" becomes a danda
>>> r.text == "कख।"
True
>>> r.offsets.to_source(2, 3)            # the danda came from the "|" at index 3
(3, 4)
>>> r.offsets.to_target(1, 2)            # the deleted ZWNJ -> empty span where it was
(1, 1)
"""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from typing import NamedTuple

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

from indichallu.languages import load_languages
from indichallu.text.offsets import OffsetMap, OffsetMapBuilder

# Options IndicNormalizerFactory.get_normalizer accepts (library defaults apply if omitted).
INDICNLP_OPTIONS = frozenset(
    {"remove_nuktas", "nasals_mode", "do_normalize_chandras", "do_normalize_vowel_ending"}
)


class Normalized(NamedTuple):
    text: str
    offsets: OffsetMap  # maps `text` back to the original input


# --- NFC -------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _composing_starters() -> frozenset[str]:
    """Starters (ccc=0) that can compose with the PRECEDING character under NFC.

    These are the second characters of two-character canonical decompositions that are
    themselves starters, e.g. Bengali U+09BE (in U+09CB = U+09C7 U+09BE), Tamil U+0BBE,
    Malayalam U+0D3E, plus the Hangul medial vowels and final consonants, which compose
    algorithmically. NFC may merge text across these, so they are never chunk starts.
    """
    out = set()
    for cp in range(0x110000):
        decomposition = unicodedata.decomposition(chr(cp))
        if not decomposition or decomposition.startswith("<"):
            continue  # unassigned, no decomposition, or compatibility-only
        parts = decomposition.split()
        if len(parts) == 2:
            second = chr(int(parts[1], 16))
            if unicodedata.combining(second) == 0:
                out.add(second)
    out.update(chr(c) for c in range(0x1161, 0x1176))  # Hangul jungseong (V)
    out.update(chr(c) for c in range(0x11A8, 0x11C3))  # Hangul jongseong (T)
    return frozenset(out)


def _nfc_boundary_before(ch: str) -> bool:
    """True if NFC never combines ``ch`` with anything before it.

    Decided by the first character of ``ch``'s canonical decomposition (as ICU's
    hasCompBoundaryBefore does): if that is a starter (ccc=0) that can't compose
    backwards, nothing before ``ch`` can merge into it or be reordered past it. This also
    makes characters NFC itself changes, like U+0958 (क़ -> क + nukta), chunk starts, so
    their neighbours keep exact 1:1 offsets.
    """
    first = unicodedata.normalize("NFD", ch)[0]
    return unicodedata.combining(first) == 0 and first not in _composing_starters()


def nfc(text: str) -> Normalized:
    """Unicode NFC with an exact offset map.

    Unchanged chunks are copied 1:1; a chunk that NFC changes (e.g. U+0958 -> क + nukta,
    or Bengali ে + া -> ো) is replaced as a whole.
    """
    builder = OffsetMapBuilder(text)
    start = 0
    for i in range(1, len(text) + 1):
        if i == len(text) or _nfc_boundary_before(text[i]):
            chunk = text[start:i]
            normalized = unicodedata.normalize("NFC", chunk)
            if normalized == chunk:
                builder.copy(len(chunk))
            else:
                builder.replace(len(chunk), normalized)
            start = i
    offsets = builder.build()
    if offsets.target != unicodedata.normalize("NFC", text):  # pragma: no cover - guard
        raise AssertionError("chunked NFC diverged from unicodedata.normalize")
    return Normalized(offsets.target, offsets)


# --- aligning a black-box transformation -----------------------------------------------


def align(source: str, target: str) -> OffsetMap:
    """Offset map from ``target`` back to ``source`` via a character-level diff.

    For transformations that don't report offsets (e.g. IndicNLP). Matching runs are
    mapped exactly; each differing region is one replacement chunk.
    """
    builder = OffsetMapBuilder(source)
    matcher = SequenceMatcher(None, source, target, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            builder.copy(i2 - i1)
        else:  # replace / delete / insert
            builder.replace(i2 - i1, target[j1:j2])
    offsets = builder.build()
    assert offsets.target == target
    return offsets


# --- IndicNLP --------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _indicnlp_normalizer(lang: str, options: tuple[tuple[str, object], ...]):
    return IndicNormalizerFactory().get_normalizer(lang, **dict(options))


def _check_lang(lang: str) -> None:
    supported = load_languages()
    if lang not in supported:
        raise ValueError(f"unsupported language {lang!r}; expected one of {sorted(supported)}")


def indicnlp_normalize(text: str, lang: str, **options: object) -> Normalized:
    """Run the IndicNLP normaliser for ``lang`` and align its output to ``text``."""
    _check_lang(lang)
    unknown = set(options) - INDICNLP_OPTIONS
    if unknown:
        raise TypeError(f"unknown IndicNLP options {sorted(unknown)}")
    normalizer = _indicnlp_normalizer(lang, tuple(sorted(options.items())))
    out = normalizer.normalize(text)
    return Normalized(out, align(text, out))


# --- full pipeline ---------------------------------------------------------------------


def normalize(text: str, lang: str, *, indicnlp: bool = True, **options: object) -> Normalized:
    """NFC, then (by default) IndicNLP normalisation; offsets map back to ``text``."""
    _check_lang(lang)
    step1 = nfc(text)
    if not indicnlp:
        return step1
    step2 = indicnlp_normalize(step1.text, lang, **options)
    return Normalized(step2.text, step1.offsets.then(step2.offsets))
