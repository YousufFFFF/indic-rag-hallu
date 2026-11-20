"""Indic digit normalisation: native decimal digits -> ASCII, with exact offsets.

``"आयु २१ वर्ष"`` and ``"आयु 21 वर्ष"`` must compare equal, so claims and passages are
compared after converting every decimal digit to ASCII (DECISIONS D-013). Each digit is
one code point in and one out, so the offset map is exact character by character.

Scope
-----
* ``"indic"`` (default): the decimal digits of the configured languages
  (``configs/languages.yaml``): Devanagari, Bengali, Gurmukhi, Gujarati, Tamil, Telugu,
  Kannada, Malayalam (Hindi and Marathi share Devanagari).
* ``"all"``: any Unicode decimal digit (category Nd), e.g. also Arabic-Indic or
  full-width digits.

Not converted: non-decimal numerals such as Tamil ``௰`` (10), ``௱`` (100), ``௲`` (1000)
or Bengali currency numerators. They are not positional digits, so a 1:1 swap would
change the value; parsing them belongs to the typed NUMERICAL verifier.

>>> r = to_ascii_digits("आयु २१ वर्ष, ২০২৬ ও ௨௦௨௬")
>>> r.text
'आयु 21 वर्ष, 2026 ও 2026'
>>> r.offsets.source_text(4, 6)
'२१'
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache
from typing import Literal

from indichallu.languages import load_languages
from indichallu.text.normalize import Normalized
from indichallu.text.offsets import OffsetMapBuilder

Scope = Literal["indic", "all"]


@lru_cache(maxsize=1)
def indic_digit_table() -> dict[str, str]:
    """Native digit -> ASCII digit for every configured language's script."""
    table: dict[str, str] = {}
    for lang in load_languages().values():
        for value, ch in enumerate(lang.digits):
            table[ch] = str(value)
    return table


def _check_scope(scope: object) -> None:
    if scope not in ("indic", "all"):
        raise ValueError(f"scope must be 'indic' or 'all', got {scope!r}")


def _ascii_digit(ch: str, scope: Scope) -> str | None:
    if "0" <= ch <= "9":
        return None  # already ASCII
    if scope == "indic":
        return indic_digit_table().get(ch)
    if unicodedata.category(ch) == "Nd":
        return str(unicodedata.decimal(ch))
    return None


def to_ascii_digits(text: str, scope: Scope = "indic") -> Normalized:
    """Replace native decimal digits with ASCII ones; offsets map back exactly."""
    _check_scope(scope)
    builder = OffsetMapBuilder(text)
    for ch in text:
        ascii_digit = _ascii_digit(ch, scope)
        if ascii_digit is None:
            builder.copy()
        else:
            builder.replace(1, ascii_digit)
    offsets = builder.build()
    return Normalized(offsets.target, offsets)


def has_native_digits(text: str, scope: Scope = "indic") -> bool:
    """True if ``text`` contains any digit that :func:`to_ascii_digits` would convert."""
    _check_scope(scope)  # same validation as to_ascii_digits, even for empty text
    return any(_ascii_digit(ch, scope) is not None for ch in text)
