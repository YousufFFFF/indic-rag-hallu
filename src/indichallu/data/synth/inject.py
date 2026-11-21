"""Typed error injection with known spans (M0 task T28; CLAUDE.md §4.4 "typed injection").

Each injector takes a faithful sentence and returns a changed sentence plus the exact span
(on the NEW sentence) and the error type, or ``None`` if it doesn't apply:

* :func:`inject_temporal`: a year (1000-2099) shifted by 1-30 years, or a Hindi month
  name swapped for another month -> TEMPORAL;
* :func:`inject_numerical`: a non-year integer changed to a different positive value ->
  NUMERICAL.

Numbers keep their digit script: "२१" becomes e.g. "२७", never "27". Numbers written with
separators (``1,00,000``, ``3.5``) are left alone in M0. ENTITY/LOCATION injection needs
IndicNER (M1). Everything is driven by a seeded ``random.Random``.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from indichallu.data.schema import SpanType
from indichallu.detection.aggregate import HINDI_MONTHS
from indichallu.text.numerals import indic_digit_table, to_ascii_digits

PLAIN_INT = re.compile(r"(?<![\d.,])\d+(?![\d.,]*\d)")
MONTHS = tuple(m for m in HINDI_MONTHS if m not in ("फ़रवरी", "सितम्बर", "अक्तूबर", "नवम्बर", "दिसम्बर"))
CANONICAL_MONTH = {
    "फ़रवरी": "फरवरी",
    "सितम्बर": "सितंबर",
    "अक्तूबर": "अक्टूबर",
    "नवम्बर": "नवंबर",
    "दिसम्बर": "दिसंबर",
}


@dataclass(frozen=True)
class Injection:
    text: str  # the changed sentence
    start: int  # span of the injected value in ``text``
    end: int
    type: SpanType
    original: str  # the value it replaced
    replacement: str


def _numbers(text: str) -> list[tuple[int, int, int]]:
    """(start, end, value) of plain integers in any digit script (offsets are 1:1)."""
    ascii_text = to_ascii_digits(text).text
    return [(m.start(), m.end(), int(m.group())) for m in PLAIN_INT.finditer(ascii_text)]


def _in_script_of(value: int, original: str) -> str:
    """Write ``value`` in the digit script of ``original`` (ASCII if it was ASCII)."""
    table = indic_digit_table()
    native = next((ch for ch in original if ch in table), None)
    digits = str(value)
    if native is None:
        return digits
    zero = ord(native) - int(table[native])
    return "".join(chr(zero + int(d)) for d in digits)


def _replace(text: str, start: int, end: int, new: str, typ: SpanType) -> Injection:
    out = text[:start] + new + text[end:]
    return Injection(out, start, start + len(new), typ, text[start:end], new)


def is_year(value: int, width: int) -> bool:
    return width == 4 and 1000 <= value <= 2099


def inject_temporal(text: str, rng: random.Random) -> Injection | None:
    years = [(s, e, v) for s, e, v in _numbers(text) if is_year(v, e - s)]
    months = [
        (m.start(), m.end(), m.group())
        for m in re.finditer("|".join(sorted(HINDI_MONTHS, key=len, reverse=True)), text)
    ]
    options = [("year", y) for y in years] + [("month", m) for m in months]
    if not options:
        return None
    kind, (s, e, v) = rng.choice(options)
    if kind == "year":
        shift = rng.choice([d for d in range(-30, 31) if d != 0])
        return _replace(text, s, e, _in_script_of(v + shift, text[s:e]), SpanType.TEMPORAL)
    current = CANONICAL_MONTH.get(v, v)
    return _replace(text, s, e, rng.choice([m for m in MONTHS if m != current]), SpanType.TEMPORAL)


def inject_numerical(text: str, rng: random.Random) -> Injection | None:
    nums = [(s, e, v) for s, e, v in _numbers(text) if not is_year(v, e - s)]
    if not nums:
        return None
    s, e, v = rng.choice(nums)
    spread = max(2, v // 2)
    candidates = [x for x in range(max(1, v - spread), v + spread + 1) if x != v]
    return _replace(
        text, s, e, _in_script_of(rng.choice(candidates), text[s:e]), SpanType.NUMERICAL
    )


def inject(text: str, rng: random.Random, prefer: SpanType | None = None) -> Injection | None:
    """Try the preferred type first, then the other; None if neither applies."""
    order = [inject_temporal, inject_numerical]
    if prefer is SpanType.NUMERICAL:
        order.reverse()
    for fn in order:
        result = fn(text, rng)
        if result is not None:
            return result
    return None
