"""Offset maps between an original string and a transformed copy of it.

CLAUDE.md §2: never compute spans on normalised text without a map back to the original;
span metrics use ORIGINAL character offsets. Every text transformation in this package
returns an :class:`OffsetMap` alongside its output.

Model
-----
A transformation turns ``source`` into ``target`` left to right as a sequence of edits:

* **copy** n characters unchanged (exact 1:1 alignment),
* **replace** n source characters with any replacement string (n >= 0, replacement may be
  empty). Every target character a replacement produces is aligned to the WHOLE consumed
  source chunk, because the chunk can't be split meaningfully (e.g. ``"क़"`` as two code
  points -> one, or ``"२"`` -> ``"2"``). Deletions and insertions are the special cases
  with an empty replacement or ``n = 0``.

Internally each target character ``j`` stores the half-open source interval
``[lo[j], hi[j])`` it came from (an inserted character has ``lo == hi``). A span maps to
the hull (smallest enclosing span) of its characters' intervals, so mappings are always
contiguous and maps compose. Lookups are linear scans: texts here are answers and
passages, short enough that clarity beats bisection (and the intervals of a composed map
need not be monotone, which bisection would require).

All offsets are Python ``str`` indices (Unicode code points), spans are half-open.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class OffsetMap:
    """Alignment from ``target`` (transformed) back to ``source`` (original)."""

    source: str
    target: str
    lo: tuple[int, ...]
    hi: tuple[int, ...]

    def __post_init__(self) -> None:
        n, m = len(self.source), len(self.target)
        if len(self.lo) != m or len(self.hi) != m:
            raise ValueError(f"need one source interval per target char ({m})")
        for j, (a, b) in enumerate(zip(self.lo, self.hi, strict=True)):
            if not 0 <= a <= b <= n:
                raise ValueError(f"target char {j}: bad source interval [{a},{b}) for n={n}")

    # --- construction ----------------------------------------------------------------

    @classmethod
    def identity(cls, text: str) -> OffsetMap:
        idx = tuple(range(len(text)))
        return cls(text, text, idx, tuple(i + 1 for i in idx))

    # --- mapping ---------------------------------------------------------------------

    def _check_span(self, start: int, end: int, length: int, which: str) -> None:
        if not 0 <= start <= end <= length:
            raise ValueError(f"{which} span [{start},{end}) out of range for length {length}")

    def _source_boundary(self, p: int) -> int:
        """Source position corresponding to the target boundary ``p``."""
        if p < len(self.target):
            return self.lo[p]
        return max(self.hi) if self.target else 0

    def _target_boundary(self, p: int) -> int:
        """Target position corresponding to the source boundary ``p``: the index of the
        first target char that comes from at or after ``p`` (insertions at ``p`` count as
        before it)."""
        for j, b in enumerate(self.hi):
            if b > p:
                return j
        return len(self.target)

    def to_source(self, start: int, end: int) -> tuple[int, int]:
        """Map a target span to the smallest source span that produced it.

        Spans inside copied text map exactly; a span that touches part of a replacement
        expands to the whole replaced source chunk. An empty span maps to a boundary.
        """
        self._check_span(start, end, len(self.target), "target")
        if start == end:
            p = self._source_boundary(start)
            return p, p
        return min(self.lo[start:end]), max(self.hi[start:end])

    def to_target(self, start: int, end: int) -> tuple[int, int]:
        """Map a source span to the target span of everything derived from it.

        Included: target characters whose source interval overlaps the span, and
        characters inserted strictly inside it. A span that was deleted entirely maps to
        the empty target span where it used to be.
        """
        self._check_span(start, end, len(self.source), "source")
        if start < end:
            hits = [
                j
                for j, (a, b) in enumerate(zip(self.lo, self.hi, strict=True))
                if (a < end and b > start) or (a == b and start < a < end)
            ]
            if hits:
                return hits[0], hits[-1] + 1
        p = self._target_boundary(start)
        return p, p

    def source_text(self, start: int, end: int) -> str:
        """The original text behind target span ``[start, end)``."""
        s, e = self.to_source(start, end)
        return self.source[s:e]

    # --- composition -----------------------------------------------------------------

    def then(self, other: OffsetMap) -> OffsetMap:
        """Compose ``self`` (A -> B) with ``other`` (B -> C) into a map C -> A.

        ``other.source`` must equal ``self.target``.
        """
        if other.source != self.target:
            raise ValueError("cannot compose: other.source is not this map's target")
        lo: list[int] = []
        hi: list[int] = []
        for a, b in zip(other.lo, other.hi, strict=True):
            s, e = self.to_source(a, b)
            lo.append(s)
            hi.append(e)
        return OffsetMap(self.source, other.target, tuple(lo), tuple(hi))


def compose(maps: Sequence[OffsetMap]) -> OffsetMap:
    """Compose a chain of maps applied in order: ``maps[0]`` first."""
    if not maps:
        raise ValueError("need at least one map")
    out = maps[0]
    for m in maps[1:]:
        out = out.then(m)
    return out


class OffsetMapBuilder:
    """Build an :class:`OffsetMap` by consuming ``source`` left to right.

    >>> b = OffsetMapBuilder("आयु २१ वर्ष")  # "आयु" is 3 code points: आ य ु
    >>> b.copy(4); b.replace(1, "2"); b.replace(1, "1"); b.copy_rest()
    >>> m = b.build(); m.target
    'आयु 21 वर्ष'
    >>> m.source_text(4, 6)
    '२१'
    """

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self._parts: list[str] = []
        self._lo: list[int] = []
        self._hi: list[int] = []

    @property
    def remaining(self) -> int:
        return len(self.source) - self.pos

    def _need(self, n: int) -> None:
        if n < 0 or n > self.remaining:
            raise ValueError(f"cannot consume {n} chars at {self.pos}; {self.remaining} left")

    def copy(self, n: int = 1) -> None:
        """Copy the next ``n`` source characters unchanged."""
        self._need(n)
        self._parts.append(self.source[self.pos : self.pos + n])
        self._lo.extend(range(self.pos, self.pos + n))
        self._hi.extend(range(self.pos + 1, self.pos + n + 1))
        self.pos += n

    def replace(self, n: int, replacement: str) -> None:
        """Consume ``n`` source characters and emit ``replacement`` in their place."""
        self._need(n)
        self._parts.append(replacement)
        self._lo.extend([self.pos] * len(replacement))
        self._hi.extend([self.pos + n] * len(replacement))
        self.pos += n

    def delete(self, n: int = 1) -> None:
        self.replace(n, "")

    def insert(self, text: str) -> None:
        self.replace(0, text)

    def copy_rest(self) -> None:
        self.copy(self.remaining)

    def build(self) -> OffsetMap:
        if self.remaining:
            raise ValueError(f"{self.remaining} source chars not consumed")
        return OffsetMap(self.source, "".join(self._parts), tuple(self._lo), tuple(self._hi))
