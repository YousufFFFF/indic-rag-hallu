"""Unicode + Indic script normalisation that keeps an offset map to the original text.

Pipeline (docs/DECISIONS.md D-012), each step returning an :class:`OffsetMap`:

1. **NFC** (:func:`nfc`): exact Unicode NFC, with an alignment built by normalising the
   text in independent chunks split at NFC-safe boundaries.
2. **IndicNLP** (:func:`indicnlp_normalize`): the language's IndicNLP normaliser
   (nukta forms decomposed, ZWJ/ZWNJ removed, ``|`` -> danda, ``:`` after a letter ->
   visarga, ...), aligned to its input with a lockstep minimum-edit alignment
   (:func:`align`) because the library returns no offsets.

:func:`normalize` runs both and composes the maps, so a span found on the normalised
text maps straight back to the original. Apply the same pipeline to everything that is
compared (claims and passages); the output is a matching form, not guaranteed NFC
(IndicNLP decomposes e.g. U+0929 ``ऩ``, which NFC keeps composed).

>>> r = normalize("क\u200cख|", "hi")    # ZWNJ removed, "|" becomes a danda
>>> r.text == "कख।"
True
>>> r.offsets.to_source(2, 3)            # the danda came from the "|" at index 3
(3, 4)
>>> r.offsets.to_target(1, 2)            # the deleted ZWNJ -> empty span where it was
(1, 1)
"""

from __future__ import annotations

import unicodedata
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


# Alignment costs (DECISIONS D-012). Insert/delete cost 1. Substituting a RELATED
# character (same base letter after NFKD, e.g. ऩ -> न; same digit value, e.g. २ -> 2; or
# whitespace -> whitespace) costs 1. Any other substitution costs 2, the same as a
# deletion plus an insertion, so unrelated characters are only aligned when that keeps
# more matches (LCS-style). This stops alignments from shifting by a char to swallow a
# neighbouring separator (cost-1 subs tie with that), and lets e.g. "ऩ" -> "न" + nukta win
# over pairing "न" with an unrelated deleted char.
INDEL_COST = 1
RELATED_SUB_COST = 1
SUB_COST = 2

# Initial half-width of the band around the diagonal. It doubles until the banded optimum
# is provably global (see _edit_ops), so cumulative drift is handled...
ALIGN_BAND = 32
# ...within a work budget: one alignment pass fills (rows x band width) cells. If the next
# pass would exceed this, align() falls back to one coarse chunk for the differing region
# (pathological input, e.g. thousands of deleted ZWNJs followed by as many expansions).
# That is still safe, since every span maps to a region containing its true source, just
# less precise. Time and memory stay bounded (DECISIONS D-012).
MAX_ALIGN_CELLS = 2_000_000

_EQ, _SUB, _DEL, _INS = 0, 1, 2, 3


@lru_cache(maxsize=4096)
def _base(ch: str) -> str:
    return unicodedata.normalize("NFKD", ch)[:1] or ch


def _sub_cost(x: str, y: str) -> int:
    if _base(x) == _base(y):
        return RELATED_SUB_COST
    dx, dy = unicodedata.digit(x, None), unicodedata.digit(y, None)
    if dx is not None and dx == dy:
        return RELATED_SUB_COST
    if x.isspace() and y.isspace():
        return RELATED_SUB_COST
    return SUB_COST


def _banded_ops(a: str, b: str, band: int) -> tuple[int, list[int]]:
    """Optimal alignment restricted to diagonals j - i within ``band`` of [0, m - n].

    Returns (cost, ops). Ties prefer match/substitute, then delete, then insert, which
    keeps 1:1 substitutions (e.g. digits) as substitutions. Memory: two rows of costs plus
    one byte of backpointer per band cell.
    """
    n, m = len(a), len(b)
    lo_d = min(0, m - n) - band  # allowed range of j - i
    hi_d = max(0, m - n) + band
    width = hi_d - lo_d + 1
    inf = SUB_COST * (n + m) + 1
    prev: list[int] = []
    row = [inf] * width
    back = [bytearray(width) for _ in range(n + 1)]
    for i in range(n + 1):
        if i:
            prev, row = row, [inf] * width
        # only cells with 0 <= j <= m
        for k in range(max(0, -i - lo_d), min(width - 1, m - i - lo_d) + 1):
            j = i + lo_d + k
            if i == 0 and j == 0:
                row[k] = 0
                continue
            best, op = inf, _EQ
            if i and j:  # diagonal: same k in the previous row
                same = a[i - 1] == b[j - 1]
                c = prev[k] + (0 if same else _sub_cost(a[i - 1], b[j - 1]))
                if c < best:
                    best, op = c, (_EQ if same else _SUB)
            if i and k + 1 < width:  # delete a[i-1]: from (i-1, j)
                c = prev[k + 1] + INDEL_COST
                if c < best:
                    best, op = c, _DEL
            if j and k:  # insert b[j-1]: from (i, j-1)
                c = row[k - 1] + INDEL_COST
                if c < best:
                    best, op = c, _INS
            row[k], back[i][k] = best, op
    final_cost = row[m - n - lo_d]
    ops: list[int] = []
    i, j = n, m
    while i or j:
        op = back[i][j - i - lo_d]
        ops.append(op)
        if op in (_EQ, _SUB):
            i, j = i - 1, j - 1
        elif op == _DEL:
            i -= 1
        else:
            j -= 1
    ops.reverse()
    return final_cost, ops


def _edit_ops(a: str, b: str, max_cells: int | None = None) -> list[int] | None:
    """Globally optimal alignment of ``a`` to ``b`` under the costs above, or None.

    Runs the banded alignment and doubles the band until the result is provably optimal.
    Every insert/delete costs 1 and only they change the diagonal ``d = j - i``, so a path
    that leaves the band reaches some diagonal ``d`` with ``|d| > band + max(0, D)`` or
    ``d < min(0, D) - band`` (``D = m - n``) and costs at least ``|d| + |D - d|`` >=
    ``|D| + 2*band + 2``. If the banded optimum costs less than that, nothing outside the
    band can beat it (Ukkonen's band-doubling argument). Returns None if a pass would
    need more than ``max_cells`` cells (rows x band width; default ``MAX_ALIGN_CELLS``).
    """
    n, m = len(a), len(b)
    max_cells = MAX_ALIGN_CELLS if max_cells is None else max_cells
    band = ALIGN_BAND
    while True:
        if (n + 1) * (abs(m - n) + 2 * band + 1) > max_cells:
            return None
        cost, ops = _banded_ops(a, b, band)
        covers_all = min(0, m - n) - band <= -n and max(0, m - n) + band >= m
        if covers_all or cost < abs(m - n) + 2 * band + 2:
            return ops
        band *= 2


def align(source: str, target: str) -> OffsetMap:
    """Offset map from ``target`` back to ``source`` for a black-box transformation.

    For transformations that don't report offsets (e.g. IndicNLP), which make small local
    edits. Uses a globally optimal (LCS-style) alignment that walks both strings in step,
    so repeated text keeps its provenance. A longest-common-substring diff can attribute an
    output word to a later identical source word (DECISIONS D-012).

    Chunks: unchanged text maps 1:1. Each substitution starts its own chunk, and inserts
    or deletes that directly follow it join that chunk (an expansion like ऩ -> न + nukta,
    or a contraction like ে + া -> ো). Inserts/deletes with no substitution before them
    form a chunk of their own (e.g. deleted ZWNJs). If optimality would need a band wider
    than ``MAX_ALIGN_CELLS`` allows, the differing middle becomes one chunk: coarse but safe.

    >>> m = align("राम:रामः", "रामःरामः")   # only the first ":" became a visarga
    >>> m.to_source(0, 4), m.to_source(4, 8)
    ((0, 4), (4, 8))
    """
    builder = OffsetMapBuilder(source)
    if source == target:
        return OffsetMap.identity(source)
    n, m = len(source), len(target)
    p = 0  # common prefix and suffix are copied in lockstep
    while p < min(n, m) and source[p] == target[p]:
        p += 1
    s = 0
    while s < min(n, m) - p and source[n - 1 - s] == target[m - 1 - s]:
        s += 1
    builder.copy(p)

    mid_target = target[p : m - s]
    j = 0
    chunk_src, chunk_tgt = 0, []  # the open chunk

    def flush() -> None:
        nonlocal chunk_src, chunk_tgt
        if chunk_src or chunk_tgt:
            builder.replace(chunk_src, "".join(chunk_tgt))
        chunk_src, chunk_tgt = 0, []

    ops = _edit_ops(source[p : n - s], mid_target)
    if ops is None:  # pathological drift: one coarse but safe chunk for the middle
        builder.replace(n - s - p, mid_target)
        ops = []
    for op in ops:
        if op == _EQ:
            flush()
            builder.copy(1)
            j += 1
        elif op == _SUB:
            flush()  # a substitution always starts a new chunk
            chunk_src, chunk_tgt = 1, [mid_target[j]]
            j += 1
        elif op == _DEL:
            chunk_src += 1
        else:  # _INS
            chunk_tgt.append(mid_target[j])
            j += 1
    flush()
    builder.copy(s)
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


def normalize(
    text: str,
    lang: str,
    *,
    indicnlp: bool = True,
    ascii_digits: bool = False,
    **options: object,
) -> Normalized:
    """NFC, then (by default) IndicNLP, then (optionally) Indic digits -> ASCII.

    Offsets always map back to ``text``. ``ascii_digits`` is off by default so the
    pipeline's output stays in the source script; matching code turns it on
    (DECISIONS D-013).
    """
    _check_lang(lang)
    result = nfc(text)
    if indicnlp:
        step = indicnlp_normalize(result.text, lang, **options)
        result = Normalized(step.text, result.offsets.then(step.offsets))
    if ascii_digits:
        from indichallu.text.numerals import to_ascii_digits  # avoids an import cycle

        step = to_ascii_digits(result.text)
        result = Normalized(step.text, result.offsets.then(step.offsets))
    return result
