"""T06: offset maps between original and transformed text."""

import random

import pytest

from indichallu.text.offsets import OffsetMap, OffsetMapBuilder, compose

ZWNJ = "\u200c"


def digits_to_ascii(text: str, zero: int) -> OffsetMap:
    """Test helper: replace one script's digits with ASCII, char by char."""
    b = OffsetMapBuilder(text)
    for ch in text:
        if zero <= ord(ch) <= zero + 9:
            b.replace(1, str(ord(ch) - zero))
        else:
            b.copy()
    return b.build()


# --- basics ----------------------------------------------------------------------------


def test_identity_maps_every_span_to_itself():
    text = "यह योजना राज्य सरकार द्वारा चलाई जाती है।"
    m = OffsetMap.identity(text)
    for a in range(len(text) + 1):
        for b in range(a, len(text) + 1):
            assert m.to_source(a, b) == (a, b)
            assert m.to_target(a, b) == (a, b)


@pytest.mark.parametrize(
    ("script", "zero"),
    [("Devanagari", 0x0966), ("Bengali", 0x09E6), ("Tamil", 0x0BE6), ("Telugu", 0x0C66)],
)
def test_digit_replacement_maps_exactly(script, zero):
    native = chr(zero + 2) + chr(zero + 1)  # "21" in the script's digits
    text = f"आयु {native} वर्ष"
    m = digits_to_ascii(text, zero)
    assert m.target == "आयु 21 वर्ष"
    s, e = m.to_source(*_find(m.target, "21"))
    assert m.source[s:e] == native
    assert m.to_target(s, e) == _find(m.target, "21")
    # one digit alone maps to exactly one digit
    assert m.source_text(5, 6) == chr(zero + 1)


def test_worked_example_span_maps_back_to_original_digits():
    """Slide 17 answer written with Devanagari digits; detection runs on ASCII digits."""
    original = "यह योजना राज्य सरकार द्वारा चलाई जाती है और आवेदक की आयु २१ वर्ष होनी चाहिए।"
    m = digits_to_ascii(original, 0x0966)
    claim = "आवेदक की आयु 21 वर्ष होनी चाहिए"
    s, e = m.to_source(*_find(m.target, claim))
    assert original[s:e] == "आवेदक की आयु २१ वर्ष होनी चाहिए"


def test_expansion_one_to_two():
    """U+0958 (क़) decomposes to क + nukta: 1 source char -> 2 target chars."""
    b = OffsetMapBuilder("xक़y")
    b.copy()
    b.replace(1, "क़")
    b.copy()
    m = b.build()
    assert m.target == "xक़y"
    assert m.to_source(1, 2) == (1, 2)  # half of the expansion -> the whole source char
    assert m.to_source(2, 3) == (1, 2)
    assert m.to_target(1, 2) == (1, 3)  # the source char -> both target chars
    assert m.to_source(0, 4) == (0, 3)


def test_contraction_two_to_one():
    b = OffsetMapBuilder("xक़y")
    b.copy()
    b.replace(2, "क़")
    b.copy()
    m = b.build()
    assert m.to_source(1, 2) == (1, 3)
    assert m.to_target(1, 2) == (1, 2)  # part of a chunk -> the whole merged char
    assert m.to_target(2, 3) == (1, 2)


def test_deletion():
    text = f"क्{ZWNJ}ष"  # ZWNJ removed by normalisation
    b = OffsetMapBuilder(text)
    b.copy(2)
    b.delete()
    b.copy()
    m = b.build()
    assert m.target == "क्ष"
    assert m.to_source(0, 3) == (0, 4)  # span across the deletion includes it
    assert m.to_target(2, 3) == (2, 2)  # the deleted char -> empty span where it was
    assert m.to_source(2, 3) == (3, 4)


def test_insertion():
    b = OffsetMapBuilder("ab")
    b.copy()
    b.insert(" ")
    b.copy()
    m = b.build()
    assert m.target == "a b"
    assert m.to_source(1, 2) == (1, 1)  # inserted char has no source
    assert m.to_source(0, 3) == (0, 2)
    assert m.to_target(0, 2) == (0, 3)  # insertion strictly inside is included
    assert m.to_target(0, 1) == (0, 1)


def test_empty_and_fully_deleted_strings():
    b = OffsetMapBuilder(ZWNJ * 3)
    b.delete(3)
    m = b.build()
    assert m.target == ""
    assert m.to_source(0, 0) == (0, 0)
    assert m.to_target(0, 3) == (0, 0)
    assert OffsetMap.identity("").to_source(0, 0) == (0, 0)


# --- composition -----------------------------------------------------------------------


def test_compose_digits_then_deletion_points_to_original():
    original = f"आयु २१{ZWNJ} वर्ष"
    m1 = digits_to_ascii(original, 0x0966)  # "आयु 21\u200c वर्ष"
    b = OffsetMapBuilder(m1.target)
    for ch in m1.target:
        b.delete() if ch == ZWNJ else b.copy()
    m2 = b.build()  # "आयु 21 वर्ष"
    m = m1.then(m2)
    assert m.source == original and m.target == "आयु 21 वर्ष"
    s, e = m.to_source(*_find(m.target, "21"))
    assert original[s:e] == "२१"


def test_compose_is_associative():
    m1, m2, m3 = _random_chain(random.Random(7), 3)
    left = m1.then(m2).then(m3)
    right = m1.then(m2.then(m3))
    assert (left.lo, left.hi) == (right.lo, right.hi)
    assert compose([m1, m2, m3]) == left


def test_insertion_inside_an_earlier_expansion():
    """Composition must not assume monotone intervals (regression for the design)."""
    b1 = OffsetMapBuilder("xक़y")
    b1.copy()
    b1.replace(1, "क़")
    b1.copy()
    m1 = b1.build()
    b2 = OffsetMapBuilder(m1.target)
    b2.copy(2)
    b2.insert("|")  # lands between the two halves of the expansion
    b2.copy_rest()
    m = m1.then(b2.build())
    assert m.target == "xक|़y"
    # The inserted "|" has no source; it sits at the start of the expanded char. With
    # monotone-only bookkeeping this composition raised (hi went 1, 2, 1).
    assert m.to_source(2, 3) == (1, 1)
    assert m.to_source(1, 4) == (1, 2)
    assert m.to_source(0, 5) == (0, 3)


def test_compose_rejects_mismatched_maps():
    with pytest.raises(ValueError):
        OffsetMap.identity("ab").then(OffsetMap.identity("abc"))
    with pytest.raises(ValueError):
        compose([])


# --- property test against an independent reference ------------------------------------

ALPHABET = "कखगघचजटडतदनपबमयरलवसह्ािीुूेैोौंःअआइ" + "অআকখগ" + "தமிழ்" + "తెలుగు" + "०१२३४५" + ZWNJ + " "


def _random_map(rng: random.Random, source: str):
    """Random edits; also returns, per target char, the ground-truth source chunk."""
    b = OffsetMapBuilder(source)
    truth: list[tuple[int, int, bool]] = []  # (start, end, is_copy)
    while b.remaining:
        op = rng.random()
        n = rng.randint(1, min(3, b.remaining))
        if op < 0.5:
            truth += [(b.pos + k, b.pos + k + 1, True) for k in range(n)]
            b.copy(n)
        elif op < 0.8:
            rep = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 3)))
            truth += [(b.pos, b.pos + n, False)] * len(rep)
            b.replace(n, rep)
        elif op < 0.9:
            b.delete(n)
        else:
            rep = rng.choice(ALPHABET)
            truth.append((b.pos, b.pos, False))
            b.insert(rep)
    return b.build(), truth


def _random_chain(rng: random.Random, k: int):
    text = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(5, 25)))
    maps = []
    for _ in range(k):
        m, _ = _random_map(rng, text)
        maps.append(m)
        text = m.target
    return maps


@pytest.mark.parametrize("seed", range(200))
def test_random_maps_match_ground_truth(seed):
    rng = random.Random(seed)
    source = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 30)))
    m, truth = _random_map(rng, source)
    assert len(truth) == len(m.target)
    for j, (s, e, is_copy) in enumerate(truth):
        assert (m.lo[j], m.hi[j]) == (s, e)
        if is_copy:
            assert m.source[s:e] == m.target[j]
    # every non-empty target span maps to the hull of its characters' chunks, and back
    for _ in range(20):
        if not m.target:
            break
        a = rng.randrange(len(m.target))
        b = rng.randint(a + 1, len(m.target))
        s, e = m.to_source(a, b)
        chunks = truth[a:b]
        assert (s, e) == (min(c[0] for c in chunks), max(c[1] for c in chunks))
        # Round trip covers the span, once inserted chars at its edges are trimmed off:
        # an insertion exactly at a source boundary is not attributed to either side.
        while a < b and truth[a][0] == truth[a][1]:
            a += 1
        while a < b and truth[b - 1][0] == truth[b - 1][1]:
            b -= 1
        if a < b:
            ta, tb = m.to_target(*m.to_source(a, b))
            assert ta <= a and b <= tb


@pytest.mark.parametrize("seed", range(100))
def test_random_composition_matches_set_reference(seed):
    """Composed map == per-character reference built with plain sets."""
    m1, m2 = _random_chain(random.Random(1000 + seed), 2)
    m = m1.then(m2)
    for j in range(len(m.target)):
        mids = range(m2.lo[j], m2.hi[j])
        if mids:
            pos = {i for k in mids for i in range(m1.lo[k], m1.hi[k])}
            bounds = [m1.lo[k] for k in mids] + [m1.hi[k] for k in mids]
            expected = (min(bounds), max(bounds))
            assert (m.lo[j], m.hi[j]) == expected
            assert pos <= set(range(*expected))
        else:  # inserted by m2: empty interval at the corresponding boundary
            assert m.lo[j] == m.hi[j]


# --- errors ----------------------------------------------------------------------------


def test_builder_guards():
    b = OffsetMapBuilder("abc")
    with pytest.raises(ValueError):
        b.copy(4)
    b.copy(1)
    with pytest.raises(ValueError):
        b.build()  # 2 chars not consumed
    with pytest.raises(ValueError):
        b.replace(-1, "x")


def test_out_of_range_spans_and_intervals_rejected():
    m = OffsetMap.identity("abc")
    for bad in [(-1, 1), (2, 1), (0, 4)]:
        with pytest.raises(ValueError):
            m.to_source(*bad)
        with pytest.raises(ValueError):
            m.to_target(*bad)
    with pytest.raises(ValueError):
        OffsetMap("ab", "x", (0,), (3,))
    with pytest.raises(ValueError):
        OffsetMap("ab", "xy", (0,), (1,))


def _find(text: str, sub: str) -> tuple[int, int]:
    i = text.index(sub)
    return i, i + len(sub)
