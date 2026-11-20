"""T08: Indic digits -> ASCII with exact offsets (CLAUDE.md §3 row 2)."""

import random
import unicodedata

import pytest

from indichallu.languages import load_languages
from indichallu.text.normalize import normalize
from indichallu.text.numerals import has_native_digits, indic_digit_table, to_ascii_digits

LANGS = load_languages()


@pytest.mark.parametrize(
    ("script", "year"),
    [
        ("Devanagari", "२०२६"),
        ("Bengali", "২০২৬"),
        ("Tamil", "௨௦௨௬"),
        ("Telugu", "౨౦౨౬"),
    ],
)
def test_named_scripts_convert_and_map_back_exactly(script, year):
    text = f"वर्ष {year} में"
    r = to_ascii_digits(text)
    assert r.text == "वर्ष 2026 में"
    i = r.text.index("2026")
    assert r.offsets.source_text(i, i + 4) == year
    for k in range(4):  # each digit maps to exactly its own source digit
        assert r.offsets.to_source(i + k, i + k + 1) == (i + k, i + k + 1)
        assert r.offsets.to_target(i + k, i + k + 1) == (i + k, i + k + 1)


@pytest.mark.parametrize("code", list(LANGS))
def test_every_configured_language(code):
    digits = LANGS[code].digits
    r = to_ascii_digits(digits)
    assert r.text == "0123456789"
    assert r.offsets.lo == tuple(range(10)) and r.offsets.hi == tuple(range(1, 11))


def test_table_covers_the_eight_digit_blocks():
    table = indic_digit_table()
    assert len(table) == 80  # hi and mr share Devanagari
    for ch, ascii_digit in table.items():
        assert unicodedata.digit(ch) == int(ascii_digit)


def test_text_and_ascii_digits_are_untouched():
    text = "आयु 21 वर्ष, Rs. 1,00,000.50 तक।"
    r = to_ascii_digits(text)
    assert r.text == text
    assert r.offsets.lo == tuple(range(len(text)))
    assert not has_native_digits(text)


def test_mixed_script_number():
    r = to_ascii_digits("२1৩")
    assert r.text == "213"


def test_non_decimal_tamil_numerals_are_not_converted():
    text = "௰ ௱ ௲"  # 10, 100, 1000: not positional digits
    assert to_ascii_digits(text).text == text
    assert to_ascii_digits(text, scope="all").text == text


def test_scope_all_also_converts_other_decimal_digits():
    text = "٢٠ ２０ २०"  # Arabic-Indic, full-width, Devanagari
    assert to_ascii_digits(text).text == "٢٠ ２０ 20"
    assert to_ascii_digits(text, scope="all").text == "20 20 20"
    assert has_native_digits("٢", scope="all") and not has_native_digits("٢")


@pytest.mark.parametrize("bad", ["arabic", "", "ALL", None])
def test_both_functions_reject_bad_scopes_consistently(bad):
    """Codex review on #13: has_native_digits must not silently treat a bad scope as 'all'."""
    for text in ("٢٠ २०", ""):
        with pytest.raises(ValueError):
            to_ascii_digits(text, scope=bad)
        with pytest.raises(ValueError):
            has_native_digits(text, scope=bad)


@pytest.mark.parametrize("seed", range(100))
def test_random_text_is_length_preserving_and_exact(seed):
    rng = random.Random(seed)
    pool = list(indic_digit_table()) + list("0123456789 अकখটமతಕമગਕ,.") + ["\u200c"]
    text = "".join(rng.choice(pool) for _ in range(rng.randint(0, 40)))
    r = to_ascii_digits(text)
    assert len(r.text) == len(text)
    assert r.offsets.lo == tuple(range(len(text)))
    for ch_in, ch_out in zip(text, r.text, strict=True):
        if ch_in in indic_digit_table():
            assert ch_out == indic_digit_table()[ch_in]
        else:
            assert ch_out == ch_in


# --- pipeline integration --------------------------------------------------------------


def test_normalize_keeps_native_digits_by_default():
    assert "२१" in normalize("आयु २१ वर्ष", "hi").text


def test_worked_example_with_native_digits_and_noise():
    """Slide-17 answer with Devanagari digits, a ZWNJ and a pipe. After the full pipeline
    the claim reads with ASCII digits and maps back to the original slice."""
    zwnj = "\u200c"
    original = f"यह योजना राज्य सरकार द्वारा चलाई जाती है और आवेदक{zwnj} की आयु २१ वर्ष होनी चाहिए|"
    r = normalize(original, "hi", ascii_digits=True)
    claim = "आवेदक की आयु 21 वर्ष होनी चाहिए"
    i = r.text.index(claim)
    s, e = r.offsets.to_source(i, i + len(claim))
    assert original[s:e] == f"आवेदक{zwnj} की आयु २१ वर्ष होनी चाहिए"
    j = r.text.index("21")
    assert r.offsets.source_text(j, j + 2) == "२१"


@pytest.mark.parametrize("code", list(LANGS))
def test_full_pipeline_digits_in_every_language(code):
    lang = LANGS[code]
    original = (
        f"{lang.native_name} {lang.digits[2]}{lang.digits[0]}{lang.digits[2]}{lang.digits[6]}"
    )
    r = normalize(original, code, ascii_digits=True)
    i = r.text.index("2026")
    assert r.offsets.source_text(i, i + 4) == original[-4:]
