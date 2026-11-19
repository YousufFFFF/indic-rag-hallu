"""T03: configs/languages.yaml matches the Stage I deck and the Unicode standard."""

import unicodedata

import pytest
from pydantic import ValidationError

from indichallu.languages import Family, enabled_languages, load_languages

# Deck slide 5, in slide order.
DECK_LANGUAGES = ["hi", "bn", "ta", "te", "mr", "kn", "ml", "gu", "pa"]
DECK_ENDONYMS = {
    "hi": "हिन्दी",
    "bn": "বাংলা",
    "ta": "தமிழ்",
    "te": "తెలుగు",
    "mr": "मराठी",
    "kn": "ಕನ್ನಡ",
    "ml": "മലയാളം",
    "gu": "ગુજરાતી",
    "pa": "ਪੰਜਾਬੀ",
}

LANGS = load_languages()


def test_exactly_the_nine_deck_languages():
    assert list(LANGS) == DECK_LANGUAGES


def test_m0_enables_hindi_only():
    assert list(enabled_languages()) == ["hi"]


@pytest.mark.parametrize("code", DECK_LANGUAGES)
def test_endonym_matches_deck(code):
    assert LANGS[code].native_name == DECK_ENDONYMS[code]


@pytest.mark.parametrize("code", DECK_LANGUAGES)
def test_digits_are_the_scripts_digits_0_to_9(code):
    lang = LANGS[code]
    for value, ch in enumerate(lang.digits):
        assert unicodedata.digit(ch) == value
        assert unicodedata.name(ch).startswith(f"{lang.unicode_script} DIGIT ")


@pytest.mark.parametrize("code", DECK_LANGUAGES)
def test_endonym_is_written_in_the_configured_block(code):
    lang = LANGS[code]
    first, last = lang.unicode_block
    for ch in lang.native_name:
        assert first <= ord(ch) <= last, f"{ch!r} U+{ord(ch):04X} outside block"
        assert unicodedata.name(ch).startswith(lang.unicode_script + " ")


@pytest.mark.parametrize("code", DECK_LANGUAGES)
def test_block_start_belongs_to_the_script(code):
    """The first assigned code point of each Indic block names its script."""
    lang = LANGS[code]
    first, last = lang.unicode_block
    names = (unicodedata.name(chr(cp), "") for cp in range(first, last + 1))
    first_named = next(n for n in names if n)
    assert first_named.startswith(lang.unicode_script + " ")


def test_families():
    dravidian = {c for c, lang in LANGS.items() if lang.family is Family.DRAVIDIAN}
    assert dravidian == {"ta", "te", "kn", "ml"}


def test_indicqa_files_follow_repo_naming():
    for code, lang in LANGS.items():
        assert lang.indicqa_file == f"data/indicqa.{code}.json"


def test_resource_level_not_filled_without_a_source():
    # CLAUDE.md: never fabricate. Filled in M1 from a verified citation only.
    assert all(lang.resource_level is None for lang in LANGS.values())


# --- the loader rejects malformed configs ---------------------------------------------


def _write(tmp_path, text):
    p = tmp_path / "languages.yaml"
    p.write_text(text, encoding="utf-8")
    return p


GOOD_ENTRY = """
    name: Hindi
    native_name: हिन्दी
    iso639_3: hin
    script: Deva
    unicode_script: DEVANAGARI
    unicode_block: [0x0900, 0x097F]
    digit_zero: 0x0966
    family: indo-aryan
    indictrans2_code: hin_Deva
    indicqa_file: data/indicqa.hi.json
    resource_level: null
    enabled: true
"""


def test_loader_accepts_good_entry(tmp_path):
    langs = load_languages(_write(tmp_path, "languages:\n  hi:" + GOOD_ENTRY))
    assert langs["hi"].digits == "०१२३४५६७८९"


def test_loader_rejects_unknown_key(tmp_path):
    bad = "languages:\n  hi:" + GOOD_ENTRY + "    typo_field: 1\n"
    with pytest.raises(ValidationError):
        load_languages(_write(tmp_path, bad))


def test_loader_rejects_inconsistent_indictrans2_code(tmp_path):
    bad = "languages:\n  hi:" + GOOD_ENTRY.replace("hin_Deva", "hin_Beng")
    with pytest.raises(ValidationError):
        load_languages(_write(tmp_path, bad))


def test_loader_rejects_digits_outside_block(tmp_path):
    bad = "languages:\n  hi:" + GOOD_ENTRY.replace("0x0966", "0x09E6")
    with pytest.raises(ValidationError):
        load_languages(_write(tmp_path, bad))


def test_loader_rejects_wrong_top_level(tmp_path):
    with pytest.raises(ValueError):
        load_languages(_write(tmp_path, "langs: {}\n"))
