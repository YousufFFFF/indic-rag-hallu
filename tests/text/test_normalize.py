"""T07: NFC + IndicNLP normalisation with offset maps back to the original."""

import random
import unicodedata

import pytest
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

from indichallu.languages import load_languages
from indichallu.text.normalize import (
    _composing_starters,
    align,
    indicnlp_normalize,
    nfc,
    normalize,
)
from indichallu.text.offsets import OffsetMapBuilder

ZWNJ, ZWJ, NBSP = "‌", "‍", " "
LANGS = load_languages()

# --- NFC -------------------------------------------------------------------------------


def test_nfc_of_already_normal_text_is_an_exact_identity():
    text = "यह योजना राज्य सरकार द्वारा चलाई जाती है।"
    r = nfc(text)
    assert r.text == text
    assert r.offsets.lo == tuple(range(len(text)))


def test_nfc_expands_devanagari_nukta_letter():
    """U+0958 is a composition exclusion: NFC turns it into क + nukta (1 -> 2)."""
    r = nfc("xक़y")
    assert r.text == "xक़y"
    assert r.offsets.to_source(1, 2) == (1, 2)
    assert r.offsets.to_source(2, 3) == (1, 2)
    assert r.offsets.to_target(0, 1) == (0, 1)  # neighbours stay exact


@pytest.mark.parametrize(
    ("decomposed", "composed", "script"),
    [
        ("কো", "কো", "Bengali ো"),
        ("கொ", "கொ", "Tamil ொ"),
        ("കൊ", "കൊ", "Malayalam ൊ"),
        ("ಕೀ", "ಕೀ", "Kannada ೀ"),
        ("కై", "కై", "Telugu ై"),
    ],
)
def test_nfc_composes_two_part_vowels(decomposed, composed, script):
    """These vowel parts look like separate starters but NFC merges them."""
    r = nfc(decomposed)
    assert r.text == composed == unicodedata.normalize("NFC", decomposed)
    assert r.offsets.to_source(0, 1) == (0, 1)  # consonant exact
    assert r.offsets.to_source(1, 2) == (1, 3)  # composed vowel <- both parts


def test_nfc_reorders_combining_marks():
    text = "ạ́b"  # acute then dot-below; NFC puts dot-below first and composes
    r = nfc(text)
    assert r.text == unicodedata.normalize("NFC", text)
    assert r.offsets.source_text(0, len(r.text) - 1) == "ạ́"
    assert r.offsets.to_source(len(r.text) - 1, len(r.text)) == (3, 4)


def test_composing_starter_set_contains_known_two_part_vowel_marks():
    s = _composing_starters()
    for cp in (0x09BE, 0x09D7, 0x0BBE, 0x0BD7, 0x0CD5, 0x0CD6, 0x0D3E, 0x0D57, 0x1161):
        assert chr(cp) in s, f"U+{cp:04X}"
    assert "क" not in s and "a" not in s


INDIC_POOL = [chr(c) for c in range(0x0900, 0x0D80) if unicodedata.name(chr(c), "")] + list(
    "abc ̣́̀" + ZWNJ + ZWJ + NBSP + "|:"
)


@pytest.mark.parametrize("seed", range(300))
def test_nfc_matches_unicodedata_and_maps_every_char(seed):
    rng = random.Random(seed)
    text = "".join(rng.choice(INDIC_POOL) for _ in range(rng.randint(0, 20)))
    r = nfc(text)
    assert r.text == unicodedata.normalize("NFC", text)
    for j, ch in enumerate(r.text):
        src = r.offsets.source_text(j, j + 1)
        assert ch in unicodedata.normalize("NFC", src)  # each char comes from its chunk


# --- align (black-box transforms) ------------------------------------------------------


@pytest.mark.parametrize("seed", range(200))
def test_align_reconstructs_target_and_keeps_unchanged_text_exact(seed):
    rng = random.Random(10_000 + seed)
    source = "".join(rng.choice(INDIC_POOL) for _ in range(rng.randint(0, 30)))
    b = OffsetMapBuilder(source)
    while b.remaining:  # random local edits, like a normaliser makes
        n = rng.randint(1, min(2, b.remaining))
        if rng.random() < 0.7:
            b.copy(n)
        else:
            b.replace(n, "".join(rng.choice(INDIC_POOL) for _ in range(rng.randint(0, 2))))
    target = b.build().target
    m = align(source, target)
    assert m.target == target
    for j in range(len(target)):
        s, e = m.to_source(j, j + 1)
        if e - s == 1 and m.source[s] == target[j]:
            continue  # exact copy
        assert s <= e  # part of a replaced chunk (or an insertion)


# --- IndicNLP --------------------------------------------------------------------------


def test_indicnlp_output_is_exactly_the_librarys():
    text = f"क{ZWNJ}ख| आयु{NBSP}२१ वर्ष। रामः राम:"
    expected = IndicNormalizerFactory().get_normalizer("hi").normalize(text)
    assert indicnlp_normalize(text, "hi").text == expected


def test_indicnlp_hindi_edits_map_back():
    text = f"क{ZWNJ}ख|"
    r = indicnlp_normalize(text, "hi")
    assert r.text == "कख।"
    assert r.offsets.to_target(1, 2) == (1, 1)  # deleted ZWNJ
    assert r.offsets.source_text(2, 3) == "|"  # danda <- pipe
    assert r.offsets.source_text(0, 2) == f"क{ZWNJ}ख"


def test_indicnlp_colon_after_letter_becomes_visarga():
    r = indicnlp_normalize("राम:", "hi")
    assert r.text == "रामः"
    assert r.offsets.source_text(3, 4) == ":"


@pytest.mark.parametrize("code", list(LANGS))
def test_every_language_round_trips_its_endonym(code):
    """Each configured language normalises (via its own IndicNLP normaliser) and maps back."""
    lang = LANGS[code]
    text = f"{lang.native_name}{ZWNJ} {lang.digits[2]}{lang.digits[1]}|"
    r = normalize(text, code)
    assert r.offsets.source == text
    i = r.text.index(lang.digits[2] + lang.digits[1])
    assert r.offsets.source_text(i, i + 2) == lang.digits[2] + lang.digits[1]


def test_unsupported_language_and_options_are_rejected():
    with pytest.raises(ValueError):
        normalize("abc", "en")
    with pytest.raises(ValueError):
        indicnlp_normalize("abc", "or")  # IndicNLP knows Odia, our config does not
    with pytest.raises(TypeError):
        indicnlp_normalize("abc", "hi", remove_nukta=True)  # typo


def test_indicnlp_options_are_passed_through():
    text = "क़"  # क + nukta
    assert indicnlp_normalize(text, "hi").text == text
    assert indicnlp_normalize(text, "hi", remove_nuktas=True).text == "क"


# --- full pipeline ---------------------------------------------------------------------


def test_pipeline_is_nfc_then_indicnlp():
    text = f"xक़{ZWNJ}y|"
    r = normalize(text, "hi")
    step = (
        IndicNormalizerFactory().get_normalizer("hi").normalize(unicodedata.normalize("NFC", text))
    )
    assert r.text == step
    assert normalize(text, "hi", indicnlp=False).text == unicodedata.normalize("NFC", text)


def test_worked_example_span_survives_normalisation():
    """Slide-17 answer with typical noise: a ZWNJ, a precomposed nukta letter (ज़), a pipe."""
    original = f"यह योजना राज्य सरकार द्वारा चलाई जाती है और आवेदक{ZWNJ} की आयु 21 वर्ष होनी चाहिए ज़रूरी|"
    r = normalize(original, "hi")
    claim = "आवेदक की आयु 21 वर्ष होनी चाहिए"
    i = r.text.index(claim)
    s, e = r.offsets.to_source(i, i + len(claim))
    assert original[s:e] == f"आवेदक{ZWNJ} की आयु 21 वर्ष होनी चाहिए"
    j = r.text.index("ज़")  # decomposed ज़ in the output
    assert r.offsets.to_source(j, j + 2) == (original.index("ज़"), original.index("ज़") + 1)


def _hard_pool() -> list[str]:
    """Every BMP char with a canonical decomposition, every combining mark, the composing
    starters, Hangul jamo and the Indic blocks: the inputs most likely to break chunking."""
    decomposable = [
        chr(c)
        for c in range(0x10000)
        if (d := unicodedata.decomposition(chr(c))) and not d.startswith("<")
    ]
    marks = [chr(c) for c in range(0x10000) if unicodedata.combining(chr(c))]
    jamo = [chr(c) for c in range(0x1100, 0x1200)]
    return decomposable + marks + sorted(_composing_starters()) + jamo + INDIC_POOL


def test_nfc_matches_unicodedata_on_hard_inputs():
    pool, rng = _hard_pool(), random.Random(2026)
    for _ in range(3000):
        text = "".join(rng.choice(pool) for _ in range(rng.randint(0, 10)))
        assert nfc(text).text == unicodedata.normalize("NFC", text), ascii(text)


def test_nfc_keeps_neighbours_of_changed_chars_exact():
    """A char NFC rewrites (U+095B ज़ -> ज + nukta) must not drag its neighbours along."""
    r = nfc("a \u095bb")
    assert r.offsets.to_source(1, 2) == (1, 2)  # the space
    assert r.offsets.to_source(2, 4) == (2, 3)  # ज + nukta <- U+095B only
    assert r.offsets.to_source(4, 5) == (3, 4)  # "b"


# --- provenance in repetitive text (Codex review on PR #11) ----------------------------


def _word_spans(text: str, sep: str = " ") -> list[tuple[int, int]]:
    spans, start = [], 0
    for i, ch in enumerate(text + sep):
        if ch == sep:
            spans.append((start, i))
            start = i + 1
    return spans


def _assert_wordwise_provenance(source: str, m, sep: str = " ") -> None:
    """Output word k must map back inside source word k: no cross-occurrence mix-ups."""
    src_words, out_words = _word_spans(source, sep), _word_spans(m.target, sep)
    assert len(src_words) == len(out_words)
    for (a, b), (s0, e0) in zip(out_words, src_words, strict=True):
        if a == b:
            continue  # word normalised away entirely
        s, e = m.to_source(a, b)
        assert s0 <= s and e <= e0, (m.target[a:b], source[s0:e0], source[s:e])


def test_codex_case_repeated_word_keeps_its_own_occurrence():
    r = indicnlp_normalize("राम:रामः", "hi")
    assert r.text == "रामःरामः"
    assert r.offsets.to_source(0, 4) == (0, 4)  # was (4, 8) with a longest-match diff
    assert r.offsets.to_source(4, 8) == (4, 8)
    assert r.offsets.to_source(3, 4) == (3, 4)  # the visarga <- the first ":"


@pytest.mark.parametrize("code", list(LANGS))
@pytest.mark.parametrize("seed", range(10))
def test_indicnlp_repetitive_text_keeps_wordwise_provenance(code, seed):
    """Real IndicNLP on text built from few repeated words with noise, in every language."""
    lang, rng = LANGS[code], random.Random(seed)
    vocab = [lang.native_name, lang.digits[:3], lang.native_name[:2], lang.native_name[1:]]
    noise = ["", "", ":", "|", ZWNJ, ZWJ]
    words = [rng.choice(noise) + rng.choice(vocab) + rng.choice(noise) for _ in range(40)]
    source = " ".join(words)
    r = indicnlp_normalize(source, code)
    _assert_wordwise_provenance(source, r.offsets)


@pytest.mark.parametrize("seed", range(200))
def test_align_repetitive_text_keeps_wordwise_provenance(seed):
    """Random local edits inside words of highly repetitive text."""
    rng = random.Random(50_000 + seed)
    vocab = ["राम", "रामः", "राम:", "कख", "ক\u09cb"]
    source = "/".join(rng.choice(vocab) for _ in range(rng.randint(1, 30)))
    b = OffsetMapBuilder(source)
    for ch in source:
        if ch == "/" or rng.random() < 0.8:
            b.copy()
        else:  # substitute, delete or expand this char
            b.replace(1, rng.choice(["", "ः", "\u093c", "xy", "र"]))
    target = b.build().target
    _assert_wordwise_provenance(source, align(source, target), sep="/")


def test_align_keeps_pure_substitutions_exact():
    m = align("आयु २१ वर्ष", "आयु 21 वर्ष")
    assert m.to_source(4, 5) == (4, 5) and m.to_source(5, 6) == (5, 6)


def test_align_identity_and_extremes():
    assert align("राम", "राम").lo == (0, 1, 2)
    assert align("", "").target == ""
    assert align("राम", "").to_target(0, 3) == (0, 0)
    assert align("", "राम").to_source(0, 3) == (0, 0)
