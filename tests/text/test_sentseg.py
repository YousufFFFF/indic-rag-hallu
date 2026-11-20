"""T09: sentence segmentation and tokens with offsets into the original text."""

import random

import pytest
from indicnlp.tokenize.sentence_tokenize import sentence_split

from indichallu.languages import load_languages
from indichallu.text.normalize import normalize
from indichallu.text.sentseg import Segment, sentences, tokens

LANGS = load_languages()
ZWNJ = "\u200c"


def _nonspace(s: str) -> str:
    return "".join(c for c in s if not c.isspace())


def _check_segments(text: str, segs: list[Segment]) -> None:
    prev_end = 0
    for s in segs:
        assert text[s.start : s.end] == s.text
        assert s.text and s.text == s.text.strip()
        assert s.start >= prev_end  # ordered, non-overlapping
        prev_end = s.end


# --- sentences -------------------------------------------------------------------------


def test_worked_example_two_sentences():
    text = "यह योजना राज्य सरकार द्वारा चलाई जाती है। आवेदक की आयु 21 वर्ष होनी चाहिए।"
    segs = sentences(text, "hi")
    _check_segments(text, segs)
    assert [s.text for s in segs] == [
        "यह योजना राज्य सरकार द्वारा चलाई जाती है।",
        "आवेदक की आयु 21 वर्ष होनी चाहिए।",
    ]


def test_pipe_typed_for_danda_still_splits_and_spans_are_original():
    text = f"पहला वाक्य{ZWNJ} है| दूसरा वाक्य है|"
    segs = sentences(text, "hi")
    _check_segments(text, segs)
    assert [s.text for s in segs] == [f"पहला वाक्य{ZWNJ} है|", "दूसरा वाक्य है|"]


def _texts(text: str, lang: str = "hi") -> list[str]:
    segs = sentences(text, lang)
    _check_segments(text, segs)
    return [s.text for s in segs]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # "." is a delimiter only when the text has no danda (hi, bn, pa)
        ("वह आया. फिर गया.", ["वह आया.", "फिर गया."]),
        ("वह आया. फिर गया।", ["वह आया. फिर गया।"]),
        # one-syllable words commonly end sentences: they are NOT treated as initials
        ("वह यहाँ है. फिर में.", ["वह यहाँ है.", "फिर में."]),
        # abbreviations from IndicNLP's list and bare-letter initials don't end a sentence
        ("डॉ. शर्मा आए. वे खुश थे.", ["डॉ. शर्मा आए.", "वे खुश थे."]),
        ("श्री. क. ख. वर्मा आए.", ["श्री. क. ख. वर्मा आए."]),
        # numbers: ordinals and decimals don't split; other delimiters after digits DO
        ("आयु 21. वर्ष है।", ["आयु 21. वर्ष है।"]),
        ("दर 3.5 है. ठीक.", ["दर 3.5 है.", "ठीक."]),
        ("उसकी आयु 21। अब वह बड़ा है।", ["उसकी आयु 21।", "अब वह बड़ा है।"]),
        # runs of delimiters and closing quotes/brackets stay with their sentence
        ('क्या?! "हाँ।" (ठीक।) अंत।', ["क्या?!", '"हाँ।"', "(ठीक।)", "अंत।"]),
        # original whitespace (newlines) is kept inside spans, trimmed at the ends
        ("राम आया.\n\nश्याम\nगया.", ["राम आया.", "श्याम\nगया."]),
        ("  अंत  ", ["अंत"]),
    ],
)
def test_sentence_rules(text, expected):
    assert _texts(text) == expected


def test_regression_indicnlp_drops_text():
    """IndicNLP's sentence_split keeps only 6 of these 19 non-space chars; we keep all."""
    text = "आया\n\n!\tडॉ.\n\n21.\n\nडॉ.\t21.\n\n डॉ. "
    lib = sentence_split(normalize(text, "hi").text, "hi")
    assert len(_nonspace("".join(lib))) < len(_nonspace(text))  # the library bug
    assert _nonspace("".join(_texts(text))) == _nonspace(text)


def test_empty_and_whitespace_only_text():
    assert sentences("", "hi") == []
    assert sentences("  \n ", "hi") == []


@pytest.mark.parametrize("code", list(LANGS))
def test_every_language_splits_two_sentences(code):
    name = LANGS[code].native_name
    text = f"{name} {name}. {name} {name}?"
    segs = sentences(text, code)
    _check_segments(text, segs)
    assert [s.text for s in segs] == [f"{name} {name}.", f"{name} {name}?"]


SENT_POOL = ["राम", "श्याम", "डॉ.", "21.", "आया", "२१", "|", "।", ".", "?", "!", ZWNJ, "क़", '"']
WS_POOL = [" ", " ", " ", "\n", "  ", "\n\n", "\t"]


@pytest.mark.parametrize("seed", range(200))
def test_sentences_never_lose_text(seed):
    """Sentences are ordered, non-overlapping, trimmed, and together contain every
    non-space character of the original, in order."""
    rng = random.Random(seed)
    text = "".join(rng.choice(SENT_POOL) + rng.choice(WS_POOL) for _ in range(rng.randint(0, 25)))
    if not normalize(text, "hi").text.strip():
        assert _texts(text) == []  # nothing but characters normalisation deletes
    else:
        assert _nonspace("".join(_texts(text))) == _nonspace(text)


def test_edge_zwnj_is_kept_inside_the_sentence():
    zwnj = chr(0x200C)
    text = f"राम आया।{zwnj} श्याम गया{zwnj}। {zwnj}"
    assert _texts(text) == [f"राम आया।{zwnj}", f"श्याम गया{zwnj}। {zwnj}"]
    assert _texts(f"  {zwnj}  ") == []


@pytest.mark.parametrize("seed", range(150))
def test_agrees_with_indicnlp_where_the_library_is_lossless(seed):
    """With a danda present (IndicNLP skips its buggy period-merging pass there) and no
    digits or repeated delimiters, our sentences equal the library's."""
    rng = random.Random(seed)
    words = ["राम", "श्याम", "आया", "गया", "क़ानून", "डॉ", "क्षत्रिय", f"क{ZWNJ}ख"]
    parts = []
    for _ in range(rng.randint(1, 12)):
        parts.append(" ".join(rng.choice(words) for _ in range(rng.randint(1, 4))))
        parts.append(rng.choice(["।", "?", "!", "॥"]) + rng.choice(WS_POOL))
    text = "".join(parts) + "अंत।"
    norm = normalize(text, "hi")
    lib = [p.strip() for p in sentence_split(norm.text, "hi") if p.strip()]
    ours = [normalize(t, "hi").text for t in _texts(text)]
    assert ours == lib


# --- tokens ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("आवेदक की आयु 21 वर्ष होनी चाहिए।", ["आवेदक", "की", "आयु", "21", "वर्ष", "होनी", "चाहिए", "।"]),
        ("क्षत्रिय क़ानून", ["क्षत्रिय", "क़ानून"]),  # virama, nukta stay inside words
        (f"क्ष{ZWNJ}ता", [f"क्ष{ZWNJ}ता"]),  # ZWNJ inside a word
        ("रु. 1,00,000.50 तक", ["रु", ".", "1,00,000.50", "तक"]),  # Indian grouping
        ("२,५००.७५ और ২০২৬", ["२,५००.७५", "और", "২০২৬"]),  # native digits too
        ("पंचायत-स्तर, (2026)!", ["पंचायत", "-", "स्तर", ",", "(", "2026", ")", "!"]),
        ("21. वर्ष, 3,", ["21", ".", "वर्ष", ",", "3", ","]),  # trailing separators split
        ("", []),
    ],
)
def test_tokens(text, expected):
    toks = tokens(text)
    assert [t.text for t in toks] == expected
    _check_segments(text, toks)


@pytest.mark.parametrize("code", list(LANGS))
def test_endonym_is_one_token_in_every_language(code):
    name = LANGS[code].native_name
    assert [t.text for t in tokens(f" {name}। ")] == [name, "।"]


def test_window_offsets_are_absolute():
    text = "यह योजना राज्य सरकार द्वारा चलाई जाती है और आवेदक की आयु 21 वर्ष होनी चाहिए।"
    claim = "आवेदक की आयु 21 वर्ष होनी चाहिए"
    s = text.index(claim)
    toks = tokens(text, s, s + len(claim))
    assert toks[0].start == s and toks[-1].end == s + len(claim)
    assert [t.text for t in toks] == claim.split()
    with pytest.raises(ValueError):
        tokens(text, 5, 2)


@pytest.mark.parametrize("seed", range(150))
def test_tokens_partition_the_non_space_characters(seed):
    rng = random.Random(seed)
    pool = list("कखगािीुे्ंः़") + ["क़", ZWNJ] + list("0123२३৪,.।|-!? \n")
    text = "".join(rng.choice(pool) for _ in range(rng.randint(0, 60)))
    toks = tokens(text)
    _check_segments(text, toks)
    covered = [False] * len(text)
    for t in toks:
        assert not any(c.isspace() for c in t.text)
        for k in range(t.start, t.end):
            covered[k] = True
    assert all(covered[i] or text[i].isspace() for i in range(len(text)))
