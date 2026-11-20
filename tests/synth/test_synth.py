"""T28-T30: typed error injection and the pilot set."""

import random

import pytest

from indichallu.data.loaders.indicqa import IndicQA, Passage, QAPair
from indichallu.data.schema import RetrievedPassage, SpanType, Split
from indichallu.data.split import SplitConfig, split_dataset
from indichallu.data.synth.inject import inject, inject_numerical, inject_temporal
from indichallu.data.synth.pilot import answer_sentence, build_pilot

R = {Split.TRAIN: 0.6, Split.VAL: 0.2, Split.TEST: 0.2}


@pytest.mark.parametrize("seed", range(30))
def test_numerical_injection_changes_exactly_the_number(seed):
    s = "आवेदक की आयु 21 वर्ष होनी चाहिए।"
    r = inject_numerical(s, random.Random(seed))
    assert r.type is SpanType.NUMERICAL and r.original == "21"
    assert r.text[r.start : r.end] == r.replacement != "21" and int(r.replacement) > 0
    assert r.text[: r.start] == s[: s.index("21")] and r.text[r.end :] == s[s.index("21") + 2 :]


@pytest.mark.parametrize("seed", range(30))
def test_native_digits_stay_native(seed):
    r = inject_numerical("उसकी आयु २१ वर्ष है।", random.Random(seed))
    assert all("०" <= ch <= "९" for ch in r.replacement) and r.replacement != "२१"


@pytest.mark.parametrize("seed", range(30))
def test_temporal_injection_year_or_month(seed):
    rng = random.Random(seed)
    y = inject_temporal("भारत को 1947 में स्वतंत्रता मिली।", rng)
    assert y.type is SpanType.TEMPORAL and y.original == "1947"
    assert 1917 <= int(y.replacement) <= 1977 and y.replacement != "1947"
    m = inject_temporal("यह अगस्त में मनाया जाता है।", rng)
    assert (
        m.original == "अगस्त"
        and m.replacement != "अगस्त"
        and m.text[m.start : m.end] == m.replacement
    )


def test_years_are_not_numerical_and_unsafe_numbers_are_skipped():
    rng = random.Random(0)
    assert inject_numerical("भारत को 1947 में स्वतंत्रता मिली।", rng) is None
    assert inject_numerical("कुल 1,00,000 रुपये और दर 3.5 है।", rng) is None
    assert inject("कोई संख्या नहीं।", rng) is None
    assert inject("आयु 21 वर्ष, सन 1947।", rng, prefer=SpanType.NUMERICAL).type is SpanType.NUMERICAL


def test_answer_sentence():
    text = "पहला वाक्य है। आवेदक की आयु 21 वर्ष है। तीसरा।"
    s = text.index("21")
    assert answer_sentence(text, "hi", s, s + 2) == "आवेदक की आयु 21 वर्ष है।"
    assert answer_sentence(text, "hi", 0, len(text)) is None  # crosses sentences


def _ds():
    passages, questions = [], []
    for i in range(20):
        pid = f"hi-p{i:010d}"
        text = f"परिचय वाक्य {i} है। योजना {i + 1} वर्ष पुरानी है।"
        passages.append(Passage(pid, "hi", f"लेख {i}", text))
        ans = str(i + 1)
        start = text.index(f"योजना {ans}") + len("योजना ")
        questions.append(
            QAPair(
                f"hi-q{i:06d}",
                "hi",
                f"योजना {i} कितनी पुरानी है?",
                pid,
                "SHORT",
                True,
                "unique",
                ans,
                start,
                start + len(ans),
                start,
            )
        )
    return IndicQA("hi", passages, questions)


def test_pilot_records_are_valid_balanced_and_split_by_passage():
    ds = _ds()
    split = split_dataset(ds, SplitConfig(13, R))
    retrieved = {}
    for q in ds.questions:
        retrieved[q.qa_id] = [
            RetrievedPassage(
                passage_id=q.gold_passage_id,
                text=ds.passage(q.gold_passage_id).text,
                score=0.9,
                rank=1,
                is_gold=True,
            )
        ]
    records, stats = build_pilot(ds, split, retrieved)
    used = [r for r in records if r.split in (Split.VAL, Split.TEST)]
    assert len(used) == len(records) == 2 * (stats["used (val)"] + stats["used (test)"])
    for f, h in zip(records[::2], records[1::2], strict=True):
        assert f.id.endswith("-f") and h.id.endswith("-h") and f.id[:-2] == h.id[:-2]
        assert f.labels.response.value == "faithful" and h.labels.response.value == "hallucinated"
        (span,) = h.labels.spans
        assert h.answer[: span.start] == f.answer[: span.start]  # one local edit
        assert f.answer in ds.passage(f.retrieved[0].passage_id).text  # verbatim from the passage
        assert split.of_passage(f.retrieved[0].passage_id) == f.split
    assert not any(r.split is Split.TRAIN for r in records)


def test_pilot_skips_questions_whose_gold_passage_was_not_retrieved():
    ds = _ds()
    split = split_dataset(ds, SplitConfig(13, R))
    other = ds.passages[0]
    retrieved = {
        q.qa_id: [
            RetrievedPassage(
                passage_id=other.passage_id,
                text=other.text,
                score=0.5,
                rank=1,
                is_gold=other.passage_id == q.gold_passage_id,
            )
        ]
        for q in ds.questions
    }
    records, stats = build_pilot(ds, split, retrieved)
    assert stats["skipped: gold passage not retrieved"] > 0
    assert all(any(p.is_gold for p in r.retrieved) for r in records)
