"""T11: IndicQA loader (tiny fixture; plus the real file when it has been downloaded)."""

import json
from pathlib import Path

import pytest

from indichallu.data.download import DEFAULT_DATA_ROOT, load_pinned_dataset
from indichallu.data.loaders.indicqa import (
    MAX_SHIFT,
    load_indicqa,
    passage_id,
    realign_answer,
)

CTX = "योजना के लाभ राज्य सरकार द्वारा दिए जाते हैं। आवेदक की आयु 21 वर्ष होनी चाहिए।"
CTX2 = "राम आया। राम गया। श्याम रुका।"


def _qa(qid, question, text, start, category="SHORT"):
    return {
        "id": qid,
        "question": question,
        "category": category,
        "answers": [{"text": text, "answer_start": start}],
    }


def _write(tmp_path, data):
    p = tmp_path / "indicqa.hi.json"
    p.write_text(json.dumps({"version": "1.0", "data": data}, ensure_ascii=False), "utf-8")
    return p


@pytest.fixture
def fixture_file(tmp_path):
    s = CTX.index("राज्य सरकार")
    data = [
        {
            "title": "योजना",
            "paragraphs": [
                {
                    "context": CTX,
                    "qas": [
                        _qa(0, "लाभ कौन देता है?", "राज्य सरकार", s),  # exact
                        _qa(1, "आयु क्या है?", "21 वर्ष", CTX.index("21 वर्ष") - 6),  # shifted
                        _qa(2, "आवेदक कौन है?", "", None, "NO"),  # unanswerable
                        _qa(3, "किसकी आयु?", " आवेदक ", CTX.index("आवेदक") - 1),  # padded
                    ],
                },
                {
                    "context": CTX2,
                    "qas": [
                        _qa(4, "पहले कौन आया?", "राम", 0),  # repeated answer, exact
                        _qa(5, "कौन गया?", "राम", CTX2.index("राम गया") - 3),  # nearest
                    ],
                },
            ],
        },
        {"title": "दोहराव", "paragraphs": [{"context": CTX, "qas": [_qa(6, "फिर?", "हैं", 0)]}]},
    ]
    return _write(tmp_path, data)


def test_passages_are_deduplicated_with_content_ids(fixture_file):
    ds = load_indicqa(fixture_file, "hi")
    assert [p.passage_id for p in ds.passages] == [passage_id("hi", CTX), passage_id("hi", CTX2)]
    assert ds.stats["duplicate_contexts"] == 1
    assert ds.passages[0].title == "योजना"  # first title wins
    assert all(p.passage_id.startswith("hi-p") and len(p.passage_id) == 14 for p in ds.passages)


def test_questions_link_to_their_gold_passage(fixture_file):
    ds = load_indicqa(fixture_file, "hi")
    by_id = {q.qa_id: q for q in ds.questions}
    assert list(by_id) == [f"hi-q{i:06d}" for i in range(7)]
    assert by_id["hi-q000004"].gold_passage_id == passage_id("hi", CTX2)
    assert by_id["hi-q000006"].gold_passage_id == passage_id("hi", CTX)  # duplicate context


@pytest.mark.parametrize(
    ("qa_id", "status", "answer"),
    [
        ("hi-q000000", "exact", "राज्य सरकार"),
        ("hi-q000001", "unique", "21 वर्ष"),
        ("hi-q000002", "unanswerable", None),
        ("hi-q000003", "exact", "आवेदक"),  # trimmed; shift by the leading space
        ("hi-q000004", "exact", "राम"),
        ("hi-q000005", "nearest", "राम"),
    ],
)
def test_answer_offsets_are_verified(fixture_file, qa_id, status, answer):
    ds = load_indicqa(fixture_file, "hi")
    q = next(q for q in ds.questions if q.qa_id == qa_id)
    assert q.answer_status == status and q.answer_text == answer
    assert q.answerable is (answer is not None)
    if q.answer_start is not None:
        assert ds.passage(q.gold_passage_id).text[q.answer_start : q.answer_end] == answer
    if status == "unanswerable":
        assert q.answer_start is None and q.category == "NO"


def test_realign_rules():
    ctx = "क ख क ग" + " " * 30 + "क"
    assert realign_answer(ctx, "ख", 2) == (2, "exact")
    assert realign_answer(ctx, "ग", 0) == (6, "unique")
    assert realign_answer(ctx, "क", 1) == (4, "nearest")  # 4 is 3 after, next is 34 after
    assert realign_answer(ctx, "घ", 0) == (None, "not_found")
    assert realign_answer("कxक", "क", 1) == (2, "nearest")  # the hit before the hint is ignored
    assert realign_answer("क क", "क", None) == (None, "ambiguous")  # no hint
    assert realign_answer("क" + " " * (MAX_SHIFT + 5) + "क", "क", 1)[1] == "ambiguous"
    assert realign_answer("x क कy", "क", 0) == (None, "ambiguous")  # 2 and 4: margin < 10


def test_bad_input_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_indicqa(_write(tmp_path, []), "en")
    p = tmp_path / "x.json"
    p.write_text("[]", "utf-8")
    with pytest.raises(ValueError):
        load_indicqa(p, "hi")
    two = [{"title": "", "paragraphs": [{"context": CTX, "qas": [_qa(1, "?", "", None)] * 2}]}]
    with pytest.raises(ValueError):  # duplicate question id
        load_indicqa(_write(tmp_path, two), "hi")


# --- the real file (runs only where it has been downloaded; CI skips) -----------------


def _real_hindi_file() -> Path:
    path = load_pinned_dataset().local_path("hi", DEFAULT_DATA_ROOT)
    if not path.exists():
        pytest.skip("Hindi IndicQA not downloaded (python scripts/download_indicqa.py)")
    return path


def test_real_hindi_file():
    ds = load_indicqa(_real_hindi_file(), "hi")
    s = ds.stats
    assert (s["passages"], s["questions"], s["unanswerable"]) == (261, 1547, 495)
    assert s["exact"] + s["unique"] + s["nearest"] + s["ambiguous"] + s["not_found"] == 1052
    assert s["not_found"] == 0 and s["ambiguous"] <= 5
    for q in ds.questions:
        assert q.answerable is (q.category == "SHORT")
        if q.answer_start is not None:
            text = ds.passage(q.gold_passage_id).text
            assert text[q.answer_start : q.answer_end] == q.answer_text
