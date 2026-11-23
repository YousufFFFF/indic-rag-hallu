"""T45 / D-029: refusals are recognised, never scored as claims, and reported separately."""

import numpy as np
import pytest
from pydantic import ValidationError

from indichallu.annotate.spotcheck import spotcheck_markdown
from indichallu.data.schema import Record, RetrievedPassage
from indichallu.detection.aggregate import Thresholds
from indichallu.detection.detector import Detector
from indichallu.detection.refusal import is_refusal_sentence, refusal_sentences
from indichallu.eval.detection import response_level_metrics
from indichallu.eval.refusals import answer_available, answered, refusal_metrics

# Real Qwen3-4B answers from the T36 spot-check run (20261121-112642-pipeline-cf50a16a).
REFUSALS = [
    "दिए गए अनुच्छेदों में इस प्रश्न का उत्तर नहीं है।",
    "दिए गए अनुच्छेदों में तिब्बत विज्ञान संस्थान के बारे में कोई जानकारी नहीं है। इसलिए इस प्रश्न का उत्तर नहीं है।",
    "दिए गए अनुच्छेदों में भगवान विष्णु की तेली मंदिर का निर्माण किस शासक ने कराया था "
    "इसका कोई उल्लेख नहीं है।\n\nदिए गए अनुच्छेदों में इस प्रश्न का उत्तर नहीं है।",
]
ANSWERS = [
    "वेद चार प्रकार के होते हैं।",
    "यज्ञ की विधियों का वर्णन यजुर्वेद में किया गया है।",
    "ऋग्वेद में महिलाओं के नाम का उच्चारण पाए जाते हैं।",
]


class CountingNLI:
    def __init__(self):
        self.pairs = 0

    def predict(self, pairs):
        self.pairs += len(pairs)
        return np.tile([0.9, 0.08, 0.02], (len(pairs), 1))


def rec(answer, gold_shown=True, answerable=True, rid="q1", **extra):
    p = RetrievedPassage(
        passage_id="hi-p1", text="वेद चार हैं।", score=0.9, rank=1, is_gold=gold_shown
    )
    return Record(
        id=f"hi-{rid}",
        lang="hi",
        split="val",
        source="injected",
        question="प्रश्न?",
        retrieved=[p],
        answer=answer,
        answerable=answerable,
        **extra,
    )


def detector(nli, refusals="separate"):
    return Detector(nli, "hi", Thresholds(0.5, 0.5), refusals=refusals)


@pytest.mark.parametrize("answer", REFUSALS)
def test_real_refusals_are_detected_and_get_no_claims(answer):
    nli = CountingNLI()
    out = detector(nli).detect(rec(answer))
    assert out.refusal is True and out.claims == [] and nli.pairs == 0


@pytest.mark.parametrize("answer", ANSWERS)
def test_real_answers_are_not_refusals(answer):
    out = detector(CountingNLI()).detect(rec(answer))
    assert out.refusal is False and out.claims


def test_a_constitutional_article_is_not_a_reference_to_the_passages():
    # "अनुच्छेद" also means a constitution's article; only "the given passages" counts.
    assert not is_refusal_sentence("संविधान के अनुच्छेद 370 में इसका उल्लेख नहीं है।", "hi")


def test_a_partial_refusal_keeps_the_answer_and_drops_the_refusal_sentence():
    answer = "वेद चार प्रकार के होते हैं। दिए गए अनुच्छेदों में उनके रचयिता का उल्लेख नहीं है।"
    out = detector(CountingNLI()).detect(rec(answer))
    assert out.refusal is False
    assert [c.text for c in out.claims] == ["वेद चार प्रकार के होते हैं"]


def test_the_m0_mode_is_unchanged():
    out = detector(CountingNLI(), refusals="claims").detect(rec(REFUSALS[0]))
    assert out.refusal is None and len(out.claims) == 1  # scored like any answer, as in M0


def test_languages_without_patterns_never_detect_refusals():
    assert refusal_sentences("any text at all.", "ta") == []
    with pytest.raises(ValueError, match="refusals must be"):
        Detector(CountingNLI(), "hi", Thresholds(0.5, 0.5), refusals="drop")


def test_schema_keeps_refusals_consistent():
    with pytest.raises(ValidationError, match="refusal has no claims"):
        Record.model_validate(
            {
                **detector(CountingNLI()).detect(rec(ANSWERS[0])).model_dump(),
                "refusal": True,
            }
        )
    with pytest.raises(ValidationError, match="refusal cannot have"):
        rec(
            REFUSALS[0],
            labels={"response": "refusal", "spans": [{"start": 0, "end": 3, "type": "NUMERICAL"}]},
        )


def test_correct_refusal_means_the_answer_was_not_available():
    assert answer_available(rec(ANSWERS[0]))
    assert not answer_available(rec(ANSWERS[0], gold_shown=False))
    assert not answer_available(rec(ANSWERS[0], answerable=False))  # gold shown, no answer
    assert answer_available(rec(ANSWERS[0], answerable=None))  # unknown: assume available


def test_refusal_metrics_and_detection_metrics_never_mix():
    d = detector(CountingNLI())
    faithful = {"response": "faithful", "spans": []}
    records = [
        d.detect(rec(REFUSALS[0], gold_shown=False, rid="a", labels=faithful)),  # correct
        d.detect(rec(REFUSALS[0], gold_shown=True, rid="b", labels=faithful)),  # incorrect
        d.detect(rec(ANSWERS[0], gold_shown=False, rid="c", labels=faithful)),  # should refuse
        d.detect(rec(ANSWERS[0], gold_shown=True, rid="d", labels=faithful)),
    ]
    m = refusal_metrics(records)
    assert (m["n_refused"], m["n_correct_refusals"], m["n_incorrect_refusals"]) == (2, 1, 1)
    assert m["refusal_rate"] == 0.5 and m["refusal_precision"] == 0.5
    assert m["n_answer_unavailable"] == 2 and m["refusal_recall"] == 0.5
    with pytest.raises(ValueError, match="refusals among the records"):
        response_level_metrics(records)
    assert [r.id for r in answered(records)] == ["hi-c", "hi-d"]
    assert response_level_metrics(answered(records))["n"] == 2


def test_unchecked_records_cannot_get_refusal_metrics():
    out = detector(CountingNLI(), refusals="claims").detect(rec(ANSWERS[0]))
    with pytest.raises(ValueError, match="not checked for refusals"):
        refusal_metrics([out])


def test_the_spotcheck_sheet_says_refusal_and_whether_it_was_right():
    d = detector(CountingNLI())
    sheet = spotcheck_markdown(
        [d.detect(rec(REFUSALS[0], gold_shown=False)), d.detect(rec(REFUSALS[0], rid="q2"))]
    )
    assert "Detector: **REFUSAL** (correct); no claims scored." in sheet
    assert "Detector: **REFUSAL** (incorrect: the answer was available)" in sheet
