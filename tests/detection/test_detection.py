"""T24-T27, T33-T34: decomposition, alignment, aggregation, B1, tuning, worked example."""

import numpy as np
import pytest

from indichallu.data.schema import Record, SpanType, Verdict
from indichallu.detection.aggregate import Thresholds, decide, guess_type
from indichallu.detection.align import Aligner, premise_windows
from indichallu.detection.decompose import decompose
from indichallu.detection.detector import Detector, b1_claims, load_detection_config
from indichallu.detection.tune import redecide, tune_thresholds
from indichallu.text.sentseg import tokens

PASSAGE = "…योजना के लाभ राज्य सरकार द्वारा दिए जाते हैं।"
ANSWER = "यह योजना राज्य सरकार द्वारा चलाई जाती है और आवेदक की आयु 21 वर्ष होनी चाहिए।"
CLAIM_1 = "योजना राज्य सरकार द्वारा चलाई जाती है"
CLAIM_2 = "आवेदक की आयु 21 वर्ष होनी चाहिए"


class OverlapNLI:
    """Deterministic stand-in for CI: entailment = share of hypothesis tokens found in the
    premise; contradiction when the hypothesis has a number the premise contradicts."""

    def __init__(self):
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        out = []
        for premise, hyp in pairs:
            h = [t.text for t in tokens(hyp)]
            p = {t.text for t in tokens(premise)}
            e = sum(t in p for t in h) / max(len(h), 1)
            nums_h = {t for t in h if t.isdigit()}
            nums_p = {t for t in p if t.isdigit()}
            c = 0.8 if nums_h and nums_p and not nums_h & nums_p else 0.0
            e = min(e, 1 - c)
            out.append([e, 1 - e - c, c])
        return np.array(out)


def record(answer=ANSWER, passages=(PASSAGE,), split="test", labels=None, rid=1):
    return Record.model_validate(
        {
            "id": f"hi-q{rid:06d}",
            "lang": "hi",
            "split": split,
            "source": "injected",
            "question": "इस योजना के लिए पात्रता क्या है?",
            "retrieved": [
                {
                    "passage_id": f"hi-p{i}",
                    "text": t,
                    "score": 0.9 - i / 10,
                    "rank": i + 1,
                    "is_gold": i == 0,
                }
                for i, t in enumerate(passages)
            ],
            "answer": answer,
            "labels": labels,
        }
    )


def detector(**kw):
    return Detector(OverlapNLI(), "hi", Thresholds(0.5, 0.5), **kw)


# --- decomposition (T24) ---------------------------------------------------------------


def test_worked_example_decomposes_into_the_deck_claims():
    claims = decompose(ANSWER, "hi")
    assert [c.text for c in claims] == [CLAIM_1, CLAIM_2]
    for c in claims:
        assert ANSWER[c.start : c.end] == c.text  # offsets into the original answer


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("राम और श्याम स्कूल गए।", ["राम और श्याम स्कूल गए"]),  # short noun phrase stays whole
        ("वह आया। वह गया।", ["वह आया", "वह गया"]),  # short: demonstrative kept
        ("मौसम अच्छा था; हम घूमने गए।", ["मौसम अच्छा था", "हम घूमने गए"]),
        ("दिए गए अनुच्छेदों में इस प्रश्न का उत्तर नहीं है।", ["दिए गए अनुच्छेदों में इस प्रश्न का उत्तर नहीं है"]),
        ("", []),
    ],
)
def test_decomposition_rules(answer, expected):
    assert [c.text for c in decompose(answer, "hi")] == expected


def test_unknown_language_falls_back_to_sentences():
    text = "நான் வந்தேன் மற்றும் அவர் போனார். சரி."
    assert [c.text for c in decompose(text, "ta")] == ["நான் வந்தேன் மற்றும் அவர் போனார்", "சரி"]


# --- alignment (T25) -------------------------------------------------------------------


def test_every_claim_is_scored_against_every_passage():
    rec = record(passages=(PASSAGE, "हिमालय उत्तर में है।", "क्रिकेट खेल है।"))
    out = Aligner(OverlapNLI(), "hi").score([CLAIM_1, CLAIM_2], rec.retrieved)
    assert len(out) == 2 and all(len(row) == 3 for row in out)
    assert [s.passage_id for s in out[0]] == ["hi-p0", "hi-p1", "hi-p2"]
    for row in out:
        for s in row:
            assert s.entail + s.neutral + s.contradict == pytest.approx(1.0)


def test_long_passages_are_windowed_and_digits_normalised():
    sents = [f"वाक्य संख्या {i} यहाँ है।" for i in range(10)] + ["आवेदक की आयु २१ वर्ष होनी चाहिए।"]
    long_passage = " ".join(sents)
    assert len(premise_windows(long_passage, "hi", 3)) == len(sents) - 2
    rec = record(passages=(long_passage,))
    (row,) = Aligner(OverlapNLI(), "hi").score([CLAIM_2], rec.retrieved)
    assert row[0].entail == pytest.approx(1.0)  # found in the last window, २१ == 21


# --- aggregation (T26) -----------------------------------------------------------------


def ps(pid, e, c):
    from indichallu.data.schema import PassageScores

    return PassageScores(passage_id=pid, entail=e, neutral=1 - e - c, contradict=c)


@pytest.mark.parametrize(
    ("scores", "verdict", "best"),
    [
        ([ps("a", 0.2, 0.1), ps("b", 0.7, 0.0)], Verdict.SUPPORTED, "b"),
        ([ps("a", 0.2, 0.6), ps("b", 0.3, 0.0)], Verdict.CONTRADICTED, "a"),
        ([ps("a", 0.2, 0.1), ps("b", 0.3, 0.2)], Verdict.UNSUPPORTED, "b"),
        ([ps("a", 0.6, 0.0), ps("b", 0.0, 0.9)], Verdict.SUPPORTED, "a"),  # support wins
    ],
)
def test_decision_rule(scores, verdict, best):
    c = decide("x", 0, 1, scores, Thresholds(0.5, 0.5))
    assert c.verdict is verdict and c.best_passage_id == best
    assert len(c.passage_scores) == 2


def test_guess_type():
    assert guess_type("आयु 21 वर्ष") is SpanType.NUMERICAL
    assert guess_type("आयु २१ वर्ष") is SpanType.NUMERICAL
    assert guess_type("1947 में स्वतंत्रता") is SpanType.TEMPORAL
    assert guess_type("15 अगस्त को") is SpanType.TEMPORAL
    assert guess_type("राम दिल्ली गए") is None
    with pytest.raises(ValueError):
        Thresholds(1.5, 0.5)


# --- the detector, B1 and the worked example (T27, T33) --------------------------------


def test_worked_example_regression():
    """Deck slide 17: claim 1 SUPPORTED, claim 2 UNSUPPORTED (NUMERICAL), offsets on the
    age clause. CI version with the deterministic stand-in NLI; the real model is the
    slow test below."""
    out = detector().detect(record())
    c1, c2 = out.claims
    assert (c1.text, c1.verdict, c1.best_passage_id) == (CLAIM_1, Verdict.SUPPORTED, "hi-p0")
    assert (c2.text, c2.verdict, c2.type) == (CLAIM_2, Verdict.UNSUPPORTED, SpanType.NUMERICAL)
    assert ANSWER[c2.start : c2.end] == CLAIM_2


def test_b1_uses_whole_sentences():
    assert [c.text for c in b1_claims(ANSWER, "hi")] == [ANSWER[:-1]]
    out = detector(claim_fn=b1_claims).detect(record())
    assert len(out.claims) == 1  # one sentence-level verdict, no localisation


def test_detector_uses_only_retrieved_evidence():
    no_evidence = record(passages=("क्रिकेट भारत में लोकप्रिय खेल है।",))
    assert all(c.verdict is not Verdict.SUPPORTED for c in detector().detect(no_evidence).claims)
    with pytest.raises(ValueError):
        Detector(OverlapNLI(), "mr", Thresholds()).detect(record())


def test_m0_detection_config():
    cfg = load_detection_config("m0")
    assert len(cfg["nli"]["revision"]) == 40 and cfg["nli"]["licence"] == "MIT"
    assert set(cfg["thresholds"]) == {"entail", "contradict"}


# --- threshold tuning (T34) ------------------------------------------------------------


def _labelled(split, rid, hallucinated):
    span = {
        "start": ANSWER.index(CLAIM_2),
        "end": ANSWER.index(CLAIM_2) + len(CLAIM_2),
        "type": "NUMERICAL",
    }
    labels = (
        {"response": "hallucinated", "spans": [span]}
        if hallucinated
        else {"response": "faithful", "spans": []}
    )
    answer = ANSWER if hallucinated else CLAIM_1 + "।"
    return detector().detect(record(answer=answer, split=split, labels=labels, rid=rid))


def test_tuning_refuses_anything_but_val():
    with pytest.raises(ValueError, match="val split only"):
        tune_thresholds([_labelled("test", 1, True)])
    with pytest.raises(ValueError, match="val split only"):
        tune_thresholds([_labelled("val", 1, True), _labelled("train", 2, False)])


def test_tuning_finds_thresholds_that_separate_the_classes():
    val = [_labelled("val", i, i % 2 == 0) for i in range(6)]
    th, table = tune_thresholds(val, entail_grid=(0.3, 0.5, 0.9), contradict_grid=(0.5,))
    assert len(table) == 3
    best = max(table, key=lambda r: r["span_f1"] or 0)
    assert best["span_f1"] == 1.0 and th.entail in (0.3, 0.5)
    # re-deciding with the tuned thresholds reproduces those verdicts without new NLI calls
    assert all(redecide(r, th).claims for r in val)


@pytest.mark.slow
def test_worked_example_with_the_real_nli_model():
    """Slide 17 with the real M0 aligner (mDeBERTa, pinned). If the real model disagrees
    with the deck, that is reported as a finding (DECISIONS / PROGRESS), never patched."""
    cfg = load_detection_config("m0")
    out = Detector.from_config(cfg, "hi").detect(record())
    c1, c2 = out.claims
    print({c.text: (c.verdict.value, c.scores.model_dump(), c.type) for c in out.claims})
    assert c1.verdict is Verdict.SUPPORTED
    assert c2.verdict is Verdict.UNSUPPORTED and c2.type is SpanType.NUMERICAL
