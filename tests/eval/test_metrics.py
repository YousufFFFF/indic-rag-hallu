"""T16-T18, T20: retrieval, detection, span and per-language metrics (hand-computed)."""

import math

import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

from indichallu.data.schema import Record
from indichallu.eval.detection import binary_metrics, claim_score, response_level_metrics
from indichallu.eval.fairness import by_language
from indichallu.eval.retrieval import (
    Query,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieval_metrics,
    retrieval_metrics_from_records,
)
from indichallu.eval.spans import span_metrics, span_metrics_by_type

# --- retrieval (T16) -------------------------------------------------------------------


def q(*flags, n=1):
    return Query(tuple(flags), n)


def test_single_gold_metrics_by_hand():
    at3 = q(False, False, True, False, False)
    assert recall_at_k(at3, 1) == 0 and recall_at_k(at3, 3) == 1
    assert ndcg_at_k(at3, 5) == pytest.approx(1 / math.log2(4))  # 0.5
    assert reciprocal_rank(at3) == pytest.approx(1 / 3)
    missed = q(False, False, False)
    assert (
        recall_at_k(missed, 3) == 0 and ndcg_at_k(missed, 3) == 0 and reciprocal_rank(missed) == 0
    )


def test_multi_gold_ndcg_by_hand():
    two = q(True, False, True, n=2)
    dcg = 1 + 1 / math.log2(4)
    idcg = 1 + 1 / math.log2(3)
    assert ndcg_at_k(two, 3) == pytest.approx(dcg / idcg)
    assert recall_at_k(two, 2) == 0.5


def test_aggregate_and_guards():
    m = retrieval_metrics([q(True, False), q(False, True), q(False, False)], ks=(1, 2))
    assert m["n_queries"] == 3 and m["n_excluded"] == 0
    assert m["recall@1"] == pytest.approx(1 / 3) and m["recall@2"] == pytest.approx(2 / 3)
    assert m["mrr"] == pytest.approx((1 + 0.5 + 0) / 3)
    with pytest.raises(ValueError, match="retrieve more"):
        retrieval_metrics([q(True)], ks=(1, 5))
    assert retrieval_metrics([q(True, False)], ks=(5,), corpus_size=2)["recall@5"] == 1.0
    assert retrieval_metrics([q(True, n=0)], ks=(1,))["n_excluded"] == 1
    assert retrieval_metrics([], ks=(1,))["mrr"] is None


def _record(lang="hi", gold_rank=None, k=3, answer="उत्तर", claims=(), labels=None, rid=1):
    return Record.model_validate(
        {
            "id": f"{lang}-q{rid:06d}",
            "lang": lang,
            "split": "test",
            "source": "injected",
            "question": "प्रश्न?",
            "retrieved": [
                {
                    "passage_id": f"p{i}",
                    "text": "t",
                    "score": 1 - i / 10,
                    "rank": i,
                    "is_gold": i == gold_rank,
                }
                for i in range(1, k + 1)
            ],
            "answer": answer,
            "claims": list(claims),
            "labels": labels,
        }
    )


def test_from_records_uses_is_gold_and_rank():
    recs = [
        _record(gold_rank=1, rid=1),
        _record(gold_rank=3, rid=2),
        _record(gold_rank=None, rid=3),
    ]
    m = retrieval_metrics_from_records(recs, ks=(1, 3))
    assert m["recall@1"] == pytest.approx(1 / 3) and m["recall@3"] == pytest.approx(2 / 3)
    assert m["mrr"] == pytest.approx((1 + 1 / 3) / 3)


# --- detection (T17) -------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_binary_metrics_agree_with_sklearn(seed):
    import random

    rng = random.Random(seed)
    y = [rng.random() < 0.3 for _ in range(60)]
    p = [rng.random() < 0.4 for _ in range(60)]
    if not any(y) or all(y) or not any(p):
        pytest.skip("degenerate draw")
    m = binary_metrics(y, p)
    assert m["precision"] == pytest.approx(precision_score(y, p))
    assert m["recall"] == pytest.approx(recall_score(y, p))
    assert m["f1"] == pytest.approx(f1_score(y, p))
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, p))


def test_binary_metrics_by_hand_and_undefined_cases():
    m = binary_metrics([True, True, False, False], [True, False, True, False], [0.9, 0.4, 0.6, 0.1])
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == m["recall"] == m["f1"] == m["balanced_accuracy"] == 0.5
    assert m["auroc"] == pytest.approx(0.75)  # 3 of 4 positive/negative pairs ordered right
    none = binary_metrics([False, False], [False, False], [0.1, 0.2])
    assert none["precision"] is None and none["recall"] is None and none["f1"] is None
    assert none["auroc"] is None and none["balanced_accuracy"] is None
    missed = binary_metrics([True, False], [False, False])
    assert missed["recall"] == 0.0 and missed["precision"] is None and missed["f1"] == 0.0
    with pytest.raises(ValueError):
        binary_metrics([True], [True, False])


ANSWER = "यह योजना राज्य सरकार द्वारा चलाई जाती है और आवेदक की आयु 21 वर्ष होनी चाहिए।"
C1 = "योजना राज्य सरकार द्वारा चलाई जाती है"
C2 = "आवेदक की आयु 21 वर्ष होनी चाहिए"


def span(text, sub):
    s = text.index(sub)
    return s, s + len(sub)


def claim(sub, verdict, entail=None, typ=None, best="p1"):
    s, e = span(ANSWER, sub)
    c = {
        "text": sub,
        "start": s,
        "end": e,
        "verdict": verdict,
        "best_passage_id": best,
        "type": typ,
    }
    if verdict == "UNSUPPORTED":
        c["best_passage_id"] = None
    if entail is not None:
        c["scores"] = {"entail": entail, "neutral": 1 - entail, "contradict": 0.0}
    return c


def labelled(claims, gold_sub=None, gold_type="NUMERICAL", rid=1, lang="hi"):
    labels = {"response": "faithful", "spans": []}
    if gold_sub:
        s, e = span(ANSWER, gold_sub)
        labels = {"response": "hallucinated", "spans": [{"start": s, "end": e, "type": gold_type}]}
    return _record(lang=lang, gold_rank=1, answer=ANSWER, claims=claims, labels=labels, rid=rid)


def test_worked_example_is_a_perfect_detection():
    rec = labelled(
        [claim(C1, "SUPPORTED", 0.9), claim(C2, "UNSUPPORTED", 0.1, "NUMERICAL")], gold_sub=C2
    )
    m = response_level_metrics([rec, labelled([claim(C1, "SUPPORTED", 0.95)], rid=2)])
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (1, 1, 0, 0)
    assert m["f1"] == 1.0 and m["auroc"] == 1.0
    s = span_metrics([rec])
    assert s["span_f1"] == 1.0 and s["token_iou"] == 1.0 and s["exact_f1"] == 1.0


def test_claim_score():
    from indichallu.data.schema import Claim

    assert claim_score(Claim.model_validate(claim(C1, "SUPPORTED", 0.8))) == pytest.approx(0.2)
    assert claim_score(Claim.model_validate(claim(C2, "UNSUPPORTED"))) == 1.0


def test_unlabelled_records_are_refused():
    with pytest.raises(ValueError, match="no gold labels"):
        response_level_metrics([_record()])


# --- spans (T18) -----------------------------------------------------------------------


def test_partial_overlap_by_hand():
    """Gold = the age clause; predicted = only '21 वर्ष' inside it."""
    gs, ge = span(ANSWER, C2)
    ps, pe = span(ANSWER, "21 वर्ष")
    pred = {"text": "21 वर्ष", "start": ps, "end": pe, "verdict": "UNSUPPORTED", "type": "NUMERICAL"}
    rec = labelled([pred], gold_sub=C2)
    s = span_metrics([rec])
    assert s["span_precision"] == 1.0
    assert s["span_recall"] == pytest.approx((pe - ps) / (ge - gs))
    assert s["token_iou"] == pytest.approx(2 / 7)  # 2 of the clause's 7 tokens
    assert s["exact_f1"] == 0.0  # not the same span


def test_per_type_breakdown():
    rec = labelled([claim(C2, "UNSUPPORTED", typ="ENTITY")], gold_sub=C2, gold_type="NUMERICAL")
    by = span_metrics_by_type([rec])
    assert set(by) == {"ALL", "ENTITY", "NUMERICAL", "TEMPORAL", "LOCATION"}
    assert by["ALL"]["span_f1"] == 1.0  # located correctly...
    assert by["NUMERICAL"]["span_recall"] == 0.0  # ...but typed wrongly
    assert by["ENTITY"]["span_precision"] == 0.0
    assert by["TEMPORAL"]["span_f1"] is None  # no gold, no predictions: undefined


def test_contradicted_claims_count_as_hallucinated_spans():
    rec = labelled([claim(C2, "CONTRADICTED", typ="NUMERICAL")], gold_sub=C2)
    assert span_metrics([rec])["exact_f1"] == 1.0


# --- fairness (T20) --------------------------------------------------------------------


def test_by_language_pooled_macro_worst():
    hi = [
        labelled([claim(C2, "UNSUPPORTED", 0.1, "NUMERICAL")], gold_sub=C2, rid=i) for i in (1, 2)
    ]
    mr = [
        labelled([claim(C1, "SUPPORTED", 0.9)], gold_sub=C2, rid=3, lang="mr"),  # missed
        labelled([claim(C2, "UNSUPPORTED", 0.2, "NUMERICAL")], gold_sub=C2, rid=4, lang="mr"),
    ]
    out = by_language(hi + mr, lambda rs: span_metrics(rs))
    assert set(out["per_language"]) == {"hi", "mr"}
    assert out["per_language"]["hi"]["exact_recall"] == 1.0
    assert out["per_language"]["mr"]["exact_recall"] == 0.5
    assert out["pooled"]["exact_recall"] == 0.75
    assert out["macro"]["exact_recall"] == 0.75 and out["macro_n"]["exact_recall"] == 2
    assert out["worst"]["exact_recall"] == {"lang": "mr", "value": 0.5}
    assert "n_records" not in out["macro"] and "gold_spans" not in out["macro"]


def test_by_language_with_one_language_and_undefined_values():
    out = by_language([labelled([], rid=1)], lambda rs: span_metrics(rs))
    assert list(out["per_language"]) == ["hi"]
    assert out["macro"]["span_f1"] is None and out["macro_n"]["span_f1"] == 0


def test_duplicate_predicted_spans_are_counted():
    """Codex review on #21: two predictions with the same offsets are two predictions."""
    c = claim(C2, "UNSUPPORTED", typ="NUMERICAL")
    rec = labelled([c, c], gold_sub=C2)
    s = span_metrics([rec])
    assert s["predicted_spans"] == 2 and s["gold_spans"] == 1
    assert s["exact_precision"] == 0.5 and s["exact_recall"] == 1.0
    assert s["exact_f1"] == pytest.approx(2 / 3)
