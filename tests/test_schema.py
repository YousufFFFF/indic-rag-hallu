"""T04: record schema (CLAUDE.md §6) — validation rules and JSONL round-trip."""

import json

import pytest
from pydantic import ValidationError

from indichallu.data.schema import (
    Claim,
    Labels,
    Record,
    RecordParseError,
    ResponseLabel,
    SpanType,
    Verdict,
    read_jsonl,
    record_to_json,
    write_jsonl,
)

# Deck slide 17 worked example. Offsets are computed from the strings, never typed.
PASSAGE = "…योजना के लाभ राज्य सरकार द्वारा दिए जाते हैं।"
ANSWER = "यह योजना राज्य सरकार द्वारा चलाई जाती है और आवेदक की आयु 21 वर्ष होनी चाहिए।"
CLAIM_1 = "योजना राज्य सरकार द्वारा चलाई जाती है"
CLAIM_2 = "आवेदक की आयु 21 वर्ष होनी चाहिए"


def span_of(sub: str, text: str = ANSWER) -> tuple[int, int]:
    start = text.index(sub)
    return start, start + len(sub)


def make_record(**overrides) -> dict:
    s1, e1 = span_of(CLAIM_1)
    s2, e2 = span_of(CLAIM_2)
    rec = {
        "id": "hi-000001",
        "lang": "hi",
        "split": "test",
        "source": "natural",
        "question": "इस योजना के लिए पात्रता क्या है?",
        "retrieved": [
            {"passage_id": "hi-p1", "text": PASSAGE, "score": 0.71, "rank": 1, "is_gold": True},
            {
                "passage_id": "hi-p2",
                "text": "अन्य अनुच्छेद।",
                "score": 0.40,
                "rank": 2,
                "is_gold": False,
            },
        ],
        "generator": {
            "model": "qwen3:4b",
            "revision": "1cfa9a720891",
            "decoding": {"temperature": 0.0, "max_new_tokens": 256, "seed": 0},
            "backend": "ollama",
        },
        "answer": ANSWER,
        "claims": [
            {
                "text": CLAIM_1,
                "start": s1,
                "end": e1,
                "verdict": "SUPPORTED",
                "best_passage_id": "hi-p1",
                "scores": {"entail": 0.9, "neutral": 0.08, "contradict": 0.02},
                "passage_scores": [
                    {"passage_id": "hi-p1", "entail": 0.9, "neutral": 0.08, "contradict": 0.02},
                    {"passage_id": "hi-p2", "entail": 0.1, "neutral": 0.8, "contradict": 0.1},
                ],
            },
            {
                "text": CLAIM_2,
                "start": s2,
                "end": e2,
                "verdict": "UNSUPPORTED",
                "best_passage_id": None,
                "type": "NUMERICAL",
                "typed_checks": {"numeric": "mismatch"},
            },
        ],
        "labels": {
            "response": "hallucinated",
            "spans": [{"start": s2, "end": e2, "type": "NUMERICAL"}],
        },
    }
    rec.update(overrides)
    return rec


# --- valid records ---------------------------------------------------------------------


def test_worked_example_validates():
    r = Record.model_validate(make_record())
    assert [c.verdict for c in r.claims] == [Verdict.SUPPORTED, Verdict.UNSUPPORTED]
    c2 = r.claims[1]
    assert r.answer[c2.start : c2.end] == CLAIM_2
    assert c2.type is SpanType.NUMERICAL
    assert r.labels.response is ResponseLabel.HALLUCINATED


def test_minimal_record_before_detection():
    """Retrieval + answer only: no claims, no labels yet."""
    r = Record.model_validate(make_record(claims=[], labels=None))
    assert r.claims == [] and r.labels is None


def test_injected_record_needs_no_generator():
    Record.model_validate(make_record(source="injected", generator=None))


def test_decoding_extras_are_kept():
    r = Record.model_validate(make_record())
    assert r.generator.decoding.model_dump()["seed"] == 0


# --- JSONL I/O -------------------------------------------------------------------------


def test_jsonl_round_trip_is_lossless(tmp_path):
    records = [Record.model_validate(make_record(id=f"hi-{i:06d}")) for i in range(3)]
    path = tmp_path / "sub" / "records.jsonl"
    assert write_jsonl(records, path) == 3
    assert list(read_jsonl(path)) == records
    # writing what we read gives identical bytes
    path2 = tmp_path / "again.jsonl"
    write_jsonl(read_jsonl(path), path2)
    assert path.read_bytes() == path2.read_bytes()


def test_jsonl_keeps_indic_text_unescaped(tmp_path):
    path = tmp_path / "r.jsonl"
    write_jsonl([Record.model_validate(make_record())], path)
    raw = path.read_text(encoding="utf-8")
    assert CLAIM_2 in raw and "\\u" not in raw
    assert raw.count("\n") == 1


def test_read_jsonl_skips_blank_lines(tmp_path):
    line = record_to_json(Record.model_validate(make_record()))
    path = tmp_path / "r.jsonl"
    path.write_text(f"\n{line}\n\n", encoding="utf-8")
    assert len(list(read_jsonl(path))) == 1


def test_read_jsonl_reports_line_number(tmp_path):
    good = record_to_json(Record.model_validate(make_record()))
    bad = json.dumps(make_record(lang="xx"), ensure_ascii=False)
    path = tmp_path / "r.jsonl"
    path.write_text(f"{good}\n{bad}\n", encoding="utf-8")
    with pytest.raises(RecordParseError) as exc:
        list(read_jsonl(path))
    assert exc.value.line_no == 2


def test_failed_write_leaves_no_partial_file(tmp_path):
    path = tmp_path / "r.jsonl"

    def boom():
        yield Record.model_validate(make_record())
        raise RuntimeError("generator crashed")

    with pytest.raises(RuntimeError):
        write_jsonl(boom(), path)
    assert list(tmp_path.iterdir()) == []


# --- rejected records ------------------------------------------------------------------


def _claims_with(**changes):
    rec = make_record()
    rec["claims"][0].update(changes)
    return rec


@pytest.mark.parametrize(
    "rec",
    [
        pytest.param(make_record(unexpected="x"), id="unknown-field"),
        pytest.param(make_record(id="mr-000001"), id="id-prefix-not-lang"),
        pytest.param(make_record(split="dev"), id="bad-split"),
        pytest.param(make_record(retrieved=[]), id="no-retrieved-passages"),
        pytest.param(make_record(source="natural", generator=None), id="natural-without-generator"),
        pytest.param(_claims_with(start=1), id="claim-text-not-answer-slice"),
        pytest.param(_claims_with(end=10_000), id="claim-span-out-of-bounds"),
        pytest.param(_claims_with(start=3, end=3), id="empty-span"),
        pytest.param(_claims_with(best_passage_id=None), id="supported-without-evidence"),
        pytest.param(_claims_with(best_passage_id="hi-p999"), id="evidence-not-retrieved"),
        pytest.param(_claims_with(type="NUMERICAL"), id="supported-claim-with-error-type"),
        pytest.param(
            _claims_with(scores={"entail": 0.9, "neutral": 0.9, "contradict": 0.0}),
            id="nli-not-summing-to-one",
        ),
        pytest.param(
            _claims_with(
                passage_scores=[{"passage_id": "x", "entail": 1, "neutral": 0, "contradict": 0}]
            ),
            id="score-matrix-unknown-passage",
        ),
    ],
)
def test_invalid_records_are_rejected(rec):
    with pytest.raises(ValidationError):
        Record.model_validate(rec)


def test_contradicted_claim_needs_evidence():
    with pytest.raises(ValidationError):
        Claim(text="x", start=0, end=1, verdict=Verdict.CONTRADICTED)
    Claim(text="x", start=0, end=1, verdict=Verdict.CONTRADICTED, best_passage_id="p")


def test_retrieved_ranks_must_be_1_to_n():
    rec = make_record()
    rec["retrieved"][1]["rank"] = 3
    with pytest.raises(ValidationError):
        Record.model_validate(rec)


def test_retrieved_ids_must_be_unique():
    rec = make_record()
    rec["retrieved"][1]["passage_id"] = "hi-p1"
    with pytest.raises(ValidationError):
        Record.model_validate(rec)


def test_gold_span_out_of_bounds():
    rec = make_record()
    rec["labels"]["spans"][0]["end"] = len(ANSWER) + 1
    with pytest.raises(ValidationError):
        Record.model_validate(rec)


@pytest.mark.parametrize(
    "labels",
    [
        pytest.param({"response": "hallucinated", "spans": []}, id="hallucinated-without-spans"),
        pytest.param(
            {"response": "faithful", "spans": [{"start": 0, "end": 2, "type": "ENTITY"}]},
            id="faithful-with-spans",
        ),
        pytest.param(
            {
                "response": "hallucinated",
                "spans": [
                    {"start": 0, "end": 5, "type": "ENTITY"},
                    {"start": 4, "end": 8, "type": "LOCATION"},
                ],
            },
            id="overlapping-spans",
        ),
        pytest.param(
            {"response": "hallucinated", "spans": [{"start": 0, "end": 2, "type": "OPINION"}]},
            id="unknown-span-type",
        ),
    ],
)
def test_invalid_labels_are_rejected(labels):
    with pytest.raises(ValidationError):
        Labels.model_validate(labels)


def test_taxonomy_is_exactly_the_deck_four():
    assert {t.value for t in SpanType} == {"ENTITY", "NUMERICAL", "TEMPORAL", "LOCATION"}


def test_verdicts_are_exactly_the_deck_three():
    assert {v.value for v in Verdict} == {"SUPPORTED", "UNSUPPORTED", "CONTRADICTED"}
