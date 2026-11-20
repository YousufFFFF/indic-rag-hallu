"""T38: demo rendering (pure HTML; no Gradio needed)."""

from indichallu.data.schema import Record
from indichallu.viz import render_answer, render_claims, render_passages

ANSWER = "यह योजना राज्य सरकार द्वारा चलाई जाती है और आवेदक की आयु 21 वर्ष होनी चाहिए।"
C1, C2 = "योजना राज्य सरकार द्वारा चलाई जाती है", "आवेदक की आयु 21 वर्ष होनी चाहिए"


def rec():
    s1, s2 = ANSWER.index(C1), ANSWER.index(C2)
    return Record.model_validate(
        {
            "id": "hi-demo",
            "lang": "hi",
            "split": "test",
            "source": "injected",
            "question": "पात्रता?",
            "retrieved": [
                {
                    "passage_id": "p1",
                    "text": "<b>योजना</b> के लाभ।",
                    "score": 0.8,
                    "rank": 1,
                    "is_gold": True,
                },
                {"passage_id": "p2", "text": "अन्य।", "score": 0.4, "rank": 2, "is_gold": False},
            ],
            "answer": ANSWER,
            "claims": [
                {
                    "text": C1,
                    "start": s1,
                    "end": s1 + len(C1),
                    "verdict": "SUPPORTED",
                    "best_passage_id": "p1",
                    "scores": {"entail": 0.79, "neutral": 0.2, "contradict": 0.01},
                },
                {
                    "text": C2,
                    "start": s2,
                    "end": s2 + len(C2),
                    "verdict": "UNSUPPORTED",
                    "best_passage_id": "p2",
                    "type": "NUMERICAL",
                },
            ],
        }
    )


def test_answer_highlights_every_claim_and_keeps_the_rest():
    html = render_answer(rec())
    assert html.count("<mark") == 2
    assert "SUPPORTED · evidence [1] · entail 0.79" in html
    assert "UNSUPPORTED · NUMERICAL · evidence [2]" in html
    assert "यह " in html and "और" in html and "।" in html  # text outside claims kept


def test_claims_table_and_passages_are_escaped():
    rows = render_claims(rec())
    assert rows[0] == [C1, "SUPPORTED", "—", "[1]", "0.79"]
    assert rows[1] == [C2, "UNSUPPORTED", "NUMERICAL", "[2]", "—"]
    html = render_passages(rec())
    assert "&lt;b&gt;योजना&lt;/b&gt;" in html and "gold" in html
