"""T37: the README results block is generated only from results/m0/pilot.json."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_tables", ROOT / "scripts" / "make_tables.py")
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)


def summary():
    def det(f1):
        span = {"span_f1": f1, "token_iou": 0.5, "exact_f1": None}
        by_type = {t: {"span_f1": None} for t in ("ENTITY", "LOCATION")}
        by_type.update({"NUMERICAL": {"span_f1": 0.9}, "TEMPORAL": {"span_f1": 0.7}})
        return {
            "thresholds": {"entail": 0.4, "contradict": 0.85},
            "response": {
                "pooled": {
                    "f1": f1,
                    "precision": 0.9,
                    "recall": 0.1,
                    "balanced_accuracy": 0.75,
                    "auroc": 0.8,
                    "positives": 6,
                    "n": 12,
                }
            },
            "spans": {"pooled": span},
            "spans_by_type": by_type,
        }

    retrieval = {
        k: 0.5 for k in ("recall@1", "recall@3", "recall@5", "recall@10", "ndcg@10", "mrr")
    }
    return {
        "label": "pilot — synthetic Hindi data only",
        "run_id": "20261120-000000-m0-pilot-deadbeef",
        "git_commit": "0123456789abcdef",
        "embedder": {"name": "me5-small", "model_id": "intfloat/multilingual-e5-small"},
        "nli": {"model_id": "org/nli", "device": "cuda"},
        "pilot_records": {"val": 10, "test": 12},
        "pilot_set": {"used (val)": 5, "used (test)": 6, "injected NUMERICAL": 7},
        "retrieval_test": {"pooled": {**retrieval, "n_queries": 300}},
        "detection_test": {"ours": det(0.8), "b1_sentence_nli": det(0.6)},
    }


def test_render_uses_only_summary_values_and_labels_the_pilot():
    md = mt.render(summary())
    assert "pilot — synthetic Hindi data only" in md and "0123456" in md
    assert "| `me5-small` | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |" in md
    row = (
        "Ours (decompose → align → aggregate) | 0.900 | 0.100 | 0.800 | 0.750 | 0.800 | 0.800 "
        "| 0.500 | — |"
    )
    assert row + " 0.40 / 0.85 |" in md
    assert "| NUMERICAL | 0.900 | 0.900 |" in md and "| ENTITY | — | — |" in md
    assert "(device: cuda)" in md
    # 6 of 12 hallucinated: flagging everything gives P = 0.5, R = 1, F1 = 2/3
    assert "gives response F1 0.667 on this test set (6 of 12 records hallucinated)" in md


def test_replace_block_and_check(tmp_path, monkeypatch):
    doc = f"x\n{mt.BEGIN}\nold\n{mt.END}\ny\n"
    out = mt.replace_block(doc, "new\n")
    assert out == f"x\n{mt.BEGIN}\nnew\n{mt.END}\ny\n"
    with pytest.raises(ValueError):
        mt.replace_block("no markers", "x")
    s, r, card = tmp_path / "s.json", tmp_path / "R.md", tmp_path / "card.md"
    s.write_text(json.dumps(summary()), "utf-8")
    r.write_text(doc, "utf-8")
    card.write_text(mt.PILOT_BEGIN + "\n" + mt.PILOT_END + "\n", "utf-8")
    monkeypatch.setattr(mt, "SUMMARY", s)
    monkeypatch.setattr(mt, "README", r)
    monkeypatch.setattr(mt, "DATASET_CARD", card)
    assert mt.main(["--check"]) == 1  # stale
    assert mt.main([]) == 0
    assert mt.main(["--check"]) == 0


def test_committed_readme_matches_committed_summary():
    """CI-safe: if a summary is committed, the README block must be exactly its rendering."""
    if not mt.SUMMARY.exists():
        pytest.skip("no M0 summary committed yet")
    assert mt.main(["--check"]) == 0


def test_pilot_set_block_lists_every_count():
    md = mt.render_pilot_set(summary())
    assert "**10 val / 12 test**" in md and "| injected NUMERICAL | 7 |" in md
