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
            "response": {"pooled": {"f1": f1, "balanced_accuracy": 0.75, "auroc": 0.8}},
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
        "nli": {"model_id": "org/nli"},
        "pilot_records": {"val": 10, "test": 12},
        "retrieval_test": {"pooled": {**retrieval, "n_queries": 300}},
        "detection_test": {"ours": det(0.8), "b1_sentence_nli": det(0.6)},
    }


def test_render_uses_only_summary_values_and_labels_the_pilot():
    md = mt.render(summary())
    assert "pilot — synthetic Hindi data only" in md and "0123456" in md
    assert "| `me5-small` | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |" in md
    row = "Ours (decompose → align → aggregate) | 0.800 | 0.750 | 0.800 | 0.800 | 0.500 | — |"
    assert row + " 0.40 / 0.85 |" in md
    assert "| NUMERICAL | 0.900 | 0.900 |" in md and "| ENTITY | — | — |" in md


def test_replace_block_and_check(tmp_path, monkeypatch):
    doc = f"x\n{mt.BEGIN}\nold\n{mt.END}\ny\n"
    out = mt.replace_block(doc, "new\n")
    assert out == f"x\n{mt.BEGIN}\nnew\n{mt.END}\ny\n"
    with pytest.raises(ValueError):
        mt.replace_block("no markers", "x")
    s, r = tmp_path / "s.json", tmp_path / "R.md"
    s.write_text(json.dumps(summary()), "utf-8")
    r.write_text(doc, "utf-8")
    monkeypatch.setattr(mt, "SUMMARY", s)
    monkeypatch.setattr(mt, "README", r)
    assert mt.main(["--check"]) == 1  # stale
    assert mt.main([]) == 0
    assert mt.main(["--check"]) == 0


def test_committed_readme_matches_committed_summary():
    """CI-safe: if a summary is committed, the README block must be exactly its rendering."""
    if not mt.SUMMARY.exists():
        pytest.skip("no M0 summary committed yet")
    assert mt.main(["--check"]) == 0
