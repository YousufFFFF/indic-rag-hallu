"""T37: the outcome panel is generated only from results/m0/pilot.json; gaps are errors."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "make_outcome_panel", ROOT / "scripts" / "make_outcome_panel.py"
)
mp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mp)


def summary():
    def det(f1, span):
        resp = {
            "per_language": {"hi": {"f1": f1}},
            "pooled": {"f1": f1},
            "macro": {"f1": f1},
            "macro_n": {"f1": 1},
            "worst": {"f1": {"lang": "hi", "value": f1}},
        }
        return {"response": resp, "spans": {"pooled": {"span_f1": span}}}

    return {
        "label": "pilot — synthetic Hindi data only",
        "run_id": "20261120-000000-m0-pilot-deadbeef",
        "git_commit": "0123456789abcdef",
        "retrieval_test": {"pooled": {"recall@10": 0.8}},
        "detection_test": {"ours": det(0.75, 0.6), "b1_sentence_nli": det(0.5, 0.25)},
    }


def test_panel_values_come_from_the_summary():
    p = mp.panel(summary())
    assert p["macro_f1"] == 0.75 and p["span_f1"] == 0.6 and p["recall@10"] == 0.8
    assert p["delta_response_f1"] == pytest.approx(0.25)
    assert p["delta_span_f1"] == pytest.approx(0.35) and p["n_languages"] == 1
    md = mp.render(p)
    assert "| 0.750 | 0.600 | 0.800 | +0.250 / +0.350 | 1 |" in md
    assert "pilot — synthetic Hindi data only" in md and "0123456" in md


def test_missing_metric_is_an_error_not_a_number():
    s = copy.deepcopy(summary())
    s["detection_test"]["ours"]["response"]["macro"]["f1"] = None
    with pytest.raises(mp.MissingResult):
        mp.panel(s)
    del s["detection_test"]["b1_sentence_nli"]
    with pytest.raises(mp.MissingResult):
        mp.panel(s)


def test_main_writes_checks_and_fails_loudly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mp, "SUMMARY", tmp_path / "pilot.json")
    monkeypatch.setattr(mp, "OUT", tmp_path / "outcome_panel")
    monkeypatch.setattr(mp, "ROOT", tmp_path)
    assert mp.main([]) == 2  # no summary yet
    (tmp_path / "pilot.json").write_text(json.dumps(summary()), "utf-8")
    assert mp.main(["--check"]) == 1
    assert mp.main([]) == 0
    assert mp.main(["--check"]) == 0
    assert json.loads((tmp_path / "outcome_panel.json").read_text("utf-8"))["n_languages"] == 1


def test_committed_panel_matches_committed_summary():
    if not mp.SUMMARY.exists():
        pytest.skip("no M0 summary committed yet")
    assert mp.main(["--check"]) == 0
