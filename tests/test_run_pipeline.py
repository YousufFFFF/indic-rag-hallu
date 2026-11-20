"""T32: one command, retrieve -> generate -> detect, smoke-tested with mocks (no models)."""

import functools
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from indichallu.data.schema import RetrievedPassage, Split, Verdict, read_jsonl
from indichallu.generation.backends import MockBackend
from indichallu.generation.generate import Generator
from indichallu.runs import start_run

ROOT = Path(__file__).resolve().parents[1]
PASSAGE = "…योजना के लाभ राज्य सरकार द्वारा दिए जाते हैं।"
ANSWER = "यह योजना राज्य सरकार द्वारा चलाई जाती है और आवेदक की आयु 21 वर्ष होनी चाहिए।"


class EntailIfAllWordsInPremise:
    """Deterministic NLI stand-in: entail if every claim word occurs in the premise."""

    def predict(self, pairs):
        rows = []
        for premise, hyp in pairs:
            ok = all(w in premise for w in hyp.split() if w not in {"चलाई", "जाती"})
            rows.append([0.9, 0.08, 0.02] if ok else [0.05, 0.9, 0.05])
        return np.array(rows)


def load(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tiny_repo(path: Path) -> Path:
    path.mkdir()
    git = ["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "init"], check=True)
    return path


def test_pipeline_writes_detected_records_and_a_spotcheck_sheet(tmp_path, monkeypatch):
    rp = load("run_pipeline")
    ga = rp.ga
    passages = [
        SimpleNamespace(passage_id="hi-p1", text=PASSAGE),
        SimpleNamespace(passage_id="hi-p2", text="अन्य अनुच्छेद।"),
    ]
    q = SimpleNamespace(qa_id="hi-q1", question="पात्रता?", gold_passage_id="hi-p1", answerable=True)
    hits = [
        RetrievedPassage(
            passage_id=p.passage_id, text=p.text, score=0.9 - i / 10, rank=i + 1, is_gold=i == 0
        )
        for i, p in enumerate(passages)
    ]

    class FakeRetriever:
        @classmethod
        def build(cls, *a, **k):
            return cls()

        def retrieve(self, queries, k, gold_passage_ids):
            return [hits[:k]]

    for name, value in {
        "load_pinned_dataset": lambda: SimpleNamespace(repo="r", revision="v"),
        "verified_local_path": lambda d, lang: None,
        "load_indicqa": lambda path, lang: SimpleNamespace(passages=passages, questions=[q]),
        "split_dataset": lambda ds: SimpleNamespace(of_passage=lambda pid: Split.TEST),
        "load_embedder_config": lambda name: SimpleNamespace(model_id="m", revision="r"),
        "Embedder": lambda cfg, cache_dir: None,
        "Retriever": FakeRetriever,
        "Generator": lambda cfg: Generator(
            cfg, MockBackend(lambda prompt: ANSWER), cache_dir=tmp_path / "cache"
        ),
    }.items():
        monkeypatch.setattr(ga, name, value)
    monkeypatch.setattr(rp, "load_nli", lambda cfg: EntailIfAllWordsInPremise())
    monkeypatch.setattr(rp, "PILOT", tmp_path / "absent.json")
    monkeypatch.setattr(
        rp,
        "start_run",
        functools.partial(
            start_run, results_root=tmp_path / "runs", repo=tiny_repo(tmp_path / "g")
        ),
    )

    assert rp.main(["--embedder", "x", "--k", "3", "--sample", "1"]) == 0

    (run_dir,) = (tmp_path / "runs").iterdir()
    (rec,) = read_jsonl(run_dir / "detected.jsonl")
    assert [c.verdict for c in rec.claims] == [Verdict.SUPPORTED, Verdict.UNSUPPORTED]
    assert rec.claims[0].best_passage_id == "hi-p1"
    assert len(rec.claims[0].passage_scores) == 2  # every shown passage scored
    meta = json.loads((run_dir / "meta.json").read_text("utf-8"))
    assert meta["thresholds"]["source"] == "configs/detection/m0.yaml"
    sheet = (run_dir / "spotcheck.md").read_text("utf-8")
    assert "`hi-q1`" in sheet and ANSWER in sheet and "- [ ] faithful" in sheet
    assert "| SUPPORTED | [1] | 0.90 |" in sheet and "UNSUPPORTED (NUMERICAL)" in sheet
    assert "(gold)" in sheet


def test_pilot_thresholds_are_read_from_the_summary(tmp_path):
    rp = load("run_pipeline")
    assert rp.pilot_thresholds(tmp_path / "none.json") is None
    s = tmp_path / "pilot.json"
    s.write_text(
        json.dumps(
            {"detection_test": {"ours": {"thresholds": {"entail": 0.3, "contradict": 0.7}}}}
        ),
        "utf-8",
    )
    th = rp.pilot_thresholds(s)
    assert (th.entail, th.contradict) == (0.3, 0.7)
