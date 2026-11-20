"""scripts/generate_answers.py: records hold only the evidence the generator was shown."""

import functools
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from indichallu.data.schema import RetrievedPassage, Split, read_jsonl
from indichallu.generation.backends import MockBackend
from indichallu.generation.generate import Generator
from indichallu.runs import start_run

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "generate_answers.py"


def load_script():
    spec = importlib.util.spec_from_file_location("generate_answers", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tiny_repo(path: Path) -> Path:
    """A one-commit git repo, so run provenance works however the tests are checked out."""
    path.mkdir()
    git = ["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "init"], check=True)
    return path


def test_unshown_passages_stay_out_of_the_evidence(tmp_path, monkeypatch):
    mod = load_script()
    passages = [SimpleNamespace(passage_id=f"hi-p{i}", text=f"अनुच्छेद {i}।") for i in range(1, 6)]
    question = SimpleNamespace(
        qa_id="hi-q1", question="प्रश्न?", gold_passage_id="hi-p5", answerable=True
    )
    hits = [
        RetrievedPassage(
            passage_id=p.passage_id, text=p.text, score=1 - i / 10, rank=i + 1, is_gold=i == 4
        )
        for i, p in enumerate(passages)
    ]
    backend = MockBackend()

    class FakeRetriever:
        @classmethod
        def build(cls, *a, **k):
            return cls()

        def retrieve(self, queries, k, gold_passage_ids):
            return [hits[:k]]

    patches = {
        "load_pinned_dataset": lambda: SimpleNamespace(repo="r", revision="v"),
        "verified_local_path": lambda d, lang: None,
        "load_indicqa": lambda path, lang: SimpleNamespace(passages=passages, questions=[question]),
        "split_dataset": lambda ds: SimpleNamespace(of_passage=lambda pid: Split.TEST),
        "load_embedder_config": lambda name: SimpleNamespace(model_id="m", revision="r"),
        "Embedder": lambda cfg, cache_dir: None,
        "Retriever": FakeRetriever,
        "Generator": lambda cfg: Generator(cfg, backend, cache_dir=tmp_path / "cache"),
        "start_run": functools.partial(
            start_run, results_root=tmp_path / "runs", repo=tiny_repo(tmp_path / "repo")
        ),
    }
    for name, value in patches.items():
        monkeypatch.setattr(mod, name, value)

    assert mod.main(["--embedder", "x", "--k", "5"]) == 0

    (run_dir,) = (tmp_path / "runs").iterdir()
    (record,) = read_jsonl(run_dir / "records.jsonl")
    assert [p.passage_id for p in record.retrieved] == ["hi-p1", "hi-p2", "hi-p3"]
    (prompt, _) = backend.calls[0]
    assert "अनुच्छेद 3" in prompt and "अनुच्छेद 4" not in prompt
    (row,) = [json.loads(x) for x in (run_dir / "retrieval.jsonl").read_text("utf-8").splitlines()]
    assert row["gold_rank"] == 5 and len(row["hits"]) == 5 and "text" not in row["hits"][0]


def test_sample_sizes_must_be_non_negative():
    mod = load_script()
    ids = [f"hi-{i}" for i in range(5)]
    assert len(mod.sample_ids(ids, 2, 13)) == 2 and len(mod.sample_ids(ids, None, 13)) == 5
    with pytest.raises(ValueError):
        mod.sample_ids(ids, -1, 13)
    for bad in ("-1", "0"):
        with pytest.raises(SystemExit):
            mod.main(["--sample", bad])
