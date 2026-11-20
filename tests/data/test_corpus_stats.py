"""T13: corpus statistics and the auto-generated dataset card."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from indichallu.data.loaders.indicqa import IndicQA, Passage, QAPair
from indichallu.data.schema import Split
from indichallu.data.split import SplitConfig
from indichallu.data.stats import (
    BEGIN,
    DATASET_CARD,
    END,
    STATS_JSON,
    compute_language_stats,
    render_tables,
    replace_block,
)
from indichallu.languages import load_languages

ROOT = Path(__file__).resolve().parents[2]
LANGS = load_languages()
R = {Split.TRAIN: 0.6, Split.VAL: 0.2, Split.TEST: 0.2}


def tiny_ds() -> IndicQA:
    passages = [
        Passage(f"hi-p{i:010d}", "hi", f"लेख {i}", f"वाक्य एक है। वाक्य दो {i} है।") for i in range(10)
    ]
    questions = []
    for i, p in enumerate(passages):
        for k, answerable in enumerate((True, False)):
            questions.append(
                QAPair(
                    qa_id=f"hi-q{2 * i + k:06d}",
                    lang="hi",
                    question=f"प्रश्न {i} {k}?",
                    gold_passage_id=p.passage_id,
                    category="SHORT" if answerable else "NO",
                    answerable=answerable,
                    answer_status="unique" if answerable else "unanswerable",
                    answer_text="वाक्य" if answerable else None,
                    answer_start=0 if answerable else None,
                    answer_end=5 if answerable else None,
                    given_answer_start=3 if answerable else None,
                )
            )
    ds = IndicQA("hi", passages, questions)
    ds.stats.update({"unique": 10, "unanswerable": 10})
    return ds


def test_compute_language_stats():
    s = compute_language_stats(tiny_ds(), LANGS["hi"], SplitConfig(13, R))
    assert (s["passages"], s["questions"], s["answerable"], s["unanswerable"]) == (10, 20, 10, 10)
    assert s["answer_offsets"] == {
        "exact": 0,
        "unique": 10,
        "nearest": 0,
        "ambiguous": 0,
        "not_found": 0,
    }
    assert s["median_passage_sentences"] == 2.0
    assert s["median_question_tokens"] == 4.0  # प्रश्न, i, k, ?
    assert sum(v["passages"] for v in s["splits"].values()) == 10
    assert [s["splits"][k]["passages"] for k in ("train", "val", "test")] == [6, 2, 2]
    json.dumps(s)  # plain JSON types only


def test_render_marks_missing_languages_and_is_deterministic():
    s = compute_language_stats(tiny_ds(), LANGS["hi"], SplitConfig(13, R))
    stats = {
        "provenance": {"split_ratios": {"test": 0.2, "train": 0.6, "val": 0.2}},
        "languages": {"hi": s},
    }
    md = render_tables(stats, LANGS)
    assert md == render_tables(stats, LANGS)
    assert "| Hindi (`hi`) | indo-aryan | yes | 10 | 20 | 10 (50.0%) | 10 (50.0%) |" in md
    assert "| Tamil (`ta`) | dravidian | — | not computed" in md
    assert "train 0.6 / val 0.2 / test 0.2" in md


def test_replace_block_keeps_everything_outside_the_markers():
    doc = f"head\n{BEGIN}\nold\n{END}\ntail\n"
    out = replace_block(doc, "new\n")
    assert out == f"head\n{BEGIN}\nnew\n{END}\ntail\n"
    assert replace_block(out, "new\n") == out  # idempotent
    with pytest.raises(ValueError):
        replace_block("no markers", "x")
    with pytest.raises(ValueError):
        replace_block(doc + BEGIN, "x")


def test_committed_card_matches_committed_stats():
    """The card's tables must be exactly what the committed JSON renders to. This runs in
    CI without any data, so the card can never drift from the numbers."""
    stats = json.loads(STATS_JSON.read_text(encoding="utf-8"))
    card = DATASET_CARD.read_text(encoding="utf-8")
    assert replace_block(card, render_tables(stats, LANGS)) == card


def test_committed_stats_cover_all_languages_with_provenance():
    stats = json.loads(STATS_JSON.read_text(encoding="utf-8"))
    assert set(stats["languages"]) == set(LANGS)
    prov = stats["provenance"]
    assert len(prov["revision"]) == 40 and set(prov["file_sha256"]) == set(LANGS)
    for s in stats["languages"].values():
        o = s["answer_offsets"]
        assert sum(o.values()) == s["answerable"]  # every answerable question accounted for
        assert sum(v["questions"] for v in s["splits"].values()) == s["questions"]


def test_render_check_cli_passes_on_the_committed_card():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "corpus_stats.py"), "render", "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_compute_refuses_a_cached_file_that_differs_from_its_pin(tmp_path):
    """Codex review on #18: a changed local file must not be analysed as the pinned one."""
    import importlib.util

    from indichallu.data.download import ChecksumError, load_pinned_dataset

    spec = importlib.util.spec_from_file_location(
        "corpus_stats", ROOT / "scripts" / "corpus_stats.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    path = load_pinned_dataset().local_path("hi", tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"version": "1.0", "data": []}), "utf-8")
    with pytest.raises(ChecksumError):
        module.compute(["hi"], data_root=tmp_path)
    assert module.compute(["bn"], data_root=tmp_path)["languages"] == {}  # missing: skipped
