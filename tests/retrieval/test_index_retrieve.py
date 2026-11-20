"""T15: FAISS passage index and top-k retrieval."""

import hashlib
import json

import numpy as np
import pytest

from indichallu.data.schema import Record, RetrievedPassage
from indichallu.retrieval.embed import Embedder, EmbedderConfig
from indichallu.retrieval.index import PassageIndex, passages_fingerprint
from indichallu.retrieval.retrieve import Retriever, gold_rank


def unit_rows(n, d, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


IDS = [f"hi-p{i:010d}" for i in range(50)]


# --- index -----------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 5, 50, 80])
def test_search_matches_brute_force(k):
    p, q = unit_rows(50, 16, 0), unit_rows(7, 16, 1)
    index = PassageIndex.build(p, IDS)
    hits = index.search(q, k)
    scores = q @ p.T
    for row, s in zip(hits, scores, strict=True):
        expected = sorted(range(50), key=lambda i: (-s[i], i))[: min(k, 50)]
        assert [pid for pid, _ in row] == [IDS[i] for i in expected]
        assert np.allclose([sc for _, sc in row], s[expected], atol=1e-5)


def test_ties_are_broken_by_corpus_position_even_at_the_cut_off():
    """Many identical passages: which ones make the top-k must not be FAISS's choice."""
    base = unit_rows(1, 8, 3)
    p = np.repeat(base, 40, axis=0)  # 40 identical vectors
    p[[5, 30]] = unit_rows(2, 8, 4)  # two different ones
    index = PassageIndex.build(p, IDS[:40])
    for k in (1, 3, 10, 25):
        row = index.search(base, k)[0]
        tied = [i for i in range(40) if i not in (5, 30)]
        assert [pid for pid, _ in row] == [IDS[i] for i in tied[:k]]


def test_input_validation():
    p = unit_rows(3, 4)
    with pytest.raises(ValueError, match="unique"):
        PassageIndex.build(p, ["a", "a", "b"])
    with pytest.raises(ValueError, match="one vector per passage"):
        PassageIndex.build(p, ["a", "b"])
    with pytest.raises(ValueError, match="finite"):
        PassageIndex.build(np.full((1, 4), np.nan, dtype=np.float32), ["a"])
    index = PassageIndex.build(p, ["a", "b", "c"])
    with pytest.raises(ValueError, match="shape"):
        index.search(unit_rows(1, 5), 2)
    with pytest.raises(ValueError, match="k must"):
        index.search(unit_rows(1, 4), 0)
    assert index.search(np.zeros((0, 4), dtype=np.float32), 2) == []


def test_save_load_round_trip(tmp_path):
    p, q = unit_rows(50, 16), unit_rows(3, 16, 9)
    index = PassageIndex.build(p, IDS, meta={"model_id": "org/m", "revision": "r" * 40})
    index.save(tmp_path / "idx")
    loaded = PassageIndex.load(tmp_path / "idx", expect={"model_id": "org/m"})
    assert loaded.passage_ids == IDS and loaded.meta["size"] == 50
    assert loaded.search(q, 5) == index.search(q, 5)
    index.save(tmp_path / "idx")  # overwriting an existing index works
    assert sorted(f.name for f in (tmp_path / "idx").iterdir()) == [
        "ids.json",
        "index.faiss",
        "meta.json",
    ]
    assert [f.name for f in tmp_path.iterdir()] == ["idx"]  # no temp dirs left


def test_load_rejects_mismatches_and_damage(tmp_path):
    index = PassageIndex.build(unit_rows(5, 4), IDS[:5], meta={"model_id": "org/m"})
    d = index.save(tmp_path / "idx")
    with pytest.raises(ValueError, match="expected"):
        PassageIndex.load(d, expect={"model_id": "org/other"})
    (d / "ids.json").write_text(json.dumps(IDS[:4]), "utf-8")
    with pytest.raises(ValueError, match="vectors but"):
        PassageIndex.load(d)
    (d / "index.faiss").write_bytes(b"garbage")
    with pytest.raises(ValueError, match="not a readable"):
        PassageIndex.load(d)
    with pytest.raises(ValueError, match="not a readable"):
        PassageIndex.load(tmp_path / "missing")


def test_passages_fingerprint_is_unambiguous():
    assert passages_fingerprint(["a"], ["bc"]) != passages_fingerprint(["ab"], ["c"])
    assert passages_fingerprint(["a", "b"], ["x", "y"]) != passages_fingerprint(
        ["b", "a"], ["y", "x"]
    )


# --- retriever -------------------------------------------------------------------------


class HashEncoder:
    def __init__(self, dim=16):
        self.dim = dim

    def encode(self, texts, batch_size):
        return np.stack(
            [
                np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:16], 16)).normal(
                    size=self.dim
                )
                for t in texts
            ]
        )


CFG = EmbedderConfig(name="fake", model_id="org/fake", revision="0" * 40, dim=16)
CORPUS = {
    "hi-p0000000001": "योजना के लाभ राज्य सरकार द्वारा दिए जाते हैं।",
    "hi-p0000000002": "हिमालय भारत के उत्तर में है।",
    "hi-p0000000003": "क्रिकेट लोकप्रिय खेल है।",
}


def make_retriever():
    return Retriever.build(Embedder(CFG, HashEncoder()), list(CORPUS), list(CORPUS.values()))


def test_retrieve_returns_ranked_schema_objects_with_gold_flags():
    r = make_retriever()
    # no prefixes in CFG: a question identical to a passage embeds identically -> rank 1
    questions = [CORPUS["hi-p0000000002"], "कुछ और", CORPUS["hi-p0000000003"]]
    golds = ["hi-p0000000002", "hi-p0000000001", None]
    results = r.retrieve(questions, k=2, gold_passage_ids=golds)
    assert all(isinstance(p, RetrievedPassage) for row in results for p in row)
    assert [p.rank for p in results[0]] == [1, 2]
    assert results[0][0].passage_id == "hi-p0000000002" and results[0][0].is_gold
    assert results[0][0].score == pytest.approx(1.0, abs=1e-5)
    assert results[0][0].text == CORPUS["hi-p0000000002"]
    assert gold_rank(results[0]) == 1
    assert not any(p.is_gold for p in results[2])  # no gold given
    assert gold_rank(results[2]) is None
    assert results[0][0].score >= results[0][1].score


def test_results_validate_inside_a_record():
    r = make_retriever()
    retrieved = r.retrieve(["आयु?"], k=3, gold_passage_ids=["hi-p0000000001"])[0]
    rec = Record(
        id="hi-q000001",
        lang="hi",
        split="test",
        source="injected",
        question="आयु?",
        retrieved=retrieved,
        answer="उत्तर",
    )
    assert [p.rank for p in rec.retrieved] == [1, 2, 3]
    assert sum(p.is_gold for p in rec.retrieved) == 1


def test_retriever_checks_index_and_embedder_agree(tmp_path):
    r = make_retriever()
    d = r.index.save(tmp_path / "idx")
    other = EmbedderConfig(name="fake", model_id="org/fake", revision="1" * 40, dim=16)
    with pytest.raises(ValueError, match="revision"):
        Retriever(Embedder(other, HashEncoder()), PassageIndex.load(d), CORPUS)
    with pytest.raises(ValueError, match="no text"):
        Retriever(Embedder(CFG, HashEncoder()), PassageIndex.load(d), {})
    with pytest.raises(ValueError, match="one gold"):
        r.retrieve(["a", "b"], k=1, gold_passage_ids=["x"])
