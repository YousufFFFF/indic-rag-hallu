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
    for _ in range(3):
        index.save(tmp_path / "idx")  # overwriting an existing index works
    names = sorted(f.name for f in (tmp_path / "idx").iterdir())
    assert names[0] == "CURRENT" and len(names) == 3  # current + previous version only
    assert all(n.startswith("v-") for n in names[1:])
    assert [f.name for f in tmp_path.iterdir()] == ["idx"]
    assert PassageIndex.load(tmp_path / "idx").search(q, 5) == index.search(q, 5)


def _version_dir(d):
    return d / (d / "CURRENT").read_text("utf-8").strip()


def test_load_rejects_mismatches_and_damage(tmp_path):
    index = PassageIndex.build(unit_rows(5, 4), IDS[:5], meta={"model_id": "org/m"})
    d = index.save(tmp_path / "idx")
    with pytest.raises(ValueError, match="expected"):
        PassageIndex.load(d, expect={"model_id": "org/other"})
    v = _version_dir(d)
    (v / "ids.json").write_text(json.dumps(IDS[:4]), "utf-8")
    with pytest.raises(ValueError, match="vectors but"):
        PassageIndex.load(d)
    (v / "index.faiss").write_bytes(b"garbage")
    with pytest.raises(ValueError, match="not a readable"):
        PassageIndex.load(d)
    with pytest.raises(ValueError, match="not a readable"):
        PassageIndex.load(tmp_path / "missing")
    (d / "CURRENT").write_text("../../etc", "utf-8")
    with pytest.raises(ValueError, match="bad CURRENT"):
        PassageIndex.load(d)


def test_failed_save_keeps_the_previous_index(tmp_path, monkeypatch):
    """Codex review on #20: replacing an index must never lose the old one."""
    import faiss

    old = PassageIndex.build(unit_rows(5, 4, 1), IDS[:5], meta={"model_id": "org/m"})
    d = old.save(tmp_path / "idx")
    q = unit_rows(1, 4, 2)
    new = PassageIndex.build(unit_rows(5, 4, 3), IDS[:5], meta={"model_id": "org/new"})

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(faiss, "write_index", boom)
    with pytest.raises(RuntimeError):
        new.save(d)
    loaded = PassageIndex.load(d)  # the previous index is intact and current
    assert loaded.meta["model_id"] == "org/m" and loaded.search(q, 3) == old.search(q, 3)
    assert len(list(d.glob("v-*"))) == 1 and not list(d.glob(".CURRENT.*"))  # no leftovers


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
    with pytest.raises(ValueError, match="different embedder config"):
        Retriever(Embedder(other, HashEncoder()), PassageIndex.load(d), CORPUS)
    with pytest.raises(ValueError, match="no text"):
        Retriever(Embedder(CFG, HashEncoder()), PassageIndex.load(d), {})
    with pytest.raises(ValueError, match="one gold"):
        r.retrieve(["a", "b"], k=1, gold_passage_ids=["x"])


def test_retriever_rejects_changed_passage_texts(tmp_path):
    """Codex review on #20: same ids but different text must not pass as the indexed corpus."""
    r = make_retriever()
    d = r.index.save(tmp_path / "idx")
    changed = {**CORPUS, "hi-p0000000002": "हिमालय भारत के दक्षिण में है।"}
    with pytest.raises(ValueError, match="fingerprint"):
        Retriever(Embedder(CFG, HashEncoder()), PassageIndex.load(d), changed)
    Retriever(Embedder(CFG, HashEncoder()), PassageIndex.load(d), CORPUS)  # exact texts: ok


def test_retriever_requires_normalised_embeddings_and_built_metadata():
    """Codex review on #20: inner product is cosine only for unit vectors."""
    raw = EmbedderConfig(
        name="fake", model_id="org/fake", revision="0" * 40, dim=16, normalize=False
    )
    with pytest.raises(ValueError, match="normalise"):
        Retriever.build(Embedder(raw, HashEncoder()), list(CORPUS), list(CORPUS.values()))
    bare = PassageIndex.build(unit_rows(3, 16), list(CORPUS))  # no embedder metadata
    with pytest.raises(ValueError, match="Retriever.build"):
        Retriever(Embedder(CFG, HashEncoder()), bare, CORPUS)


@pytest.mark.parametrize(
    "change",
    [{"query_prefix": "query: "}, {"passage_prefix": "passage: "}, {"max_seq_length": 64}],
)
def test_retriever_rejects_any_vector_affecting_config_change(tmp_path, change):
    """Codex review on #20: same model and revision but different prefixes -> different vectors."""
    d = make_retriever().index.save(tmp_path / "idx")
    other = EmbedderConfig(**{**CFG.__dict__, **change})
    with pytest.raises(ValueError, match="different embedder config"):
        Retriever(Embedder(other, HashEncoder()), PassageIndex.load(d), CORPUS)


def test_retriever_keeps_a_frozen_snapshot_of_the_passages(tmp_path):
    """Codex review on #20: editing the caller's mapping later must not change results."""
    d = make_retriever().index.save(tmp_path / "idx")
    corpus = dict(CORPUS)
    r = Retriever(Embedder(CFG, HashEncoder()), PassageIndex.load(d), corpus)
    corpus["hi-p0000000002"] = "बदला हुआ पाठ"
    hit = r.retrieve([CORPUS["hi-p0000000002"]], k=1)[0][0]
    assert hit.text == CORPUS["hi-p0000000002"]
    with pytest.raises(TypeError):
        r.passages["hi-p0000000002"] = "x"  # read-only


def test_load_retries_when_a_concurrent_save_removes_its_version(tmp_path, monkeypatch):
    """Codex review on #20: a reader that read CURRENT and then lost the race against two
    further saves (its version deleted) must re-read CURRENT and load the new index."""
    from indichallu.retrieval import index as index_module

    first = PassageIndex.build(unit_rows(5, 4, 1), IDS[:5], meta={"model_id": "v1"})
    d = first.save(tmp_path / "i")
    real_read_version = index_module._read_version
    calls = {"n": 0}

    def racing_read_version(files):
        calls["n"] += 1
        if calls["n"] == 1:  # between reading CURRENT and opening files, two saves happen
            PassageIndex.build(unit_rows(5, 4, 2), IDS[:5], meta={"model_id": "v2"}).save(d)
            PassageIndex.build(unit_rows(5, 4, 3), IDS[:5], meta={"model_id": "v3"}).save(d)
        return real_read_version(files)

    monkeypatch.setattr(index_module, "_read_version", racing_read_version)
    loaded = PassageIndex.load(d)
    assert loaded.meta["model_id"] == "v3" and calls["n"] == 2


def test_broken_version_is_not_retried_forever(tmp_path):
    d = PassageIndex.build(unit_rows(5, 4), IDS[:5]).save(tmp_path / "i")
    (_version_dir(d) / "meta.json").write_text("{not json", "utf-8")
    with pytest.raises(ValueError, match="not a readable"):
        PassageIndex.load(d)


def test_cleanup_never_deletes_the_version_current_names(tmp_path, monkeypatch):
    """Codex review on #20: a second writer publishes between our switch and our cleanup;
    our cleanup must not delete the version CURRENT now names."""
    from indichallu.retrieval import index as index_module

    d = PassageIndex.build(unit_rows(5, 4, 1), IDS[:5], meta={"model_id": "a"}).save(tmp_path / "i")
    other = PassageIndex.build(unit_rows(5, 4, 2), IDS[:5], meta={"model_id": "other"})
    real_pointer = index_module._read_pointer
    state = {"published": False}

    def pointer_with_concurrent_save(directory):
        if not state["published"]:  # first pointer read during our cleanup
            state["published"] = True
            other.save(directory)  # the other writer publishes now
        return real_pointer(directory)

    mine = PassageIndex.build(unit_rows(5, 4, 3), IDS[:5], meta={"model_id": "mine"})
    monkeypatch.setattr(index_module, "_read_pointer", pointer_with_concurrent_save)
    mine.save(d)
    monkeypatch.setattr(index_module, "_read_pointer", real_pointer)
    assert PassageIndex.load(d).meta["model_id"] == "other"  # still loadable
