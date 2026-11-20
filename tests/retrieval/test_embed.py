"""T14: embedder wrapper (fake encoder in CI; a real small model under -m slow)."""

import hashlib
import unicodedata

import numpy as np
import pytest

from indichallu.retrieval.embed import (
    Embedder,
    EmbedderConfig,
    load_embedder_config,
)


class FakeEncoder:
    """Deterministic vectors from a hash of each input; records what it was asked."""

    def __init__(self, dim: int, scale: float = 3.0):
        self.dim, self.scale, self.calls = dim, scale, []

    def encode(self, texts, batch_size):
        self.calls.append(list(texts))
        out = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "little")
            out.append(np.random.default_rng(seed).normal(size=self.dim) * self.scale)
        return np.array(out, dtype=np.float64)


CFG = EmbedderConfig(
    name="fake",
    model_id="org/fake",
    revision="0" * 40,
    dim=8,
    query_prefix="query: ",
    passage_prefix="passage: ",
)


def test_prefixes_are_applied_per_kind():
    enc = FakeEncoder(8)
    e = Embedder(CFG, enc)
    e.embed_queries(["आयु क्या है?"])
    e.embed_passages(["आवेदक की आयु 21 वर्ष है।"])
    assert enc.calls == [["query: आयु क्या है?"], ["passage: आवेदक की आयु 21 वर्ष है।"]]
    with pytest.raises(ValueError):
        e.embed(["x"], "document")


def test_output_is_float32_normalised_and_shaped():
    v = Embedder(CFG, FakeEncoder(8)).embed_passages(["क", "ख", "ग"])
    assert v.dtype == np.float32 and v.shape == (3, 8)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-6)
    raw = Embedder(EmbedderConfig(**{**CFG.__dict__, "normalize": False}), FakeEncoder(8))
    assert not np.allclose(np.linalg.norm(raw.embed_passages(["क"]), axis=1), 1.0)


def test_text_is_nfc_normalised_before_encoding():
    enc = FakeEncoder(8)
    decomposed = "क़"  # क + nukta (U+0958 decomposes to this; NFC keeps it as is)
    bengali = "কো"  # NFC composes to U+0995 U+09CB
    Embedder(CFG, enc).embed_passages([decomposed, bengali])
    assert enc.calls[0] == [
        "passage: " + unicodedata.normalize("NFC", t) for t in (decomposed, bengali)
    ]
    assert enc.calls[0][1] == "passage: কো"


def test_empty_input():
    enc = FakeEncoder(8)
    v = Embedder(CFG, enc).embed_passages([])
    assert v.shape == (0, 8) and v.dtype == np.float32 and enc.calls == []


def test_wrong_shape_or_bad_values_are_rejected():
    class Bad:
        def __init__(self, out):
            self.out = out

        def encode(self, texts, batch_size):
            return self.out

    with pytest.raises(ValueError, match="shape"):
        Embedder(CFG, Bad(np.ones((1, 7)))).embed_passages(["क"])
    with pytest.raises(ValueError, match="non-finite"):
        Embedder(CFG, Bad(np.full((1, 8), np.nan))).embed_passages(["क"])
    with pytest.raises(ValueError, match="zero vector"):
        Embedder(CFG, Bad(np.zeros((1, 8)))).embed_passages(["क"])


# --- cache -----------------------------------------------------------------------------


def test_cache_avoids_recomputation(tmp_path):
    enc = FakeEncoder(8)
    e = Embedder(CFG, enc, cache_dir=tmp_path)
    first = e.embed_passages(["क", "ख"])
    second = e.embed_passages(["क", "ख"])
    assert len(enc.calls) == 1 and np.array_equal(first, second)
    assert len(list((tmp_path / "fake").glob("*.npy"))) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"revision": "1" * 40},
        {"passage_prefix": "p: "},
        {"max_seq_length": 256},
        {"normalize": False},
    ],
)
def test_any_config_change_misses_the_cache(tmp_path, change):
    enc = FakeEncoder(8)
    Embedder(CFG, enc, cache_dir=tmp_path).embed_passages(["क"])
    Embedder(EmbedderConfig(**{**CFG.__dict__, **change}), enc, cache_dir=tmp_path).embed_passages(
        ["क"]
    )
    assert len(enc.calls) == 2


def test_texts_order_and_kind_are_part_of_the_key(tmp_path):
    enc = FakeEncoder(8)
    e = Embedder(CFG, enc, cache_dir=tmp_path)
    e.embed_passages(["क", "ख"])
    e.embed_passages(["ख", "क"])
    e.embed_queries(["क", "ख"])
    e.embed_passages(["कख"])  # concatenation must not collide with ["क", "ख"]
    assert len(enc.calls) == 4


def test_corrupt_cache_entry_is_recomputed(tmp_path):
    enc = FakeEncoder(8)
    e = Embedder(CFG, enc, cache_dir=tmp_path)
    good = e.embed_passages(["क"])
    (cache_file,) = (tmp_path / "fake").glob("*.npy")
    np.save(cache_file, np.zeros((5, 3), dtype=np.float32))  # wrong shape
    again = e.embed_passages(["क"])
    assert len(enc.calls) == 2 and np.array_equal(good, again)


# --- config ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "dim", "qp", "pp"),
    [
        ("bge-m3", 1024, "", ""),
        ("me5-large", 1024, "query: ", "passage: "),
        ("me5-small", 384, "query: ", "passage: "),
    ],
)
def test_configured_embedders(name, dim, qp, pp):
    c = load_embedder_config(name)
    assert (c.dim, c.query_prefix, c.passage_prefix, c.normalize) == (dim, qp, pp, True)
    assert len(c.revision) == 40


def test_unknown_embedder():
    with pytest.raises(KeyError):
        load_embedder_config("nope")


# --- a real model (opt-in: pytest -m slow; downloads ~470 MB once) -------------------


@pytest.mark.slow
def test_real_small_model_ranks_the_relevant_hindi_passage_first(tmp_path):
    e = Embedder(load_embedder_config("me5-small"), cache_dir=tmp_path)
    passages = [
        "योजना के लाभ राज्य सरकार द्वारा दिए जाते हैं।",
        "हिमालय भारत के उत्तर में स्थित एक पर्वत श्रृंखला है।",
        "क्रिकेट भारत में सबसे लोकप्रिय खेल है।",
    ]
    q = e.embed_queries(["इस योजना के लाभ कौन देता है?"])
    p = e.embed_passages(passages)
    assert p.shape == (3, 384) and np.allclose(np.linalg.norm(p, axis=1), 1.0, atol=1e-5)
    scores = (p @ q.T).ravel()
    assert int(np.argmax(scores)) == 0, scores
    assert np.array_equal(e.embed_passages(passages), p)  # deterministic / cached
