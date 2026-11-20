"""T21-T23: prompts, backends, generation cache (no network: mock backend + local server)."""

import http.server
import json
import threading

import pytest

from indichallu.data.schema import RetrievedPassage
from indichallu.generation.backends import MockBackend, OllamaBackend, clean_answer
from indichallu.generation.generate import (
    GenerationCache,
    Generator,
    GeneratorConfig,
    cache_key,
    load_generator_config,
)
from indichallu.generation.prompts import build_prompt, format_passages, load_template


def rp(rank, text, gold=False):
    return RetrievedPassage(
        passage_id=f"p{rank}", text=text, score=1 - rank / 10, rank=rank, is_gold=gold
    )


PASSAGES = [rp(2, "हिमालय उत्तर में है।"), rp(1, "योजना के लाभ राज्य सरकार द्वारा दिए जाते हैं।", True)]

# --- prompts ---------------------------------------------------------------------------


def test_hindi_template_has_one_slot_each_and_the_no_answer_rule():
    t = load_template("hi")
    assert t.count("{passages}") == 1 and t.count("{question}") == 1
    assert "दिए गए अनुच्छेदों में इस प्रश्न का उत्तर नहीं है।" in t


def test_prompt_numbers_passages_in_rank_order_in_the_source_language():
    prompt = build_prompt("hi", "  योजना के लाभ कौन देता है? ", PASSAGES)
    assert prompt.index("[1] योजना के लाभ") < prompt.index("[2] हिमालय")
    assert "प्रश्न: योजना के लाभ कौन देता है?\n" in prompt
    assert "{passages}" not in prompt and "{question}" not in prompt


def test_braces_in_passages_are_not_template_fields():
    prompt = build_prompt("hi", "प्रश्न?", [rp(1, "सूत्र {x} और {question}")])
    assert "सूत्र {x} और {question}" in prompt


def test_prompt_guards():
    with pytest.raises(ValueError):
        build_prompt("hi", "प्रश्न?", [])
    with pytest.raises(FileNotFoundError):
        load_template("xx")
    assert format_passages([rp(1, "  क  ")]) == "[1] क"


# --- backends --------------------------------------------------------------------------


def test_clean_answer_drops_a_think_block():
    assert clean_answer("<think>\nसोच\n</think>\n\nउत्तर है।  ") == "उत्तर है।"
    assert clean_answer("उत्तर") == "उत्तर"


class _FakeOllama(http.server.BaseHTTPRequestHandler):
    bodies: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
        type(self).bodies.append((self.path, body))
        reply = {
            "message": {"role": "assistant", "content": "राज्य सरकार।"},
            "prompt_eval_count": 42,
            "eval_count": 5,
            "done_reason": "stop",
        }
        data = json.dumps(reply, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_ollama():
    _FakeOllama.bodies = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1", _FakeOllama.bodies
    server.shutdown()


def test_ollama_backend_sends_fixed_decoding_and_utf8(fake_ollama):
    url, bodies = fake_ollama
    decoding = load_generator_config("m0").decoding
    gen = OllamaBackend("qwen3:4b", url).generate("प्रश्न: योजना?", decoding)
    path, body = bodies[0]
    assert path == "/api/chat"  # .../v1 is stripped to the server root
    assert body["messages"] == [{"role": "user", "content": "प्रश्न: योजना?"}]  # Hindi intact
    assert body["think"] is False and body["stream"] is False
    assert body["options"] == {
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "num_predict": 256,
        "num_ctx": 8192,
    }
    assert gen.text == "राज्य सरकार।" and gen.prompt_tokens == 42 and gen.completion_tokens == 5


# --- config and cache ------------------------------------------------------------------


def test_m0_config_is_pinned_and_fixed():
    cfg = load_generator_config("m0")
    assert cfg.model == "qwen3:4b" and len(cfg.ollama_digest) == 64
    assert len(cfg.hf_revision) == 40 and cfg.licence == "Apache-2.0"
    assert cfg.decoding["temperature"] == 0.0 and cfg.decoding["think"] is False
    assert cfg.record_info()["revision"] == cfg.ollama_digest


def cfg(**changes):
    base = dict(
        backend="mock",
        model="m",
        decoding={"temperature": 0.0, "max_new_tokens": 8},
        prompt_template="hi",
        context_passages=3,
        ollama_digest="d" * 64,
    )
    return GeneratorConfig(**{**base, **changes})


def test_generator_caches_and_resumes(tmp_path):
    backend = MockBackend(lambda p: f"उत्तर: {len(p)}")
    g = Generator(cfg(), backend, cache_dir=tmp_path)
    first, cached = g.generate("प्रश्न")
    assert not cached and first.text == "उत्तर: 6"
    again, cached = g.generate("प्रश्न")
    assert cached and again == first and len(backend.calls) == 1
    # a new process (new Generator) reads the same cache file
    fresh_backend = MockBackend()
    resumed, cached = Generator(cfg(), fresh_backend, cache_dir=tmp_path).generate("प्रश्न")
    assert cached and resumed.text == "उत्तर: 6" and fresh_backend.calls == []


@pytest.mark.parametrize(
    "change",
    [
        {"decoding": {"temperature": 0.7, "max_new_tokens": 8}},
        {"ollama_digest": "e" * 64},
        {"model": "n"},
    ],
)
def test_any_identity_change_misses_the_cache(tmp_path, change):
    backend = MockBackend()
    Generator(cfg(), backend, cache_dir=tmp_path).generate("प्रश्न")
    other = cfg(**change)
    Generator(other, backend, cache_dir=tmp_path).generate("प्रश्न")
    assert len(backend.calls) == 2


def test_torn_last_line_is_ignored(tmp_path):
    path = tmp_path / "c.jsonl"
    cache = GenerationCache(path)
    cache.put("k1", {"generation": {"text": "क", "raw": "क"}})
    with path.open("a", encoding="utf-8") as f:
        f.write('{"key": "k2", "generation": {"te')  # crash mid-write
    reloaded = GenerationCache(path)
    assert len(reloaded) == 1 and reloaded.get("k1")["generation"]["text"] == "क"


def test_cache_key_depends_on_prompt_and_identity():
    ident = cfg().identity()
    assert cache_key(ident, "a") != cache_key(ident, "b")
    assert cache_key(ident, "a") != cache_key(cfg(model="x").identity(), "a")
