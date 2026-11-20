"""Gradio demo (M0 task T38): ask in Hindi, see which parts of the answer the evidence supports.

    docker compose run --rm -p 7860:7860 dev python demo/app.py   # then open localhost:7860

Two modes:

* **Check an answer** (seconds): paste any Hindi answer. The demo retrieves evidence for the
  question from the Hindi IndicQA corpus and marks every claim SUPPORTED / UNSUPPORTED /
  CONTRADICTED, linked to its evidence passage.
* **Generate and check** (minutes on CPU): the local generator (Ollama) answers from the
  retrieved passages first.

Thresholds are the val-tuned ones from results/m0/pilot.json when present (D-024/D-025).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import gradio as gr

from indichallu.data.download import load_pinned_dataset, verified_local_path
from indichallu.data.loaders.indicqa import load_indicqa
from indichallu.data.schema import Record
from indichallu.detection.aggregate import Thresholds
from indichallu.detection.detector import Detector, load_detection_config
from indichallu.generation.generate import Generator, load_generator_config
from indichallu.generation.prompts import build_prompt
from indichallu.retrieval.embed import REPO_ROOT, Embedder, load_embedder_config
from indichallu.retrieval.retrieve import Retriever
from indichallu.viz import render_answer, render_claims, render_passages

LANG = "hi"
EMBEDDER = os.environ.get("DEMO_EMBEDDER", "me5-small")
PILOT = REPO_ROOT / "results" / "m0" / "pilot.json"
EXAMPLE_Q = "इस योजना के लिए पात्रता क्या है?"
EXAMPLE_A = "यह योजना राज्य सरकार द्वारा चलाई जाती है और आवेदक की आयु 21 वर्ष होनी चाहिए।"


@lru_cache(maxsize=1)
def pipeline():
    ds = load_indicqa(verified_local_path(load_pinned_dataset(), LANG), LANG)
    retriever = Retriever.build(
        Embedder(
            load_embedder_config(EMBEDDER), cache_dir=REPO_ROOT / "data" / "interim" / "embeddings"
        ),
        [p.passage_id for p in ds.passages],
        [p.text for p in ds.passages],
    )
    cfg = load_detection_config("m0")
    th = None
    if PILOT.exists():
        th = Thresholds(
            **json.loads(PILOT.read_text("utf-8"))["detection_test"]["ours"]["thresholds"]
        )
    detector = Detector.from_config(cfg, LANG, thresholds=th)
    return retriever, detector


def check(question: str, answer: str, generate: bool):
    question = (question or "").strip()
    if not question:
        raise gr.Error("प्रश्न लिखें / enter a question")
    retriever, detector = pipeline()
    hits = retriever.retrieve([question], k=3)[0]
    if generate:
        gen_cfg = load_generator_config("m0")
        answer = Generator(gen_cfg).generate(build_prompt(LANG, question, hits))[0].text
    if not (answer or "").strip():
        raise gr.Error("उत्तर लिखें या 'Generate' चुनें / enter an answer or choose Generate")
    record = Record(
        id="hi-demo",
        lang=LANG,
        split="test",
        source="natural" if generate else "injected",
        question=question,
        retrieved=hits,
        answer=answer.strip(),
        generator=load_generator_config("m0").record_info() if generate else None,
    )
    checked = detector.detect(record)
    return render_answer(checked), render_claims(checked), render_passages(checked)


with gr.Blocks(title="IndicRAG-Hallu demo") as demo:
    gr.Markdown(
        "## Does the retrieved evidence support this answer?\n"
        "Hindi IndicQA corpus · claims are checked against the top-3 retrieved passages "
        "(green = SUPPORTED, red = UNSUPPORTED, orange = CONTRADICTED; hover for evidence). "
        "**Pilot (M0).**"
    )
    q = gr.Textbox(label="प्रश्न (question)", value=EXAMPLE_Q)
    a = gr.Textbox(label="उत्तर (answer to check)", value=EXAMPLE_A, lines=3)
    g = gr.Checkbox(label="Generate the answer with the local LLM instead (slow on CPU)")
    btn = gr.Button("Check", variant="primary")
    out_answer = gr.HTML()
    out_claims = gr.Dataframe(headers=["claim", "verdict", "type", "evidence", "entail"], wrap=True)
    out_passages = gr.HTML()
    btn.click(check, [q, a, g], [out_answer, out_claims, out_passages])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
