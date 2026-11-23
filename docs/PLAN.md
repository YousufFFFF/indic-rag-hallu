# PLAN — Hallucination Detection in RAG for Low-Resource Indic Languages

*Last updated: 2026-11-24 · Status: M0 complete (pilot; HF Space optional); M1 planned in §3, waiting on Q14–Q16.*

**Team:** Mohammed Yousuf Ansari (lead) · Mukta Dhage · Sneha Shinde · Srushti Kondhalkar
**Guide:** Prof. [GUIDE NAME — open question Q7]

Sources: [CLAUDE.md](../CLAUDE.md) (standing brief) and [stage1_deck.md](stage1_deck.md) (Stage I commitments). Models and datasets are listed in [RESOURCES.md](RESOURCES.md).

---

## M0 status (2026-11-21)

| Tasks | Status | Where |
|---|---|---|
| T01–T14 | Done | merged to `main` |
| T15 index + retrieval | Done | #20 |
| T16–T18, T20 metrics | Done | #21 |
| T19 retrieval eval | **Done with a deviation:** `multilingual-e5-small` as a labelled stand-in; bge-m3 / mE5-large downloads not approved yet, so the embedder choice moves to M1 (T4) | #26 (pilot run) |
| T21–T23 generation | Done (Ollama digest pinned; Hindi prompt approved by Yousuf) | #22 |
| T24–T27, T33, T34 detection, B1, tuning | Done (slide-17 regression passes with the real NLI model); window reduction fixed to the most entailing window (D-027) | #23, #25 |
| GPU path | NLI on CUDA when available; Kaggle/Colab notebook for the pilot (D-026) | #24, #27 |
| T28 NUMERICAL/TEMPORAL injection, T30 pilot set, T35 pilot eval, T37 README table, T38 demo, T39 README | Done; results generated from `results/m0/pilot.json` (Kaggle T4 run) | #26 |
| T29 ENTITY/LOCATION injection | **Moved to M1:** needs IndicNER (gated; access to be requested) | — |
| T31 generation | **Partial:** Ollama does not see the laptop GPU, so M0 generates a small CPU sample (~10 answers); the full val+test run moves to a GPU in M1 | bundle 5 |
| T32 one-command pipeline, T36 review sheet, T37 outcome panel, T40 related work, T41 dataset card | Done | bundle 5 |
| T36 spot-check | Done: 10 CPU answers reviewed by Yousuf; the detector missed the one hallucination and scores refusals as SUPPORTED (PROGRESS 2026-11-21) | — |
| T41 fresh-clone check | Done: a fresh clone of `main` (8ef6a40) builds, passes the full suite, downloads IndicQA with verified hashes, and every `--check` (README, dataset card, outcome panel, corpus stats) is up to date | — |
| T42 HF Space | Open (optional for M0; needs the HF account) | — |

**Pilot and spot-check findings that shape M1:** off-the-shelf multilingual NLI entails most injected number/date swaps, so recall is low; the typed verifiers (C2) are the first M1 priority. Refusals need explicit handling (they are scored as entailed claims), and the annotation guidelines must cover answers that repeat the question's presupposition. The span-F1 threshold objective (D-024) is degenerate for claim-level predictions against number-level gold spans; M1 fixes its objective before its own evaluation (PROGRESS 2026-11-21).

## 1. Traceability matrix

Every item in the Stage I deck, mapped to where it is built, how it is tested, and what metric or artefact shows it is done. "S#" = deck slide number. "CL#" = row in CLAUDE.md §3. Milestones: **M0** Hindi slice (by 10 Dec 2026) · **M1** Jan · **M2** Feb · **M3** Mar · **M4** Apr–May 2027.

Paths are relative to `src/indichallu/` unless they start with `configs/`, `scripts/`, `tests/`, `docs/`, `demo/`, `analysis/`, `paper/` or `stretch/`.

### 1.1 Framing, motivation and failure modes (S1, S4–S8)

| ID | Deck item | Module / file | Test | Metric / evidence | M |
|---|---|---|---|---|---|
| D01 | S1 one-line framing: "verify whether an LLM answer … is supported by its retrieved evidence" | `README.md`, `detection/aggregate.py` (verdict semantics) | `tests/test_worked_example.py` | README states the contract; verdicts are evidence-relative only | M0 |
| D02 | S1 pipeline strip QUERY → RETRIEVE → CONTEXT → GENERATE → DETECT | `scripts/run_pipeline.py` (one stage per step) | `tests/test_pipeline_smoke.py` (mock models, CPU) | Every stage's output is present in the record JSONL | M0 |
| D03 | S1 title slide team/guide placeholders | Stage II deck, `README.md` authors | — | Names filled (Q7, Q9) | M0 |
| D04 | S4 definition: "fluent text that no source supports" | `docs/annotation_guidelines.md` §definitions | — | Definition used verbatim in guidelines and paper | M2 |
| D05 | S4 illustrative Hindi example (scheme eligibility, age 21 unsupported) | `tests/test_worked_example.py` | same | Passes (also D57) | M0 |
| D06 | S4 high-stakes domains (health, legal, agri, govt schemes) | `docs/dataset_card.md` domain breakdown; corpus metadata `domain` field | `tests/test_schema.py` (field allowed) | Per-domain counts in dataset card (where source provides domain) | M2 |
| D07 | S5 motivation 01 — sparse corpora | `scripts/corpus_stats.py` → `docs/dataset_card.md` | `tests/test_corpus_stats.py` | Per-language #passages, #tokens, #QA table (auto-generated) | M0 (hi) / M1 |
| D08 | S5 motivation 02 — script & morphology | `text/normalize.py`, `text/offsets.py`, `text/numerals.py` | `tests/text/test_normalize.py`, `test_offsets.py`, `test_numerals.py` | Offset round-trip on all 9 scripts | M0 (hi + digit tables for all) |
| D09 | S5 motivation 03 — weak retrieval | `retrieval/`, `eval/retrieval.py` | `tests/eval/test_retrieval_metrics.py` | Recall@k/nDCG@k/MRR per language (D84–D86) | M0 (hi) / M1 |
| D10 | S5 motivation 04 — no evaluation resource | `data/`, `annotate/app.py`, HF dataset release | `tests/test_schema.py` | Released gold test set (C1) | M2 / M4 |
| D11 | S5 languages in scope: hi, bn, ta, te, mr, kn, ml, gu, pa; "final set fixed by corpus availability" | `configs/languages.yaml` (code, script, digit block, resource level, enabled flag); `data/loaders/indicqa.py` | `tests/test_languages_config.py` | Corpus stats table lists all 9; enabled set recorded in `DECISIONS.md` | M0 (config, hi enabled) / M1 |
| D12 | S6 RAG: retriever encodes query + corpus into one space, returns top-k | `retrieval/embed.py`, `retrieval/index.py`, `retrieval/retrieve.py` | `tests/retrieval/test_index.py`, `test_retrieve.py` | Retrieval metrics (D84–D86) | M0 |
| D13 | S6 context window: passages concatenated into the prompt | `generation/prompts/<lang>.txt`, `generation/generate.py::assemble_prompt` | `tests/generation/test_prompt_assembly.py` | Prompt stored in each record | M0 |
| D14 | S6 generator conditions on evidence but is not forced to stay inside it | `generation/backends.py` | `tests/generation/test_backends.py` (mock) | Cached generations in JSONL | M0 |
| D15 | S7 failure A — retrieval miss | `retrieved[].is_gold`, `gold_retrieved@k` flag in record; `analysis/` | `tests/retrieval/test_retrieve.py` | Hallucination rate split by gold-retrieved vs not; C4 "distractors only" condition | M0 (flag) / M3 (C4) |
| D16 | S7 failure B — partial grounding (supported + invented spans in one sentence) | `detection/decompose.py` (sub-sentence claims) | `tests/detection/test_decompose.py`; worked example | Span F1; % responses with mixed verdicts | M0 |
| D17 | S7 failure C — context contradiction / cross-lingual drift | CONTRADICTED verdict in `detection/aggregate.py`; C4 condition "gold passage translated to English" | `tests/detection/test_aggregate.py` | Per-verdict P/R; C4 drift condition results | M0 (verdict) / M3 (C4) |
| D18 | S8 problem: detection is response-level → we need span-level | `detection/` claim spans; `eval/spans.py` | `tests/eval/test_spans.py` | Span F1, token IoU, exact match | M0 |
| D19 | S8 problem: English-centric alignment | Indic-tuned aligner (C3), typed verifiers (C2), translate-test B5 comparison (C6) | `tests/detection/typed/*` | Ours vs B5 Δ per language | M1 / M3 |
| D20 | S8 problem: no Indic ground truth | Gold human-annotated test set (C1) | `tests/test_agreement.py` | κ/α reported; #annotated responses | M2 |

### 1.2 Objectives (S9)

| ID | Deck item | Module / file | Test | Metric / evidence | M |
|---|---|---|---|---|---|
| D21 | Obj 01 — build the RAG pipeline over an Indic corpus | `retrieval/`, `generation/`, `scripts/run_pipeline.py` | `tests/test_pipeline_smoke.py` | One-command run produces `results/runs/<id>/` | M0 |
| D22 | Obj 02 — create ground truth, response + span level | `data/schema.py`, `data/synth/`, `annotate/app.py`, `docs/annotation_guidelines.md` | `tests/test_schema.py`, `tests/test_agreement.py` | #annotated responses, κ/α | M0 (synthetic) / M2 (gold) |
| D23 | Obj 03 — decompose into atomic claims, align each against evidence | `detection/decompose.py`, `detection/align.py` | `tests/detection/*` | Claim score matrix saved per record | M0 |
| D24 | Obj 04 — evaluate per language and per error type | `eval/*`, `eval/fairness.py`, `scripts/run_eval.py` | `tests/eval/*` | Every table has per-language and per-type rows + pooled | M0 (hi) / M3 |
| D25 | Obj 05 — retrieval quality & resource level vs hallucination rate | `analysis/retrieval_vs_halluc.ipynb` (reads `results/` only); C4 experiment config `configs/experiments/c4_*.yaml` | `tests/test_c4_conditions.py` (condition builder) | Figures from `results/`; hallucination rate per C4 condition and per resource level | M3 |
| D26 | Scope note: [INSERT LANGUAGE COUNT], [INSERT DATASET SIZE] | `scripts/corpus_stats.py`, `scripts/make_outcome_panel.py` | — | Filled from generated stats only | M3 |

### 1.3 Literature and gap (S11–S13, S24)

| ID | Deck item | Module / file | Test | Metric / evidence | M |
|---|---|---|---|---|---|
| D27 | S11 RAGTruth (Niu et al. 2024) — response + span labels | `data/schema.py` label format; `data/synth/translate_ragtruth.py` | `tests/test_schema.py` | Citation verified in `docs/related_work.md` | M0 (schema) / M1 |
| D28 | S11 Chang et al. 2024 — hallucination & coverage errors in RAG | `docs/related_work.md` | — | Citation verified | M1 |
| D29 | S11 Benkirane et al. 2024 — MT hallucination, low/high resource | `docs/related_work.md` | — | Citation verified | M1 |
| D30 | S12 HalluSearch (Abdallah & El-Beltagy 2025) — factual splitting → verification | `docs/related_work.md`; informs `detection/decompose.py` | — | Citation verified | M1 |
| D31 | S12 Multi-Hall-SA (Ralethe & Buys 2026) — ENTITY/NUMERICAL/TEMPORAL/LOCATION | `data/schema.py` `type` enum | `tests/test_schema.py` | Citation verified | M0 (enum) / M1 |
| D32 | S13 research gap statement | `docs/related_work.md`, paper intro | — | Gap restated per CLAUDE.md §4.1 | M1 |
| D33 | S13 bullet "Benchmarks exist for English and South African languages — not Indic" — **must be rewritten** (BHRAM-IL exists) | `docs/related_work.md` + Stage II deck | — | New wording approved by guide | M1 |
| D34 | S24 references list: venue details pending verification | `paper/refs.bib` | — | Every entry verified (DOI/ACL Anthology/arXiv) | M1 |

### 1.4 Methodology stages (S15) and block diagram (S16)

| ID | Deck item | Module / file | Test | Metric / evidence | M |
|---|---|---|---|---|---|
| D35 | S15-I / S16 **Indic User Query** box | `data/schema.py` `question`, `lang`; demo input | `tests/test_schema.py` | — | M0 |
| D36 | S16 **Normalisation & Script Handling** box | `text/normalize.py` (NFC + IndicNLP normaliser), `text/numerals.py` (Indic digits), `text/offsets.py` (offset maps), `text/sentseg.py` | `tests/text/*` incl. Devanagari, Bengali, Tamil, Telugu digit cases + round-trip | All spans on original offsets; tests pass | M0 |
| D37 | S15-I build per-language corpus / S16 **Vector Index (Indic corpus)** box | `data/loaders/indicqa.py`, `retrieval/index.py` (FAISS `IndexFlatIP`) | `tests/retrieval/test_index.py` | Index built per language; corpus stats | M0 |
| D38 | S15-I / S16 **Multilingual Embedder** box | `retrieval/embed.py` (sentence-transformers; bge-m3 primary, mE5-large comparison) | `tests/retrieval/test_embed.py` (tiny model) | Retrieval metrics per embedder | M0 |
| D39 | S15-I retrieve top-k, **log scores for attribution** / S16 **Top-k Passages** box | `retrieval/retrieve.py` writes `passage_id, text, score, rank, is_gold` | `tests/retrieval/test_retrieve.py` | Scores present in every downstream JSONL (asserted by `tests/test_record_integrity.py`) | M0 |
| D40 | S16 footer: **retrieval scores carried forward** so retrieval vs generation failure are separable | `data/schema.py` (retrieved list mandatory on every record); `analysis/` | `tests/test_record_integrity.py` | Analysis split by gold-retrieved | M0 / M3 |
| D41 | S15-II prompt = query + context **in the source language** / S16 **Prompt Assembly** box | `generation/prompts/hi.txt` (+ one per language) | `tests/generation/test_prompt_assembly.py` | Templates reviewed by a native speaker (sign-off in `DECISIONS.md`) | M0 (hi) / M1 |
| D42 | S15-II **fixed decoding config** / S16 **LLM Generator** box | `generation/backends.py` (HF local 4-bit; OpenAI-compatible), `configs/generation/*.yaml` | `tests/generation/test_backends.py`, `test_cache.py` | Decoding config + model revision in every record; cache hit on re-run | M0 |
| D43 | S16 **Generated Answer** box | record `answer` | `tests/test_schema.py` | — | M0 |
| D44 | S15-III / S16 **Claim Decomposition (atomic claims)** box / S17 step 01 | `detection/decompose.py` (M0: rule-based Hindi clause splitter; M1: LLM-based option) | `tests/detection/test_decompose.py` | Every claim has `start`/`end` into the original answer; claim-boundary accuracy on a small hand-checked set | M0 |
| D45 | S15-III / S16 **Claim–Evidence Alignment (entailment scoring)** / S17 step 02: each claim vs **every** passage → entail/neutral/contradict | `detection/align.py` | `tests/detection/test_align.py` (mock NLI + one slow real-model test) | Per-claim × per-passage score matrix saved | M0 |
| D46 | S15-III aggregate to response verdict keeping span attribution / S16 **Aggregation & Span Attribution** / S17 step 03 | `detection/aggregate.py` | `tests/detection/test_aggregate.py` | Claim verdict + `best_passage_id` + response label | M0 |
| D47 | S16 **SUPPORTED / UNSUPPORTED / CONTRADICTED** outputs | `detection/aggregate.py` verdict enum | `tests/detection/test_aggregate.py` | Per-verdict P/R/F1 | M0 |
| D48 | S17 **Decision threshold: [INSERT THRESHOLD]** | `configs/detection/*.yaml`; `scripts/tune_threshold.py` (validation split only) | `tests/test_no_test_tuning.py` (tuner refuses `split=test`) | Value in `DECISIONS.md` + reported | M0 |
| D49 | S15-IV / S16 **Evaluation vs. annotation** | `eval/*`, `scripts/run_eval.py` | `tests/eval/*` | All metric tables (D76–D90) | M0 / M3 |
| D50 | S15-IV compare per language and per error type | `eval/fairness.py`, per-type breakdown in `eval/detection.py` and `eval/spans.py` | `tests/eval/test_fairness.py` | Per-language × per-type tables | M0 (hi) / M3 |
| D51 | S15-IV relate retrieval quality to hallucination rate | `analysis/`, C4 experiment | `tests/test_c4_conditions.py` | C4 figures | M3 |
| D52 | S15 contract: "does the retrieved evidence say this?" | `detection/` only reads `retrieved[]`, never external knowledge | `tests/detection/test_evidence_only.py` (verdict changes when passages change, not otherwise) | — | M0 |

### 1.5 Detection mechanism worked example (S17)

| ID | Deck item | Module / file | Test | Metric / evidence | M |
|---|---|---|---|---|---|
| D57 | Worked example: claim 1 `योजना राज्य सरकार द्वारा चलाई जाती है` → SUPPORTED; claim 2 `आवेदक की आयु 21 वर्ष होनी चाहिए` → UNSUPPORTED, type NUMERICAL, span offsets on the age clause | `tests/test_worked_example.py` (CL19) | CI version with mocked NLI scores; `@slow` version with the real M0 aligner | Both pass. If the real model fails, that is reported as a finding, **not** patched with special-case code. | M0 |

### 1.6 Dataset & languages (S18)

| ID | Deck item | Module / file | Test | Metric / evidence | M |
|---|---|---|---|---|---|
| D58 | Languages in scope: [INSERT COUNT] of nine | `configs/languages.yaml` enabled set | `tests/test_languages_config.py` | Count generated into the outcome panel | M1 |
| D59 | Document corpus: [INSERT CORPUS SOURCE] | `data/loaders/`, `docs/dataset_card.md` | — | Source + revision listed (IndicQA `d7e91454f5c8`; IndicRAGSuite if used) | M0 |
| D60 | Question–answer pairs: [INSERT DATASET SIZE] | `scripts/corpus_stats.py` | `tests/test_corpus_stats.py` | Auto-generated count | M0 (hi) |
| D61 | Annotated RAG responses: [INSERT NO.] | `data/processed/gold/`, `scripts/corpus_stats.py` | — | Auto-generated count (synthetic and gold reported separately) | M2 |
| D62 | Train / val / test split: [INSERT SPLIT] | `data/split.py` — split **by passage/question group**, seeded | `tests/data/test_split_no_leakage.py` | Split sizes table; zero passage overlap asserted | M0 |
| D63 | Annotator agreement: [INSERT SCORE] | `eval/agreement.py` (Cohen's κ, Krippendorff's α, span-level) | `tests/eval/test_agreement.py` (textbook examples) | κ/α on the ≥20% double-annotated subset | M2 |
| D64 | Taxonomy ENTITY / NUMERICAL / TEMPORAL / LOCATION | `data/schema.py` `SpanType` enum; `type` on every gold + predicted span; `detection/typed/{numeric,temporal,entity}.py` | `tests/test_schema.py`; `tests/detection/typed/*` | Per-type P/R/F1 + span F1 in every results table | M0 (enum + injection) / M1 (typed verifiers) |
| D65 | Response- and span-level labelling follows RAGTruth | `data/schema.py` `labels.response`, `labels.spans` | `tests/test_schema.py` | Schema validated | M0 |
| D66 | Four error categories follow Multi-Hall-SA | `docs/annotation_guidelines.md` | — | Guidelines cite Multi-Hall-SA definitions | M2 |

### 1.7 Hardware & software (S20)

| ID | Deck item | Module / file | Test | Metric / evidence | M |
|---|---|---|---|---|---|
| D67 | GPU ≥16 GB VRAM · [INSERT GPU MODEL] | `docs/RESOURCES.md` measured VRAM; run metadata records `nvidia-smi` name | — | GPU model filled from run metadata (Q2) | M0 |
| D68 | 16 GB RAM min, ~100 GB storage, x86-64 quad-core, local or hosted notebook | `docs/RESOURCES.md` §hardware; run metadata | — | Storage usage measured at M1 | M1 |
| D69 | Python 3.11 | `pyproject.toml` `requires-python = ">=3.11,<3.12"`; CI matrix | GitHub Actions | CI green | M0 |
| D70 | PyTorch · HF Transformers & Datasets | `generation/backends.py`, `detection/align.py`, `data/loaders/` | unit tests | pinned in `pyproject.toml` | M0 |
| D71 | sentence-transformers · FAISS or ChromaDB | `retrieval/embed.py`, `retrieval/index.py` (FAISS chosen; reason in `DECISIONS.md`) | `tests/retrieval/*` | — | M0 |
| D72 | LangChain or LlamaIndex | `retrieval/orchestration.py` — thin LlamaIndex wrapper over our retriever (detector stays independent) | `tests/retrieval/test_orchestration.py` (import + equivalence with plain retriever) | — | M1 |
| D73 | IndicNLP Library | `text/normalize.py`, `text/sentseg.py` | `tests/text/*` | — | M0 |
| D74 | IndicTrans2 | `detection/baselines/b5_translate_test.py`, `data/synth/translate_ragtruth.py`, C4 translated-gold condition | `tests/detection/baselines/test_b5.py` (mock) | — | M1 |
| D75 | scikit-learn · Matplotlib · Jupyter | `eval/detection.py` (sklearn metrics), `analysis/*.ipynb`, `scripts/make_figures.py` | `tests/eval/*` | Figures in `results/figures/` | M0 / M3 |

### 1.8 Evaluation metrics (S21) — every metric per language **and** pooled

| ID | Deck item | Module / file | Test | Metric / evidence | M |
|---|---|---|---|---|---|
| D76 | Precision | `eval/detection.py` | `tests/eval/test_detection_metrics.py` (hand-computed cases) | Claim- and response-level | M0 |
| D77 | Recall | `eval/detection.py` | same | | M0 |
| D78 | F1 | `eval/detection.py` | same | | M0 |
| D79 | Balanced accuracy | `eval/detection.py` | same | | M0 |
| D80 | AUROC | `eval/detection.py` (uses continuous hallucination score, not the thresholded label) | same | | M0 |
| D81 | Span-level F1 | `eval/spans.py` (char offsets on original text) | `tests/eval/test_spans.py` | | M0 |
| D82 | Token IoU | `eval/spans.py` (tokens from `text/sentseg.py` tokenizer, mapped to original offsets) | same | | M0 |
| D83 | Exact-span match | `eval/spans.py` | same | | M0 |
| D84 | Recall@k | `eval/retrieval.py` | `tests/eval/test_retrieval_metrics.py` | k ∈ {1, 3, 5, 10} | M0 |
| D85 | nDCG@k | `eval/retrieval.py` | same | | M0 |
| D86 | MRR | `eval/retrieval.py` | same | | M0 |
| D87 | Macro-F1 over languages | `eval/fairness.py` | `tests/eval/test_fairness.py` | | M0 (works with n=1) / M3 |
| D88 | Per-language breakdown | `eval/fairness.py`, `scripts/make_tables.py` | same | | M0 |
| D89 | Worst-language F1 | `eval/fairness.py` | same | | M3 |
| D90 | "All metrics reported per language as well as pooled" | `scripts/make_tables.py` refuses to emit a table without per-language rows | `tests/test_make_tables.py` | | M0 |

### 1.9 Expected outcome (S22), limitations & future scope (S23)

| ID | Deck item | Module / file | Test | Metric / evidence | M |
|---|---|---|---|---|---|
| D91 | Outcome: working pipeline + attached detector | `scripts/run_pipeline.py`, `demo/app.py` | `tests/test_pipeline_smoke.py` | Gradio demo (local M0; HF Space M4) | M0 |
| D92 | Outcome: span-annotated Indic RAG resource | HF dataset `IndicRAG-Hallu` + datasheet | — | Released with licence | M4 |
| D93 | Outcome: claim-level verdicts | `detection/` | worked example | | M0 |
| D94 | Outcome: diagnostic retrieval-vs-generation analysis | C4 (`configs/experiments/c4_*.yaml`, `analysis/`) | `tests/test_c4_conditions.py` | | M3 |
| D95 | Results panel: MACRO-F1, SPAN F1, RECALL@K, BASELINE Δ, LANGUAGES | `scripts/make_outcome_panel.py` (reads `results/` only) | `tests/test_outcome_panel.py` (fails on missing inputs, never defaults) | Panel image/table generated | M0 (pilot) / M3 |
| D96 | "No experimental numbers claimed" — every slot filled from measured runs | `scripts/make_tables.py`, `scripts/make_outcome_panel.py` | same | No number typed by hand anywhere | all |
| D97 | Limitation: coverage bounded by corpora | `docs/dataset_card.md` limitations | — | Stated in paper | M4 |
| D98 | Limitation: manual annotation caps size | same | — | | M4 |
| D99 | Limitation: Indic entailment models are imperfect | same; C3 vs off-the-shelf comparison quantifies it | — | Aligner error analysis | M3 |
| D100 | Limitation: code-mixed/transliterated input out of scope v1 | same | — | | M4 |
| D101 | Limitation: findings domain-bound | same | — | | M4 |
| D102 | Future: code-mixed Hinglish / romanised | `stretch/codemixed/` | — | only after core | stretch |
| D103 | Future: detection → correction (regenerate unsupported span) | `stretch/correction/` | — | | stretch |
| D104 | Future: speech input | `stretch/speech/` | — | | stretch |
| D105 | Future: remaining Indic languages & domains | `configs/languages.yaml` | — | | stretch |
| D106 | Future: open benchmark release | HF dataset (D92) | — | | M4 |

### 1.10 Beyond the deck (CLAUDE.md §4) — tracked here so nothing is lost

| ID | Item | Module / file | M |
|---|---|---|---|
| X1 | B1 sentence-level multilingual NLI | `detection/baselines/b1_sentence_nli.py` | **M0** |
| X2 | B2 LettuceDetect v2 mmBERT zero-shot | `detection/baselines/b2_lettucedetect.py` | M1 |
| X3 | B3 LLM-as-judge (open model run locally or on the T4, no paid API) | `detection/baselines/b3_llm_judge.py` | M1 |
| X4 | B4 LM-Polygraph entropy / CCP | `detection/baselines/b4_uncertainty.py` | M1 |
| X5 | B5 translate-test via IndicTrans2 | `detection/baselines/b5_translate_test.py` | M1 |
| X6 | C2 typed verifiers (numeric, temporal, entity/location) | `detection/typed/` | M1 |
| X7 | C3 Indic-tuned aligner | `detection/align.py` + `scripts/train_aligner.py` | M1 |
| X8 | C4 controlled retrieval conditions | `configs/experiments/c4_*.yaml` | M3 |
| X9 | C5 cross-lingual transfer | `configs/experiments/c5_*.yaml` | M3 |
| X10 | C6 native vs translate-test | ours vs X5 | M3 |
| X11 | Synthetic data: typed injection, translated RAGTruth, natural under degraded retrieval | `data/synth/` | M0 (injection, hi) / M1 |
| X12 | Annotation guidelines + tool | `docs/annotation_guidelines.md`, `annotate/app.py` | M2 |
| X13 | Paper skeleton (ACL LaTeX) | `paper/` | M4 |

---

## 2. M0 plan — Hindi vertical slice (deadline **Thu 10 Dec 2026**)

**Goal:** one command runs Hindi IndicQA → retrieval → generation → decompose → NLI → aggregate → metrics, the slide-17 test passes, B1 vs ours is measured on a small synthetic Hindi set, and a local Gradio demo works. README results are labelled **"pilot — synthetic Hindi data only"**.

**M0 design choices** (each gets a `DECISIONS.md` entry when made; open questions in §5 can change them):
- Corpus = unique IndicQA Hindi contexts. Questions = IndicQA Hindi questions with a gold passage. Unanswerable items are kept, flagged, and excluded from retrieval metrics.
- Split by passage, 60/20/20, seeded.
- Embedders: `BAAI/bge-m3` (primary) vs `intfloat/multilingual-e5-large`. Index: FAISS exact inner product (the corpus is small).
- Generator: **`qwen3:4b` via Ollama** (Apache-2.0, 2.5 GB, fits the 4 GB laptop GPU), called through the OpenAI-compatible backend. Thinking mode is off; decoding is fixed. 7–10B generators move to M1 on a free T4.
- Environment: **Docker** for development and tests (no local Python install). Ollama runs on the host; the container reaches it at `host.docker.internal:11434`. Heavy runs (embedding, NLI at scale) go on the maintainer's free Colab/Kaggle T4.
- **Free and open-source only. No paid APIs.** LLM-as-judge (B3) and auto-labelling use open models run locally.
- Decomposition: rule-based Hindi clause splitter (sentence split + coordinators such as और / तथा / लेकिन / परंतु / किंतु and `;`), with offsets.
- Aligner + B1: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`.
- Pilot labelled set (`source=injected`): sentences from the gold passage are supported by construction; typed injections (NUMERICAL, TEMPORAL, ENTITY, LOCATION) add known spans. Known bias: copied sentences are easy to entail, so results are a pilot only.

**Proposed owners by track** (confirm in Q10): **Y** = Yousuf (infra + detection core) · **M** = Mukta (text processing + synthetic data) · **S** = Sneha (data + retrieval) · **K** = Srushti Kondhalkar (generation + evaluation + demo).

Each task is half a day or less. "Dep" = must be done first. Dates are target finish dates.

### Week 0 — today
| # | Task | Owner | Dep | Target |
|---|---|---|---|---|
| T00 | Request HF access to the auto-gated resources (IndicNER for M0; IndicTrans2 for M1); store the HF token as a Colab/Kaggle secret or a git-ignored `.env`, never in the repo; `ollama pull qwen3:4b` | Y | — | 20 Nov |

### Week 1 — foundations, data, retrieval (20–26 Nov)
| # | Task | Owner | Dep | Target |
|---|---|---|---|---|
| T01 | Docker dev environment (`Dockerfile` on `python:3.11-slim`, CPU torch, `docker compose run dev pytest`); `pyproject.toml` with pinned deps; empty `src/indichallu/` package; `pytest` runs in the container | Y | — | 20 Nov |
| T02 | GitHub Actions CI (CPU `pytest`, ruff); branch protection on `main` | Y | T01 | 21 Nov |
| T03 | `docs/PROGRESS.md`, `docs/DECISIONS.md` skeletons; `configs/languages.yaml` (9 languages, only `hi` enabled) | Y | T01 | 21 Nov |
| T04 | `data/schema.py` (record of CLAUDE.md §6: `SpanType` enum, verdict enum, JSONL read/write, validation) + tests | Y | T01 | 23 Nov |
| T05 | Run utilities: `results/runs/<run_id>/` with resolved config, git hash, model revisions, GPU name; global seeding helper + tests | Y | T01 | 23 Nov |
| T06 | `text/offsets.py`: composable offset map (normalised ↔ original) + round-trip tests | M | T01 | 21 Nov |
| T07 | `text/normalize.py`: NFC + IndicNLP normaliser with offset maps (char-level alignment since IndicNLP returns none) + Devanagari tests | M | T06 | 24 Nov |
| T08 | `text/numerals.py`: digit tables for all 8 Indic digit blocks used by the 9 languages → ASCII, with offsets; tests for Devanagari, Bengali, Tamil, Telugu | M | T06 | 25 Nov |
| T09 | `text/sentseg.py`: Hindi sentence segmentation + tokeniser with original offsets + tests | M | T06 | 26 Nov |
| T10 | `scripts/download_indicqa.py`: fetch `data/indicqa.hi.json` at revision `d7e91454f5c8`, verify checksum | S | T01 | 21 Nov |
| T11 | `data/loaders/indicqa.py`: dedupe contexts → passages with stable IDs; QA pairs with gold passage ID; unanswerable flag; tests on a tiny fixture | S | T04, T10 | 23 Nov |
| T12 | `data/split.py`: seeded split by passage + leakage test | S | T11 | 24 Nov |
| T13 | `scripts/corpus_stats.py` → auto-generated table in `docs/dataset_card.md` | S | T11 | 24 Nov |
| T14 | `retrieval/embed.py`: sentence-transformers wrapper, model-specific prefixes, embedding cache; test with a tiny model | S | T01 | 25 Nov |
| T15 | `retrieval/index.py` (FAISS build/save/load) + `retrieval/retrieve.py` (top-k, scores, ranks, `is_gold`) + tests | S | T14 | 26 Nov |
| T16 | `eval/retrieval.py`: Recall@k, nDCG@k, MRR + hand-computed tests | K | T04 | 24 Nov |
| T17 | `eval/detection.py`: P, R, F1, balanced accuracy, AUROC (claim and response level, per type) + tests | K | T04 | 25 Nov |
| T18 | `eval/spans.py`: span F1, token IoU, exact match on original offsets, per type + tests | K | T09 | 26 Nov |

### Week 2 — generation and detection (27 Nov – 3 Dec)
| # | Task | Owner | Dep | Target |
|---|---|---|---|---|
| T19 | **T4 run (Colab/Kaggle):** embed the Hindi corpus with bge-m3 and mE5-large; retrieval eval on val → `results/runs/`; pick embedder → `DECISIONS.md` | S | T12–T16 | 27 Nov |
| T20 | `eval/fairness.py`: per-language, pooled, macro-F1, worst-language (works with one language) + tests | K | T17 | 27 Nov |
| T21 | `generation/backends.py`: backend interface, OpenAI-compatible backend (Ollama), HF 4-bit backend (for the T4 in M1), mock backend; JSONL cache keyed by (model, revision, prompt hash, decoding) + tests | K | T05 | 28 Nov |
| T22 | `generation/prompts/hi.txt` + prompt assembly (query + top-k context in Hindi); native-speaker review of the template | K + Yousuf (Hindi review) | T15 | 28 Nov |
| T23 | **Laptop:** smoke-test `qwen3:4b` through Ollama from the container; record the Ollama manifest digest, measured VRAM and tokens/s in `RESOURCES.md`; `configs/generation/m0.yaml` (fixed decoding, thinking off) | K | T21 | 30 Nov |
| T24 | `detection/decompose.py`: rule-based Hindi clause splitter with offsets; tests incl. the slide-17 sentence | Y | T09 | 30 Nov |
| T25 | `detection/align.py`: claim × every passage NLI scores (label-order check against model config); score matrix; mock test + `@slow` real test | Y | T04 | 1 Dec |
| T26 | `detection/aggregate.py`: verdict rules (SUPPORTED / UNSUPPORTED / CONTRADICTED), thresholds from config, `best_passage_id`, response label + tests | Y | T25 | 1 Dec |
| T27 | `tests/test_worked_example.py`: slide-17 regression (CI mock version + `@slow` real-model version) | Y | T24–T26 | 2 Dec |
| T28 | `data/synth/inject.py` NUMERICAL + TEMPORAL: change Indic/ASCII numbers and Hindi month/day names; record the span and type; tests | M | T08 | 30 Nov |
| T29 | `data/synth/inject.py` ENTITY + LOCATION: swap IndicNER entities with same-type entities from other passages; tests | M | T00, T28 | 2 Dec |
| T30 | `scripts/build_pilot_set.py`: supported gold-passage sentences + injected variants → Hindi pilot set (val/test follow the passage split) | M | T12, T29 | 3 Dec |
| T31 | **Laptop run (Ollama):** generate answers for Hindi val+test questions (cached; resumable if interrupted) | K | T22, T23 | 2 Dec |
| T32 | `scripts/run_pipeline.py`: config → retrieve → generate → detect → JSONL, one command; smoke test with mocks | Y | T15, T21, T26 | 3 Dec |

### Week 3 — evaluation, demo, write-up (4–10 Dec)
| # | Task | Owner | Dep | Target |
|---|---|---|---|---|
| T33 | `detection/baselines/b1_sentence_nli.py` (sentence-level, no decomposition) + test | Y | T25 | 4 Dec |
| T34 | `scripts/tune_threshold.py` on **val only** (guard test); value → `DECISIONS.md` | Y | T30, T33 | 4 Dec |
| T35 | **Run:** B1 vs ours on the pilot test set → `scripts/run_eval.py` → `results/runs/` | K | T34 | 5 Dec |
| T36 | Hindi speaker spot-checks ~50 natural generations from T31 for obvious hallucinations (sanity check only; not gold, not reported as a headline number) | Yousuf | T31 | 5 Dec |
| T37 | `scripts/make_tables.py` + `scripts/make_outcome_panel.py` (read `results/` only; fail loudly on missing inputs) + tests | K | T35 | 7 Dec |
| T38 | `demo/app.py`: Gradio, Hindi question → answer with SUPPORTED spans linked to passages, UNSUPPORTED/CONTRADICTED highlighted | S | T32 | 7 Dec |
| T39 | README: slide-16 architecture diagram, one-command quickstart, auto-generated results table labelled "pilot — synthetic Hindi data only" | Y | T37 | 8 Dec |
| T40 | `docs/related_work.md` first pass: verify the five deck citations + CLAUDE.md §4.1 table; draft new wording for slide 13 | M | — | 8 Dec |
| T41 | `docs/dataset_card.md` (Hindi pilot), `PROGRESS.md` update, fresh-clone reproducibility check | S | T37 | 9 Dec |
| T42 | Buffer / fixes; optional HF Space deploy of the demo | all | — | 10 Dec |

**Critical path:** T01 → T04 → T11 → T15 → T19 · T21 → T23 → T31 · T25 → T26 → T34 → T35 → T37 → T39.
**Biggest risks:** generation speed on the 4 GB laptop (mitigation: cache, resumable runs, `qwen2.5:1.5b` fallback, or run Ollama on the T4); free-tier session limits on Colab/Kaggle (mitigation: checkpoint embeddings to disk); only one Hindi reviewer (Yousuf) for T22/T36.

---

## 3. M1 plan — typed verifiers, more languages, baselines (target **Sun 31 Jan 2027**)

**Goal (CLAUDE.md §7):** synthetic labelled data for every in-scope language, baselines B2–B5, the Indic-tuned aligner (C3) and the typed verifiers (C2), each evaluated **per language and pooled on val**. M1 reports no test numbers; test is used once, in M3.

**What M0 taught us** (PROGRESS 2026-11-21) and how M1 answers it:

| M0 finding | M1 response |
|---|---|
| Off-the-shelf NLI entails most injected number/date swaps (low recall) | Typed NUMERICAL/TEMPORAL verifiers first (C2, Phase A) |
| The span-F1 threshold objective is degenerate for claim-level predictions vs number-level gold spans | Evaluation protocol fixed **before** any M1 evaluation (T43, D-028) |
| Refusals ("the passages don't contain the answer") are scored as entailed claims | Refusal handling in schema, detector and metrics (T45) |
| The pilot's faithful answers are copied passage sentences (easy) | Natural generations and translated RAGTruth join the synthetic data (Phase B, D) |
| CPU is too slow for NLI and generation at scale | All heavy runs on the free Kaggle/Colab T4 through notebooks (D-026); the laptop is for development and tests |
| The demo takes ~10 min per check on CPU | HF Space on free hardware or a GPU notebook demo (T42 carried over) |

### Evaluation protocol for M1 (decide in T43, before any M1 number exists)

Proposed; becomes D-028 once Yousuf approves it:
- **Development data = val only.** The M0 test records were inspected while diagnosing D-027, so test stays untouched until the final M3 run, and the paper says so.
- **Threshold objective = macro-F1 over languages of response-level F1 on val.** Span metrics (character F1, token IoU, exact match) are still reported for every system but are never optimised. AUROC is reported next to every F1 because it needs no threshold.
- **Predicted spans:** a typed verifier's mismatch gives a token-level span (the number or date); otherwise the span is the whole claim. Gold spans stay as annotated.
- **Every table:** per language and pooled, macro and worst-language F1 (CLAUDE.md §3 row 16), plus the flag-everything reference.
- **Refusals are their own class:** a refusal is *correct* if the gold passage was not shown and *incorrect* otherwise. Refusals are excluded from claim-level metrics and reported separately.

### Phase A — Hindi depth (25 Nov – 8 Dec)
| # | Task | Owner | Dep | Target |
|---|---|---|---|---|
| T43 | Write D-028 (protocol above) after Yousuf approves it; `eval/` reads the objective from config | Y | — | 26 Nov |
| T44 | Generic GPU runner notebook: any `scripts/*.py` command on Kaggle/Colab, results zipped back (generalises `m0_pilot_gpu.ipynb`) | Y | — | 27 Nov |
| T45 | Refusal handling: per-language refusal patterns, `Record` field or label, detector skips refusals, metrics report correct/incorrect refusals | K | T43 | 29 Nov |
| T46 | `detection/typed/numeric.py`: digits in every in-scope script, Hindi number words (e.g. पाँच), lakh/crore and separators, compared on normalised values with offset maps | M | — | 1 Dec |
| T47 | `detection/typed/temporal.py`: years, eras, Hindi month names and date formats; month/day names for the other languages from Unicode CLDR (source recorded in RESOURCES.md) | M | — | 3 Dec |
| T48 | Fusion rule: a typed mismatch against the best-matching evidence window overrides NLI entailment, with the mismatch as a token-level span (D-029) | Y | T46, T47 | 5 Dec |
| T49 | T29 carried over: ENTITY/LOCATION injection with IndicNER (needs HF access), transliteration-aware matching for the entity verifier | S | IndicNER access | 6 Dec |
| T50 | Hindi val evaluation on GPU: ours (+C2) vs ours vs B1, with the T43 protocol | K | T44, T48 | 8 Dec |

### Phase B — all in-scope languages (9 – 22 Dec)
| # | Task | Owner | Dep | Target |
|---|---|---|---|---|
| T51 | Final language set from corpus availability and who can review prompts (Q6); `configs/languages.yaml` updated | Y | Q6 | 9 Dec |
| T52 | Per-language prompt templates (`generation/prompts/<lang>.txt`), each reviewed by a native speaker | K + reviewers | T51 | 12 Dec |
| T53 | Per-language decomposition rules (conjunctions, sentence splitting); fall back to sentence claims where no speaker can check them | M | T51 | 15 Dec |
| T54 | Typed injection for every language (digits, months, entities where IndicNER covers the language) with per-language pilot sets | M | T49, T53 | 17 Dec |
| T55 | Embedder choice on val: `bge-m3` vs `multilingual-e5-large` (each ~2.2 GB; needs approval, Q15), all languages, on GPU | S | T44, Q15 | 19 Dec |
| T56 | Natural generations on GPU for val + test questions of every language: the M0 generator first, then the M1 generators once Q14 is answered (the paper needs at least three); cached JSONL | K | T44, T52, Q14 | 22 Dec |

### Phase C — baselines (23 Dec – 12 Jan)
| # | Task | Owner | Dep | Target |
|---|---|---|---|---|
| T57 | B2: `KRLabsOrg/lettucedect-v2-mmbert-base` zero-shot (Apache-2.0); token spans mapped to original offsets | S | T44 | 29 Dec |
| T58 | B3: LLM-as-judge with an open model on the T4 (no paid API); prompt fixed before evaluation | K | T44, Q14 | 2 Jan |
| T59 | B4: LM-Polygraph token entropy and CCP from the generator's logits (needs the generator in HF transformers, not Ollama: D-030 on the Q4-vs-bf16 difference) | Y | T44 | 6 Jan |
| T60 | B5: translate-test with IndicTrans2 (indic→en, needs HF access), an English detector, and span projection back to the source offsets | S | IndicTrans2 access | 12 Jan |

### Phase D — data and aligner (13 – 31 Jan)
| # | Task | Owner | Dep | Target |
|---|---|---|---|---|
| T61 | Translated RAGTruth (`wandb/RAGTruth-processed`, upstream licence checked first): IndicTrans2 en→indic with span markers; drop items whose markers break | M | IndicTrans2 access | 18 Jan |
| T62 | Natural hallucinations under degraded retrieval (C4 conditions, small scale), auto-labelled by the B3 judge; Yousuf checks a sample to measure the judge's accuracy | K | T56, T58 | 22 Jan |
| T63 | C3: fine-tune the aligner on IndicXNLI (`Divyanshu/indicxnli`, CC0-1.0) + synthetic claim–evidence pairs from the **train** split; candidates `google/muril-base-cased`, `ai4bharat/IndicBERTv2-MLM-only`, `jhu-clsp/mmBERT-base`, `FacebookAI/xlm-roberta-large` | Y | T54, T61 | 28 Jan |
| T64 | M1 val results per language and pooled for ours, ours+C2, ours+C3, B1–B5; README block and outcome panel regenerated, labelled "val, synthetic" | K | all above | 31 Jan |
| T42 | HF Space (carried over from M0) on free hardware, or a GPU notebook demo if CPU Spaces are too slow | Y | — | 31 Jan |

**M1 is done when:** every in-scope language has a synthetic labelled set; B1–B5, ours, ours+C2 and ours+C3 have val results per language and pooled under D-028; refusals are handled; and every number is generated from `results/`.

**Downloads and access M1 needs** (all in RESOURCES.md; nothing above 10 GB without approval):
- HF access, auto-approved after accepting terms: `ai4bharat/IndicNER` (T49), `ai4bharat/indictrans2-*` (T60, T61).
- Under 10 GB, not yet approved: `BAAI/bge-m3` and `intfloat/multilingual-e5-large` (T55).
- Over 10 GB, each needs explicit approval and only runs on Kaggle/Colab: any 7–10B generator (Q14) and `FacebookAI/xlm-roberta-large` if the full repo is fetched (safetensors only is smaller).

---

## 4. Resources

See [RESOURCES.md](RESOURCES.md): exact HF IDs, revisions, licences, gating, sizes and whether each fits 16 GB in 4-bit, all checked 2026-11-19.

---

## 5. Open questions

Question numbers stay fixed so references elsewhere in this file still work.

### Answered (2026-11-19)

| # | Question | Answer |
|---|---|---|
| Q1 | Install Python 3.11 locally, or work in Docker/Colab? | **Docker.** No local Python install (T01). |
| Q2 | Which GPU, and whose account? | **Both:** the laptop's GTX 1650 Ti (4 GB) for development, Ollama and the demo, plus free Colab/Kaggle T4 (16 GB) on the maintainer's account for heavy runs. |
| Q3 | M0 generator | **`qwen3:4b` via Ollama** (Apache-2.0, 2.5 GB). `qwen2.5vl:3b` was considered and rejected: Qwen Research (non-commercial) licence. |
| Q5 | Budget for paid APIs? | **None.** Everything must be free and open source. B3 and auto-labelling use open models run locally. |
| Q6 (part) | Who reviews the Hindi prompts? | **Yousuf**, for T22 and T36. |
| Q9 | OK to show the team's names publicly? | **Yes.** |

### Still open

| # | Question | Why it matters | Blocks |
|---|---|---|---|
| Q4 | CLAUDE.md §4.4 lists `MBZUAI/Llama-3-Nanda-10B-Chat`. On HF it redirects to `IFM/Llama-3-Nanda-10B-Chat`. OK to update CLAUDE.md to the `IFM/` ID? | Configs must pin the real ID | M1 |
| Q6 | Which languages do Mukta, Sneha and Srushti speak natively? The gold set needs **two annotators per language** for the ≥20% double-annotated subset. If Yousuf is the only Hindi speaker, Hindi agreement can't be measured within the team. | Gold-set languages (C1), κ/α (D63) | M2 |
| Q7 | Guide's name, and faculty requirements and date for the Stage II review | Deck placeholders; M3 timing | — |
| Q8 | Target paper venue and deadline | Sets M4 | M4 |
| Q10 | Confirm or change the proposed owner per track (Y / M / S / K) in §2 | Task assignment | Week 1 |
| Q11 | M0 corpus: IndicQA contexts only (small, so retrieval is easy), or also add Hindi Wikipedia distractors from `ai4bharat/Indic-Rag-Suite` (download size per language not yet measured)? | Realism of retrieval metrics | T11 |
| Q12 | Licences: `ai4bharat/MSMARCO-XI` (defers to MS MARCO's non-commercial terms), `ai4bharat/IndicXNLI-Translated` and `ai4bharat/IndicBERTv2-MLM-only-XNLI` (none declared), `google/muril-large-cased` (none on HF). Use them for research with a note, or skip them? | Release licence; "free and open source" rule | M1 |
| Q13 | Turn on branch protection for `main` (require a PR + CODEOWNERS review)? CODEOWNERS alone only *requests* review. | Enforces the "never commit to main" rule | T02 |
| Q14 | Llama 3.x (incl. Nanda-10B) and Gemma 3 are free open-weight models, but their licences aren't OSI open source. Are they allowed as M1 generators, or Apache/MIT models only (e.g. Qwen3-8B, Qwen2.5-7B)? | The paper needs ≥3 generators; Nanda is the Hindi-specialised one | M1 |
| Q15 | Approve downloading `BAAI/bge-m3` (~2.3 GB) and `intfloat/multilingual-e5-large` (~2.2 GB) for the embedder choice? Both are under 10 GB but weren't approved in M0, so M0 used `multilingual-e5-small` as a stand-in. | Retrieval quality for every M1 result | T55 |
| Q16 | Approve the M1 evaluation protocol in §3 (val only; threshold objective = macro-F1 over languages of response-level F1; span metrics reported, not optimised; refusals as their own class)? It must be fixed before any M1 number exists. | Validity of every M1 result | T43 |
