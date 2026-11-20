# Resources — models, datasets and tools

Every model and dataset this project plans to use. **Checked against the Hugging Face Hub API on 2026-11-19.** Revisions are the `main` commit SHA at that date (first 12 characters); configs must pin these exact revisions.

**Column guide**
- **Gated:** `no` = open download. `auto` = you must accept terms on HF, approval is automatic. `manual` = the repo owner approves requests by hand (can take days).
- **Download:** size of the weight files we would actually fetch (or the total repo size where marked *repo*). Repo totals include duplicate formats (ONNX, fp16 copies, etc.) and overstate what we need.
- **16 GB / 4-bit:** for generators, an **estimate** of weight memory in 4-bit NF4 (≈ 0.55–0.65 GB per billion parameters) plus headroom for a 4k-token KV cache. Encoders run in fp16/fp32 and don't need quantisation. All VRAM figures are to be **measured** in M0 task T21 and replaced here.
- **M0?** = used in the Hindi vertical slice.
- ⚠ = needs your attention (see Open questions in [PLAN.md](PLAN.md#4-open-questions)).

---

## 1. Datasets

| HF ID | Role | Revision | Licence | Gated | Size | M0? | Notes |
|---|---|---|---|---|---|---|---|
| `ai4bharat/IndicQA` | Hindi (and 10 other langs) contexts + QA pairs → corpus and questions | `d7e91454f5c8` (full hash, per-file size and SHA-256 pinned in `configs/data/indicqa.yaml`) | CC-BY-4.0 | no | repo 0.38 GB; `data/indicqa.hi.json` = 2.67 MB | **yes** | Repo still ships a loading script (`IndicQA.py`). Download `data/indicqa.<lang>.json` directly. Files present for as, bn, gu, hi, kn, ml, mr, or, pa, ta, te, so all nine in-scope languages are covered. |
| `ai4bharat/Indic-Rag-Suite` | Wikipedia-derived RAG triplets: extra corpus/distractors; training pairs (M1) | `72117d48b08e` | MIT | no | repo 8.82 GB (per-language subset smaller, not yet measured) | optional | Use per-language files only. |
| `ai4bharat/MSMARCO-XI` | IndicMSMarco: retrieval evaluation (M1/M3) | `bf5cdc1f26e5` | ⚠ none declared; card says "refer to the original MS MARCO licence terms" | no | repo **55.6 GB** | no | MS MARCO's terms are research/non-commercial. Fetch the Hindi split only. Never the whole repo. |
| `Divyanshu/indicxnli` | IndicXNLI: NLI training data for the aligner (C3) | `7092c27872e9` | CC0-1.0 | no | repo 9.12 GB (*repo*) | no | Original IndicXNLI (11 langs). |
| `ai4bharat/IndicXNLI-Translated` | Alternative IndicXNLI (IndicTrans2 translations), val/test only | `4e7b9dd02aae` | ⚠ none declared | no | repo 0.31 GB | no | Use only if the licence is clarified. |
| `wandb/RAGTruth-processed` | RAGTruth (English) — source for translated synthetic data | `eb4f4b9d1b68` | MIT (per card) | no | 0.03 GB | no | Verify against upstream ParticleMedia/RAGTruth (GitHub) before release. |
| `s-nlp/PsiloQA` | Related work / possible extra Hindi eval | `375c3321b833` | CC-BY-4.0 | no | 0.10 GB | no | Not RAG; Hindi is its only Indic language. |
| `Helsinki-NLP/mu-shroom` | Related work / possible extra Hindi eval | `3f99df059b00` | CC-BY-4.0 | no | 0.02 GB | no | Closed-book QA, not RAG. |
| `sambhashana/BHRAM-IL` | Related work (must be cited; changes slide 13) | `95729e2ff831` | ⚠ not visible without accepting terms | **auto** | not visible without access | no | Request access to read the card + licence. |
| `KRLabsOrg/ragtruth-de-translated` | Reference only: example of the translate-RAGTruth recipe | `63267a938120` | MIT | no | 0.04 GB | no | Not downloaded; format reference. |

## 2. Models

### 2.1 Retrieval embedders

| HF ID | Revision | Licence | Gated | Params / download | 16 GB? | M0? | Notes |
|---|---|---|---|---|---|---|---|
| `BAAI/bge-m3` | `5617a9f61b02` | MIT | no | ~568M; `pytorch_model.bin` 2.27 GB | yes (fp16) | **yes — primary** | Dense mode via sentence-transformers. No query prefix. Supports 8,192 tokens; capped at 2,048. Full revisions in `configs/retrieval/embedders.yaml`. |
| `intfloat/multilingual-e5-large` | `3d7cfbdacd47` | MIT | no | 560M; ~2.2 GB | yes | **yes — comparison** | Needs `query: ` / `passage: ` prefixes. **Max 512 tokens: long passages are truncated.** |
| `intfloat/multilingual-e5-base` | `d12875059715` | MIT | no | 278M; ~1.1 GB | yes (CPU ok) | CI/dev | Fast option for laptop runs. |
| `intfloat/multilingual-e5-small` | `614241f622f5` | MIT | no | 118M; ~0.47 GB | yes (CPU ok) | dev | Local real-model test (`pytest -m slow`); not used for reported results. Max 512 tokens. |
| `intfloat/multilingual-e5-large-instruct` | `274baa43b0e1` | MIT | no | 560M | yes | no | Optional M1 comparison. |

### 2.2 Generators

**Policy (DECISIONS.md D-004):** free and open-source only, no paid APIs. M0 runs the generator locally through **Ollama**, using its OpenAI-compatible API. That way it fits the laptop's 4 GB GTX 1650 Ti. Larger models run in 4-bit via HF Transformers on a free Colab/Kaggle T4 (16 GB).

**M0 generator (Ollama)**

| Ollama tag | HF source model | HF revision | Licence | Gated | Params / download | Fits 4 GB laptop? | Notes |
|---|---|---|---|---|---|---|---|
| `qwen3:4b` | `Qwen/Qwen3-4B` | `1cfa9a720891` | Apache-2.0 | no | 4.02B; Ollama model layer **2.50 GB** (Q4) | yes (est.; measure at T23) | **M0 generator.** Card states 100+ languages. Hybrid "thinking" model: thinking is switched **off** for the fixed decoding config. Pin the Ollama manifest digest (recorded at T23) in run metadata alongside the HF revision. |
| `qwen2.5vl:3b` | `Qwen/Qwen2.5-VL-3B-Instruct` | `66285546d2b8` | ⚠ **Qwen Research Licence** (non-commercial) | no | 3.75B; Ollama layer 3.20 GB | yes | **Rejected for M0**: the licence isn't open source, and its vision encoder goes unused for text-only RAG. |
| `qwen2.5:1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4c` | Apache-2.0 | no | 1.54B; Ollama layer 0.99 GB | yes | Fast fallback for development and the CPU-only demo. |
| `qwen2.5vl:7b` / `qwen2.5:7b` | `Qwen/Qwen2.5-VL-7B-Instruct` / `Qwen/Qwen2.5-7B-Instruct` | `cc594898137f` / `a09a35458c70` | Apache-2.0 | no | Ollama layers 5.97 GB / 4.68 GB | no (partial CPU offload, slow) | Run on the T4 instead. |

**M1 generators (HF Transformers, 4-bit, free Colab/Kaggle T4 16 GB)** — the paper needs at least three.

| HF ID | Revision | Licence | Gated | Params / download | 16 GB / 4-bit (est.) | M0? | Notes |
|---|---|---|---|---|---|---|---|
| `Qwen/Qwen3-8B` | `b968826d9c46` | Apache-2.0 | no | 8.19B; ~16.4 GB safetensors | ~5.5–6.5 GB → fits | M1 | Open-source option at the 8B size. Download > 10 GB, so needs approval. |
| `IFM/Llama-3-Nanda-10B-Chat` | `96c2ec3845f8` | Llama 3 Community | **auto** | ~10B; ⚠ **39.9 GB** of fp32 `.bin` shards (no safetensors) | ~6–7 GB → fits | M1 | Hindi-specialised. ⚠ CLAUDE.md says `MBZUAI/Llama-3-Nanda-10B-Chat`. That ID now **redirects** to `IFM/…`, so configs should use `IFM/`. Download is > 10 GB, so needs your approval. |
| `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c70` | Apache-2.0 | no | 7.62B; ~15.2 GB safetensors | ~5–6 GB → fits | M1 | Apache-2.0, ungated. Download > 10 GB, so needs your approval. |
| `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a1` | Llama 3.1 Community | **manual** | 8.03B; ~16 GB | ~5.5–6.5 GB → fits | M1 | Request access now; manual approval can be slow. |
| `google/gemma-3-12b-it` | `96b6f1eccf38` | Gemma terms | **manual** | 12.19B; ~24 GB (*repo* 111 GB) | ~8–9.5 GB → fits, tight with long context | M1 | Multimodal checkpoint; text-only use. |
| `google/gemma-3-4b-it` | `093f9f388b31` | Gemma terms | **manual** | 4.30B; ~8.6 GB | ~3–4 GB → fits easily | fallback | Small fallback if VRAM is tight. |

### 2.3 Claim–evidence alignment (NLI) and aligner backbones

| HF ID | Revision | Licence | Gated | Params / download | 16 GB? | M0? | Notes |
|---|---|---|---|---|---|---|---|
| `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | `b5113eb38ab6` | MIT | no | 279M; ~1.1 GB | yes (CPU ok) | **yes — aligner + B1** | Trained on XNLI + multilingual-NLI-26lang, which includes **hi, bn, mr, ta** but **not** te, kn, ml, gu, pa. |
| `joeddav/xlm-roberta-large-xnli` | `b227ee8435ce` | MIT | no | 561M | yes | M1 | Second off-the-shelf NLI baseline. |
| `ai4bharat/IndicBERTv2-MLM-only-XNLI` | `feea7aac7e12` | ⚠ none declared; card is empty | no | ~278M | yes | M1 | Possibly already IndicXNLI-tuned. Licence must be clarified before use. |
| `google/muril-base-cased` | `afd9f36c7923` | Apache-2.0 | no | ~237M; 0.95 GB | yes | M1 (C3) | Aligner fine-tuning candidate. |
| `google/muril-large-cased` | `ace319f0d175` | ⚠ none declared on HF | no | repo 6.08 GB | yes | M1 (C3) | Check licence (base model is Apache-2.0). |
| `ai4bharat/IndicBERTv2-MLM-only` | `8598f13fe524` | MIT | no | ~278M; 1.12 GB | yes | M1 (C3) | Aligner candidate. |
| `jhu-clsp/mmBERT-base` | `c5955035435e` | MIT | no | ~307M; repo 3.71 GB | yes | M1 (C3) | Aligner candidate; backbone of LettuceDetect v2. |
| `FacebookAI/xlm-roberta-large` | `c23d21b0620b` | MIT | no | 561M; repo 15 GB (*repo*) | yes | M1 (C3) | Fetch safetensors only. |

### 2.4 Baselines

| HF ID | Revision | Licence | Gated | Params | 16 GB? | M0? | Notes |
|---|---|---|---|---|---|---|---|
| `KRLabsOrg/lettucedect-v2-mmbert-base` | `e4b105921ef7` | Apache-2.0 | no | 308M | yes | M1 (B2) | Card cites arXiv 2607.00895 (GroundLM 2026 @ EMNLP 2026). That matches the ID given in CLAUDE.md §4.1. Card says it is multilingual via PsiloQA (14 langs). |
| `KRLabsOrg/lettucedect-v2-qwen-2b` | `1833c76f4408` | Apache-2.0 | no | 2.27B | yes | M1 (optional B2b) | Larger LettuceDetect v2 variant. |

### 2.5 Indic tooling models

| HF ID | Revision | Licence | Gated | Params / download | 16 GB? | M0? | Notes |
|---|---|---|---|---|---|---|---|
| `ai4bharat/IndicNER` | `b56cb5e9bdf3` | MIT | **auto** | 0.67 GB | yes | **yes** (ENTITY/LOCATION injection) | Accept terms on HF first. |
| `ai4bharat/indictrans2-indic-en-1B` | `ac3daf0ecd37` | MIT | **auto** | 1.02B | yes | M1 (B5, C4) | Translate-test baseline. |
| `ai4bharat/indictrans2-en-indic-1B` | `89cd0007fd7b` | MIT | **auto** | 1.12B | yes | M1 (translated RAGTruth) | |
| `ai4bharat/indictrans2-indic-en-dist-200M` | `d66a2122ad6d` | MIT | **auto** | 228M | yes (CPU ok) | dev | Distilled; for quick tests. |

## 3. Python libraries (non-HF) — versions pinned in `pyproject.toml` at T01

| Package | Licence | Checked | Use |
|---|---|---|---|
| `indic-nlp-library` | MIT (PyPI, v0.92) | 2026-11-19 | Normalisation, sentence segmentation, transliteration |
| `torch`, `transformers`, `datasets`, `sentence-transformers`, `bitsandbytes`, `accelerate` | BSD/Apache | to check at T01 | Modelling, 4-bit loading |
| `faiss-cpu` (+ `faiss-gpu` on Colab) | MIT | to check at T01 | Vector index |
| `llama-index-core` *or* `langchain-core` | MIT | to check at T01 | Thin orchestration only (deck commitment) |
| `scikit-learn`, `matplotlib`, `jupyter`, `pydantic`, `pyyaml`, `pytest`, `gradio` | BSD/MIT/Apache | to check at T01 | Metrics, plots, schema, tests, demo |
| `lm-polygraph` | to check | M1 | Baseline B4 |
| `lettucedetect` | to check | M1 | Baseline B2 runner |

**Licence note:** Llama 3 / 3.1 (`IFM/Llama-3-Nanda-10B-Chat`, `meta-llama/…`) and Gemma 3 are free open-weight models, but their licences are not OSI open-source. Whether they're allowed under the "free and open source" rule is open question Q14 in PLAN.md.

## 4. Hardware

| Machine | GPU | Use |
|---|---|---|
| Dev laptop | NVIDIA GeForce GTX 1650 Ti, **4 GB** VRAM (driver 566.36), checked with `nvidia-smi` on 2026-11-19 | Docker dev environment, CPU tests, Ollama `qwen3:4b`, Gradio demo |
| Free Colab / Kaggle (maintainer's account) | T4, 16 GB | Embedding the corpus, NLI scoring at scale, 4-bit 7–10B generators (M1), aligner fine-tuning (M1) |

## 5. Download budget for M0

Nothing in M0 exceeds the 10 GB approval threshold:

| Item | Download |
|---|---|
| IndicQA Hindi JSON | ~3 MB |
| bge-m3 + multilingual-e5-large | ~4.5 GB |
| mDeBERTa NLI | ~1.1 GB |
| IndicNER | ~0.7 GB |
| Ollama `qwen3:4b` | 2.5 GB |
| Docker image `python:3.11-slim` + dependencies | ~1–3 GB (CPU torch) |
