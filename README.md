# Hallucination Detection in RAG for Low-Resource Indic Languages

[![CI](https://github.com/YousufFFFF/indic-rag-hallu/actions/workflows/ci.yml/badge.svg)](https://github.com/YousufFFFF/indic-rag-hallu/actions/workflows/ci.yml)

A claim-level framework that checks whether an LLM answer in an Indian language is actually supported by the passages retrieved for it, with span-level attribution.

> **The detector never asks "is this true?". It asks "does the retrieved evidence say this?"**
> Every claim either points at the retrieved passage that supports or contradicts it, or is marked unsupported. Retrieval scores are carried through every stage, so a retrieval failure can be told apart from a generation failure.

Final-year B.E. (AI & Data Science) project, JSPM's Narhe Technical Campus, Pune, 2026–27: Mohammed Yousuf Ansari, Mukta Dhage, Sneha Shinde, Srushti Kondhalkar.

**Status:** M0, the Hindi vertical slice (pilot). Languages in scope: Hindi, Bengali, Tamil, Telugu, Marathi, Kannada, Malayalam, Gujarati, Punjabi (`configs/languages.yaml`); M0 enables Hindi. See [`docs/PLAN.md`](docs/PLAN.md) for the plan and [`docs/PROGRESS.md`](docs/PROGRESS.md) for the log.

## How it works

```mermaid
flowchart LR
  subgraph IR [Input and retrieval]
    Q[Indic user query] --> N[Normalisation and script handling]
    N --> E[Multilingual embedder]
    C[(Vector index, Indic corpus)] --> K[Top-k passages with scores]
    E --> K
  end
  subgraph G [Generation]
    K --> P[Prompt: query + context]
    P --> L[LLM generator]
    L --> A[Generated answer]
  end
  subgraph D [Detection]
    A --> DC[Claim decomposition]
    DC --> AL[Claim-evidence alignment: NLI vs every passage]
    K --> AL
    AL --> AG[Aggregation and span attribution]
  end
  AG --> V{{SUPPORTED / UNSUPPORTED / CONTRADICTED}}
  V --> EV[Evaluation vs annotation]
```

Every text transformation (NFC, IndicNLP, Indic digits, sentence splitting) keeps an **offset map** back to the original string, so spans are always reported on the original answer text.

| Stage | Code |
|---|---|
| Normalisation, digits, sentences, tokens, offset maps | `src/indichallu/text/` |
| Data: IndicQA download (pinned + SHA-256), loader, leakage-free split, synthetic errors | `src/indichallu/data/` |
| Retrieval: embedder, FAISS index, top-k with scores / ranks / gold flags | `src/indichallu/retrieval/` |
| Generation: Hindi prompt, Ollama / OpenAI-compatible backends, cache | `src/indichallu/generation/` |
| Detection: decompose → align (NLI) → aggregate; baseline B1 | `src/indichallu/detection/` |
| Metrics: Recall@k / nDCG / MRR, P/R/F1 / balanced accuracy / AUROC, span F1 / token IoU / exact match, per-language and worst-language | `src/indichallu/eval/` |

## M0 results

<!-- BEGIN AUTO: m0-results -->
_Not generated yet: run `python scripts/run_pilot_eval.py`, then `python scripts/make_tables.py`._
<!-- END AUTO: m0-results -->

## Quickstart

Everything runs in Docker; no local Python install is needed (Docker Desktop with Compose v2).

```sh
docker compose build                                            # Python 3.11, CPU PyTorch, pinned deps
docker compose run --rm dev pytest                              # test suite (CPU, no downloads)
docker compose run --rm dev python scripts/download_indicqa.py  # Hindi IndicQA, pinned + verified
docker compose run --rm dev python scripts/run_pilot_eval.py --embedder me5-small   # M0 pilot run
docker compose run --rm dev python scripts/make_tables.py       # regenerate the table above
```

`run_pilot_eval.py` writes a run folder under `results/runs/` (config, git commit, model revisions, environment) and the summary `results/m0/pilot.json`, the only source of the numbers above.

## Development

- Source is bind-mounted, so code edits take effect without rebuilding. Rebuild after changing dependencies.
- Dependencies: direct pins live in `pyproject.toml`; the full frozen set is `requirements-lock.txt`. After changing a pin, run `sh scripts/lock_deps.sh` and rebuild.
- Secrets (for example `HF_TOKEN`) go in a git-ignored `.env` file, which the container loads automatically if present.
- The generator runs in [Ollama](https://ollama.com) on the host; the container reaches it at `http://host.docker.internal:11434/v1`.
- Every model and dataset, with its pinned revision and licence: [`docs/RESOURCES.md`](docs/RESOURCES.md). Every non-obvious choice: [`docs/DECISIONS.md`](docs/DECISIONS.md). Data: [`docs/dataset_card.md`](docs/dataset_card.md).

## Licence

Code: Apache-2.0 ([`LICENSE`](LICENSE)). IndicQA (AI4Bharat) is CC-BY-4.0; models keep their own licences (see `docs/RESOURCES.md`).
