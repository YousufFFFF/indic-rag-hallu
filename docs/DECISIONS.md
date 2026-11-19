# Decisions

Short ADR-style records of every non-obvious choice. Newest last. Each entry: context → decision → consequences.

---

## D-001 · Development environment: Docker (2026-11-19)

- **Context:** The dev laptop has no Python install. Docker Desktop is available. CI runs on Linux.
- **Decision:** All development and tests run in a Docker container built on `python:3.11-slim` with CPU PyTorch. No local Python install.
- **Consequences:** Local and CI environments match. GPU work doesn't run in this container: generation goes through Ollama on the host, and heavy runs go on Colab/Kaggle.

## D-002 · Compute: laptop + free cloud T4 (2026-11-19)

- **Context:** The laptop GPU is an NVIDIA GeForce GTX 1650 Ti with 4 GB VRAM (`nvidia-smi`, driver 566.36). The Stage I deck (slide 20) commits to a GPU with at least 16 GB.
- **Decision:** Use the laptop for development, the Ollama generator and the demo. Use a free Colab/Kaggle T4 (16 GB, maintainer's account) for embedding the corpus, large-scale NLI scoring, 7–10B generators and fine-tuning.
- **Consequences:** Slide 20's [INSERT GPU MODEL] is filled from run metadata. Runs on the T4 must checkpoint to survive session limits.

## D-003 · M0 generator: `qwen3:4b` via Ollama (2026-11-19)

- **Context:** M0 needs one Hindi-capable generator that fits a 4 GB GPU and has an open-source licence. Candidates checked on HF and the Ollama registry on 2026-11-19:
  - `qwen2.5vl:3b`: Qwen Research Licence (non-commercial), and a vision encoder we don't use.
  - `qwen3:4b`: Apache-2.0, 2.50 GB, card states 100+ languages.
  - `qwen2.5:1.5b`: Apache-2.0, 0.99 GB.
  - `qwen2.5:7b` / `qwen2.5vl:7b`: Apache-2.0, but 4.68 / 5.97 GB, which doesn't fit in 4 GB.
- **Decision:** `qwen3:4b` (HF source `Qwen/Qwen3-4B`, revision `1cfa9a720891`), served by Ollama and called through the OpenAI-compatible backend. Thinking mode is off. Decoding settings are fixed in `configs/generation/m0.yaml`. `qwen2.5:1.5b` is the fallback.
- **Consequences:** Run metadata records the Ollama manifest digest as well as the HF revision. White-box baseline B4 (logit-based uncertainty) needs an HF Transformers model on the T4, not Ollama. Hindi output quality is still to be measured in T23 and T36.

## D-004 · Free and open-source only (2026-11-19)

- **Context:** The maintainer's rule is that everything must be free and open source, with no paid API budget.
- **Decision:** No paid APIs. Prefer models and datasets under OSI-approved or open-data licences (Apache-2.0, MIT, CC-BY, CC0). LLM-as-judge (B3) and auto-labelling use open models run locally or on the free T4.
- **Consequences:** Open-weight models under non-OSI licences (Llama 3.x incl. Nanda, Gemma 3) and datasets without clear licences need an explicit decision (PLAN.md Q12, Q14).

## D-005 · Dependency pinning (2026-11-19)

- **Context:** CLAUDE.md requires pinned dependencies and reproducible runs. The dev container installs the CPU build of PyTorch; Colab/Kaggle use the CUDA build.
- **Decision:**
  - Direct dependencies are pinned exactly (`==`) in `pyproject.toml`.
  - The full transitive set is frozen in `requirements-lock.txt` by `scripts/lock_deps.sh`, which resolves inside the same digest-pinned base image as the `Dockerfile`.
  - `torch==2.15.0` comes from the PyTorch CPU index in Docker. GPU notebooks install the same version's CUDA build.
  - `numpy` is pinned to 2.4.6, because 2.5.x requires Python ≥ 3.12.
- **Consequences:** Changing any pin means re-running the lock script and rebuilding the image. GPU-only packages (`bitsandbytes`, `accelerate`) will go into a separate optional extra when M1 needs them.

## D-006 · Branch protection on `main` (2026-11-20)

- **Context:** CLAUDE.md §2 says never commit to `main` directly. CODEOWNERS only *requests* review. The maintainer is the only code owner, and GitHub doesn't let authors approve their own PRs.
- **Decision:** `main` requires
  - a pull request,
  - a passing `test` CI check (branch doesn't have to be up to date),
  - resolved review conversations.

  Force-pushes and deleting `main` are blocked. Required approvals are **0** and code-owner review is **not** required, so the maintainer can merge their own PRs. Admins are not forced to follow these rules (`enforce_admins: false`), as an emergency escape hatch.
- **Consequences:** Every change goes through a PR with green CI. If teammates start opening PRs, raise required approvals to 1 so the maintainer reviews them. History rewrites on `main`, like the one on 2026-11-19, now need protection lifted temporarily.

## D-007 · Language registry; M0 enables Hindi only (2026-11-20)

- **Context:** Deck slide 5 puts nine languages in scope (hi, bn, ta, te, mr, kn, ml, gu, pa), with the final set fixed by corpus availability. M0 is a Hindi vertical slice. Objective 05 needs a resource level per language.
- **Decision:**
  - `configs/languages.yaml` lists all nine languages, validated by `indichallu.languages`. Unknown keys are rejected, and IndicTrans2 tags and digit ranges are checked for consistency.
  - Only `hi` has `enabled: true` for M0.
  - Each language records its ISO 639-3 code, ISO 15924 script, Unicode block and digit-zero code point, family, IndicTrans2 tag, and IndicQA file. Every Unicode field is checked against Python's `unicodedata` in the tests, not taken on trust.
  - `resource_level` is left `null`. It will be filled in M1 from a verified, cited source (planned: the Joshi et al. 2020 taxonomy), never estimated.
- **Consequences:** Turning a language on later is a one-line config change. Analysis code must handle `resource_level` being null until M1.

## D-008 · Record schema conventions (2026-11-20)

- **Context:** Every stage reads and writes the JSONL record from CLAUDE.md §6. The span metrics and the "points back at a retrieved passage" contract only hold if the schema enforces them.
- **Decision** (`src/indichallu/data/schema.py`):
  - **Offsets** are Python `str` indices (Unicode code points) into the **original, un-normalised** answer, half-open `[start, end)`. A claim's `text` must equal `answer[start:end]`. Paraphrased claims (e.g. from LLM decomposition in M1) will need a separate field, not a looser check.
  - **Contract:** SUPPORTED and CONTRADICTED claims must name a `best_passage_id`. UNSUPPORTED claims may name one or none. Every passage ID mentioned anywhere must be in `retrieved`.
  - **Retrieval carried forward:** `retrieved` is required, non-empty, with unique IDs and ranks exactly 1..n.
  - **Score matrix:** `claims[].passage_scores` holds the claim's NLI scores against each retrieved passage (deck slide 17 step 02 / CLAUDE.md §3 row 8). This field is an addition to the §6 example. It is **all-or-nothing**: either empty (baselines with no NLI step, e.g. B2–B4) or exactly one row for every retrieved passage, so a partial matrix can't pass as valid. (From the Codex review on PR #8.)
  - **Error types:** `claims[].type` (predicted) and `labels.spans[].type` (gold) use the four-type taxonomy. A SUPPORTED claim cannot carry an error type.
  - **Response label** is `faithful` | `hallucinated`. It must agree with the spans (hallucinated ⇔ ≥1 span). Gold spans may not overlap.
  - `natural` records must name their generator. `injected`/`translated` records may omit it.
  - **Validation:** NLI probabilities must sum to 1 (±1e-3). Unknown fields are rejected everywhere, except backend-specific decoding extras.
  - **JSONL:** UTF-8 without `\u` escapes, one compact line per record, atomic writes, line numbers in parse errors.
- **Consequences:** Invalid records fail at write/read time instead of producing silently wrong metrics. Schema changes need a decision entry and a test update.

## D-009 · Run folders, provenance and seeding (2026-11-20)

- **Context:** CLAUDE.md §2 requires config-driven, seeded runs. Each run must write `results/runs/<run_id>/` with the resolved config, git commit and model revisions. Every reported number has to be traceable back to a run.
- **Decision** (`src/indichallu/runs.py`):
  - `run_id = <UTC yyyymmdd-HHMMSS>-<name>-<8-char config hash>`. Existing folders are never overwritten.
  - **`config.yaml`** holds the fully resolved config. Overrides are `dotted.key=value`, parsed as YAML. Only existing keys can be overridden, so typos fail.
  - **`meta.json`** records:
    - git commit, branch and dirty flag
    - models, each with a **mandatory** revision
    - seed and determinism flag
    - Python, platform, and versions of the tracked packages
    - CUDA version and GPU names
    - the command line
    - start/end time, and a status of `running` / `completed` / `failed` (with the error)
  - **`git.diff`** is saved when the tree is dirty. `require_clean=True` refuses to start from a dirty tree; use it for every number that gets reported.
  - **Seeding:** the config must contain `seed`, an int in [0, 2³²−1] (NumPy's limit). It is validated before the run folder is created. `set_seed` seeds Python, NumPy and PyTorch (CPU + CUDA). `deterministic=True` uses PyTorch's **strict** deterministic mode, which raises on a nondeterministic operation instead of warning. The mode is set explicitly, on or off, on every call, so the recorded `deterministic` flag always matches PyTorch's actual state. (From the Codex review on PR #9.)
  - **git in the image:** `git` is installed in the dev image, and `/workspace` is marked a git `safe.directory` (the bind-mounted repo has a different owner). Without this, provenance can't be recorded in Docker or CI.
- **Consequences:** Every experiment script goes through `start_run`. Reported results must come from runs with `dirty: false`.
