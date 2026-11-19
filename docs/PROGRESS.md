# Progress log

Dated log of what was done, what broke, and what's next. Newest first.

---

## 2026-11-20

**Done**
- T01 merged (#5).
- **T02: CI.** `.github/workflows/ci.yml` runs on every PR and on pushes to `main`.
  - It builds the same `Dockerfile` used for local development, with the Docker layer cache stored in GitHub Actions.
  - It then runs `ruff check`, `ruff format --check` and `pytest` in the container. Tests marked `slow`/`gpu` are excluded.
  - Actions are pinned by commit SHA.
  - `actionlint` 1.7.12 reports 0 errors.
- README CI badge. T02 merged (#6). Branch protection enabled on `main` (DECISIONS D-006).
- **T03: language registry.** `configs/languages.yaml` covers the 9 deck languages, with Hindi the only one enabled (D-007).
  - Loader and validation in `src/indichallu/languages.py`.
  - `tests/test_languages_config.py` checks the config against the deck and against Python's Unicode database: digits, scripts, blocks and endonyms. It also checks that malformed configs are rejected.
  - 49 tests pass in the container.
- **T04: record schema.** `src/indichallu/data/schema.py` implements the CLAUDE.md §6 record as pydantic models (D-008).
  - Enums: the 4 error types, the 3 verdicts, response label, source, split.
  - Validation of offsets against the original answer and of the evidence contract.
  - Atomic, UTF-8 JSONL read/write.
  - `tests/test_schema.py` uses the slide-17 worked example as its fixture, with offsets computed from the strings, plus 23 rejection cases, each checked to fail for the intended reason.
  - 84 tests pass in the container.
- T04 merged (#8), after fixing the Codex review finding (the claim score matrix is now all-or-nothing).
- **T05: run utilities.** `src/indichallu/runs.py` (D-009).
  - Seeding: `set_seed`. Config handling: `load_config`, `apply_overrides` (only existing keys), `config_hash`.
  - Provenance: `git_info` and `git_diff`, `environment_info` (packages, CUDA, GPUs).
  - `start_run` creates `results/runs/<run_id>/` with `config.yaml`, `meta.json`, and `git.diff` when the tree is dirty. `Run.add_model` requires a revision. The status records `completed` / `failed`.
  - `git` added to the dev image (+ `safe.directory`).
  - 113 tests pass. Git tests use a throwaway repo, so they don't depend on this repo's state.
  - Codex review fixes: 32-bit seed bound checked before the folder exists; strict, explicitly reset deterministic mode.
- T05 merged (#9), after fixing 2 Codex findings (seed range, strict deterministic mode).
- **T06: offset maps.** `src/indichallu/text/offsets.py` (D-010).
  - `OffsetMapBuilder` records copy / replace / delete / insert edits. `OffsetMap.to_source`/`to_target` map spans both ways. `then`/`compose` chain transforms.
  - `tests/text/test_offsets.py` covers:
    - Devanagari, Bengali, Tamil and Telugu digit replacement
    - the slide-17 answer written with Devanagari digits (the claim span maps back to `२१`)
    - expansion, contraction, deletion (ZWNJ), insertion
    - an insertion inside an earlier expansion
    - associativity
    - 200 random edit sequences checked against ground truth
    - 100 random compositions checked against a set-based reference
  - First design (bisection over monotone intervals) was replaced: composition can produce non-monotone intervals. A regression test covers it.
  - Doctests now run in CI (D-011). The first builder docstring example had an off-by-one; it is fixed and now executed.
  - 431 tests pass.
- T06 merged (#10), Codex: no findings.
- **T07: normalisation.** `src/indichallu/text/normalize.py` (D-012).
  - `nfc()`: exact NFC with an offset map, computed in chunks split at safe boundaries.
  - `indicnlp_normalize()`: the library's output, diff-aligned to its input.
  - `normalize()`: NFC → IndicNLP with one composed map. `align()`: a generic diff-based map for black-box transforms.
  - First boundary rule was too conservative: a NFC-changed char like U+095B dragged its neighbour into its chunk. Fixed to decide by the decomposition's first character, as ICU does. A test pins it.
  - Tests cover:
    - NFC equal to `unicodedata` on 300 random Indic strings and 3000 "hard" strings (all decomposable characters, combining marks, Hangul);
    - two-part vowels in Bengali, Tamil, Malayalam, Kannada and Telugu;
    - mark reordering;
    - IndicNLP output identical to the library's, with ZWNJ / `|` / visarga edits mapping back;
    - every configured language round-tripping its endonym and native digits;
    - the slide-17 answer with ZWNJ, `ज़` and `|` noise mapping back to the original.
  - 959 tests pass.
- T07 merged (#11) with an open Codex finding: the diff-based IndicNLP alignment mis-attributed repeated words (`राम:रामः`). **Fixed in a follow-up PR:**
  - `align()` is now a banded, globally optimal LCS-style alignment (substitution cost 2) that walks both strings in step.
  - Tests check that word k of the output maps back inside word k of the input:
    - real IndicNLP on repetitive noisy text: 90 cases, all 9 languages;
    - 200 synthetic edit cases;
    - the exact Codex case;
    - pure substitutions stay exact.
  - Out-of-suite stress:
    - IndicNLP: 0 / 1,350 failures;
    - harsh synthetic edits: 70 / 5,000 misses, all genuinely ambiguous (our alignment never matches fewer characters than the true edits).
  - First attempt with substitution cost 1 failed 8 synthetic cases by tying with a shifted path; cost 2 fixed them.
  - Second Codex finding (on #12): a fixed band missed cumulative drift (33 ZWNJs + 33 `ऩ` mapped output into the deleted ZWNJs). Fixed with provably optimal band doubling, related-character substitution cost 1, and per-substitution chunking. Regression tests: drift up to 200 chars; band doubling equals unbanded cost on 60 random cases.
  - Third Codex finding (on #12): band doubling could allocate a dense grid of hundreds of MB on pathological drift. Fixed: two rolling cost rows plus 1-byte backpointers, and a 2M-cell work budget with a coarse-but-safe fallback (exact up to 300-char drift; at most 2.7 s and < 3 MB at 20k chars). Tests: a spy asserts no pass exceeds the budget; the forced fallback is safe; moderate drift stays exact.
  - 1,321 tests pass.
  - A smoke run against the real repo recorded the commit, dirty flag, diff, model revision and package versions.

**Next**
- T08: `text/numerals.py` (Indic digits → ASCII with offsets, all 8 digit blocks), then T09: `text/sentseg.py` (sentence segmentation with offsets).

## 2026-11-19

**Done**
- Repo set up and published. Governance merged: CODEOWNERS and the "never commit to main" rule (#1). CodeRabbit config (#2).
- Planning docs merged (#4): `PLAN.md` (traceability matrix + M0 tasks), `RESOURCES.md` (all HF resources checked), `DECISIONS.md` (D-001 to D-004).
- **T01 — Docker dev environment.**
  - `Dockerfile` (base `python:3.11-slim` pinned by digest, Python 3.11.18) and `docker-compose.yml` (`dev` service, source bind-mounted, HF cache volume, Ollama on the host via `host.docker.internal`).
  - `pyproject.toml` with exact pins for direct dependencies. Full frozen set in `requirements-lock.txt` (89 packages, CPU torch 2.15.0), generated by `scripts/lock_deps.sh`.
  - Empty `src/indichallu/` package. `tests/test_environment.py` checks the Python version, the package import and the core dependency imports.
  - Verified in the container: `pytest` 3 passed; `ruff check` and `ruff format --check` clean. Image size 2.84 GB.

**Broke / learned**
- numpy 2.5.x requires Python ≥ 3.12, so it's pinned to 2.4.6, the newest release that supports 3.11.
- Main's history was rewritten to remove commit trailers. That auto-closed PR #3, which was replaced by #4.

**Next**
- T02: GitHub Actions CI (CPU pytest + ruff), branch protection on `main`.
- T03: `configs/languages.yaml`.
- T00: request IndicNER access; `ollama pull qwen3:4b`.
