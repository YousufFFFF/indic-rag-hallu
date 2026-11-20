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

## D-010 · Offset-map semantics (2026-11-20)

- **Context:** CLAUDE.md §2 says spans are computed on original offsets, and every text transformation needs a map back to the original. Normalisation can expand (U+0958 → क + nukta), contract, delete (ZWNJ) and replace (Indic → ASCII digits).
- **Decision** (`src/indichallu/text/offsets.py`):
  - A transformation is recorded left to right as **copy** (exact 1:1) or **replace n source chars with any string**. Deletion and insertion are special cases of replace.
  - Each target char stores the source interval it came from. Characters from one replacement share the **whole** chunk, because a chunk can't be split meaningfully.
  - A span maps to the **hull** of its characters' intervals:
    - spans inside copied text map exactly;
    - touching part of a replaced chunk expands to the whole chunk;
    - a fully deleted source span maps to the empty target span where it was;
    - an inserted char at the exact edge of a source span is not attributed to it (only insertions strictly inside are).
  - Maps compose (`m1.then(m2)`, `compose([...])`), so normalisation steps chain into one map back to the original.
  - Lookups are linear scans, not bisection. Composed intervals need not be monotone: an insertion can land inside an earlier expansion. Texts are short, so this costs nothing that matters.
- **Consequences:** Every text transform in `text/` returns `(output, OffsetMap)`. Spans found on normalised text are mapped with `to_source` before they are stored in a record.

## D-011 · Doctests run in CI (2026-11-20)

- **Context:** The first `OffsetMapBuilder` docstring example had an off-by-one offset. That is exactly the error class this project must avoid in documentation too.
- **Decision:** pytest runs with `--doctest-modules` over `src/` as well as `tests/`, so every `>>>` example is executed in CI.
- **Consequences:** Docstring examples must be correct and runnable. Illustrative snippets that shouldn't run use a `::` literal block instead of `>>>`.

## D-012 · Normalisation pipeline: NFC, then IndicNLP (2026-11-20)

- **Context:** Deck block diagram box 1 ("Normalisation & Script Handling"); CLAUDE.md §3 row 2: NFC + IndicNLP normaliser, all with offset maps. Python's `unicodedata` and IndicNLP return no offsets. IndicNLP (v0.92) does more than script fixes:
  - decomposes nukta letters;
  - deletes ZWJ/ZWNJ, word joiner, soft hyphen and BOM;
  - turns NBSP / zero-width space into a space;
  - `|` → danda;
  - `:` after an Indic letter → visarga.

  Its punctuation step is a no-op in this version. For an unknown language code it silently falls back to a generic normaliser.
- **Decision** (`src/indichallu/text/normalize.py`):
  - **Order: NFC, then IndicNLP.** `normalize()` composes both offset maps back to the original. The output is a *matching form*, not guaranteed NFC: IndicNLP decomposes U+0929/0931/0934, which NFC keeps composed. Claims and passages must go through the same pipeline.
  - **Exact NFC map:** normalise in chunks split before characters whose canonical decomposition starts with a starter that can't compose backwards (ICU's `hasCompBoundaryBefore`).
    - "Composing starters" (e.g. Bengali U+09BE, Tamil U+0BBE, Malayalam U+0D3E, Kannada U+0CD5, Hangul V/T) are derived from the Unicode data at first use (0.16 s), not hand-listed.
    - Every call checks the result equals `unicodedata.normalize("NFC", ...)`.
    - Stress-tested on 100k random strings from all decomposable characters, combining marks and Hangul: 0 mismatches.
  - **IndicNLP map:** IndicNLP is treated as a black box and its output aligned to its input. This is preferred over re-implementing its rules, which would drift from the library.
    - **Alignment method (revised 2026-11-20, Codex review on PR #11):** the first version used `difflib.SequenceMatcher`. Its longest-common-substring matching mis-attributed repeated text. For `राम:रामः` → `रामःरामः`, the first output word mapped to the *second* source word.
    - `align()` now uses a globally optimal alignment that walks both strings in step. Costs:
      - insert/delete 1;
      - substituting a **related** char costs 1: same NFKD base letter (ऩ→न), same digit value (२→2), or whitespace→whitespace;
      - any other substitution costs 2, i.e. LCS-like. With a flat cost of 1, unrelated nearby edits could tie with a path shifted by one char that swallowed a separator.
    - **Band doubling (Codex review on PR #12):** the alignment starts with a ±32-char band around the diagonal and doubles it until the result is *provably* globally optimal. A path that leaves the band costs ≥ `|m−n| + 2·band + 2`, because only insert/delete change the diagonal and each costs 1. A fixed band missed cumulative drift: deletions then expansions that cancel out in length, e.g. 33 ZWNJs + 33 `ऩ`.
    - **Chunking:** each substitution starts its own chunk, and the inserts/deletes right after it join it (`ऩ` → `न`+nukta is 1→2; `ে`+`া` → `ো` is 2→1). Inserts/deletes with no substitution before them are their own chunk (deleted ZWNJs). The earlier rule of "every differing region is one chunk" turned a region with no matching chars into a single chunk.
    - **Bounded work (Codex review on PR #12, second round):** a pass keeps only two rows of costs plus one byte of backpointer per band cell. Before each pass the cell count (rows × band width) is checked against `MAX_ALIGN_CELLS` = 2,000,000. Beyond that, the differing middle becomes **one coarse chunk**. That is still safe, because every span maps to a region containing its true source; it is just less precise.
      - Measured on n ZWNJs + n `ऩ`: exact up to n = 300, then coarse but safe; at most 2.7 s and < 3 MB peak even for 20,000 chars.
      - The dense full-grid version would have needed hundreds of MB.
    - The common prefix and suffix are copied directly, and identical text is a fast path. Runs of pure 1:1 substitutions stay exact per char.
    - **Verification:**
      - real IndicNLP output on repetitive noisy text, including 40-ZWNJ runs: 1,350 texts across all 9 languages, 0 cases where output word k maps outside source word k;
      - band doubling matches an unbanded alignment's cost on 60 random long-drift cases;
      - 0.02 s for 2,000 chars;
      - in a harsher synthetic stress run (25% of chars edited, with edits reusing neighbouring letters), the remaining 70/5,000 word-provenance misses are all genuine ambiguity: in every one our alignment matches at least as many characters as the true edit history (62 more, 8 equal, 0 fewer).
  - Only the 9 configured languages are accepted, so IndicNLP's silent generic fallback can't happen. IndicNLP options are passed through; their names are checked, and library defaults are used unless set.
- **Consequences:** Digit normalisation (T08) is a separate step that composes onto this map. `:` → visarga can alter text like `समय: 10`. This is IndicNLP's documented behaviour and is accepted for matching. Spans are always reported on the original.

## D-013 · Indic digits → ASCII (2026-11-20)

- **Context:** CLAUDE.md §3 row 2 requires Indic digit normalisation with offset maps. Passages and answers mix native and ASCII digits (`२१` vs `21`), which must compare equal for matching and for the NUMERICAL checks.
- **Decision** (`src/indichallu/text/numerals.py`):
  - `to_ascii_digits()` replaces each native decimal digit with its ASCII digit, one code point for one. The offset map is exact per character.
  - **Default scope `"indic"`:** the decimal digits of the configured languages, i.e. 8 digit blocks, since hi and mr share Devanagari. The table is built from `configs/languages.yaml`, not hand-listed. `scope="all"` converts any Unicode decimal digit (`Nd`), e.g. Arabic-Indic or full-width.
  - **Non-decimal numerals are not converted** (Tamil `௰` 10, `௱` 100, `௲` 1000; Bengali currency numerators). They are not positional digits, so a 1:1 swap would change the value. Parsing them is the typed NUMERICAL verifier's job (C2, M1).
  - `normalize(..., ascii_digits=True)` chains this after NFC → IndicNLP with one composed map. It is **off by default** so the pipeline's output stays in the source script; matching and detection configs turn it on.
- **Consequences:** Digit grouping and number parsing (Indian `1,00,000`, decimals, mixed scripts) are out of scope here and belong to the typed verifier.

## D-014 · Sentence segmentation and tokens (2026-11-20)

- **Context:** Claim decomposition (T24) and Token IoU (D82) need sentences and tokens with offsets into the original text (CLAUDE.md §3 rows 2, 7, 14).
  - IndicNLP's `sentence_split` returns strings without offsets.
  - Its period-merging pass **silently drops text**: on one input it kept 6 of 19 non-space characters. Even `'क. ख. डॉ. ग आया.'` is merged with a stray leading space.
  - It also never splits on `।`, `?` or `!` right after a digit (`आयु 21। अब` stays one sentence), and splits `क?!` into `क?` and `!`.
- **Decision** (`src/indichallu/text/sentseg.py`):
  - **Sentences:** we keep IndicNLP's *delimiter choice* (for hi/bn/pa text containing a danda only `? ! । ॥` end sentences, otherwise `.` too) and its *abbreviation list* (`is_acronym_abbvr`), but use our own boundary logic. Boundaries are found on the normalised text (so `|` typed for a danda works). The **cut positions** are mapped back and the original is split there, so sentences **partition the original**: no text is ever lost. A piece made only of characters normalisation deletes (e.g. an edge ZWNJ) is merged into a neighbour.
  - **Rules:**
    - `.` does not end a sentence after a digit (`21.`, `3.5`), after a listed abbreviation (`डॉ`, `श्री`, …) or after a single *bare* letter (initials `क.`, `A.`). Letters with vowel signs are excluded on purpose: `है.` and `में.` do end sentences.
    - Other delimiters after digits do split.
    - Runs of delimiters and closing quotes/brackets stay with their sentence.
    - IndicNLP's "a one-word sentence glues its neighbours" heuristic is not reproduced, because it is where the text loss happens.
  - **Tokens:** our own tokenizer on the original text, by Unicode category.
    - A word is a run of letters, marks, digits and ZWJ/ZWNJ. Python's `\w` would split at vowel signs and viramas.
    - A number keeps `,`/`.` between digits (`1,00,000.50`, `२,५००.७५`).
    - Every other non-space char is its own token.
    - `tokens(text, start, end)` returns absolute offsets for a window (e.g. a claim span).
- **Consequences:**
  - Where IndicNLP is lossless (danda text, no digits or repeated delimiters), our sentences equal its output; a test checks this.
  - Elsewhere they differ by the rules above, which are documented and tested.
  - Initials whose letter carries a vowel sign (`पी.`) do split.

## D-015 · No literal invisible characters in source (2026-11-20)

- **Context:** ZWNJ/ZWJ/NBSP had ended up written literally in 7 source and test files instead of as escapes. Behaviour was unaffected, but they are invisible in an editor and in review, and easy to break by accident.
- **Decision:** Such characters are written as escapes (`"\u200c"`) or `chr(0x200C)`. `tests/test_source_hygiene.py` fails on any Unicode format character (`Cf`), NBSP or narrow NBSP in `src/`, `tests/`, `configs/`, `docs/`, `scripts/`, the README or CLAUDE.md. I checked it fails when a ZWNJ is planted.
- **Consequences:** The data itself (`data/`, `results/`) is not scanned; its invisible characters are content.

## D-016 · Pinned, checksum-verified data downloads (2026-11-20)

- **Context:** CLAUDE.md §2 says every run must be reproducible and data is git-ignored, with download scripts committed. IndicQA's loading script is refused by recent `datasets` versions, so its per-language JSON files are fetched directly (§4.4).
- **Decision:**
  - **`configs/data/<dataset>.yaml`** records the repo, the **full 40-char revision**, the licence, and for each file its **size and SHA-256** (the Git LFS object id). For IndicQA these were read from the Hugging Face API on 2026-11-20, for all 9 configured languages.
  - **`indichallu.data.download.fetch()`**:
    - streams to a temp file while hashing, and moves the file into place only if size and hash match;
    - on any mismatch or HTTP error, raises and leaves nothing behind;
    - reuses a valid existing file without network access, and replaces a corrupt one.
  - Downloads go to `data/raw/<name>/<revision[:12]>/`, versioned on disk and git-ignored.
  - The User-Agent is generic, with no personal details.
  - **`scripts/download_indicqa.py`** downloads the *enabled* languages by default (`--lang`, `--all` otherwise).
  - **Tests** use a local HTTP server, so CI needs no network. A `network`-marked test (excluded by default; `pytest -m network`) checks the real Hindi file against its pin.
- **Consequences:** A changed upstream file fails loudly instead of silently changing results. New datasets get their own pinned config. **Code that reads cached data goes through `verified_local_path()`**, which re-checks size and SHA-256, so a file changed or corrupted after download is never analysed as the pinned revision (Codex review on PR #18). `corpus_stats.py compute` records the hash of the bytes it actually read.
- **Lesson:** my first, text-sliced reading of the HF tree listing paired each path with the *next* file's size and hash. The pins come from properly parsed JSON. The config test also rejects duplicate hashes, which catches copy-paste mistakes.

## D-017 · IndicQA loading and answer-offset realignment (2026-11-20)

- **Context:** The M0 corpus is IndicQA Hindi. I inspected the pinned file (2,673,518 bytes):
  - **Shape:** SQuAD-style, 261 articles, each with one paragraph. 261 unique contexts (1,156–8,857 chars, median 1,997).
  - **Questions:** 1,547, each with exactly one answer. Category `NO` (495) means unanswerable, marked by `text == ""` and `answer_start == null`. `SHORT` (1,052) means answerable.
  - 54 questions are not in NFC.
  - **The given `answer_start` is wrong for 894 of 1,052 answers.** Every answer text does occur in its context, always at or *after* the given offset (shift 0–60 chars, median 6, p95 28). No single unit explains the shift: bytes match 12, UTF-16 units 90, and counts of newlines, spaces, marks, viramas or ZWJ/ZWNJ at most 90 of 779.
- **Decision** (`src/indichallu/data/loaders/indicqa.py`):
  - **Passages** are deduplicated by content. The ID is `<lang>-p` + 10 hex chars of the context's SHA-256, independent of file order. The first title wins.
  - **QA pairs** get the ID `<lang>-q` + the original id (6 digits; stable at the pinned revision). Each links to its gold passage and keeps the original `answer_start`.
  - **Answer offsets are realigned and verified:**
    - `exact` (the given offset is right);
    - `unique` (the answer text occurs once in the context);
    - `nearest` (the nearest occurrence at most `MAX_SHIFT = 80` chars after the given offset, and at least `MIN_MARGIN = 10` chars closer than the next);
    - otherwise `ambiguous` / `not_found`, with **no offset**. The question is kept, but nothing relies on its answer span.
  - Answer text is whitespace-trimmed and the offset adjusted. Raw text is kept as-is: normalisation happens in the pipeline.
  - **Hindi result:** 158 exact, 675 unique, 218 nearest, **1 ambiguous**, 0 not found. Every kept offset slices out exactly its answer text.
- **Consequences:**
  - Anything using answer spans (e.g. building the pilot set, T30) must use `answer_start`/`answer_end` and skip `None`.
  - Retrieval evaluation is unaffected: the gold *passage* is the paragraph itself.
  - These counts belong in the dataset card (T13).
  - Other languages may behave differently; the loader reports counts per status, so check them before enabling a language.

## D-018 · Train/val/test split (2026-11-20)

- **Context:** CLAUDE.md §4.4: split by passage/question, so the same passage is never in both train and test. Thresholds are tuned on val only (§3 row 9).
- **Decision** (`src/indichallu/data/split.py`, `configs/data/splits.yaml`):
  - **Ratios** 60/20/20, **seed 13**, both in config.
  - **Group by article title.** A passage with no title is its own group. IndicQA Hindi has one paragraph per article, but other languages may not.
  - **Groups sharing an identical question text are merged first.** The leakage check found 3 Hindi questions repeated word for word across articles (e.g. "भारत को स्वतंत्रता किस वर्ष मिली थी?"), which the plain split put in different splits.
  - **Deterministic without `random`:** groups are ranked by `sha256("<seed>:<key>")` and cut into exact counts (largest remainder). This is independent of input order, Python version and platform. A test pins a concrete assignment.
  - Every split with a non-zero ratio gets ≥ 1 group when possible. Plain largest-remainder gives 2/1/0 for 3 groups.
  - **`check_no_leakage()`** fails if any passage is unassigned, or if an article, identical passage text or identical question text spans two splits.
- **Hindi result:** 258 groups (from 261 articles). Passages train/val/test 156/52/53; questions 950/282/315; unanswerable 303/93/99 (31–33% in each split). No leakage.
- **Consequences:** Changing the seed, ratios or grouping changes every split and every reported number, so it needs a new decision entry. Thresholds may only be tuned on `val` (enforced later in T34).

## D-019 · Corpus statistics and the auto-generated dataset card (2026-11-20)

- **Context:** CLAUDE.md §3 row 1 asks for an auto-generated per-language corpus stats table in `docs/dataset_card.md`. §2 says every number in the README, paper or deck is generated by a script from `results/`, never typed.
- **Decision:**
  - **Two steps.**
    - `scripts/corpus_stats.py compute` needs the data. It loads each downloaded language, applies the configured split, and writes `results/corpus_stats/indicqa.json`. This small file is **committed**, and records the dataset revision, each file's SHA-256, the split config and the package version.
    - `render` writes only the block between `<!-- BEGIN AUTO: corpus-stats -->` / `<!-- END AUTO -->` in the card. The prose around it is kept, and contains no numbers.
  - **Drift check:** `render --check` (also run by a pytest test, so in CI without any data) fails if the card's tables differ from what the committed JSON renders to.
  - **Contents per language:** passages, questions, answerable/unanswerable, median passage chars/tokens/sentences, median question tokens, answer-offset outcomes (D-017), and split sizes (D-018). All 9 configured languages are computed (about 22 MB downloaded in total, all CC-BY-4.0).
- **Findings from the first run, all 9 languages:**
  - Answer offsets are wrong in every language: the source offset is correct for only 11–27% of answers.
  - Realignment recovers 72–88%.
  - "Not found" answers range from 0 (hi, bn) to 43 (ml). Checked by hand: almost all are genuinely different text, not normalisation differences.
  - Passage length varies a lot (Kannada and Gujarati median under 900 chars, Hindi about 2,000).
- **Consequences:** After changing the loader, splits or tokenizer, rerun `compute` and `render` and commit both files. CI catches a stale card but not stale JSON (no data in CI), so the JSON is regenerated as part of any PR that changes those modules.

## D-020 · Embedder wrapper (2026-11-20)

- **Context:** Deck block diagram "Multilingual Embedder"; CLAUDE.md §3 row 3. M0 compares bge-m3 (primary) with multilingual-e5-large. The real runs happen on the T4 (T19); CI must stay CPU-only with no model downloads.
- **Decision** (`src/indichallu/retrieval/embed.py`, `configs/retrieval/embedders.yaml`):
  - **Config per embedder:** model id, **full pinned revision**, dimension, query/passage prefixes **taken from each model card**, normalisation, max sequence length.
    - mE5 needs `"query: "` / `"passage: "`; bge-m3 needs none.
    - bge-m3 is capped at 2,048 tokens (supports 8,192); mE5's model maximum is 512, so **long passages are truncated for mE5**. This is a known limitation of the comparison.
  - **`Embedder`** adds the prefix, NFC-normalises the text (models see the source script, not the IndicNLP matching form), calls a pluggable encoder, and checks the shape, finiteness and dtype (float32). It L2-normalises in one place for every encoder, so inner product = cosine.
  - **Disk cache:** keyed by SHA-256 over the config, the kind (query/passage) and the exact length-prefixed inputs. Any config or text change re-embeds. Writes are atomic. **A cached array must pass the same checks as fresh output** (readable, shape, float32, finite, unit length when normalising); otherwise it counts as a miss and is recomputed, never raised or returned (Codex review on PR #19).
  - **Tests:** a deterministic fake encoder in CI; a `slow` test with the real `multilingual-e5-small` (pinned, MIT), run locally: a Hindi query ranks its relevant passage first.
- **Consequences:** Passage chunking for mE5's 512-token limit is not done in M0. If mE5's numbers suffer from truncation, that is reported, not hidden.

## D-021 · Passage index and top-k retrieval (2026-11-20)

- **Context:** Deck block diagram "Vector Index" and "Top-k Passages"; CLAUDE.md §3 rows 3–4: retrieval scores are logged and carried into every record. The corpus is a few hundred passages per language.
- **Decision** (`src/indichallu/retrieval/index.py`, `retrieve.py`):
  - **Exact FAISS `IndexFlatIP`** over L2-normalised embeddings, so inner product = cosine. No approximate search is needed at this size.
  - **Deterministic results:** hits are sorted by (score desc, corpus position). To keep FAISS's unspecified choice among tied candidates at its cut-off out of the result, it is asked for extra candidates (`max(2k, k+10)`). If a tie still reaches that cut-off, the query is rescored exactly from the stored vectors. A test with 38 identical passages confirms both paths.
  - **Saved index** (saves are serialised by an exclusive `.save.lock` file, with a stale lock from a crashed writer broken after 10 min; lock-free loads retry if a save removes the version they were about to read; Codex reviews on PR #20): a directory of versioned subdirectories `v-*/` (`index.faiss`, git-ignored; `ids.json`; `meta.json`) plus a `CURRENT` file naming the active one. Saving writes a complete new version, then switches `CURRENT` with **one atomic file replace**, so a crash leaves the previous index loadable and readers never see a missing or half-written index. The previous version is kept for in-flight readers. The metadata records the embedder name, model id, revision and a passages fingerprint (SHA-256 over length-prefixed id/text pairs). `load()` checks the pointer, file integrity, size/dim consistency and any `expect`-ed metadata. (Codex review on PR #20: the first version deleted the old index before installing the new one.)
  - **`Retriever`** embeds questions, searches, and returns the schema's own `RetrievedPassage` objects (passage id, text, score, rank 1..k, is_gold). `gold_rank()` gives the gold passage's rank or None. It accepts only a consistent triple (Codex review on PR #20):
    - the embedder must **normalise** (inner product = cosine only for unit vectors);
    - the index records the **full embedder config** (model, revision, prefixes, max length, normalisation: everything that changes the vectors), and it must match exactly;
    - the passage texts must reproduce the stored **passages fingerprint**, so the same ids with changed text are rejected and evidence attribution can't point at text the vectors weren't built from. A **read-only snapshot** of those texts is kept, so later edits to the caller's mapping change nothing.
- **Consequences:** Retrieval metrics (T16) and the analysis of retrieval vs. generation failures read `is_gold` and `rank` straight from records. An index can't silently be paired with the wrong embedder.
