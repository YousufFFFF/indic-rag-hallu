# Stage I Review Deck — extracted text

Source: `Hallucination-Detection-in-RAG-for-Low-Resource-Indic-Languages.pptx` (25 slides).
Everything here is a commitment made to faculty. Bracketed `[INSERT …]` values are filled ONLY from measured runs.

## Slide 1

- HALLUCINATION  ·  RETRIEVAL-AUGMENTED GENERATION  ·  LOW-RESOURCE NLP
- Hallucination Detection in RAG
- for Low-Resource Indic Languages
- A claim-level framework for verifying whether an LLM answer in an Indian language is actually supported by its retrieved evidence
- QUERY
- RETRIEVE
- CONTEXT
- GENERATE
- DETECT
- PRESENTED BY
- Mohammed Yousuf Ansari
- [GROUP MEMBER 2]
- [GROUP MEMBER 3]
- [GROUP MEMBER 4]
- PROJECT GUIDE
- Prof. [GUIDE NAME]
- DEPARTMENT
- Artificial Intelligence & Data Science
- JSPM's Narhe Technical Campus, Pune
- ACADEMIC YEAR
- 2026 – 2027
- Stage I Review

> Speaker notes: Title slide. Introduce the team, the guide and the one-line framing: we are not asking whether an answer is true in general, we are asking whether the retrieved evidence supports it.

## Slide 2

- AGENDA
- Contents
- 01
- Introduction
- 02
- Problem Statement
- 03
- Objectives
- 04
- Literature Survey
- 05
- Research Gap
- 06
- Proposed Methodology
- 07
- Block Diagram
- 08
- Dataset & Languages
- 09
- Hardware & Software Requirements
- 10
- Evaluation Metrics
- 11
- Expected Outcome
- 12
- Limitations & Future Scope
- 13
- References
- AGENDA
- 02

> Speaker notes: Agenda. Walk it quickly — spend time on Research Gap, Block Diagram and Evaluation.

## Slide 3

- 01
- The Problem
- Why fluent output is not the same as truthful output.
- SECTION 01
- 03

> Speaker notes: Section break: the problem.

## Slide 4

- 01  ·  INTRODUCTION
- A hallucination is fluent text that
- no source supports
- Fluency ≠ factuality
- Large language models optimise for likely next tokens, not for verified statements.
- Confident by construction
- The wrong answer is delivered in exactly the same tone as the right one.
- Costly in real deployments
- Health, legal, agricultural and government-scheme advice are read as authoritative.
- ILLUSTRATIVE EXAMPLE  ·  HINDI
- QUERY
- इस योजना के लिए पात्रता क्या है?
- RETRIEVED PASSAGE
- …योजना के लाभ राज्य सरकार द्वारा दिए जाते हैं।
- MODEL ANSWER
- …आवेदक की आयु 21 वर्ष होनी चाहिए।
- UNSUPPORTED
- The age condition appears nowhere in the retrieved passage — it is generated, not grounded.
- 01 · THE PROBLEM
- 04

> Speaker notes: Define hallucination precisely: fluent text no source supports. Walk the Hindi example left to right — the age condition never appears in the retrieved passage.

## Slide 5

- 01  ·  MOTIVATION
- Everything that makes hallucination
- hard gets harder in Indic languages
- 01
- Sparse corpora
- Far less high-quality, clean, domain-specific text per language than English — so both the retriever index and the generator are thinner.
- 02
- Script & morphology
- Nine scripts, rich inflection and compounding break naive tokenisation, string matching and span alignment.
- 03
- Weak retrieval
- Multilingual embedding models are dominated by high-resource data, so relevant Indic passages are often ranked below the cut-off.
- 04
- No evaluation resource
- Almost no annotated hallucination benchmarks exist for these languages, so failures are invisible to standard evaluation.
- LANGUAGES IN SCOPE
- हिन्दी
- বাংলা
- தமிழ்
- తెలుగు
- मराठी
- ಕನ್ನಡ
- മലയാളം
- ગુજરાતી
- ਪੰਜਾਬੀ
- Final language set to be fixed by corpus availability — see Dataset slide.
- 01 · THE PROBLEM
- 05

> Speaker notes: Four compounding reasons the problem is worse in Indic languages. Note the language set is provisional and fixed by corpus availability.

## Slide 6

- 01  ·  BACKGROUND
- Retrieval-Augmented Generation
- Ground the model in fetched documents instead of trusting its parameters alone.
- QUERY
- User question
- (Indic)
- RETRIEVER
- Multilingual
- encoder + index
- PASSAGES
- Top-k retrieved
- evidence
- GENERATOR
- LLM conditioned
- on context
- ANSWER
- Generated
- response
- RETRIEVER
- Encodes the query and the corpus into one vector space, then returns the top-k nearest passages.
- CONTEXT WINDOW
- Retrieved passages are concatenated into the prompt as the evidence the answer must rest on.
- GENERATOR
- The LLM conditions on that evidence — but is never forced to stay inside it.
- RAG improves grounding. It does not guarantee it.
- 01 · THE PROBLEM
- 06

> Speaker notes: Explain RAG in one pass along the pipeline, then land the punchline: RAG improves grounding, it does not guarantee it.

## Slide 7

- 01  ·  THE GAP IN PRACTICE
- Where hallucinations still get through
- QUERY
- User question
- (Indic)
- RETRIEVER
- Multilingual
- encoder + index
- PASSAGES
- Top-k retrieved
- evidence
- GENERATOR
- LLM conditioned
- on context
- ANSWER
- Generated
- response
- A
- B
- C
- A
- Retrieval miss
- No relevant Indic passage is ranked in the top-k — the model quietly falls back on parametric memory.
- B
- Partial grounding
- The answer mixes supported spans with invented entities, numbers or dates in the same sentence.
- C
- Context contradiction
- The model overrides or mistranslates the evidence, producing cross-lingual drift away from the source.
- None of these are visible to the user — the output still reads as a confident, well-formed answer.
- 01 · THE PROBLEM
- 07

> Speaker notes: Three concrete failure modes, keyed A/B/C to the markers on the pipeline. Emphasise that none of them are visible to the end user.

## Slide 8

- 02  ·  PROBLEM STATEMENT
- Current RAG systems offer no reliable, language-aware way to verify whether a generated answer in a low-resource Indic language is actually supported by the evidence retrieved for it.
- Detection is response-level
- Existing detectors label a whole answer, hiding which span is unsupported.
- English-centric evidence
- Alignment methods assume English tokenisation, entity forms and NLI models.
- No Indic ground truth
- Without annotated Indic RAG responses, detector quality cannot be measured.
- 02 · PROBLEM STATEMENT
- 08

> Speaker notes: Read the problem statement aloud, then the three supporting facts underneath.

## Slide 9

- 03  ·  OBJECTIVES
- Objectives
- Five deliverables that together close the loop from retrieval to verified answer.
- 01
- Build the pipeline
- Assemble a RAG pipeline over an Indic document corpus for the selected low-resource languages.
- 02
- Create ground truth
- Adapt or construct an annotated set of RAG responses labelled at both response and span level.
- 03
- Design the detector
- Decompose each answer into atomic claims and align every claim against the retrieved evidence.
- 04
- Evaluate rigorously
- Measure detection quality per language and per error type — entity, numeric, temporal, location.
- 05
- Analyse the drivers
- Quantify how retrieval quality and language resource level relate to hallucination rate.
- Scope note — the number of languages and corpus size are fixed in the Dataset slide as [INSERT LANGUAGE COUNT] and [INSERT DATASET SIZE].
- 03 · OBJECTIVES
- 09

> Speaker notes: Five objectives. Each one maps to a stage in the methodology slide coming up.

## Slide 10

- 02
- Prior Work & the Gap
- Five works that get close — and what each of them leaves open.
- SECTION 02
- 10

> Speaker notes: Section break: prior work.

## Slide 11

- 04  ·  LITERATURE SURVEY  ·  1 / 2
- Literature Survey
- PAPER & YEAR
- MAIN CONTRIBUTION
- LIMITATION / GAP
- 01
- RAGTruth
- Niu et al., 2024
- Built a corpus of nearly 18,000 RAG-generated responses with human annotations for hallucination detection at both response and word/span level.
- Confined to standard RAG settings in English; low-resource Indic languages are not studied.
- 02
- Detecting Hallucination & Coverage Errors in RAG
- Chang et al., 2024
- Detects hallucination and coverage errors in RAG using evidence-based techniques together with LLM classifiers.
- Evaluated on one specific RAG setting over controversial topics; Indic low-resource languages are not addressed.
- 03
- MT Hallucination Detection for Low & High Resource Languages
- Benkirane et al., 2024
- Studies hallucination detection across both high-resource and low-resource language directions.
- Targets machine translation rather than RAG, and does not focus on Indic languages.
- 04 · LITERATURE SURVEY
- 11

> Speaker notes: Literature survey, papers 1-3. For each, state contribution then the gap in one sentence.

## Slide 12

- 04  ·  LITERATURE SURVEY  ·  2 / 2
- Literature Survey
- PAPER & YEAR
- MAIN CONTRIBUTION
- LIMITATION / GAP
- 04
- HalluSearch
- Abdallah & El-Beltagy, 2025
- Uses factual splitting, context retrieval and hallucination verification to detect hallucinated spans across 14 languages, Hindi among them.
- Limited online resources, morphological complexity and retrieval quality degrade performance on less-resourced languages.
- 05
- Multi-Hall-SA
- Ralethe & Buys, 2026
- Introduces a hallucination-detection benchmark for four low-resource South African languages, covering temporal, entity, numerical and location errors.
- Covers South African rather than Indic languages, and provides a benchmark more than a detection method.
- Common thread — every method assumes either a high-resource language or a non-Indic low-resource setting.
- 04 · LITERATURE SURVEY
- 12

> Speaker notes: Papers 4-5, then the common thread line at the bottom.

## Slide 13

- 05  ·  RESEARCH GAP
- Every dimension is covered — the intersection is not
- Coverage across the five surveyed works (n = 5)
- THE GAP
- No existing work performs span-level hallucination detection inside a RAG pipeline for low-resource Indic languages.
- Benchmarks exist for English and for South African languages — not for Indic ones.
- Hindi appears only as one of many languages, never as the design target.
- Low-resource hallucination work is framed around translation, not retrieval.
- 05 · RESEARCH GAP
- 13

> Speaker notes: The chart is derived only from the five surveyed works — no external statistic is claimed. The intersection column is zero: that is the gap.

## Slide 14

- 03
- The Proposed System
- From a retrieved passage to a per-claim verdict.
- SECTION 03
- 14

> Speaker notes: Section break: the proposed system.

## Slide 15

- 06  ·  PROPOSED METHODOLOGY
- Four stages, one contract
- Every claim the system produces must point back at a retrieved passage.
- I
- Corpus & Retrieval
- Build a per-language document corpus and index it with a multilingual embedding model.
- Retrieve top-k passages; log retrieval scores for later attribution.
- II
- Grounded Generation
- Assemble prompt = query + retrieved context in the source language.
- Generate the answer with the selected LLM under a fixed decoding configuration.
- III
- Claim-Level Verification
- Split the answer into atomic claims, then align each claim with the retrieved evidence.
- Score entailment; aggregate to a response verdict while keeping span attribution.
- IV
- Evaluation & Analysis
- Compare predicted verdicts against human annotation, per language and per error type.
- Relate retrieval quality to hallucination rate to separate the two failure sources.
- The detector never asks “is this true?” — it asks “does the retrieved evidence say this?”
- 06 · PROPOSED METHODOLOGY
- 15

> Speaker notes: Four stages. The contract line at the bottom is the core idea of the project.

## Slide 16

- 07  ·  BLOCK DIAGRAM
- System Architecture
- INPUT &
- RETRIEVAL
- GENERATION
- DETECTION
- VERDICT
- Indic
- User Query
- Normalisation &
- Script Handling
- Multilingual
- Embedder
- Vector Index
- (Indic corpus)
- Top-k
- Passages
- Prompt Assembly
- (query + context)
- LLM Generator
- Generated Answer
- Claim Decomposition
- (atomic claims)
- Claim–Evidence Alignment
- (entailment scoring)
- Aggregation &
- Span Attribution
- SUPPORTED
- UNSUPPORTED
- CONTRADICTED
- Evaluation vs. annotation
- Retrieval scores are carried forward so a retrieval failure can be told apart from a generation failure.
- 07 · BLOCK DIAGRAM
- 16

> Speaker notes: Build the architecture band by band on click: input and retrieval, generation, detection, verdict. Explain that retrieval scores are carried forward so the two failure sources stay separable.

## Slide 17

- 07  ·  DETECTION MECHANISM
- How the verdict is produced
- 01
- Decompose
- The generated answer is split into atomic, independently checkable claims.
- 02
- Align
- Each claim is scored against every retrieved passage — entail / neutral / contradict.
- 03
- Aggregate
- Claim scores roll up to a response verdict while the failing span stays attributable.
- Decision threshold: [INSERT THRESHOLD]
- WORKED EXAMPLE  ·  ILLUSTRATIVE
- GENERATED ANSWER
- यह योजना राज्य सरकार द्वारा चलाई जाती है और आवेदक की आयु 21 वर्ष होनी चाहिए।
- CLAIM 1
- योजना राज्य सरकार द्वारा चलाई जाती है
- ENTAILED BY PASSAGE
- SUPPORTED
- CLAIM 2
- आवेदक की आयु 21 वर्ष होनी चाहिए
- NO SUPPORTING EVIDENCE
- UNSUPPORTED
- 07 · DETECTION MECHANISM
- 17

> Speaker notes: Detection mechanism in detail. Walk the worked example: one claim entailed, one claim unsupported, from the same sentence.

## Slide 18

- 08  ·  DATASET & LANGUAGES
- Dataset design
- Figures below are fixed once corpus collection and annotation are complete.
- Languages in scope
- [INSERT COUNT] of nine
- Document corpus
- [INSERT CORPUS SOURCE]
- Question–answer pairs
- [INSERT DATASET SIZE]
- Annotated RAG responses
- [INSERT NO. OF RESPONSES]
- Train / val / test split
- [INSERT SPLIT]
- Annotator agreement
- [INSERT AGREEMENT SCORE]
- ANNOTATION TAXONOMY
- ENTITY
- A person, place or organisation not present in the evidence.
- NUMERICAL
- Quantities, amounts or counts that the passage never states.
- TEMPORAL
- Dates, durations or sequences invented or shifted.
- LOCATION
- Geographic references that the evidence does not support.
- Response- and span-level labelling follows RAGTruth; the four error categories follow Multi-Hall-SA.
- 08 · DATASET & LANGUAGES
- 18

> Speaker notes: Dataset design. Be explicit that the bracketed figures are unfilled until corpus collection and annotation finish.

## Slide 19

- 04
- Evaluation & Outcome
- What we will build on, what we will measure, and what we expect to show.
- SECTION 04
- 19

> Speaker notes: Section break: evaluation and outcome.

## Slide 20

- 09  ·  HARDWARE & SOFTWARE REQUIREMENTS
- Requirements
- Planned configuration — final model choice may revise the GPU requirement.
- HARDWARE
- GPU
- ≥ 16 GB VRAM  ·  [INSERT GPU MODEL]
- System memory
- 16 GB RAM minimum
- Storage
- ~100 GB for corpus, embeddings and vector indices
- Processor
- x86-64, quad-core or better
- Environment
- Local workstation or hosted GPU notebook
- SOFTWARE
- Language
- Python 3.11
- Modelling
- PyTorch  ·  HuggingFace Transformers & Datasets
- Retrieval
- sentence-transformers  ·  FAISS or ChromaDB
- Orchestration
- LangChain or LlamaIndex
- Indic tooling
- IndicNLP Library  ·  IndicTrans2
- Evaluation
- scikit-learn  ·  Matplotlib  ·  Jupyter
- 09 · HARDWARE & SOFTWARE
- 20

> Speaker notes: Requirements. Flag that the GPU line may change with final model selection.

## Slide 21

- 10  ·  EVALUATION METRICS
- What we measure, and why it is the right measure
- DETECTION QUALITY
- Precision
- Recall
- F1
- Balanced accuracy
- AUROC
- Hallucinated spans are the minority class — plain accuracy would look excellent while missing almost all of them.
- SPAN LOCALISATION
- Span-level F1
- Token IoU
- Exact-span match
- A presenter needs to know which words are unsupported, not only that something in the answer is.
- RETRIEVAL QUALITY
- Recall@k
- nDCG@k
- MRR
- Separates a retrieval failure from a generation failure — the two demand completely different fixes.
- CROSS-LINGUAL FAIRNESS
- Macro-F1 over languages
- Per-language breakdown
- Worst-language F1
- Averaging over languages stops a strong Hindi result from hiding weak performance elsewhere.
- All metrics are reported per language as well as pooled — pooled-only numbers hide exactly the failures this project is about.
- 10 · EVALUATION METRICS
- 21

> Speaker notes: Metrics with the justification for each group. The last line is the important one — per-language reporting.

## Slide 22

- 11  ·  EXPECTED OUTCOME
- Expected Outcome
- Working pipeline
- An end-to-end RAG system with an attached hallucination detector for the selected Indic languages.
- Annotated resource
- A span-annotated set of Indic RAG responses that did not previously exist for these languages.
- Span-level detection
- Verdicts at claim level rather than a single label for the whole answer.
- Diagnostic analysis
- Evidence on how much of the hallucination rate is caused by retrieval versus generation.
- RESULTS  ·  TO BE FILLED AFTER EXPERIMENTS
- MACRO-F1
- [INSERT F1]
- SPAN F1
- [INSERT SPAN F1]
- RECALL@K
- [INSERT RECALL]
- BASELINE Δ
- [INSERT DELTA]
- LANGUAGES
- [INSERT COUNT]
- No experimental numbers are claimed at this stage — every slot above is filled from measured runs.
- 11 · EXPECTED OUTCOME
- 22

> Speaker notes: Expected outcome. State clearly that no results are claimed yet; the panel is filled after experiments.

## Slide 23

- 12  ·  LIMITATIONS & FUTURE SCOPE
- What this will not do — and what comes next
- LIMITATIONS
- Coverage is bounded by what Indic corpora actually exist for each language.
- Manual span annotation is expensive, which caps dataset size.
- Entailment models for Indic languages are themselves imperfect and add error.
- Code-mixed and transliterated input is out of scope for the first version.
- Findings hold for the chosen domains and may not transfer unchanged.
- FUTURE SCOPE
- Extend to code-mixed Hinglish and Romanised Indic input.
- Move from detection to correction — regenerate the unsupported span from evidence.
- Add speech input so voice assistants inherit the same guarantee.
- Scale to the remaining Indic languages and to additional domains.
- Release the annotated set as an open benchmark for the community.
- 12 · LIMITATIONS & FUTURE SCOPE
- 23

> Speaker notes: Limitations first, honestly, then future scope. Do not oversell.

## Slide 24

- 13  ·  REFERENCES
- References
- [1]
- Niu, Y. et al. “RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models.” 2024.
- [2]
- Chang, C.-Y. et al. “Detecting Hallucination and Coverage Errors in Retrieval Augmented Generation for Controversial Topics.” 2024.
- [3]
- Benkirane, K. et al. “Machine Translation Hallucination Detection for Low and High Resource Languages using Large Language Models.” 2024.
- [4]
- Abdallah, M. and El-Beltagy, S. R. “HalluSearch: Factual Splitting, Context Retrieval and Hallucination Verification for Multilingual Span Detection.” 2025.
- [5]
- Ralethe, S. and Buys, J. “Multi-Hall-SA: A Hallucination Detection Benchmark for Low-Resource South African Languages.” 2026.
- Titles and years follow the literature survey compiled for this project; venue details are omitted pending final verification.
- 13 · REFERENCES
- 24

> Speaker notes: References from the literature survey.

## Slide 25

- QUESTIONS  ·  DISCUSSION
- Thank you
- Hallucination Detection in RAG for Low-Resource Indic Languages
- RAG / RETRIEVAL
- HALLUCINATION
- LOW-RESOURCE NLP
- INDIC LANGUAGES
- EVALUATION
- THANK YOU
- 25

> Speaker notes: Close and invite questions.
