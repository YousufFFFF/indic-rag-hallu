"""Retrieve and generate answers for IndicQA questions (M0 tasks T19/T31).

    python scripts/generate_answers.py --lang hi --split test --sample 60 --embedder me5-small

For each question: retrieve the top-k passages from the whole language corpus, build the
source-language prompt from the top ``context_passages`` of them, and generate an answer
under the fixed decoding config (``configs/generation/m0.yaml``), with every generation
cached. Writes to ``results/runs/<run_id>/``:

* ``records.jsonl``: schema-valid records (``source: natural``, no claims or labels yet)
  whose ``retrieved`` is exactly the evidence the generator was shown (scores, ranks and
  is_gold kept), so detection and annotation never credit a passage the model never saw;
* ``retrieval.jsonl``: the full top-k ranking per question, for retrieval metrics.

``--sample N`` takes a deterministic sample: questions ranked by sha256("<seed>:<qa_id>").
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence

from indichallu.data.download import load_pinned_dataset, verified_local_path
from indichallu.data.loaders.indicqa import load_indicqa
from indichallu.data.schema import Record, RetrievedPassage, Split, write_jsonl
from indichallu.data.split import split_dataset
from indichallu.generation.generate import Generator, load_generator_config
from indichallu.generation.prompts import build_prompt
from indichallu.retrieval.embed import REPO_ROOT, Embedder, load_embedder_config
from indichallu.retrieval.retrieve import Retriever, gold_rank
from indichallu.runs import start_run

EMBEDDING_CACHE = REPO_ROOT / "data" / "interim" / "embeddings"  # git-ignored


def sample_ids(ids: list[str], n: int | None, seed: int) -> list[str]:
    if n is not None and n < 0:
        raise ValueError(f"sample size must be >= 0, got {n}")
    ranked = sorted(ids, key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
    return ranked if n is None else ranked[:n]


def retrieval_row(qa_id: str, gold_passage_id: str, hits: Sequence[RetrievedPassage]) -> dict:
    """One line of retrieval.jsonl: the full top-k ranking for a question."""
    return {
        "id": qa_id,
        "gold_passage_id": gold_passage_id,
        "gold_rank": gold_rank(hits),
        "hits": [h.model_dump(mode="json", exclude={"text"}) for h in hits],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lang", default="hi")
    p.add_argument("--split", default="test", choices=[s.value for s in Split])
    p.add_argument("--sample", type=int, default=None, help="number of questions (default: all)")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--embedder", default="bge-m3")
    p.add_argument("--k", type=int, default=10, help="passages retrieved (for retrieval metrics)")
    p.add_argument("--generator", default="m0", help="configs/generation/<name>.yaml")
    p.add_argument("--name", default="generate")
    args = p.parse_args(argv)

    if args.sample is not None and args.sample < 1:
        p.error("--sample must be a positive number of questions")
    gen_cfg = load_generator_config(args.generator)
    if gen_cfg.context_passages > args.k:
        p.error("--k must be >= the generator's context_passages")
    config = {"seed": args.seed, **{k: v for k, v in vars(args).items() if k != "seed"}}

    with start_run(config, name=args.name) as run:
        dataset = load_pinned_dataset()
        ds = load_indicqa(verified_local_path(dataset, args.lang), args.lang)
        split = split_dataset(ds)
        questions = {
            q.qa_id: q
            for q in ds.questions
            if split.of_passage(q.gold_passage_id) == Split(args.split)
        }
        chosen = [questions[i] for i in sample_ids(list(questions), args.sample, args.seed)]

        emb_cfg = load_embedder_config(args.embedder)
        run.add_model("embedder", emb_cfg.model_id, revision=emb_cfg.revision, config=args.embedder)
        run.add_model(
            "generator",
            gen_cfg.model,
            revision=gen_cfg.ollama_digest or gen_cfg.hf_revision,
            backend=gen_cfg.backend,
            hf_source=gen_cfg.hf_source,
            hf_revision=gen_cfg.hf_revision,
        )
        run.meta["dataset"] = {
            "repo": dataset.repo,
            "revision": dataset.revision,
            "lang": args.lang,
        }
        run._save()

        retriever = Retriever.build(
            Embedder(emb_cfg, cache_dir=EMBEDDING_CACHE),
            [x.passage_id for x in ds.passages],
            [x.text for x in ds.passages],
        )
        retrieved = retriever.retrieve(
            [q.question for q in chosen],
            k=args.k,
            gold_passage_ids=[q.gold_passage_id for q in chosen],
        )
        generator = Generator(gen_cfg)
        records, rankings = [], []
        for i, (q, hits) in enumerate(zip(chosen, retrieved, strict=True), start=1):
            evidence = hits[: gen_cfg.context_passages]  # exactly what the model is shown
            prompt = build_prompt(args.lang, q.question, evidence)
            rankings.append(retrieval_row(q.qa_id, q.gold_passage_id, hits))
            start = time.perf_counter()
            gen, cached = generator.generate(prompt)
            records.append(
                Record(
                    id=q.qa_id,
                    lang=args.lang,
                    split=args.split,
                    source="natural",
                    question=q.question,
                    retrieved=evidence,
                    generator=gen_cfg.record_info(),
                    answer=gen.text,
                )
            )
            print(
                f"[{i}/{len(chosen)}] {q.qa_id} gold@{gold_rank(hits)} "
                f"{'cached' if cached else f'{time.perf_counter() - start:.0f}s'} "
                f"answerable={q.answerable}",
                flush=True,
            )
        write_jsonl(records, run.path("records.jsonl"))
        with run.path("retrieval.jsonl").open("w", encoding="utf-8") as f:
            for row in rankings:
                print(json.dumps(row, ensure_ascii=False), file=f)
        print(f"wrote {len(records)} records to {run.dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
