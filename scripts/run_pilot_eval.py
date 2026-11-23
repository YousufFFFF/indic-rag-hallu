"""M0 pilot evaluation: retrieval metrics + our detector vs baseline B1 (T19, T30, T34, T35).

    python scripts/run_pilot_eval.py --embedder me5-small

1. Retrieve top-k for every Hindi val/test question (whole corpus) -> Recall@k, nDCG@k,
   MRR on test, per language and pooled (T19).
2. Build the pilot set from the val/test questions whose gold passage is in the top
   ``context_passages`` (T30, D-025).
3. Run our detector and B1 on val and test with the pinned NLI model; tune each one's
   thresholds on VAL only (T34); apply them to TEST (T35).
4. Write everything to ``results/runs/<run_id>/`` and the summary that the README and
   paper tables are generated from to ``results/m0/pilot.json``.

Pilot = synthetic Hindi data only; see the bias note in ``data/synth/pilot.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from indichallu.data.download import load_pinned_dataset, verified_local_path
from indichallu.data.loaders.indicqa import load_indicqa
from indichallu.data.schema import Split, write_jsonl
from indichallu.data.split import split_dataset
from indichallu.data.synth.pilot import build_pilot
from indichallu.detection.decompose import decompose
from indichallu.detection.detector import Detector, b1_claims, load_detection_config, load_nli
from indichallu.detection.tune import redecide, tune_thresholds
from indichallu.eval.detection import response_level_metrics
from indichallu.eval.fairness import by_language
from indichallu.eval.protocol import load_protocol
from indichallu.eval.retrieval import Query, retrieval_metrics
from indichallu.eval.spans import span_metrics_by_type
from indichallu.retrieval.embed import REPO_ROOT, Embedder, load_embedder_config
from indichallu.retrieval.retrieve import Retriever
from indichallu.runs import start_run

EMBEDDING_CACHE = REPO_ROOT / "data" / "interim" / "embeddings"
SUMMARY = REPO_ROOT / "results" / "m0" / "pilot.json"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lang", default="hi")
    p.add_argument("--embedder", default="bge-m3")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--context-passages", type=int, default=3)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--detection", default="m0")
    p.add_argument("--protocol", default="m0", help="configs/eval/<name>.yaml")
    p.add_argument("--require-clean", action="store_true", help="refuse a dirty git tree")
    args = p.parse_args(argv)
    config = {"seed": args.seed, **{k: v for k, v in vars(args).items() if k != "seed"}}

    protocol = load_protocol(args.protocol)
    protocol.check_split(Split.TEST)  # the pilot evaluates test: only allowed under m0
    with start_run(config, name="m0-pilot", require_clean=args.require_clean) as run:
        run.meta["protocol"] = protocol.record()
        dataset = load_pinned_dataset()
        ds = load_indicqa(verified_local_path(dataset, args.lang), args.lang)
        split = split_dataset(ds)
        emb_cfg = load_embedder_config(args.embedder)
        det_cfg = load_detection_config(args.detection)
        run.add_model("embedder", emb_cfg.model_id, revision=emb_cfg.revision, config=args.embedder)
        run.add_model("nli", det_cfg["nli"]["model_id"], revision=det_cfg["nli"]["revision"])

        # 1. retrieval over the whole corpus, for every val/test question
        retriever = Retriever.build(
            Embedder(emb_cfg, cache_dir=EMBEDDING_CACHE),
            [x.passage_id for x in ds.passages],
            [x.text for x in ds.passages],
        )
        questions = [
            q
            for q in ds.questions
            if split.of_passage(q.gold_passage_id) in (Split.VAL, Split.TEST)
        ]
        hits = retriever.retrieve(
            [q.question for q in questions],
            k=args.k,
            gold_passage_ids=[q.gold_passage_id for q in questions],
        )
        # one gold passage per question; relevance = is_gold in rank order
        test_queries = [
            Query(tuple(p.is_gold for p in h), n_relevant=1)
            for q, h in zip(questions, hits, strict=True)
            if split.of_passage(q.gold_passage_id) is Split.TEST
        ]
        per_lang = retrieval_metrics(test_queries, ks=(1, 3, 5, 10))
        # single language in M0: pooled == the language's own value
        retrieval = {"per_language": {ds.lang: per_lang}, "pooled": per_lang}
        print("retrieval (test):", json.dumps(retrieval["pooled"]), flush=True)

        # 2. pilot set: top-`context_passages` as the evidence
        top = {q.qa_id: h[: args.context_passages] for q, h in zip(questions, hits, strict=True)}
        pilot, pilot_stats = build_pilot(ds, split, top, seed=args.seed)
        print("pilot set:", dict(pilot_stats), flush=True)

        # 3. detect with ours and B1; tune on val; evaluate on test
        nli = load_nli(det_cfg)
        results = {}
        for name, claim_fn in (("ours", decompose), ("b1_sentence_nli", b1_claims)):
            det = Detector.from_config(det_cfg, ds.lang, nli=nli, claim_fn=claim_fn)
            scored = [det.detect(r) for r in pilot]
            val = [r for r in scored if r.split is Split.VAL]
            test = [r for r in scored if r.split is Split.TEST]
            th, grid = tune_thresholds(val, objective=protocol.threshold_objective)
            decided = [redecide(r, th) for r in test]
            write_jsonl(decided, run.path(f"{name}.test.jsonl"))
            results[name] = {
                "thresholds": asdict(th),
                "response": by_language(decided, response_level_metrics),
                "spans": by_language(decided, lambda rs: span_metrics_by_type(rs)["ALL"]),
                "spans_by_type": span_metrics_by_type(decided),
                "val_grid_best": max(grid, key=lambda g: g["span_f1"] or 0),
            }
            print(
                name,
                "thresholds",
                asdict(th),
                "test response F1",
                results[name]["response"]["pooled"]["f1"],
                "span F1",
                results[name]["spans"]["pooled"]["span_f1"],
                flush=True,
            )

        summary = {
            "label": "pilot — synthetic Hindi data only",
            "run_id": run.id,
            "git_commit": run.meta["git"]["commit"],
            "dirty": run.meta["git"]["dirty"],
            "embedder": {
                "name": args.embedder,
                "model_id": emb_cfg.model_id,
                "revision": emb_cfg.revision,
            },
            "nli": {
                "model_id": det_cfg["nli"]["model_id"],
                "revision": det_cfg["nli"]["revision"],
                "device": getattr(nli, "device", None),  # D-026
            },
            "dataset": {"repo": dataset.repo, "revision": dataset.revision, "lang": args.lang},
            "retrieval_test": retrieval,
            "pilot_set": dict(pilot_stats),
            "pilot_records": {
                s.value: sum(r.split is s for r in pilot) for s in (Split.VAL, Split.TEST)
            },
            "detection_test": results,
        }
        run.path("summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), "utf-8"
        )
        SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", "utf-8"
        )
        print(f"wrote {SUMMARY} and {run.dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
