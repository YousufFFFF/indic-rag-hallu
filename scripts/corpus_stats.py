"""Corpus statistics for the dataset card.

    python scripts/corpus_stats.py compute          # needs data/raw (download_indicqa.py --all)
    python scripts/corpus_stats.py render           # writes docs/dataset_card.md tables
    python scripts/corpus_stats.py render --check   # exit 1 if the card is out of date

`compute` writes results/corpus_stats/indicqa.json (committed). `render` only reads that
JSON, so `render --check` runs in CI without any data.
"""

from __future__ import annotations

import argparse
import json
import sys

import indichallu
from indichallu.data.download import DEFAULT_DATA_ROOT, load_pinned_dataset
from indichallu.data.loaders.indicqa import load_indicqa
from indichallu.data.split import load_split_config
from indichallu.data.stats import (
    DATASET_CARD,
    STATS_JSON,
    compute_language_stats,
    render_tables,
    replace_block,
)
from indichallu.languages import load_languages


def compute(langs: list[str] | None) -> dict:
    dataset = load_pinned_dataset()
    split_config = load_split_config()
    languages = load_languages()
    out = {
        "provenance": {
            "repo": dataset.repo,
            "revision": dataset.revision,
            "licence": dataset.licence,
            "file_sha256": {},
            "split_seed": split_config.seed,
            "split_ratios": {k.value: v for k, v in split_config.ratios.items()},
            "split_group_by": split_config.group_by,
            "indichallu_version": indichallu.__version__,
        },
        "languages": {},
    }
    for code in langs or list(languages):
        path = dataset.local_path(code, DEFAULT_DATA_ROOT)
        if not path.exists():
            print(f"  {code}: skipped (not downloaded: {path})", file=sys.stderr)
            continue
        ds = load_indicqa(path, code)
        out["languages"][code] = compute_language_stats(ds, languages[code], split_config)
        out["provenance"]["file_sha256"][code] = dataset.files[code].sha256
        print(f"  {code}: {ds.stats['passages']} passages, {ds.stats['questions']} questions")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute", help="recompute stats from data/raw into the JSON")
    c.add_argument("--lang", nargs="+", help="languages (default: every configured one)")
    r = sub.add_parser("render", help="render the JSON into docs/dataset_card.md")
    r.add_argument("--check", action="store_true", help="fail if the card is out of date")
    args = parser.parse_args(argv)

    if args.cmd == "compute":
        stats = compute(args.lang)
        STATS_JSON.parent.mkdir(parents=True, exist_ok=True)
        STATS_JSON.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", "utf-8"
        )
        print(f"wrote {STATS_JSON}")
        return 0

    stats = json.loads(STATS_JSON.read_text(encoding="utf-8"))
    card = DATASET_CARD.read_text(encoding="utf-8")
    updated = replace_block(card, render_tables(stats, load_languages()))
    if args.check:
        if updated != card:
            print(f"{DATASET_CARD} is out of date: run `python scripts/corpus_stats.py render`")
            return 1
        print(f"{DATASET_CARD} is up to date")
        return 0
    DATASET_CARD.write_text(updated, encoding="utf-8")
    print(f"wrote {DATASET_CARD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
