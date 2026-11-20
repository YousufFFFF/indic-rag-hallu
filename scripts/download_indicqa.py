"""Download IndicQA files at the pinned revision and verify their SHA-256.

    python scripts/download_indicqa.py            # enabled languages (configs/languages.yaml)
    python scripts/download_indicqa.py --lang hi mr
    python scripts/download_indicqa.py --all      # all configured languages

Files land in data/raw/indicqa/<revision[:12]>/ (git-ignored). Re-running is cheap: a
file that already matches its pinned hash is not downloaded again.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from indichallu.data.download import DEFAULT_DATA_ROOT, fetch, load_pinned_dataset
from indichallu.languages import enabled_languages, load_languages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--lang", nargs="+", help="language codes (default: enabled ones)")
    group.add_argument("--all", action="store_true", help="every configured language")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args(argv)

    dataset = load_pinned_dataset()
    if args.all:
        langs = list(load_languages())
    elif args.lang:
        langs = args.lang
    else:
        langs = list(enabled_languages())
    unknown = [lang for lang in langs if lang not in dataset.files]
    if unknown:
        parser.error(f"no pinned IndicQA file for {unknown}; known: {sorted(dataset.files)}")

    print(f"{dataset.repo}@{dataset.revision[:12]} ({dataset.licence})")
    for lang in langs:
        pinned = dataset.files[lang]
        dest, downloaded = fetch(
            dataset.url(lang),
            dataset.local_path(lang, args.data_root),
            sha256=pinned.sha256,
            size=pinned.size,
        )
        status = "downloaded" if downloaded else "already present"
        print(f"  {lang}: {dest} ({pinned.size:,} bytes, sha256 ok, {status})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
