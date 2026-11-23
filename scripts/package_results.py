"""Zip the results of one GPU session so they can be copied back into the repo (T44).

    python scripts/package_results.py --out /kaggle/working/results.zip \
        --since 2026-11-25T10:00:00+00:00 --path results/m0

Includes every run folder under ``results/runs/`` whose ``meta.json`` says it started at or
after ``--since`` (so runs from earlier sessions are left out), plus any extra repo paths.
Paths inside the zip are repo-relative, so unzipping at the repo root puts each file where
it belongs. Runs that did not finish are included but reported, never hidden.
``--allow-missing`` skips (and reports) extra paths that don't exist: the runner notebook
passes it only after a command failed, so a crash still yields a zip of what was written.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def runs_since(runs_root: Path, since: datetime) -> list[tuple[Path, dict]]:
    """Run folders started at or after ``since``, oldest first, with their meta."""
    found = []
    for meta_path in sorted(runs_root.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text("utf-8"))
        if datetime.fromisoformat(meta["started_at"]) >= since:
            found.append((meta_path.parent, meta))
    return sorted(found, key=lambda pm: pm[1]["started_at"])


def package(
    out: Path,
    since: datetime,
    extra: list[str],
    root: Path = ROOT,
    allow_missing: bool = False,
) -> list[str]:
    """Write the zip; return the repo-relative paths it contains."""
    if since.tzinfo is None:
        raise ValueError("--since needs a timezone (run start times are UTC)")
    files: list[Path] = []
    for run_dir, meta in runs_since(root / "results" / "runs", since):
        status = meta.get("status")
        print(f"run {run_dir.name}: {status}" + ("" if status == "completed" else "  <- check"))
        files += [f for f in sorted(run_dir.rglob("*")) if f.is_file()]
    for rel in extra:
        p = (root / rel).resolve()
        if not p.is_relative_to(root.resolve()):
            raise ValueError(f"{rel}: outside the repository")
        if not p.exists():
            if allow_missing:
                print(f"{rel}: not found, skipped")
                continue
            raise FileNotFoundError(f"{rel}: not found")
        files += [p] if p.is_file() else [f for f in sorted(p.rglob("*")) if f.is_file()]
    if not files:
        raise RuntimeError("nothing to package: no runs since --since and no --path")
    names = sorted({f.resolve().relative_to(root.resolve()).as_posix() for f in files})
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.write(root / name, name)
    return names


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True, help="zip file to write")
    p.add_argument("--since", required=True, help="ISO time with timezone, e.g. ...+00:00")
    p.add_argument("--path", action="append", default=[], help="extra repo path (repeatable)")
    p.add_argument("--allow-missing", action="store_true", help="skip --path that doesn't exist")
    args = p.parse_args(argv)
    try:
        names = package(
            args.out,
            datetime.fromisoformat(args.since),
            args.path,
            allow_missing=args.allow_missing,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    size = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} ({len(names)} files, {size:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
