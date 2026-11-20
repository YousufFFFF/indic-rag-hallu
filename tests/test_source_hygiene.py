"""Repo hygiene: no invisible characters written literally in source, config or docs.

Characters like ZWNJ/ZWJ (U+200C/U+200D), BOM or NBSP are invisible in an editor and in
review, so tests and code must spell them as escapes (e.g. "\\u200c") or chr(0x200C).
"""

import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("src", "tests", "configs", "docs", "scripts", "README.md", "CLAUDE.md")
SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".md", ".json", ".txt"}


def _files():
    for entry in SCANNED:
        path = ROOT / entry
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from (p for p in path.rglob("*") if p.is_file() and p.suffix in SUFFIXES)


def _invisible(ch: str) -> bool:
    return unicodedata.category(ch) == "Cf" or ch in (chr(0x00A0), chr(0x202F))


@pytest.mark.parametrize("path", sorted(_files()), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_literal_invisible_characters(path):
    bad = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for col, ch in enumerate(line, start=1):
            if _invisible(ch):
                bad.append(f"line {n} col {col}: U+{ord(ch):04X} {unicodedata.name(ch, '?')}")
    assert not bad, "write these as escapes or chr():\n" + "\n".join(bad)
