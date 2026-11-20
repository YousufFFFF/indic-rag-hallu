"""Deterministic, leakage-free train/val/test splits (DECISIONS D-018).

Splits are made over *groups* (by default the source article title; a passage with no
title is its own group), so a passage, every question about it, and every other passage
of the same article land in exactly one split. Groups that share an identical question
text are merged first (IndicQA Hindi repeats 3 questions across articles, e.g.
"भारत को स्वतंत्रता किस वर्ष मिली थी?"), so no question appears in two splits.

Assignment does not use :mod:`random`: groups are ranked by ``sha256("<seed>:<key>")``
and cut into exact counts (largest-remainder rounding of the ratios). The result
depends only on the seed, the ratios and the set of group keys: not on input order,
Python version or platform.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from indichallu.data.loaders.indicqa import IndicQA, Passage
from indichallu.data.schema import Split

REPO_ROOT = Path(__file__).resolve().parents[3]
SPLIT_CONFIG = REPO_ROOT / "configs" / "data" / "splits.yaml"
SPLITS = (Split.TRAIN, Split.VAL, Split.TEST)
GroupBy = Literal["title", "passage"]


class LeakageError(AssertionError):
    """The same passage, article or question text appears in more than one split."""


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    ratios: dict[Split, float]
    group_by: GroupBy = "title"

    def __post_init__(self) -> None:
        if set(self.ratios) != set(SPLITS):
            raise ValueError(f"ratios must give train, val and test, got {sorted(self.ratios)}")
        if any(r < 0 for r in self.ratios.values()):
            raise ValueError("ratios must be non-negative")
        if abs(sum(self.ratios.values()) - 1.0) > 1e-9:
            raise ValueError(f"ratios must sum to 1, got {sum(self.ratios.values())}")
        if self.group_by not in ("title", "passage"):
            raise ValueError(f"group_by must be 'title' or 'passage', got {self.group_by!r}")


def load_split_config(path: str | Path = SPLIT_CONFIG) -> SplitConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    ratios = {Split(k): float(v) for k, v in raw["ratios"].items()}
    return SplitConfig(seed=int(raw["seed"]), ratios=ratios, group_by=raw.get("group_by", "title"))


def group_key(passage: Passage, group_by: GroupBy) -> str:
    if group_by == "title" and passage.title.strip():
        return f"title:{passage.title.strip()}"
    return f"passage:{passage.passage_id}"


def split_counts(n: int, ratios: dict[Split, float]) -> dict[Split, int]:
    """Exact counts per split summing to ``n`` (largest-remainder rounding).

    Every split with a non-zero ratio gets at least one item when ``n`` allows, so a
    small dataset can't silently end up with an empty test split.

    >>> split_counts(261, {Split.TRAIN: 0.6, Split.VAL: 0.2, Split.TEST: 0.2})
    {<Split.TRAIN: 'train'>: 157, <Split.VAL: 'val'>: 52, <Split.TEST: 'test'>: 52}
    """
    exact = {s: n * ratios[s] for s in SPLITS}
    counts = {s: int(exact[s]) for s in SPLITS}
    by_remainder = sorted(SPLITS, key=lambda s: (-(exact[s] - counts[s]), SPLITS.index(s)))
    for s in by_remainder[: n - sum(counts.values())]:
        counts[s] += 1
    wanted = [s for s in SPLITS if ratios[s] > 0]
    if n >= len(wanted):
        for s in wanted:
            if counts[s] == 0:
                donor = max(SPLITS, key=lambda d: (counts[d], -SPLITS.index(d)))
                counts[donor] -= 1
                counts[s] += 1
    return counts


def _rank(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def assign_groups(keys: set[str], config: SplitConfig) -> dict[str, Split]:
    """Map every group key to a split: rank by hash, then cut into exact counts."""
    ranked = sorted(keys, key=lambda k: (_rank(config.seed, k), k))
    counts = split_counts(len(ranked), config.ratios)
    out: dict[str, Split] = {}
    i = 0
    for s in SPLITS:
        for key in ranked[i : i + counts[s]]:
            out[key] = s
        i += counts[s]
    return out


@dataclass(frozen=True)
class SplitAssignment:
    config: SplitConfig
    passage_split: dict[str, Split]  # passage_id -> split

    def of_passage(self, passage_id: str) -> Split:
        return self.passage_split[passage_id]

    def passages(self, split: Split) -> list[str]:
        return sorted(p for p, s in self.passage_split.items() if s == split)


def _merge_by_shared_questions(ds: IndicQA, key_of: dict[str, str]) -> dict[str, str]:
    """Union groups that contain an identical question text, so the same question can't
    appear in two splits (IndicQA Hindi repeats a few questions across articles)."""
    parent: dict[str, str] = {k: k for k in key_of.values()}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    first_group_of_question: dict[str, str] = {}
    for q in ds.questions:
        text = q.question.strip()
        g = key_of[q.gold_passage_id]
        if text in first_group_of_question:
            a, b = find(first_group_of_question[text]), find(g)
            if a != b:
                parent[max(a, b)] = min(a, b)  # deterministic representative
        else:
            first_group_of_question[text] = g
    return {pid: find(k) for pid, k in key_of.items()}


def split_dataset(ds: IndicQA, config: SplitConfig | None = None) -> SplitAssignment:
    """Assign every passage (and so its questions) to a split, without leakage."""
    config = config or load_split_config()
    key_of = {p.passage_id: group_key(p, config.group_by) for p in ds.passages}
    key_of = _merge_by_shared_questions(ds, key_of)
    group_split = assign_groups(set(key_of.values()), config)
    return SplitAssignment(config, {pid: group_split[k] for pid, k in key_of.items()})


def check_no_leakage(ds: IndicQA, assignment: SplitAssignment) -> None:
    """Raise :class:`LeakageError` if anything that must stay together is split up.

    Checks: every passage is assigned; no article title spans two splits (when grouping
    by title); no identical passage text or identical question text in two splits.
    """
    missing = {p.passage_id for p in ds.passages} - set(assignment.passage_split)
    if missing:
        raise LeakageError(f"{len(missing)} passages have no split")
    problems = []
    by_title: dict[str, set[Split]] = defaultdict(set)
    by_text: dict[str, set[Split]] = defaultdict(set)
    for p in ds.passages:
        s = assignment.of_passage(p.passage_id)
        by_text[p.text].add(s)
        if assignment.config.group_by == "title" and p.title.strip():
            by_title[p.title.strip()].add(s)
    problems += [f"article {t!r} in {sorted(s)}" for t, s in by_title.items() if len(s) > 1]
    problems += [f"passage text in {sorted(s)}" for s in by_text.values() if len(s) > 1]
    by_question: dict[str, set[Split]] = defaultdict(set)
    for q in ds.questions:
        by_question[q.question.strip()].add(assignment.of_passage(q.gold_passage_id))
    problems += [f"question {t!r} in {sorted(s)}" for t, s in by_question.items() if len(s) > 1]
    if problems:
        raise LeakageError(f"{len(problems)} leaks, e.g. " + "; ".join(problems[:5]))


def split_stats(ds: IndicQA, assignment: SplitAssignment) -> dict[Split, Counter]:
    """Per split: passages, questions, answerable and unanswerable questions."""
    stats = {s: Counter() for s in SPLITS}
    for p in ds.passages:
        stats[assignment.of_passage(p.passage_id)]["passages"] += 1
    for q in ds.questions:
        c = stats[assignment.of_passage(q.gold_passage_id)]
        c["questions"] += 1
        c["answerable" if q.answerable else "unanswerable"] += 1
    return stats
