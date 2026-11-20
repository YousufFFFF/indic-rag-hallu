"""IndicQA loader: SQuAD-style JSON -> passages + QA pairs with a gold passage.

File format (verified on the pinned Hindi file, DECISIONS D-017)::

    {"version": "1.0", "data": [{"title": ..., "paragraphs": [
        {"context": ..., "qas": [{"id": int, "question": ..., "category": "SHORT" | "NO",
                                  "answers": [{"text": ..., "answer_start": int | None}]}]}]}]}

* Unanswerable questions (category ``NO``) have one answer with ``text == ""`` and
  ``answer_start is None``.
* **The given ``answer_start`` is often wrong.** In the Hindi file 894 of 1,052
  answers do not start at their offset. The answer text is always in the context,
  always at or AFTER the given offset (shift 0-60 chars, median 6), and no single unit
  (bytes, UTF-16, marks, whitespace) explains the shift. Offsets are therefore
  realigned (see :func:`realign_answer`) and every QA pair records how its offset was
  obtained.

Passage IDs are content hashes (``hi-p`` + 10 hex chars of SHA-256 of the context), so
identical contexts merge and IDs don't depend on file order. QA IDs are
``hi-q`` + the original numeric id, which is stable at the pinned revision.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from indichallu.languages import load_languages

# Realignment window, from the Hindi data: shifts are 0..60 chars; 80 leaves headroom.
MAX_SHIFT = 80
# A nearest occurrence must be at least this much closer than the next one to be taken.
MIN_MARGIN = 10

AnswerStatus = Literal["exact", "unique", "nearest", "ambiguous", "not_found", "unanswerable"]


@dataclass(frozen=True)
class Passage:
    passage_id: str
    lang: str
    title: str
    text: str


@dataclass(frozen=True)
class QAPair:
    qa_id: str
    lang: str
    question: str
    gold_passage_id: str
    category: str
    answerable: bool
    answer_status: AnswerStatus
    answer_text: str | None  # whitespace-trimmed; None if unanswerable
    answer_start: int | None  # verified offset into the gold passage text, or None
    answer_end: int | None
    given_answer_start: int | None  # as found in the file, for the record


@dataclass
class IndicQA:
    lang: str
    passages: list[Passage]
    questions: list[QAPair]
    stats: Counter = field(default_factory=Counter)

    def passage(self, passage_id: str) -> Passage:
        return self._by_id[passage_id]

    def __post_init__(self) -> None:
        self._by_id = {p.passage_id: p for p in self.passages}


def passage_id(lang: str, text: str) -> str:
    return f"{lang}-p{hashlib.sha256(text.encode('utf-8')).hexdigest()[:10]}"


def realign_answer(
    context: str, text: str, given_start: int | None
) -> tuple[int | None, AnswerStatus]:
    """Find where ``text`` really starts in ``context``.

    Order: the given offset if it is right (``exact``); the only occurrence
    (``unique``); otherwise the nearest occurrence at most ``MAX_SHIFT`` chars after the
    given offset, if it is at least ``MIN_MARGIN`` closer than the next candidate
    (``nearest``). Anything else is ``ambiguous`` (or ``not_found``) and gets no offset.
    """
    if given_start is not None and context.startswith(text, given_start):
        return given_start, "exact"
    hits = []
    i = context.find(text)
    while i != -1:
        hits.append(i)
        i = context.find(text, i + 1)
    if not hits:
        return None, "not_found"
    if len(hits) == 1:
        return hits[0], "unique"
    if given_start is None:
        return None, "ambiguous"
    window = sorted(h - given_start for h in hits if 0 <= h - given_start <= MAX_SHIFT)
    if not window:
        return None, "ambiguous"
    if len(window) > 1 and window[1] - window[0] < MIN_MARGIN:
        return None, "ambiguous"
    return given_start + window[0], "nearest"


def _trim(text: str) -> tuple[str, int]:
    """Strip whitespace; return the stripped text and how many chars were cut in front."""
    stripped = text.lstrip()
    lead = len(text) - len(stripped)
    return stripped.rstrip(), lead


def load_indicqa(path: str | Path, lang: str) -> IndicQA:
    if lang not in load_languages():
        raise ValueError(f"unsupported language {lang!r}")
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("data"), list):
        raise ValueError(f"{path}: expected SQuAD-style JSON with a 'data' list")

    passages: dict[str, Passage] = {}
    questions: list[QAPair] = []
    seen_qa: set[str] = set()
    stats: Counter = Counter()
    for article in raw["data"]:
        title = article.get("title") or ""
        for para in article["paragraphs"]:
            context = para["context"]
            pid = passage_id(lang, context)
            existing = passages.get(pid)
            if existing is None:
                passages[pid] = Passage(pid, lang, title, context)
            elif existing.text != context:  # pragma: no cover - 40-bit hash collision
                raise ValueError(f"passage id collision for {pid}")
            else:
                stats["duplicate_contexts"] += 1
            for qa in para["qas"]:
                qa_id = f"{lang}-q{int(qa['id']):06d}"
                if qa_id in seen_qa:
                    raise ValueError(f"{path}: duplicate question id {qa['id']}")
                seen_qa.add(qa_id)
                answers = qa.get("answers") or []
                if len(answers) != 1:
                    raise ValueError(f"{path}: question {qa['id']} has {len(answers)} answers")
                given = answers[0].get("answer_start")
                text, lead = _trim(answers[0].get("text") or "")
                if not text:
                    start, end, status, answer = None, None, "unanswerable", None
                else:
                    hint = None if given is None else given + lead
                    start, status = realign_answer(context, text, hint)
                    end = None if start is None else start + len(text)
                    answer = text
                stats[status] += 1
                questions.append(
                    QAPair(
                        qa_id=qa_id,
                        lang=lang,
                        question=qa["question"],
                        gold_passage_id=pid,
                        category=qa.get("category", ""),
                        answerable=answer is not None,
                        answer_status=status,
                        answer_text=answer,
                        answer_start=start,
                        answer_end=end,
                        given_answer_start=given,
                    )
                )
    stats["passages"] = len(passages)
    stats["questions"] = len(questions)
    return IndicQA(lang, list(passages.values()), questions, stats)
