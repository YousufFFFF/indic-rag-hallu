"""The M0 Hindi pilot set: faithful vs injected answers (M0 task T30, DECISIONS D-025).

For every answerable question with a verified answer offset (D-017):

* **faithful answer** = the gold-passage sentence that contains the answer, copied
  verbatim. It is supported by the gold passage by construction;
* **hallucinated answer** = the same sentence with one injected TEMPORAL or NUMERICAL
  error (:mod:`indichallu.data.synth.inject`), labelled with the exact span and type.

The evidence is the question's retrieved passages, and a question is used only if its
gold passage is among them. The contract is faithfulness to the *retrieved* evidence, so
a faithful answer whose passage wasn't retrieved would be mislabelled.

Known bias (pilot only): faithful answers are verbatim passage sentences, so they are
easy to entail, and injected errors are single-token edits. This set measures whether the
pipeline works end to end, not how good the detector is on natural hallucinations. That
needs the human gold set (M2).
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping, Sequence

from indichallu.data.loaders.indicqa import IndicQA
from indichallu.data.schema import Record, RetrievedPassage, SpanType, Split
from indichallu.data.split import SplitAssignment
from indichallu.data.synth.inject import inject
from indichallu.text.sentseg import sentences


def answer_sentence(passage_text: str, lang: str, start: int, end: int) -> str | None:
    """The sentence of the passage that contains the whole answer span, or None."""
    for s in sentences(passage_text, lang):
        if s.start <= start and end <= s.end:
            return s.text
    return None


def build_pilot(
    ds: IndicQA,
    split: SplitAssignment,
    retrieved: Mapping[str, Sequence[RetrievedPassage]],
    splits: Sequence[Split] = (Split.VAL, Split.TEST),
    seed: int = 13,
) -> tuple[list[Record], Counter]:
    """Pilot records for questions in ``retrieved`` (qa_id -> top-k), plus counts of why
    questions were skipped."""
    rng = random.Random(seed)
    stats: Counter = Counter()
    records: list[Record] = []
    prefer = [SpanType.TEMPORAL, SpanType.NUMERICAL]
    for q in sorted(ds.questions, key=lambda q: q.qa_id):
        s = split.of_passage(q.gold_passage_id)
        if s not in splits or q.qa_id not in retrieved:
            continue
        if not q.answerable or q.answer_start is None:
            stats["skipped: no verified answer"] += 1
            continue
        hits = list(retrieved[q.qa_id])
        if not any(p.is_gold for p in hits):
            stats["skipped: gold passage not retrieved"] += 1
            continue
        sentence = answer_sentence(
            ds.passage(q.gold_passage_id).text, ds.lang, q.answer_start, q.answer_end
        )
        if sentence is None:
            stats["skipped: answer crosses a sentence boundary"] += 1
            continue
        injected = inject(sentence, rng, prefer=prefer[len(records) // 2 % 2])
        if injected is None:
            stats["skipped: nothing to inject (no number, year or month)"] += 1
            continue
        common = {
            "lang": ds.lang,
            "split": s.value,
            "source": "injected",
            "question": q.question,
            "retrieved": [p.model_dump() for p in hits],
        }
        records.append(
            Record.model_validate(
                {
                    **common,
                    "id": f"{q.qa_id}-f",
                    "answer": sentence,
                    "labels": {"response": "faithful", "spans": []},
                }
            )
        )
        records.append(
            Record.model_validate(
                {
                    **common,
                    "id": f"{q.qa_id}-h",
                    "answer": injected.text,
                    "labels": {
                        "response": "hallucinated",
                        "spans": [
                            {
                                "start": injected.start,
                                "end": injected.end,
                                "type": injected.type.value,
                            }
                        ],
                    },
                }
            )
        )
        stats[f"used ({s.value})"] += 1
        stats[f"injected {injected.type.value}"] += 1
    return records, stats
