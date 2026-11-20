"""T36 spot-check sheet: natural generations + detector verdicts, for a Hindi speaker.

A sanity check only (PLAN T36): the reviewer ticks whether each answer is faithful to the
passages the generator was shown. It is not gold data and is never a headline number.
"""

from __future__ import annotations

from collections.abc import Sequence

from indichallu.data.schema import Record


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", r"\|")


def spotcheck_markdown(records: Sequence[Record], title: str = "T36 spot-check") -> str:
    out = [
        f"# {title}",
        "",
        "For each answer: is everything it says supported by passages [1]–[k] below? Tick one "
        "box and note any unsupported part. The detector's verdicts are shown for comparison; "
        "judge the answer yourself first. (Sanity check only: not gold, not a reported result.)",
        "",
    ]
    for n, r in enumerate(records, start=1):
        gold = next((p.rank for p in r.retrieved if p.is_gold), None)
        out += [
            f"## {n}. `{r.id}`",
            "",
            f"**प्रश्न:** {r.question}",
            "",
            f"**उत्तर:** {r.answer}",
            "",
            f"Gold passage shown to the generator: {f'yes, [{gold}]' if gold else 'no'}",
            "",
        ]
        if r.claims:
            rank = {p.passage_id: p.rank for p in r.retrieved}
            out += ["| claim | detector | evidence | entail |", "|---|---|---|---:|"]
            for c in r.claims:
                ev = f"[{rank[c.best_passage_id]}]" if c.best_passage_id in rank else "—"
                ent = f"{c.scores.entail:.2f}" if c.scores else "—"
                typ = f" ({c.type.value})" if c.type else ""
                out.append(f"| {_cell(c.text)} | {c.verdict.value}{typ} | {ev} | {ent} |")
            out.append("")
        out += [
            "- [ ] faithful",
            "- [ ] hallucinated (part not supported by the passages)",
            "- [ ] not an answer / refusal",
            "- Note:",
            "",
            "<details><summary>Passages shown to the generator</summary>",
            "",
        ]
        for p in r.retrieved:
            out += [f"**[{p.rank}]** {'(gold) ' if p.is_gold else ''}{_cell(p.text)}", ""]
        out += ["</details>", ""]
    return "\n".join(out)
