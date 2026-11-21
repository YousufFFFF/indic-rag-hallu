"""HTML rendering of a checked answer (used by the demo; pure, so it is tested in CI).

Each claim span of the ORIGINAL answer is highlighted by verdict and linked to its
evidence passage: SUPPORTED green, UNSUPPORTED red, CONTRADICTED orange. Text outside
claims is shown unhighlighted. Everything is HTML-escaped.
"""

from __future__ import annotations

from html import escape

from indichallu.data.schema import Record, Verdict

COLOURS = {
    Verdict.SUPPORTED: "#d1fae5",
    Verdict.UNSUPPORTED: "#fee2e2",
    Verdict.CONTRADICTED: "#ffedd5",
}


def render_answer(record: Record) -> str:
    """The answer with claim spans highlighted; titles name the verdict and evidence."""
    rank = {p.passage_id: p.rank for p in record.retrieved}
    out, pos = [], 0
    for c in sorted(record.claims, key=lambda c: c.start):
        if c.start < pos:  # overlapping claims: keep the first, show the rest in the table
            continue
        out.append(escape(record.answer[pos : c.start]))
        ev = f" · evidence [{rank[c.best_passage_id]}]" if c.best_passage_id in rank else ""
        typ = f" · {c.type.value}" if c.type else ""
        entail = f" · entail {c.scores.entail:.2f}" if c.scores else ""
        title = escape(f"{c.verdict.value}{typ}{ev}{entail}", quote=True)
        out.append(
            f'<mark style="background:{COLOURS[c.verdict]};padding:0 2px" title="{title}">'
            f"{escape(record.answer[c.start : c.end])}</mark>"
        )
        pos = c.end
    out.append(escape(record.answer[pos:]))
    return '<p style="font-size:1.15em;line-height:1.8">' + "".join(out) + "</p>"


def render_claims(record: Record) -> list[list[str]]:
    """Rows for a claims table: claim, verdict, type, evidence passage, entail score."""
    rank = {p.passage_id: p.rank for p in record.retrieved}
    rows = []
    for c in record.claims:
        ev = f"[{rank[c.best_passage_id]}]" if c.best_passage_id in rank else "—"
        rows.append(
            [
                c.text,
                c.verdict.value,
                c.type.value if c.type else "—",
                ev,
                f"{c.scores.entail:.2f}" if c.scores else "—",
            ]
        )
    return rows


def render_passages(record: Record) -> str:
    items = []
    for p in sorted(record.retrieved, key=lambda p: p.rank):
        gold = " · gold" if p.is_gold else ""
        items.append(
            f"<li><b>[{p.rank}]</b> <small>score {p.score:.3f}{gold}</small><br>"
            f"{escape(p.text)}</li>"
        )
    return "<ol style='list-style:none;padding-left:0'>" + "".join(items) + "</ol>"
