"""Cross-lingual reporting: per-language, pooled, macro and worst-language (slide 21).

"All metrics are reported per language as well as pooled" (CLAUDE.md §3 row 16).
:func:`by_language` runs any metric function once per language and once on all
records pooled, then adds the macro average over languages and the worst language for
every numeric metric. It works with a single language too (M0).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from indichallu.data.schema import Record

MetricFn = Callable[[Sequence[Record]], dict[str, Any]]


def by_language(records: Sequence[Record], metric_fn: MetricFn) -> dict[str, Any]:
    """``{"per_language": {lang: metrics}, "pooled": metrics, "macro": {...}, "worst": {...}}``.

    ``macro[m]`` is the unweighted mean of metric ``m`` over the languages where it is
    defined (``None`` values are skipped, and ``macro_n[m]`` says how many languages
    contributed). ``worst[m]`` is ``{"lang": ..., "value": ...}`` for the lowest value.
    Counts (keys starting with ``n`` or ending in a count name) are not averaged.
    """
    groups: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        groups[r.lang].append(r)
    per_language = {lang: metric_fn(rs) for lang, rs in sorted(groups.items())}
    out: dict[str, Any] = {
        "per_language": per_language,
        "pooled": metric_fn(list(records)),
        "macro": {},
        "macro_n": {},
        "worst": {},
    }
    keys = {k for m in per_language.values() for k, v in m.items() if _is_rate(k, v)}
    for key in sorted(keys):
        values = {
            lang: m[key] for lang, m in per_language.items() if isinstance(m.get(key), float | int)
        }
        if not values:
            out["macro"][key], out["macro_n"][key], out["worst"][key] = None, 0, None
            continue
        out["macro"][key] = sum(values.values()) / len(values)
        out["macro_n"][key] = len(values)
        lang = min(values, key=lambda lg: (values[lg], lg))
        out["worst"][key] = {"lang": lang, "value": values[lang]}
    return out


_COUNT_KEYS = {"tp", "fp", "fn", "tn", "positives", "gold_spans", "predicted_spans"}


def _is_rate(key: str, value: Any) -> bool:
    """Metrics worth averaging: floats/None that aren't counts."""
    return (
        not key.startswith("n_")
        and key != "n"
        and key not in _COUNT_KEYS
        and (value is None or isinstance(value, float))
    )
