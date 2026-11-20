"""Evaluation metrics (deck slide 21): retrieval, detection, span localisation, fairness.

Every metric is reported per language AND pooled (CLAUDE.md §3 row 16; see
:mod:`indichallu.eval.fairness`). Metrics return ``None`` when undefined on the data
(e.g. AUROC with one class) rather than a made-up number.
"""
