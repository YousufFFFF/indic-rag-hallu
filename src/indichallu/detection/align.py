"""Claim-evidence alignment: every claim vs EVERY retrieved passage (deck slide 17, step 02).

An NLI model gives P(entail), P(neutral), P(contradict) for (premise=evidence,
hypothesis=claim). IndicQA passages are often longer than mDeBERTa's 512 tokens, and
truncation would silently drop evidence. So each passage is split into windows of up to
``window_sentences`` consecutive sentences, every window is scored, and the passage gets
the probabilities of its most decisive window, the one with the highest
max(entail, contradict) (DECISIONS D-024).

Premise and hypothesis go through the same normalisation (NFC, IndicNLP, ASCII digits),
so "२१" in a passage matches "21" in a claim.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from indichallu.data.schema import PassageScores, RetrievedPassage
from indichallu.text.normalize import normalize
from indichallu.text.sentseg import sentences

LABELS = ("entailment", "neutral", "contradiction")


class NLIModel(Protocol):
    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """(n, 3) probabilities in the order entail, neutral, contradict."""
        ...


def resolve_device(device: str | None, cuda_available: bool) -> str:
    """``None``/``"auto"`` -> CUDA when available, else CPU; anything else is used as given.

    >>> resolve_device(None, True), resolve_device("auto", False), resolve_device("cpu", True)
    ('cuda', 'cpu', 'cpu')
    """
    if device in (None, "auto"):
        return "cuda" if cuda_available else "cpu"
    return device


class HFNLIModel:
    """A Hugging Face sequence-classification NLI model at a pinned revision.

    Runs on a CUDA GPU when one is available (D-026): on the project's CPU laptop the
    pilot's ~12k pairs take many hours, on a free T4 minutes. Same weights, same inputs;
    only floating-point rounding differs between devices.
    """

    def __init__(
        self,
        model_id: str,
        revision: str,
        max_length: int = 512,
        batch_size: int = 16,
        device: str | None = None,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = resolve_device(device, torch.cuda.is_available())
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision)
        self.model.to(self.device)
        self.model.eval()
        id2label = {int(i): name.lower() for i, name in self.model.config.id2label.items()}
        try:  # map the model's own label order to (entail, neutral, contradict)
            self.order = [
                next(i for i, n in id2label.items() if n.startswith(lab[:5])) for lab in LABELS
            ]
        except StopIteration as e:
            raise ValueError(f"{model_id}: unexpected NLI labels {id2label}") from e
        self.max_length, self.batch_size = max_length, batch_size

    def _fit(self, premise: str, hypothesis: str) -> list[str]:
        """The premise as one or more pieces that each fit ``max_length`` together with the
        hypothesis, so truncation never silently drops evidence (Codex review on #23)."""
        tok = self.tokenizer
        hyp_ids = tok(hypothesis, add_special_tokens=False)["input_ids"]
        budget = self.max_length - len(hyp_ids) - tok.num_special_tokens_to_add(pair=True)
        prem_ids = tok(premise, add_special_tokens=False)["input_ids"]
        spans = premise_chunk_spans(len(prem_ids), budget)
        if len(spans) == 1:
            if budget < 1:
                warnings.warn(
                    f"claim of {len(hyp_ids)} tokens fills max_length={self.max_length}; "
                    "the pair will be truncated",
                    stacklevel=2,
                )
            return [premise]
        return [tok.decode(prem_ids[a:b]) for a, b in spans]

    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        expanded, owner = [], []
        for i, (premise, hypothesis) in enumerate(pairs):
            for piece in self._fit(premise, hypothesis):
                expanded.append((piece, hypothesis))
                owner.append(i)
        out = []
        for i in range(0, len(expanded), self.batch_size):
            batch = expanded[i : i + self.batch_size]
            enc = self.tokenizer(
                [p for p, _ in batch],
                [h for _, h in batch],
                truncation="longest_first",  # only reachable for absurdly long claims
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            with self.torch.no_grad():
                logits = self.model(**enc.to(self.device)).logits
            probs = self.torch.softmax(logits.float(), dim=-1)[:, self.order]
            out.append(probs.cpu().numpy())
        probs = np.concatenate(out) if out else np.zeros((0, 3))
        return reduce_most_decisive(probs, owner, len(pairs))


CHUNK_OVERLAP = 32  # tokens shared by consecutive premise chunks


def chunk_spans(n: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """[start, end) token ranges of length <= ``size`` covering 0..n, consecutive ranges
    overlapping by ``overlap`` tokens.

    >>> chunk_spans(10, 4, 1)
    [(0, 4), (3, 7), (6, 10)]
    """
    if n <= size:
        return [(0, n)]
    if not 0 <= overlap < size:
        raise ValueError("need 0 <= overlap < size")
    spans, start = [], 0
    while True:
        end = min(start + size, n)
        spans.append((start, end))
        if end == n:
            return spans
        start = end - overlap


def premise_chunk_spans(n: int, budget: int) -> list[tuple[int, int]]:
    """Token ranges of an ``n``-token premise that each fit ``budget`` tokens.

    The overlap shrinks for small budgets (at most a quarter of the budget), so any
    positive budget is honoured. Only when the claim alone fills ``max_length``
    (budget < 1) is nothing left to chunk into: the whole premise is returned.

    >>> premise_chunk_spans(10, 4)
    [(0, 4), (3, 7), (6, 10)]
    >>> premise_chunk_spans(5, 0)
    [(0, 5)]
    """
    if n <= budget or budget < 1:
        return [(0, n)]
    return chunk_spans(n, budget, min(CHUNK_OVERLAP, budget // 4))


def reduce_most_decisive(probs: np.ndarray, owner: Sequence[int], n: int) -> np.ndarray:
    """Per original pair, the row of its most decisive piece: highest max(entail, contradict)."""
    best = np.full((n, 3), np.nan)
    for row, i in zip(probs, owner, strict=True):
        if np.isnan(best[i, 0]) or max(row[0], row[2]) > max(best[i, 0], best[i, 2]):
            best[i] = row
    return best


def premise_windows(text: str, lang: str, window_sentences: int = 3, stride: int = 1) -> list[str]:
    """Windows of up to ``window_sentences`` consecutive sentences, starting every
    ``stride`` sentences; the last window always reaches the final sentence."""
    if not 1 <= stride <= window_sentences:
        raise ValueError("need 1 <= stride <= window_sentences")
    sents = [s.text for s in sentences(text, lang)] or [text]
    if len(sents) <= window_sentences:
        return [" ".join(sents)]
    last = len(sents) - window_sentences
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return [" ".join(sents[i : i + window_sentences]) for i in starts]


class Aligner:
    def __init__(
        self,
        nli: NLIModel,
        lang: str,
        window_sentences: int = 3,
        ascii_digits: bool = True,
        window_stride: int = 1,
    ):
        self.nli, self.lang = nli, lang
        self.window_sentences, self.ascii_digits = window_sentences, ascii_digits
        self.window_stride = window_stride
        self._windows: dict[str, list[str]] = {}
        self._pair_cache: dict[tuple[str, str], np.ndarray] = {}  # identical pairs scored once

    def _norm(self, text: str) -> str:
        return normalize(text, self.lang, ascii_digits=self.ascii_digits).text

    def _passage_windows(self, passage: RetrievedPassage) -> list[str]:
        if passage.passage_id not in self._windows:
            self._windows[passage.passage_id] = [
                self._norm(w)
                for w in premise_windows(
                    passage.text, self.lang, self.window_sentences, self.window_stride
                )
            ]
        return self._windows[passage.passage_id]

    def _predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        todo = list(dict.fromkeys(p for p in pairs if p not in self._pair_cache))
        if todo:
            for pair, row in zip(todo, self.nli.predict(todo), strict=True):
                self._pair_cache[pair] = np.asarray(row, dtype=float)
        return np.array([self._pair_cache[p] for p in pairs]) if pairs else np.zeros((0, 3))

    def score(
        self, claims: Sequence[str], passages: Sequence[RetrievedPassage]
    ) -> list[list[PassageScores]]:
        """For each claim, one PassageScores per passage (in the given passage order)."""
        hyps = [self._norm(c) for c in claims]
        pairs, owners = [], []
        for ci, h in enumerate(hyps):
            for pi, p in enumerate(passages):
                for w in self._passage_windows(p):
                    pairs.append((w, h))
                    owners.append((ci, pi))
        probs = self._predict(pairs)
        best: dict[tuple[int, int], np.ndarray] = {}
        for owner, pr in zip(owners, probs, strict=True):
            current = best.get(owner)
            if current is None or max(pr[0], pr[2]) > max(current[0], current[2]):
                best[owner] = pr
        out = []
        for ci in range(len(claims)):
            row = []
            for pi, p in enumerate(passages):
                e, n, c = (float(x) for x in best[(ci, pi)])
                total = e + n + c
                row.append(
                    PassageScores(
                        passage_id=p.passage_id,
                        entail=e / total,
                        neutral=n / total,
                        contradict=c / total,
                    )
                )
            out.append(row)
        return out
