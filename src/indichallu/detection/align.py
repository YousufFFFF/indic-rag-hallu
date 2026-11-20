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


class HFNLIModel:
    """A Hugging Face sequence-classification NLI model at a pinned revision."""

    def __init__(self, model_id: str, revision: str, max_length: int = 512, batch_size: int = 16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision)
        self.model.eval()
        id2label = {int(i): name.lower() for i, name in self.model.config.id2label.items()}
        try:  # map the model's own label order to (entail, neutral, contradict)
            self.order = [
                next(i for i, n in id2label.items() if n.startswith(lab[:5])) for lab in LABELS
            ]
        except StopIteration as e:
            raise ValueError(f"{model_id}: unexpected NLI labels {id2label}") from e
        self.max_length, self.batch_size = max_length, batch_size

    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        out = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            enc = self.tokenizer(
                [p for p, _ in batch],
                [h for _, h in batch],
                truncation="only_first",
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            with self.torch.no_grad():
                logits = self.model(**enc).logits
            out.append(self.torch.softmax(logits, dim=-1)[:, self.order].numpy())
        return np.concatenate(out) if out else np.zeros((0, 3))


def premise_windows(text: str, lang: str, window_sentences: int = 3) -> list[str]:
    """Overlapping windows of up to ``window_sentences`` consecutive sentences."""
    sents = [s.text for s in sentences(text, lang)] or [text]
    if len(sents) <= window_sentences:
        return [" ".join(sents)]
    return [
        " ".join(sents[i : i + window_sentences]) for i in range(len(sents) - window_sentences + 1)
    ]


class Aligner:
    def __init__(
        self, nli: NLIModel, lang: str, window_sentences: int = 3, ascii_digits: bool = True
    ):
        self.nli, self.lang = nli, lang
        self.window_sentences, self.ascii_digits = window_sentences, ascii_digits
        self._windows: dict[str, list[str]] = {}

    def _norm(self, text: str) -> str:
        return normalize(text, self.lang, ascii_digits=self.ascii_digits).text

    def _passage_windows(self, passage: RetrievedPassage) -> list[str]:
        if passage.passage_id not in self._windows:
            self._windows[passage.passage_id] = [
                self._norm(w)
                for w in premise_windows(passage.text, self.lang, self.window_sentences)
            ]
        return self._windows[passage.passage_id]

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
        probs = self.nli.predict(pairs) if pairs else np.zeros((0, 3))
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
