"""The detector (ours) and baseline B1, on records (deck slides 16-17; CLAUDE.md §4.2).

* :class:`Detector` = decompose (clause-level claims) -> align (every claim vs every
  retrieved passage) -> aggregate (three verdicts with evidence attribution).
* Baseline **B1** = the same NLI model and thresholds, but sentence-level claims with no
  decomposition: :func:`b1_claims`. Comparing them isolates what decomposition adds.

The evidence is only the record's ``retrieved`` passages, never anything else
(CLAUDE.md §1: "does the retrieved evidence say this?").
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import yaml

from indichallu.data.schema import Record
from indichallu.detection.aggregate import Thresholds, decide
from indichallu.detection.align import Aligner, HFNLIModel, NLIModel
from indichallu.detection.decompose import TRAILING, decompose
from indichallu.text.sentseg import Segment, sentences

REPO_ROOT = Path(__file__).resolve().parents[3]
DETECTION_CONFIGS = REPO_ROOT / "configs" / "detection"

ClaimFn = Callable[[str, str], list[Segment]]


def b1_claims(answer: str, lang: str) -> list[Segment]:
    """Baseline B1: one claim per sentence (trailing punctuation trimmed)."""
    out = []
    for s in sentences(answer, lang):
        end = s.end
        while end > s.start and (answer[end - 1] in TRAILING or answer[end - 1].isspace()):
            end -= 1
        if end > s.start:
            out.append(Segment(s.start, end, answer[s.start : end]))
    return out


def load_detection_config(name: str = "m0") -> dict:
    return yaml.safe_load((DETECTION_CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))


def load_nli(cfg: dict) -> HFNLIModel:
    n = cfg["nli"]
    return HFNLIModel(
        n["model_id"],
        n["revision"],
        n.get("max_length", 512),
        n.get("batch_size", 16),
        device=n.get("device"),  # None/"auto": CUDA if available (D-026)
    )


class Detector:
    def __init__(
        self,
        nli: NLIModel,
        lang: str,
        thresholds: Thresholds,
        claim_fn: ClaimFn = decompose,
        window_sentences: int = 3,
        ascii_digits: bool = True,
        window_stride: int = 1,
    ):
        self.aligner = Aligner(nli, lang, window_sentences, ascii_digits, window_stride)
        self.lang, self.thresholds, self.claim_fn = lang, thresholds, claim_fn

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        lang: str,
        nli: NLIModel | None = None,
        claim_fn: ClaimFn = decompose,
        thresholds: Thresholds | None = None,
    ) -> Detector:
        return cls(
            nli or load_nli(cfg),
            lang,
            thresholds or Thresholds(**cfg["thresholds"]),
            claim_fn,
            cfg["nli"].get("window_sentences", 3),
            cfg.get("normalize", {}).get("ascii_digits", True),
            cfg["nli"].get("window_stride", 1),
        )

    def detect(self, record: Record) -> Record:
        """A copy of ``record`` with ``claims`` filled in (labels untouched)."""
        if record.lang != self.lang:
            raise ValueError(f"detector is for {self.lang!r}, record is {record.lang!r}")
        segs = self.claim_fn(record.answer, record.lang)
        matrix = self.aligner.score([s.text for s in segs], record.retrieved)
        claims = [
            decide(s.text, s.start, s.end, row, self.thresholds)
            for s, row in zip(segs, matrix, strict=True)
        ]
        return Record.model_validate(
            {**record.model_dump(), "claims": [c.model_dump() for c in claims]}
        )
