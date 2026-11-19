"""Record schema: one JSONL line per example (CLAUDE.md §6).

Conventions enforced here (docs/DECISIONS.md D-008):

* **Offsets** are Python ``str`` indices, i.e. Unicode code points, into the ORIGINAL,
  un-normalised ``answer``. ``start`` is inclusive, ``end`` exclusive. A claim's ``text``
  must equal ``answer[start:end]`` exactly.
* **The contract** (CLAUDE.md §1): a SUPPORTED or CONTRADICTED claim must name the
  retrieved passage it rests on (``best_passage_id``); an UNSUPPORTED claim may name the
  closest passage or none. Every passage ID a record mentions must be in ``retrieved``.
* **Retrieval is carried forward**: every record has at least one retrieved passage with
  its score and rank, so retrieval failures stay separable from generation failures.
* Unknown fields are rejected everywhere, so typos fail loudly.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    ValidationError,
    model_validator,
)

PROB_SUM_TOLERANCE = 1e-3

Prob = Annotated[float, Field(ge=0.0, le=1.0)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- enums ---------------------------------------------------------------------------


class SpanType(StrEnum):
    """Error taxonomy (deck slide 18, after Multi-Hall-SA)."""

    ENTITY = "ENTITY"
    NUMERICAL = "NUMERICAL"
    TEMPORAL = "TEMPORAL"
    LOCATION = "LOCATION"


class Verdict(StrEnum):
    """Claim-level verdicts (deck slide 16)."""

    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"


class ResponseLabel(StrEnum):
    """Response-level label (RAGTruth-style)."""

    FAITHFUL = "faithful"
    HALLUCINATED = "hallucinated"


class Source(StrEnum):
    NATURAL = "natural"
    INJECTED = "injected"
    TRANSLATED = "translated"


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


TypedCheckResult = Literal["match", "mismatch"] | None


# --- building blocks -----------------------------------------------------------------


class Span(_Strict):
    """Half-open character span ``[start, end)`` into the original answer."""

    start: NonNegativeInt
    end: PositiveInt

    @model_validator(mode="after")
    def _non_empty(self) -> Span:
        if self.end <= self.start:
            raise ValueError(f"span end ({self.end}) must be greater than start ({self.start})")
        return self


class RetrievedPassage(_Strict):
    passage_id: str = Field(min_length=1)
    text: str
    score: float
    rank: PositiveInt
    is_gold: bool


class Decoding(BaseModel):
    """Fixed decoding config. Backend-specific extras (top_p, seed, ...) are allowed."""

    model_config = ConfigDict(extra="allow")

    temperature: float = Field(ge=0.0)
    max_new_tokens: PositiveInt


class GeneratorInfo(_Strict):
    model: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    decoding: Decoding
    backend: str | None = None


class NLIScores(_Strict):
    entail: Prob
    neutral: Prob
    contradict: Prob

    @model_validator(mode="after")
    def _sums_to_one(self) -> NLIScores:
        total = self.entail + self.neutral + self.contradict
        if abs(total - 1.0) > PROB_SUM_TOLERANCE:
            raise ValueError(f"NLI probabilities must sum to 1, got {total:.6f}")
        return self


class PassageScores(NLIScores):
    """One cell row of the claim x passage score matrix (deck slide 17, step 02)."""

    passage_id: str = Field(min_length=1)


class TypedChecks(_Strict):
    numeric: TypedCheckResult = None
    temporal: TypedCheckResult = None
    entity: TypedCheckResult = None


class Claim(Span):
    """A predicted atomic claim with its verdict and evidence attribution."""

    text: str = Field(min_length=1)
    verdict: Verdict
    best_passage_id: str | None = None
    scores: NLIScores | None = None
    passage_scores: list[PassageScores] = Field(default_factory=list)
    typed_checks: TypedChecks = Field(default_factory=TypedChecks)
    type: SpanType | None = None

    @model_validator(mode="after")
    def _contract(self) -> Claim:
        if self.verdict is not Verdict.UNSUPPORTED and self.best_passage_id is None:
            raise ValueError(f"a {self.verdict} claim must name its best_passage_id")
        if self.verdict is Verdict.SUPPORTED and self.type is not None:
            raise ValueError("an error type is only meaningful for non-SUPPORTED claims")
        ids = [p.passage_id for p in self.passage_scores]
        if len(ids) != len(set(ids)):
            raise ValueError("passage_scores has duplicate passage_ids")
        return self


class GoldSpan(Span):
    type: SpanType


class Labels(_Strict):
    response: ResponseLabel
    spans: list[GoldSpan] = Field(default_factory=list)

    @model_validator(mode="after")
    def _response_matches_spans(self) -> Labels:
        if self.response is ResponseLabel.HALLUCINATED and not self.spans:
            raise ValueError("a hallucinated response needs at least one labelled span")
        if self.response is ResponseLabel.FAITHFUL and self.spans:
            raise ValueError("a faithful response cannot have hallucinated spans")
        ordered = sorted(self.spans, key=lambda s: (s.start, s.end))
        for a, b in zip(ordered, ordered[1:], strict=False):
            if b.start < a.end:
                raise ValueError(f"gold spans overlap: [{a.start},{a.end}) and [{b.start},{b.end})")
        return self


# --- the record ----------------------------------------------------------------------


class Record(_Strict):
    id: str = Field(pattern=r"^[a-z]{2}-[0-9A-Za-z_-]+$")
    lang: str = Field(pattern=r"^[a-z]{2}$")
    split: Split
    source: Source
    question: str = Field(min_length=1)
    retrieved: list[RetrievedPassage] = Field(min_length=1)
    generator: GeneratorInfo | None = None
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    labels: Labels | None = None

    @model_validator(mode="after")
    def _consistency(self) -> Record:
        if not self.id.startswith(f"{self.lang}-"):
            raise ValueError(f"id {self.id!r} must start with '{self.lang}-'")
        if self.source is Source.NATURAL and self.generator is None:
            raise ValueError("a natural record must say which generator produced it")

        ids = [p.passage_id for p in self.retrieved]
        if len(ids) != len(set(ids)):
            raise ValueError("retrieved passages have duplicate passage_ids")
        ranks = [p.rank for p in self.retrieved]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(f"retrieved ranks must be 1..n in order, got {ranks}")
        known = set(ids)

        n = len(self.answer)
        for i, c in enumerate(self.claims):
            if c.end > n:
                raise ValueError(f"claim {i} span [{c.start},{c.end}) exceeds answer length {n}")
            if self.answer[c.start : c.end] != c.text:
                raise ValueError(
                    f"claim {i} text does not match answer[{c.start}:{c.end}] "
                    f"({self.answer[c.start : c.end]!r} != {c.text!r})"
                )
            if c.best_passage_id is not None and c.best_passage_id not in known:
                raise ValueError(f"claim {i} best_passage_id {c.best_passage_id!r} not retrieved")
            unknown = {p.passage_id for p in c.passage_scores} - known
            if unknown:
                raise ValueError(f"claim {i} scores unknown passages {sorted(unknown)}")

        if self.labels is not None:
            for s in self.labels.spans:
                if s.end > n:
                    raise ValueError(f"gold span [{s.start},{s.end}) exceeds answer length {n}")
        return self


# --- JSONL I/O -----------------------------------------------------------------------


class RecordParseError(ValueError):
    """A JSONL line failed to parse or validate; carries the file and line number."""

    def __init__(self, path: Path, line_no: int, cause: Exception):
        self.path, self.line_no, self.cause = path, line_no, cause
        super().__init__(f"{path}:{line_no}: {cause}")


def record_to_json(record: Record) -> str:
    """One compact JSON line. Indic text is written as-is (UTF-8), not \\u-escaped."""
    data: dict[str, Any] = record.model_dump(mode="json")
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def read_jsonl(path: str | Path) -> Iterator[Record]:
    """Yield validated records. Blank lines are skipped; bad lines raise RecordParseError."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield Record.model_validate_json(line)
            except ValidationError as e:
                raise RecordParseError(path, line_no, e) from e


def write_jsonl(records: Iterable[Record], path: str | Path) -> int:
    """Write records atomically (temp file + rename). Returns the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            for record in records:
                f.write(record_to_json(record))
                f.write("\n")
                count += 1
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return count
