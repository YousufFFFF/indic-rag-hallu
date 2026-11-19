"""Load and validate ``configs/languages.yaml`` (the languages in scope)."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Repo root is three levels up from src/indichallu/languages.py (editable install / Docker).
DEFAULT_LANGUAGES_PATH = Path(__file__).resolve().parents[2] / "configs" / "languages.yaml"

CodePoint = Annotated[int, Field(ge=0, le=0x10FFFF)]


class Family(StrEnum):
    INDO_ARYAN = "indo-aryan"
    DRAVIDIAN = "dravidian"


class Language(BaseModel):
    """One language entry. Unknown keys are rejected so typos fail loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(pattern=r"^[a-z]{2}$")
    name: str
    native_name: str
    iso639_3: str = Field(pattern=r"^[a-z]{3}$")
    script: str = Field(pattern=r"^[A-Z][a-z]{3}$")
    unicode_script: str = Field(pattern=r"^[A-Z]+$")
    unicode_block: tuple[CodePoint, CodePoint]
    digit_zero: CodePoint
    family: Family
    indictrans2_code: str
    indicqa_file: str
    resource_level: int | None = None
    enabled: bool

    @field_validator("unicode_block")
    @classmethod
    def _block_is_ordered(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[0] > v[1]:
            raise ValueError(f"unicode_block start {v[0]:#x} is after end {v[1]:#x}")
        return v

    @model_validator(mode="after")
    def _consistent_codes(self) -> Language:
        if self.indictrans2_code != f"{self.iso639_3}_{self.script}":
            raise ValueError(
                f"indictrans2_code {self.indictrans2_code!r} != {self.iso639_3}_{self.script}"
            )
        if (
            not self.unicode_block[0]
            <= self.digit_zero
            <= self.digit_zero + 9
            <= self.unicode_block[1]
        ):
            raise ValueError("digits 0-9 must lie inside unicode_block")
        return self

    @property
    def digits(self) -> str:
        """The ten native digits, 0 through 9."""
        return "".join(chr(self.digit_zero + i) for i in range(10))


def load_languages(path: str | Path = DEFAULT_LANGUAGES_PATH) -> dict[str, Language]:
    """Return all languages in the config, keyed by ISO 639-1 code, in file order."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"languages"}:
        raise ValueError("languages config must have exactly one top-level key: 'languages'")
    entries = raw["languages"]
    if not isinstance(entries, dict) or not entries:
        raise ValueError("'languages' must be a non-empty mapping of code -> entry")
    return {code: Language(code=code, **fields) for code, fields in entries.items()}


def enabled_languages(path: str | Path = DEFAULT_LANGUAGES_PATH) -> dict[str, Language]:
    """Return only the languages with ``enabled: true``."""
    return {code: lang for code, lang in load_languages(path).items() if lang.enabled}
