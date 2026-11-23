"""Evaluation protocols: which split a milestone develops on, what thresholds optimise,
whether test may be touched, and how refusals count (DECISIONS D-024/D-025, D-028).

A protocol is fixed before the numbers it governs exist. Scripts load one by name from
``configs/eval/<name>.yaml``, record it in the run, and call :meth:`Protocol.check_split`
before evaluating a split, so an M1 script cannot touch test by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from indichallu.data.schema import Split
from indichallu.detection.tune import OBJECTIVES

EVAL_CONFIGS = Path(__file__).resolve().parents[3] / "configs" / "eval"
REFUSAL_MODES = ("claims", "separate")


@dataclass(frozen=True)
class Protocol:
    name: str
    decision: str  # the DECISIONS.md entry that fixed it
    dev_split: Split  # thresholds are tuned here
    allow_test: bool  # False until the final run (M3)
    threshold_objective: str  # one of detection.tune.OBJECTIVES
    refusals: str  # "claims": scored like any answer; "separate": own class (T45)

    def __post_init__(self) -> None:
        if self.threshold_objective not in OBJECTIVES:
            raise ValueError(
                f"protocol {self.name!r}: objective {self.threshold_objective!r} not in "
                f"{OBJECTIVES}"
            )
        if self.refusals not in REFUSAL_MODES:
            raise ValueError(f"protocol {self.name!r}: refusals must be one of {REFUSAL_MODES}")
        if self.dev_split is Split.TEST:
            raise ValueError(f"protocol {self.name!r}: thresholds can't be developed on test")
        if not isinstance(self.allow_test, bool):  # "false" as a string is truthy (Codex, #32)
            raise ValueError(
                f"protocol {self.name!r}: allow_test must be a YAML boolean (true/false), "
                f"got {self.allow_test!r}"
            )

    def check_split(self, split: Split | str) -> None:
        """Raise if this protocol forbids evaluating ``split``."""
        if Split(split) is Split.TEST and not self.allow_test:
            raise PermissionError(
                f"protocol {self.name!r} ({self.decision}) keeps test untouched until the "
                "final run; evaluate on val"
            )

    def record(self) -> dict:
        """What a run stores about the protocol it followed."""
        return {
            "name": self.name,
            "decision": self.decision,
            "dev_split": self.dev_split.value,
            "allow_test": self.allow_test,
            "threshold_objective": self.threshold_objective,
            "refusals": self.refusals,
        }


def load_protocol(name: str, root: Path = EVAL_CONFIGS) -> Protocol:
    raw = yaml.safe_load((root / f"{name}.yaml").read_text(encoding="utf-8"))
    if raw.get("name") != name:
        raise ValueError(f"configs/eval/{name}.yaml must say name: {name}")
    return Protocol(
        name=raw["name"],
        decision=raw["decision"],
        dev_split=Split(raw["dev_split"]),
        allow_test=raw["allow_test"],  # validated as a real boolean, never coerced
        threshold_objective=raw["threshold_objective"],
        refusals=raw["refusals"],
    )
