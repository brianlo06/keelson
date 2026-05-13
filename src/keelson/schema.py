from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Claim:
    subject: str
    predicate: str
    object: str
    valid_from: str
    omega: float
    source_phrase: str
    supersedes_hint: str | None = None

    @classmethod
    def from_json(cls, d: dict) -> Claim:
        return cls(
            subject=d["subject"],
            predicate=d["predicate"],
            object=d["object"],
            valid_from=d.get("valid_from", "unknown"),
            omega=float(d["omega"]),
            source_phrase=d.get("source_phrase", ""),
            supersedes_hint=d.get("supersedes_hint"),
        )

    def to_json(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "valid_from": self.valid_from,
            "omega": self.omega,
            "source_phrase": self.source_phrase,
            "supersedes_hint": self.supersedes_hint,
        }


@dataclass
class ExtractionResult:
    claims: list[Claim] = field(default_factory=list)
    raw_response: str = ""
    extraction_failed: bool = False
    failure_reason: str | None = None
