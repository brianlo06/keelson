"""Types for the read-path eval.

A read-path example sets up a fresh store via a sequence of SetupActions, then
runs a natural-language query through Retriever + Composer and scores the
response against:
  - must_mention:        strings/phrases that should appear in the response
  - must_not_mention:    strings/phrases that should NOT appear
  - expected_confidence: 'high' | 'medium' | 'low' | 'none' (or null = don't check)
  - min_cited_claims:    minimum number of cited claims (or null)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SetupAction:
    type: Literal["claim", "turn"]
    # claim fields
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    omega: float = 0.85
    supersedes_hint: str | None = None
    object_entity: str | None = None
    # turn fields
    role: str | None = None
    content: str | None = None

    @classmethod
    def from_json(cls, d: dict) -> SetupAction:
        return cls(
            type=d["type"],
            subject=d.get("subject"),
            predicate=d.get("predicate"),
            object=d.get("object"),
            omega=float(d.get("omega", 0.85)),
            supersedes_hint=d.get("supersedes_hint"),
            object_entity=d.get("object_entity"),
            role=d.get("role"),
            content=d.get("content"),
        )


@dataclass(frozen=True)
class ReadExpected:
    must_mention: tuple[str, ...] = ()
    must_not_mention: tuple[str, ...] = ()
    expected_confidence: str | None = None
    min_cited_claims: int | None = None

    @classmethod
    def from_json(cls, d: dict) -> ReadExpected:
        return cls(
            must_mention=tuple(d.get("must_mention", [])),
            must_not_mention=tuple(d.get("must_not_mention", [])),
            expected_confidence=d.get("expected_confidence"),
            min_cited_claims=d.get("min_cited_claims"),
        )


@dataclass(frozen=True)
class ReadExample:
    id: str
    category: str
    difficulty: Literal["easy", "medium", "hard"]
    setup: tuple[SetupAction, ...]
    query: str
    expected: ReadExpected
    notes: str = ""

    @classmethod
    def from_json(cls, d: dict) -> ReadExample:
        return cls(
            id=d["id"],
            category=d["category"],
            difficulty=d.get("difficulty", "easy"),
            setup=tuple(SetupAction.from_json(s) for s in d.get("setup", [])),
            query=d["query"],
            expected=ReadExpected.from_json(d.get("expected", {})),
            notes=d.get("notes", ""),
        )
