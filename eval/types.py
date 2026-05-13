from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ExpectedClaim:
    subject: str
    predicate: str
    object_pattern: str
    omega_range: tuple[float, float]
    valid_from_window: str = "any"
    supersedes_hint: str | None = None
    required: bool = True

    @classmethod
    def from_json(cls, d: dict) -> ExpectedClaim:
        return cls(
            subject=d["subject"],
            predicate=d["predicate"],
            object_pattern=d["object"],
            omega_range=tuple(d["omega_range"]),
            valid_from_window=d.get("valid_from_window", "any"),
            supersedes_hint=d.get("supersedes_hint"),
            required=d.get("required", True),
        )


@dataclass(frozen=True)
class ForbiddenClaim:
    predicate_pattern: str
    object_pattern: str | None = None

    @classmethod
    def from_json(cls, d: dict) -> ForbiddenClaim:
        return cls(
            predicate_pattern=d["predicate_pattern"],
            object_pattern=d.get("object_pattern"),
        )


@dataclass(frozen=True)
class EvalExample:
    id: str
    category: str
    difficulty: Literal["easy", "medium", "hard", "adversarial"]
    user_message: str
    expected_claims: tuple[ExpectedClaim, ...]
    must_not_extract: tuple[ForbiddenClaim, ...] = ()
    context: tuple[dict, ...] = ()
    memory_state: tuple[dict, ...] = ()
    notes: str = ""

    @classmethod
    def from_json(cls, d: dict) -> EvalExample:
        exp = d.get("expected", {})
        return cls(
            id=d["id"],
            category=d["category"],
            difficulty=d.get("difficulty", "easy"),
            user_message=d["user_message"],
            expected_claims=tuple(
                ExpectedClaim.from_json(c) for c in exp.get("claims", [])
            ),
            must_not_extract=tuple(
                ForbiddenClaim.from_json(c) for c in exp.get("must_not_extract", [])
            ),
            context=tuple(d.get("context", [])),
            memory_state=tuple(d.get("memory_state", [])),
            notes=d.get("notes", ""),
        )
