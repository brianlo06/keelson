"""Compose — RetrievalResult → natural-language response with ω-aware hedging.

The composer is the speech layer. It decides how to talk about a claim based on
its ω (provenance):
  - ω ≥ 0.85  : direct statement       ("You work at Anthropic.")
  - ω ≥ 0.40  : "you mentioned" hedge  ("You mentioned working at Anthropic.")
  - ω < 0.40  : inference disclaimer   ("From context I inferred...")

Falls back to transcript hits when no structured claims match.

Like Retriever's parser, the composer is pluggable: this rule-based version
runs offline; an LLM-backed composer can replace it once the API is wired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from keelson.retrieval import QuerySpec, RetrievalResult
from keelson.store import ClaimRow, KeelsonStore, TurnRow


@dataclass
class ComposedResponse:
    text: str
    cited_claims: list[int] = field(default_factory=list)
    cited_turns: list[int] = field(default_factory=list)
    confidence: str = "high"  # "high" | "medium" | "low" | "none"


Composer = Callable[[RetrievalResult, "KeelsonStore | None"], ComposedResponse]


def _subject_name(store: KeelsonStore | None, claim: ClaimRow) -> str:
    """Resolve a claim's subject_id back to a human-readable name."""
    if store is None:
        return f"entity#{claim.subject_id}"
    ent = store.get_entity(claim.subject_id)
    if ent is None:
        return f"entity#{claim.subject_id}"
    name = ent.canonical_name
    if name == "user":
        return "You"
    if name.startswith("user."):
        return f"Your {name.split('.', 1)[1]}"
    return name.title() if name.islower() else name


def _conjugate(predicate: str, second_person: bool) -> str:
    """Map a canonical 3rd-person-singular predicate to the form for the
    subject we're rendering. Stub for the rule-based composer; an LLM-backed
    composer would just write natural language."""
    verb = predicate.replace("_", " ")
    if not second_person:
        return verb
    # "works at" -> "work at"; "lives in" -> "live in"; "loves" -> "love"
    parts = verb.split(" ", 1)
    head = parts[0]
    rest = (" " + parts[1]) if len(parts) > 1 else ""
    if head.endswith("ies") and len(head) > 3:
        head = head[:-3] + "y"
    elif head.endswith("es") and head[-3:-2] in {"s", "x", "z"}:
        head = head[:-2]
    elif head.endswith("s") and not head.endswith("ss"):
        head = head[:-1]
    return head + rest


def _format_claim(store: KeelsonStore | None, claim: ClaimRow) -> str:
    """One-line natural rendering of a single claim, ω-aware."""
    subject = _subject_name(store, claim)
    is_you = subject == "You"
    verb = _conjugate(claim.predicate, second_person=is_you)
    obj = claim.object_text

    if claim.omega >= 0.85:
        return f"{subject} {verb} {obj}."
    elif claim.omega >= 0.40:
        # Always hedge from the USER's perspective — they're the one who said it.
        if is_you:
            return f"You mentioned {verb} {obj}."
        return f"You mentioned that {subject.lower()} {verb} {obj}."
    else:
        actor = "you" if is_you else subject.lower()
        return f"I inferred from context that {actor} {verb} {obj} (low confidence, ω={claim.omega:.2f})."


def _format_history_pair(store: KeelsonStore | None, current: ClaimRow | None, past: list[ClaimRow]) -> str:
    """Compose a history-style answer: current state + past states."""
    parts = []
    if current is not None:
        parts.append(f"Currently: {_format_claim(store, current)}")
    if past:
        past_objs = ", ".join(p.object_text for p in past)
        parts.append(f"Previously: {past_objs}.")
    return " ".join(parts) if parts else ""


def _overall_confidence(claims: list[ClaimRow]) -> str:
    if not claims:
        return "none"
    best = max(c.omega for c in claims)
    if best >= 0.85:
        return "high"
    if best >= 0.40:
        return "medium"
    return "low"


def rule_based_compose(result: RetrievalResult, store: KeelsonStore | None = None) -> ComposedResponse:
    """Default composer. Dispatches by intent."""
    spec = result.spec

    if spec.intent in ("lookup", "list"):
        return _compose_lookup(result, store)
    if spec.intent == "history":
        return _compose_history(result, store)
    if spec.intent == "entity_lookup":
        return _compose_entity(result, store)
    return _compose_search(result, store)


def _compose_lookup(result: RetrievalResult, store: KeelsonStore | None) -> ComposedResponse:
    if not result.claims:
        return _fallback_or_empty(result, store)

    # Lookup: if there's exactly one live claim, render it. If many, list them.
    if len(result.claims) == 1:
        c = result.claims[0]
        return ComposedResponse(
            text=_format_claim(store, c),
            cited_claims=[c.id],
            confidence=_overall_confidence([c]),
        )

    lines = [_format_claim(store, c) for c in result.claims]
    return ComposedResponse(
        text="\n".join(lines),
        cited_claims=[c.id for c in result.claims],
        confidence=_overall_confidence(result.claims),
    )


def _compose_history(result: RetrievalResult, store: KeelsonStore | None) -> ComposedResponse:
    if not result.history:
        return _fallback_or_empty(result, store)

    # Split live from past
    current = next((c for c in result.history if c.valid_to is None), None)
    past = [c for c in result.history if c.valid_to is not None]

    text = _format_history_pair(store, current, past)
    cited = [c.id for c in result.history]
    confidence = _overall_confidence(result.history)
    return ComposedResponse(text=text, cited_claims=cited, confidence=confidence)


def _compose_entity(result: RetrievalResult, store: KeelsonStore | None) -> ComposedResponse:
    if not result.claims:
        return _fallback_or_empty(result, store)

    lines = [_format_claim(store, c) for c in result.claims]
    return ComposedResponse(
        text="\n".join(lines),
        cited_claims=[c.id for c in result.claims],
        confidence=_overall_confidence(result.claims),
    )


def _compose_search(result: RetrievalResult, store: KeelsonStore | None) -> ComposedResponse:
    if result.transcript_hits:
        lines = [
            "I don't have a structured note on that, but you mentioned this:",
        ]
        for hit in result.transcript_hits[:3]:
            lines.append(f'  "{hit.content}"  ({hit.ts[:10]})')
        return ComposedResponse(
            text="\n".join(lines),
            cited_turns=[h.id for h in result.transcript_hits[:3]],
            confidence="medium",
        )
    return ComposedResponse(text="I don't have anything on that.", confidence="none")


def _fallback_or_empty(result: RetrievalResult, store: KeelsonStore | None) -> ComposedResponse:
    """No structured claims matched — try transcript fallback."""
    if result.transcript_hits:
        lines = [
            "Nothing in my structured notes, but from past conversations:",
        ]
        for hit in result.transcript_hits[:2]:
            lines.append(f'  "{hit.content}"  ({hit.ts[:10]})')
        return ComposedResponse(
            text="\n".join(lines),
            cited_turns=[h.id for h in result.transcript_hits[:2]],
            confidence="low",
        )
    return ComposedResponse(text="I don't have anything on that.", confidence="none")
