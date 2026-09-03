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

import os
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

from keelson.retrieval import RetrievalResult
from keelson.store import ClaimRow, KeelsonStore

if TYPE_CHECKING:
    import anthropic

COMPOSE_MODEL = "claude-opus-4-7"


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


# --- LLM-backed composer -----------------------------------------------------


_LLM_SYSTEM = """You compose responses for a personal memory system. You receive
a user query and structured facts retrieved from memory. Write a brief, natural
response that honors the provenance scalar ω (omega) per fact:

  ω ≥ 0.85   direct statement       e.g. "You work at Anthropic."
  0.40-0.84  hedged                 e.g. "You mentioned working at Anthropic."
  ω < 0.40   inference disclaimer   e.g. "From context I gather you ran a marathon."

For claims about other people (user.mom, user.partner, etc.), ω is typically
0.5 — the user reported it, you don't know it firsthand. Use phrasing like
"You told me your mom retired" rather than "Your mom retired."

For a history view (live + past values for the same predicate), say what's
current first, then "Previously: ..." with the past values.

Be concise. Don't pad. Don't add information not in the retrieved facts.
If facts are empty, say "I don't have anything on that." — nothing else.
"""


def _facts_summary(result: RetrievalResult, store: KeelsonStore | None) -> str:
    lines: list[str] = []
    for c in result.claims:
        subject = _subject_name(store, c)
        lines.append(f"  CLAIM: {subject} {c.predicate} {c.object_text}  (ω={c.omega:.2f})")
    for c in result.history:
        status = "live" if c.valid_to is None else "past"
        subject = _subject_name(store, c)
        lines.append(f"  HISTORY ({status}): {subject} {c.predicate} {c.object_text}  (ω={c.omega:.2f})")
    for t in result.transcript_hits[:3]:
        lines.append(f"  TURN ({t.ts[:10]}): {t.content!r}")
    return "\n".join(lines)


def llm_compose(
    result: RetrievalResult,
    store: KeelsonStore | None = None,
    client: "anthropic.Anthropic | None" = None,
) -> ComposedResponse:
    """LLM-backed composer. Calls Claude to write natural-language responses
    with ω-aware hedging. Auto-falls-back to rule_based_compose when no API
    key is available and no client is passed.
    """
    if client is None and not os.environ.get("ANTHROPIC_API_KEY"):
        return rule_based_compose(result, store=store)

    # Lazy import — only when we actually need the SDK
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    facts = _facts_summary(result, store)
    user_msg = f"User asked: {result.spec.raw}\n\nRetrieved facts:\n{facts or '  (none)'}"

    response = client.messages.create(
        model=COMPOSE_MODEL,
        max_tokens=512,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": _LLM_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_msg}],
    )

    text = next(
        (block.text for block in response.content if block.type == "text"),
        "",
    ).strip()
    if not text:
        text = "I don't have anything on that."

    confidence = _overall_confidence(result.claims + result.history)
    return ComposedResponse(
        text=text,
        cited_claims=[c.id for c in result.claims + result.history],
        cited_turns=[t.id for t in result.transcript_hits[:3]],
        confidence=confidence,
    )
