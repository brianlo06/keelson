"""Retrieval — natural-language query → structured claim lookup + transcript fallback.

The retrieval pipeline:
  1. Parser:  natural-language query  -> QuerySpec
  2. Lookup:  QuerySpec               -> RetrievalResult (claims + transcript hits)

Parser is pluggable. The default `rule_based_parse` covers the common shapes
("where do I work", "what did I tell you about X", "tell me about Y"). Plug in
an LLM-backed parser by passing it to Retriever.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Literal

from keelson.store import ClaimRow, KeelsonStore, TurnRow

Intent = Literal[
    "lookup",         # current state of (subject, predicate) — live claim only
    "history",        # past states of (subject, predicate) — include superseded
    "entity_lookup",  # everything we know about an entity
    "list",           # all values for (subject, predicate)
    "search",         # no structured intent — fall back to FTS
    "unknown",
]


@dataclass(frozen=True)
class QuerySpec:
    intent: Intent
    subjects: tuple[str, ...] = ()
    predicates: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    raw: str = ""


@dataclass
class RetrievalResult:
    spec: QuerySpec
    claims: list[ClaimRow] = field(default_factory=list)
    history: list[ClaimRow] = field(default_factory=list)
    transcript_hits: list[TurnRow] = field(default_factory=list)


# --- parser ------------------------------------------------------------------

QueryParser = Callable[[str], QuerySpec]


# Maps natural-language tokens to canonical predicates the store uses.
# When the user asks "where do I LIVE", we look up the `lives_in` predicate.
PREDICATE_HINTS: dict[str, tuple[str, ...]] = {
    "work": ("works_at", "role", "role_at"),
    "live": ("lives_in",),
    "love": ("loves",),
    "like": ("likes",),
    "hate": ("hates",),
    "eat": ("diet", "likes_food"),
    "do": ("did", "does"),
    "age": ("age",),
}

# Relational subjects: "my mom" -> "user.mom"
RELATIONAL_TERMS = {
    "mom", "mother", "dad", "father", "sister", "brother",
    "wife", "husband", "partner", "spouse", "child", "kid",
    "son", "daughter", "boss", "manager", "coworker", "friend",
    "therapist", "doctor", "vet",
}

STOPWORDS = {
    "a", "an", "the", "i", "you", "we", "me", "my", "your", "our",
    "what", "where", "who", "when", "why", "how", "do", "did", "does",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "to", "of", "in", "on", "at", "by", "for",
    "with", "about", "from", "as", "and", "or", "but", "not", "no",
    "tell", "remember", "know", "think", "say", "said", "this", "that",
    "it", "its", "any", "some", "all", "anything", "everything", "much",
    "more", "most", "ever", "still", "last", "first", "again", "really",
    "actually", "currently",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _strip_punctuation(text: str) -> str:
    return text.strip().rstrip("?.!,;:").strip()


def _normalize_relational(noun: str) -> str:
    n = noun.lower().strip()
    if n in RELATIONAL_TERMS:
        # Canonical form for related-person subjects
        # "mom" / "mother" -> "user.mom" (mother gets normalized too)
        return f"user.{'mom' if n == 'mother' else 'dad' if n == 'father' else n}"
    return noun


def rule_based_parse(query: str) -> QuerySpec:
    """Pattern-matching parser. Handles common shapes; falls back to search.

    Order matters — more specific patterns first.
    """
    q = _strip_punctuation(query)
    q_lower = q.lower()

    def _resolve_target(raw_target: str) -> str:
        """Strip 'my '/'your ' and normalize relational terms to user.<role>."""
        t = re.sub(r"^(?:my|your)\s+", "", raw_target.strip())
        if t in RELATIONAL_TERMS:
            return _normalize_relational(t)
        return t

    # "tell me about <X>" / "what do you know about <X>" → entity_lookup
    m = re.match(
        r"^(?:tell me (?:everything |all |what you know )?about|"
        r"what (?:do you|did you) know about|"
        r"what (?:can you tell|do you remember) (?:me )?about) ([\w ']+)$",
        q_lower,
    )
    if m:
        target = m.group(1).strip()
        return QuerySpec(
            intent="entity_lookup",
            subjects=(_resolve_target(target),),
            keywords=tuple(t for t in _tokenize(target) if t not in STOPWORDS),
            raw=query,
        )

    # "what did you (last) (tell|say) (me) about (my X | X)" → entity + recency
    m = re.match(
        r"^what did (?:you|we) (?:last |ever )?(?:tell|say|mention)(?: me)? about "
        r"([\w ']+)$",
        q_lower,
    )
    if m:
        target = m.group(1).strip()
        return QuerySpec(
            intent="entity_lookup",
            subjects=(_resolve_target(target),),
            keywords=tuple(t for t in _tokenize(target) if t not in STOPWORDS),
            raw=query,
        )

    # "where did I used to <verb>" / "what did I used to <verb>" → history
    m = re.match(
        r"^(?:where|what|who) did i (?:use to|used to|previously) (\w+)",
        q_lower,
    )
    if m:
        verb = m.group(1)
        predicates = PREDICATE_HINTS.get(verb, (verb,))
        return QuerySpec(intent="history", subjects=("user",), predicates=predicates, raw=query)

    # "what do I love/like/hate/...?" → list (plural-verb intent)
    # This must come BEFORE the generic "what/where/who do i" pattern, since
    # "what do I love" matches both and the list intent is the better fit.
    m = re.match(r"^what do i (\w+)", q_lower)
    if m:
        verb = m.group(1)
        predicates = PREDICATE_HINTS.get(verb, (f"{verb}s",))  # love -> loves
        return QuerySpec(intent="list", subjects=("user",), predicates=predicates, raw=query)

    # "where/who do I <verb>" → live lookup (singular fact)
    m = re.match(r"^(?:where|who) do i (\w+)", q_lower)
    if m:
        verb = m.group(1)
        predicates = PREDICATE_HINTS.get(verb, (verb,))
        return QuerySpec(intent="lookup", subjects=("user",), predicates=predicates, raw=query)

    # "what/where/who did I <verb>" → past-tense lookup
    m = re.match(r"^(?:what|where|who) did i (\w+)", q_lower)
    if m:
        verb = m.group(1)
        predicates = PREDICATE_HINTS.get(verb, (verb,))
        return QuerySpec(intent="lookup", subjects=("user",), predicates=predicates, raw=query)

    # "what is my <noun>" / "how old am I" → identity lookup
    m = re.match(r"^(?:what is|what's) my (\w+)", q_lower)
    if m:
        attr = m.group(1)
        return QuerySpec(intent="lookup", subjects=("user",), predicates=(attr,), raw=query)
    if q_lower in {"how old am i", "what's my age"}:
        return QuerySpec(intent="lookup", subjects=("user",), predicates=("age",), raw=query)

    # Default: extract content words as keywords for FTS fallback
    keywords = tuple(t for t in _tokenize(q) if t not in STOPWORDS and len(t) > 1)
    return QuerySpec(intent="search", keywords=keywords, raw=query)


# --- retriever ---------------------------------------------------------------


@dataclass
class Retriever:
    """Pluggable retrieval over a KeelsonStore.

    Parser is a callable: str -> QuerySpec. Default is rule-based; pass an
    LLM-backed parser for higher accuracy when the API is available.
    """

    store: KeelsonStore
    parser: QueryParser = field(default=rule_based_parse)
    transcript_limit: int = 5
    omega_floor: float = 0.25

    def query(self, natural_query: str) -> RetrievalResult:
        spec = self.parser(natural_query)
        return self._fulfill(spec)

    def _fulfill(self, spec: QuerySpec) -> RetrievalResult:
        result = RetrievalResult(spec=spec)

        if spec.intent in ("lookup", "list"):
            for subject in spec.subjects or ("user",):
                for predicate in spec.predicates or (None,):
                    claims = self.store.live_claims(
                        subject=subject,
                        predicate=predicate,
                        omega_min=self.omega_floor,
                    )
                    result.claims.extend(claims)

        elif spec.intent == "history":
            for subject in spec.subjects or ("user",):
                for predicate in spec.predicates:
                    result.history.extend(self.store.claim_history(subject, predicate))

        elif spec.intent == "entity_lookup":
            for subject in spec.subjects:
                # Live claims where the entity is the subject
                result.claims.extend(
                    self.store.live_claims(subject=subject, omega_min=self.omega_floor)
                )
                # Live claims where the entity appears as a (text) object
                result.claims.extend(self._object_text_claims(subject))

        # Always also collect transcript hits — useful for fuzzy fallback
        if spec.keywords:
            query = " OR ".join(spec.keywords)
            try:
                result.transcript_hits = self.store.search_turns(query, limit=self.transcript_limit)
            except Exception:
                # FTS5 can choke on certain inputs (reserved tokens, empty); ignore.
                result.transcript_hits = []

        return result

    def _object_text_claims(self, target: str) -> list[ClaimRow]:
        """Claims whose object_text contains `target` (case-insensitive).

        Cheap path: we do a LIKE scan. For a larger corpus we'd add an FTS
        index over object_text; for v1, this is fine.
        """
        like = f"%{target}%"
        rows = self.store.conn.execute(
            """
            SELECT * FROM claims
            WHERE forgotten = 0 AND valid_to IS NULL
              AND lower(object_text) LIKE lower(?)
            ORDER BY asserted_ts DESC
            """,
            (like,),
        ).fetchall()
        from keelson.store import _to_claim  # local import: private helper
        return [_to_claim(r) for r in rows]
