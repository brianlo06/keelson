"""Scoring for the read-path eval.

Each example produces a ReadScore with:
  - mention_score:    fraction of must_mention strings that appear in response
  - forbidden_hit:    True if any must_not_mention appears (sets composite to 0)
  - confidence_match: True if response.confidence matches expected (if checked)
  - citation_ok:      True if cited_claims >= min_cited_claims (if checked)
  - composite:        weighted aggregate

Strings are matched case-insensitively as substrings. A more sophisticated
matcher could use the holistic judge or BM25, but for a personal-memory
read eval, substring is the right grain — we care that the model produced
the right words, not the right embedding cluster.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from keelson.compose import ComposedResponse
from eval.read_types import ReadExample, ReadExpected


@dataclass
class ReadScore:
    example_id: str
    category: str
    mention_score: float
    forbidden_hit: bool
    confidence_match: bool
    citation_ok: bool
    composite: float
    response_text: str = ""
    failures: list[str] = field(default_factory=list)


def _contains(haystack: str, needle: str) -> bool:
    return needle.lower() in haystack.lower()


def score_read(example: ReadExample, response: ComposedResponse) -> ReadScore:
    failures: list[str] = []

    exp = example.expected
    text = response.text

    # Mention score
    if exp.must_mention:
        hits = sum(1 for m in exp.must_mention if _contains(text, m))
        mention_score = hits / len(exp.must_mention)
        if hits < len(exp.must_mention):
            missing = [m for m in exp.must_mention if not _contains(text, m)]
            failures.append(f"missing must_mention: {missing}")
    else:
        mention_score = 1.0

    # Forbidden hit
    forbidden = [m for m in exp.must_not_mention if _contains(text, m)]
    forbidden_hit = bool(forbidden)
    if forbidden:
        failures.append(f"forbidden mention(s) present: {forbidden}")

    # Confidence match
    if exp.expected_confidence is None:
        confidence_match = True
    else:
        confidence_match = response.confidence == exp.expected_confidence
        if not confidence_match:
            failures.append(
                f"confidence mismatch: expected {exp.expected_confidence!r}, "
                f"got {response.confidence!r}"
            )

    # Citation count
    if exp.min_cited_claims is None:
        citation_ok = True
    else:
        cited = len(response.cited_claims)
        citation_ok = cited >= exp.min_cited_claims
        if not citation_ok:
            failures.append(
                f"too few cited claims: expected >= {exp.min_cited_claims}, got {cited}"
            )

    if forbidden_hit:
        composite = 0.0
    else:
        composite = (
            0.60 * mention_score
            + 0.20 * (1.0 if confidence_match else 0.0)
            + 0.20 * (1.0 if citation_ok else 0.0)
        )

    return ReadScore(
        example_id=example.id,
        category=example.category,
        mention_score=mention_score,
        forbidden_hit=forbidden_hit,
        confidence_match=confidence_match,
        citation_ok=citation_ok,
        composite=composite,
        response_text=text,
        failures=failures,
    )


def aggregate_read(scores: list[ReadScore]) -> dict[str, float]:
    n = max(1, len(scores))
    return {
        "n": float(n),
        "composite": sum(s.composite for s in scores) / n,
        "mention": sum(s.mention_score for s in scores) / n,
        "confidence": sum(1 for s in scores if s.confidence_match) / n,
        "citation": sum(1 for s in scores if s.citation_ok) / n,
        "forbidden_hits": float(sum(1 for s in scores if s.forbidden_hit)),
    }
