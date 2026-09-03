"""Composite scoring for claim extraction.

Match algorithm:
1. For each EXPECTED claim, find the highest-similarity EXTRACTED claim via judge.
2. Score the match across dimensions (predicate, object, omega calibration).
3. Penalize unmatched expected (recall). Penalize unmatched extracted (precision).
4. Hard floor: if any MUST_NOT_EXTRACT pattern is hit, composite = 0.

Weights bias toward precision because false-positive claims pollute memory
permanently; false-negative claims are recoverable from the transcript (Keelson
keeps it!).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from keelson.schema import Claim
from eval.judge import Judge, JudgeVerdict
from eval.types import EvalExample, ExpectedClaim, ForbiddenClaim

MATCH_THRESHOLD = 0.7


@dataclass
class ClaimMatch:
    expected: ExpectedClaim
    extracted: Claim | None
    verdict: JudgeVerdict | None
    score: float


@dataclass
class ExampleScore:
    example_id: str
    category: str
    recall: float
    precision: float
    omega_calibration: float
    predicate_correctness: float
    composite: float
    forbidden_hit: bool
    matches: list[ClaimMatch] = field(default_factory=list)
    fabricated: list[Claim] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def _omega_in_range(omega: float, lo: float, hi: float) -> bool:
    return lo <= omega <= hi


def _matches_forbidden(claim: Claim, rule: ForbiddenClaim) -> bool:
    if rule.subject_pattern is not None and not re.search(
        rule.subject_pattern, claim.subject, re.IGNORECASE
    ):
        return False
    if not re.search(rule.predicate_pattern, claim.predicate, re.IGNORECASE):
        return False
    if rule.object_pattern is None:
        return True
    return bool(re.search(rule.object_pattern, claim.object, re.IGNORECASE))


def score_example(
    example: EvalExample,
    extracted: list[Claim],
    judge: Judge,
) -> ExampleScore:
    failures: list[str] = []

    # Hard floor: forbidden claims
    forbidden_hits = [
        c for c in extracted if any(_matches_forbidden(c, rule) for rule in example.must_not_extract)
    ]
    if forbidden_hits:
        failures.append(
            f"emitted {len(forbidden_hits)} forbidden claim(s): "
            + ", ".join(f"{c.predicate}={c.object!r}" for c in forbidden_hits)
        )

    # Match each expected claim against extracted via judge
    matches: list[ClaimMatch] = []
    used_extracted: set[int] = set()

    for expected in example.expected_claims:
        best_idx: int | None = None
        best_verdict: JudgeVerdict | None = None
        best_score = 0.0

        for idx, claim in enumerate(extracted):
            if idx in used_extracted:
                continue
            verdict = judge.compare(expected, claim)
            if verdict.overall > best_score:
                best_score = verdict.overall
                best_verdict = verdict
                best_idx = idx

        if best_idx is not None and best_score >= MATCH_THRESHOLD:
            used_extracted.add(best_idx)
            matches.append(ClaimMatch(
                expected=expected,
                extracted=extracted[best_idx],
                verdict=best_verdict,
                score=best_score,
            ))
        else:
            matches.append(ClaimMatch(
                expected=expected,
                extracted=None,
                verdict=best_verdict,
                score=best_score,
            ))
            if expected.required:
                failures.append(
                    f"missed required claim: {expected.subject} {expected.predicate} {expected.object_pattern!r}"
                )

    fabricated = [c for idx, c in enumerate(extracted) if idx not in used_extracted]
    if fabricated:
        failures.append(
            f"fabricated {len(fabricated)} claim(s): "
            + ", ".join(f"{c.subject} {c.predicate}={c.object!r} (ω={c.omega})" for c in fabricated)
        )

    # Aggregate
    required_expected = [m for m in matches if m.expected.required]
    matched_required = sum(1 for m in required_expected if m.extracted is not None)
    if not required_expected:
        # No required claims to recall — vacuously perfect.
        recall = 1.0
    else:
        recall = matched_required / len(required_expected)

    if not extracted:
        # No extractions emitted. Only count as precision=1.0 if there was
        # nothing to extract (chit-chat case); otherwise the absence is a
        # recall failure, not a precision win.
        precision = 1.0 if not required_expected else 0.0
    else:
        precision = 1.0 - (len(fabricated) / len(extracted))

    omega_scores = []
    predicate_scores = []
    for m in matches:
        if m.extracted is None:
            continue
        lo, hi = m.expected.omega_range
        omega_scores.append(1.0 if _omega_in_range(m.extracted.omega, lo, hi) else 0.0)
        if m.verdict is not None:
            predicate_scores.append(m.verdict.predicate_match)
    # Vacuously perfect when there's nothing to score (e.g. chit-chat correctly
    # extracts no claims — there are no ωs or predicates to validate).
    omega_calibration = sum(omega_scores) / len(omega_scores) if omega_scores else 1.0
    predicate_correctness = (
        sum(predicate_scores) / len(predicate_scores) if predicate_scores else 1.0
    )

    if forbidden_hits:
        composite = 0.0
    else:
        composite = (
            0.30 * recall
            + 0.50 * precision
            + 0.10 * omega_calibration
            + 0.10 * predicate_correctness
        )

    return ExampleScore(
        example_id=example.id,
        category=example.category,
        recall=recall,
        precision=precision,
        omega_calibration=omega_calibration,
        predicate_correctness=predicate_correctness,
        composite=composite,
        forbidden_hit=bool(forbidden_hits),
        matches=matches,
        fabricated=fabricated,
        failures=failures,
    )


def aggregate(scores: list[ExampleScore]) -> dict[str, float]:
    n = max(1, len(scores))
    return {
        "n": float(n),
        "composite": sum(s.composite for s in scores) / n,
        "recall": sum(s.recall for s in scores) / n,
        "precision": sum(s.precision for s in scores) / n,
        "omega_calibration": sum(s.omega_calibration for s in scores) / n,
        "predicate_correctness": sum(s.predicate_correctness for s in scores) / n,
        "forbidden_hits": float(sum(1 for s in scores if s.forbidden_hit)),
    }
