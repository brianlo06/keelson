"""Unit tests for the scoring logic — no API calls."""

from __future__ import annotations

from unittest.mock import MagicMock

from keelson.schema import Claim
from eval.judge import JudgeVerdict
from eval.score import score_example
from eval.types import EvalExample, ExpectedClaim, ForbiddenClaim


def _example(
    expected: list[ExpectedClaim],
    forbidden: list[ForbiddenClaim] = (),
) -> EvalExample:
    return EvalExample(
        id="t",
        category="test",
        difficulty="easy",
        user_message="...",
        expected_claims=tuple(expected),
        must_not_extract=tuple(forbidden),
    )


def _claim(subject="user", predicate="works_at", obj="X", omega=0.85) -> Claim:
    return Claim(
        subject=subject,
        predicate=predicate,
        object=obj,
        valid_from="now",
        omega=omega,
        source_phrase="",
    )


def _judge_perfect():
    judge = MagicMock()
    judge.compare.return_value = JudgeVerdict(
        overall=1.0, subject_match=1.0, predicate_match=1.0, object_match=1.0, reason="exact",
    )
    return judge


def _judge_no_match():
    judge = MagicMock()
    judge.compare.return_value = JudgeVerdict(
        overall=0.1, subject_match=0.1, predicate_match=0.1, object_match=0.1, reason="no",
    )
    return judge


def test_perfect_match_full_score():
    expected = ExpectedClaim(
        subject="user", predicate="works_at", object_pattern="Anthropic",
        omega_range=(0.7, 1.0), supersedes_hint="works_at", required=True,
    )
    ex = _example([expected])
    extracted = [_claim(predicate="works_at", obj="Anthropic", omega=0.85)]

    score = score_example(ex, extracted, _judge_perfect())

    assert score.composite >= 0.99
    assert score.recall == 1.0
    assert score.precision == 1.0
    assert score.omega_calibration == 1.0


def test_fabrication_destroys_precision():
    expected = ExpectedClaim(
        subject="user", predicate="works_at", object_pattern="Anthropic",
        omega_range=(0.7, 1.0), supersedes_hint=None, required=True,
    )
    ex = _example([expected])
    extracted = [
        _claim(predicate="works_at", obj="Anthropic"),
        _claim(predicate="loves", obj="purple"),  # fabricated
    ]

    score = score_example(ex, extracted, _judge_perfect())

    # 1 fabricated of 2 extracted → precision = 0.5
    assert abs(score.precision - 0.5) < 0.01
    assert len(score.fabricated) == 1


def test_missed_required_kills_recall():
    expected = ExpectedClaim(
        subject="user", predicate="works_at", object_pattern="Anthropic",
        omega_range=(0.7, 1.0), supersedes_hint=None, required=True,
    )
    ex = _example([expected])

    score = score_example(ex, [], _judge_no_match())

    assert score.recall == 0.0
    assert score.composite < 0.5


def test_forbidden_claim_is_hard_floor():
    forbidden = ForbiddenClaim(predicate_pattern="^works_at$", object_pattern="Google")
    ex = _example([], forbidden=[forbidden])
    extracted = [_claim(predicate="works_at", obj="Google")]

    score = score_example(ex, extracted, _judge_perfect())

    assert score.composite == 0.0
    assert score.forbidden_hit is True


def test_forbidden_subject_pattern_scopes_the_rule():
    """Regression-006 shape (FINDINGS.md): `lives_in: Berlin` is forbidden for
    the USER but fine for the sister — the rule must match on subject."""
    forbidden = ForbiddenClaim(
        subject_pattern="^user$", predicate_pattern="^lives_in$", object_pattern="Berlin",
    )
    ex = _example([], forbidden=[forbidden])

    # Correct attribution: the sister lives in Berlin — no hit.
    ok = score_example(
        ex, [_claim(subject="user.sister", predicate="lives_in", obj="Berlin", omega=0.5)],
        _judge_perfect(),
    )
    assert ok.forbidden_hit is False

    # Mis-attribution: the user does not live in Berlin — hard floor.
    bad = score_example(
        ex, [_claim(subject="user", predicate="lives_in", obj="Berlin", omega=0.5)],
        _judge_perfect(),
    )
    assert bad.forbidden_hit is True
    assert bad.composite == 0.0


def test_forbidden_subject_pattern_parses_from_json():
    rule = ForbiddenClaim.from_json(
        {"subject_pattern": "^user$", "predicate_pattern": "lives_in", "object_pattern": "Berlin"}
    )
    assert rule.subject_pattern == "^user$"
    # Older rules without the field still load (subject unrestricted).
    legacy = ForbiddenClaim.from_json({"predicate_pattern": "lives_in"})
    assert legacy.subject_pattern is None


def test_omega_out_of_range_hurts_calibration():
    expected = ExpectedClaim(
        subject="user", predicate="works_at", object_pattern="Anthropic",
        omega_range=(0.85, 1.0), supersedes_hint=None, required=True,
    )
    ex = _example([expected])
    extracted = [_claim(predicate="works_at", obj="Anthropic", omega=0.25)]  # too low

    score = score_example(ex, extracted, _judge_perfect())

    assert score.omega_calibration == 0.0
    # but the claim itself still matched
    assert score.recall == 1.0
    assert score.precision == 1.0


def test_empty_extraction_against_chitchat_perfect_score():
    """User says chit-chat, model correctly extracts nothing."""
    ex = _example([])
    score = score_example(ex, [], _judge_perfect())

    # No expected claims to miss → recall = 1.0
    # No extracted claims to fabricate → precision = 1.0
    assert score.recall == 1.0
    assert score.precision == 1.0
    assert score.composite >= 0.8


def test_chitchat_with_fabrication_caught():
    """User says chit-chat, but model invents a claim — must penalize."""
    ex = _example([])
    extracted = [_claim(predicate="likes", obj="conversations")]

    score = score_example(ex, extracted, _judge_no_match())

    # precision = 1 - (1 fabricated / 1 total) = 0
    assert score.precision == 0.0
    assert len(score.fabricated) == 1
