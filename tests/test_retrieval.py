"""Tests for retrieval + compose — all offline, no API."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from keelson.compose import rule_based_compose
from keelson.retrieval import QuerySpec, Retriever, rule_based_parse
from keelson.store import KeelsonStore


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmp:
        s = KeelsonStore(Path(tmp) / "test.db")
        yield s
        s.close()


# --- parser unit tests -------------------------------------------------------


def test_parse_tell_me_about_entity():
    spec = rule_based_parse("Tell me about Jamie")
    assert spec.intent == "entity_lookup"
    assert spec.subjects == ("jamie",)


def test_parse_tell_me_about_relational():
    spec = rule_based_parse("Tell me about my mom")
    assert spec.intent == "entity_lookup"
    assert spec.subjects == ("user.mom",)


def test_parse_what_did_you_last_tell_me():
    spec = rule_based_parse("What did you last tell me about my dad?")
    assert spec.intent == "entity_lookup"
    assert spec.subjects == ("user.dad",)


def test_parse_where_do_i_work():
    spec = rule_based_parse("Where do I work?")
    assert spec.intent == "lookup"
    assert spec.subjects == ("user",)
    assert "works_at" in spec.predicates


def test_parse_where_did_i_used_to_work():
    spec = rule_based_parse("Where did I used to work?")
    assert spec.intent == "history"
    assert spec.subjects == ("user",)
    assert "works_at" in spec.predicates


def test_parse_what_do_i_love():
    spec = rule_based_parse("What do I love?")
    assert spec.intent == "list"
    assert spec.subjects == ("user",)
    assert "loves" in spec.predicates


def test_parse_unknown_falls_back_to_search():
    spec = rule_based_parse("Remember that climbing trip in Yosemite?")
    assert spec.intent == "search"
    assert "climbing" in spec.keywords
    assert "yosemite" in spec.keywords


# --- retrieval integration tests --------------------------------------------


def test_retrieve_live_claim(store):
    store.write_claim("user", "works_at", "Anthropic", omega=0.85,
                      supersedes_hint="works_at")

    r = Retriever(store)
    result = r.query("Where do I work?")
    assert len(result.claims) == 1
    assert result.claims[0].object_text == "Anthropic"


def test_retrieve_history_includes_past(store):
    store.write_claim("user", "works_at", "Google", omega=0.85,
                      supersedes_hint="works_at")
    store.write_claim("user", "works_at", "Anthropic", omega=0.85,
                      supersedes_hint="works_at")

    r = Retriever(store)
    result = r.query("Where did I used to work?")
    assert len(result.history) == 2
    objects = [c.object_text for c in result.history]
    assert "Google" in objects
    assert "Anthropic" in objects


def test_retrieve_entity_scoped(store):
    """'Tell me about Jamie' returns claims with Jamie as subject or object."""
    store.write_claim("user", "has_relationship", "Jamie",
                      omega=1.0, object_entity="Jamie")
    store.write_claim("Jamie", "lives_in", "Brooklyn", omega=0.5)

    r = Retriever(store)
    result = r.query("Tell me about Jamie")
    # Jamie-as-subject + the user-has-relationship-with-Jamie claim
    assert len(result.claims) >= 2


def test_retrieve_fts_fallback(store):
    store.append_turn("s1", "user", "Last summer I climbed in Yosemite, it was amazing.")
    store.append_turn("s1", "user", "Made pasta tonight, came out great.")

    r = Retriever(store)
    result = r.query("Remember that climbing trip?")
    assert result.spec.intent == "search"
    assert len(result.transcript_hits) >= 1
    assert "yosemite" in result.transcript_hits[0].content.lower()


def test_retrieve_omega_floor_filters_low(store):
    store.write_claim("user", "did", "marathon", omega=0.20)  # below default floor 0.25
    store.write_claim("user", "did", "wedding", omega=1.0)

    r = Retriever(store, omega_floor=0.25)
    result = r.query("What do I do?")
    objects = [c.object_text for c in result.claims]
    assert "wedding" in objects
    assert "marathon" not in objects


# --- compose tests -----------------------------------------------------------


def test_compose_lookup_high_confidence(store):
    store.write_claim("user", "works_at", "Anthropic", omega=0.95)
    r = Retriever(store)
    result = r.query("Where do I work?")
    response = rule_based_compose(result, store=store)
    assert "Anthropic" in response.text
    assert response.confidence == "high"
    # High ω → direct statement, no hedging
    assert "mentioned" not in response.text.lower()
    assert "inferred" not in response.text.lower()


def test_compose_lookup_medium_confidence_hedges(store):
    store.write_claim("user", "considers", "grad school", omega=0.55)
    r = Retriever(store)
    result = r.query("What do I consider?")
    response = rule_based_compose(result, store=store)
    # 0.4 ≤ ω < 0.85 → hedged ("you mentioned")
    assert "mentioned" in response.text.lower() or response.confidence == "medium"


def test_compose_lookup_low_confidence_inferred(store):
    store.write_claim("user", "did", "marathon (inferred)", omega=0.30)
    r = Retriever(store, omega_floor=0.0)
    result = r.query("What did I do?")
    response = rule_based_compose(result, store=store)
    assert "inferred" in response.text.lower()
    assert response.confidence == "low"


def test_compose_history_shows_current_and_past(store):
    store.write_claim("user", "works_at", "Google", omega=0.85,
                      supersedes_hint="works_at")
    store.write_claim("user", "works_at", "Anthropic", omega=0.85,
                      supersedes_hint="works_at")
    r = Retriever(store)
    result = r.query("Where did I used to work?")
    response = rule_based_compose(result, store=store)
    assert "Anthropic" in response.text
    assert "Google" in response.text
    assert "Previously" in response.text or "previously" in response.text.lower()


def test_compose_empty_returns_polite_none(store):
    r = Retriever(store)
    result = r.query("Where do I work?")
    response = rule_based_compose(result, store=store)
    assert response.confidence == "none"
    assert response.cited_claims == []


def test_compose_fts_fallback_when_no_claims(store):
    store.append_turn("s1", "user", "Last summer I climbed in Yosemite, it was amazing.")
    r = Retriever(store)
    result = r.query("Remember that climbing trip?")
    response = rule_based_compose(result, store=store)
    assert "Yosemite" in response.text
    assert response.confidence == "medium"
    assert response.cited_turns
