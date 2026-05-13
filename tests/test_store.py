"""Tests for the SQLite store — supersession, history, forget cascade, FTS."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from keelson.store import KeelsonStore


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmp:
        s = KeelsonStore(Path(tmp) / "test.db")
        yield s
        s.close()


def test_append_and_get_turn(store):
    tid = store.append_turn("s1", "user", "I work at Anthropic.")
    turn = store.get_turn(tid)
    assert turn is not None
    assert turn.session_id == "s1"
    assert turn.role == "user"
    assert turn.content == "I work at Anthropic."


def test_resolve_entity_deduplicates(store):
    e1 = store.resolve_or_create_entity("Jamie")
    e2 = store.resolve_or_create_entity("Jamie")
    e3 = store.resolve_or_create_entity("jamie")  # case-insensitive
    assert e1.id == e2.id == e3.id


def test_write_and_query_live_claim(store):
    store.write_claim("user", "works_at", "Anthropic", omega=0.85)
    live = store.live_claims(subject="user", predicate="works_at")
    assert len(live) == 1
    assert live[0].object_text == "Anthropic"
    assert live[0].omega == 0.85
    assert live[0].valid_to is None


def test_supersession_marks_old_and_keeps_history(store):
    """The Google -> Anthropic scenario."""
    old_id = store.write_claim("user", "works_at", "Google", omega=0.85)
    new_id = store.write_claim(
        "user", "works_at", "Anthropic", omega=0.85, supersedes_hint="works_at"
    )

    # Live query returns only Anthropic
    live = store.live_claims(subject="user", predicate="works_at")
    assert len(live) == 1
    assert live[0].object_text == "Anthropic"

    # History returns both, with Google marked superseded
    history = store.claim_history("user", "works_at")
    assert len(history) == 2
    by_id = {c.id: c for c in history}
    assert by_id[old_id].valid_to is not None
    assert by_id[old_id].superseded_by == new_id
    assert by_id[new_id].valid_to is None


def test_supersession_only_when_hint_provided(store):
    """Without supersedes_hint, both claims coexist as live (additive write)."""
    store.write_claim("user", "likes", "jazz", omega=0.85)
    store.write_claim("user", "likes", "punk", omega=0.85)

    live = store.live_claims(subject="user", predicate="likes")
    assert len(live) == 2


def test_supersession_collapses_multiple_live_claims(store):
    """If accumulated state has multiple live claims for one predicate,
    a new supersession write should collapse them all."""
    # Two writes without supersedes_hint accumulate
    store.write_claim("user", "works_at", "Google", omega=0.85)
    store.write_claim("user", "works_at", "Stripe", omega=0.85)
    # Third write WITH supersedes_hint should supersede both
    new_id = store.write_claim(
        "user", "works_at", "Anthropic", omega=0.95, supersedes_hint="works_at"
    )

    live = store.live_claims(subject="user", predicate="works_at")
    assert len(live) == 1
    assert live[0].id == new_id
    assert live[0].object_text == "Anthropic"

    history = store.claim_history("user", "works_at")
    assert len(history) == 3
    past = [c for c in history if c.valid_to is not None]
    assert len(past) == 2
    assert all(c.superseded_by == new_id for c in past)


def test_forget_entity_direct(store):
    """Forgetting an entity scrubs claims where it's the subject or object.

    Note: linking an object to an entity requires passing object_entity= explicitly.
    Plain string objects are not auto-resolved — the extractor would be responsible
    for tagging proper-noun objects as entities.
    """
    store.write_claim(
        "user", "has_relationship", "Jamie", omega=1.0, object_entity="Jamie"
    )
    jamie = store.resolve_or_create_entity("Jamie")
    store.write_claim("Jamie", "lives_in", "Brooklyn", omega=0.5)

    counts = store.forget_entity(jamie.id, cascade=True)

    # Entity is marked forgotten
    assert counts["entity"] == 1
    # Both claims involving Jamie (as subject, as object) are forgotten
    assert counts["direct"] == 2

    user_relationships = store.live_claims(subject="user", predicate="has_relationship")
    assert len(user_relationships) == 0


def test_forget_cascade_via_omega_origin(store):
    """Claims whose ω flowed through a forgotten entity are also scrubbed.

    The Jamie / Italian-food case:
      - User says: 'Italian food is great, Jamie introduced me.'
      - The claim user.likes=Italian has ω-origin = [Jamie's entity id]
      - Forgetting Jamie should cascade and scrub the inferred claim,
        because without Jamie we never would have learned it.

    A separately-confirmed claim (the user *also* said 'I love Italian')
    has empty omega_origin and survives.
    """
    jamie = store.resolve_or_create_entity("Jamie")
    italian_inferred = store.write_claim(
        "user", "likes", "Italian food", omega=0.5, omega_origin=[jamie.id]
    )
    italian_direct = store.write_claim(
        "user", "likes", "Italian food (confirmed solo)", omega=1.0, omega_origin=[]
    )

    counts = store.forget_entity(jamie.id, cascade=True)

    assert counts["omega_cascade"] >= 1

    inferred = store.get_claim(italian_inferred)
    direct = store.get_claim(italian_direct)
    assert inferred.forgotten is True
    assert direct.forgotten is False


def test_forget_without_cascade(store):
    """cascade=False marks the entity but leaves claims alone."""
    jamie = store.resolve_or_create_entity("Jamie")
    store.write_claim(
        "user", "has_relationship", "Jamie", omega=1.0, object_entity="Jamie"
    )

    counts = store.forget_entity(jamie.id, cascade=False)

    assert counts["entity"] == 1
    assert counts["direct"] == 0
    # Claim is still live
    rel = store.live_claims(subject="user", predicate="has_relationship")
    assert len(rel) == 1


def test_fts_search(store):
    store.append_turn("s1", "user", "I went climbing in Yosemite last summer.")
    store.append_turn("s1", "user", "Anthropic is hiring more researchers.")
    store.append_turn("s1", "user", "I love jazz, especially Coltrane.")

    hits = store.search_turns("climbing")
    assert len(hits) == 1
    assert "Yosemite" in hits[0].content

    hits = store.search_turns("anthropic OR jazz")
    assert len(hits) == 2


def test_live_claims_filter_by_omega(store):
    store.write_claim("user", "did", "marathon (inferred)", omega=0.25)
    store.write_claim("user", "did", "wedding", omega=1.0)

    high_only = store.live_claims(subject="user", predicate="did", omega_min=0.5)
    assert len(high_only) == 1
    assert high_only[0].object_text == "wedding"

    all_of_them = store.live_claims(subject="user", predicate="did", omega_min=0.0)
    assert len(all_of_them) == 2
