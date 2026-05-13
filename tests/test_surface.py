"""Tests for the proactive surfacer.

We backdate timestamps with direct SQL because the store always writes _now().
This is the cleanest way to simulate old data.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from keelson.store import KeelsonStore
from keelson.surface import Surfacer


NOW = datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmp:
        s = KeelsonStore(Path(tmp) / "test.db")
        yield s
        s.close()


def _backdate(store: KeelsonStore, table: str, row_id: int, column: str, days_ago: int) -> None:
    new_ts = (NOW - timedelta(days=days_ago)).isoformat(timespec="seconds")
    store.conn.execute(f"UPDATE {table} SET {column} = ? WHERE id = ?", (new_ts, row_id))
    store.conn.commit()


# --- stale_relationship ------------------------------------------------------


def test_stale_relationship_fires_after_threshold(store):
    jamie = store.resolve_or_create_entity("Jamie", entity_type="person")
    _backdate(store, "entities", jamie.id, "last_seen_ts", days_ago=90)

    cands = Surfacer(store, now=NOW).stale_relationships(threshold_days=60)

    assert len(cands) == 1
    assert cands[0].type == "stale_relationship"
    assert cands[0].entity_id == jamie.id
    assert "Jamie" in cands[0].message or "jamie" in cands[0].message.lower()


def test_stale_relationship_silent_within_threshold(store):
    store.resolve_or_create_entity("Jamie", entity_type="person")
    # No backdate — fresh entity
    cands = Surfacer(store, now=NOW).stale_relationships(threshold_days=60)
    assert cands == []


def test_stale_relationship_skips_user_singleton(store):
    """The 'user' singleton entity must never be flagged as stale."""
    store.resolve_or_create_entity("user", entity_type="person")
    _backdate(store, "entities", 1, "last_seen_ts", days_ago=200)
    cands = Surfacer(store, now=NOW).stale_relationships(threshold_days=60)
    assert cands == []


# --- anniversary -------------------------------------------------------------


def test_anniversary_finds_turn_from_one_year_ago(store):
    one_year_ago = (NOW - timedelta(days=365)).isoformat(timespec="seconds")
    tid = store.append_turn("s1", "user", "Yosemite climbing trip was unreal.", ts=one_year_ago)

    cands = Surfacer(store, now=NOW).anniversaries()

    assert any(c.type == "anniversary" and tid in c.turn_ids for c in cands)


def test_anniversary_ignores_non_anniversary_turns(store):
    six_months_ago = (NOW - timedelta(days=180)).isoformat(timespec="seconds")
    store.append_turn("s1", "user", "Nothing special.", ts=six_months_ago)

    cands = Surfacer(store, now=NOW).anniversaries()
    assert cands == []


# --- unresolved_contradiction ------------------------------------------------


def test_unresolved_contradiction_fires_on_multi_live(store):
    # Direct INSERTs to bypass supersession and create the bug-state intentionally
    e = store.resolve_or_create_entity("user")
    now_iso = NOW.isoformat(timespec="seconds")
    for obj in ["Google", "Anthropic"]:
        store.conn.execute(
            """
            INSERT INTO claims (subject_id, predicate, object_text, asserted_ts,
                                valid_from, omega, omega_origin)
            VALUES (?, 'works_at', ?, ?, ?, 0.85, '[]')
            """,
            (e.id, obj, now_iso, now_iso),
        )
    store.conn.commit()

    cands = Surfacer(store, now=NOW).unresolved_contradictions()

    assert len(cands) == 1
    assert cands[0].priority == 1.0
    assert "Google" in cands[0].message and "Anthropic" in cands[0].message


def test_unresolved_contradiction_silent_when_one_live(store):
    store.write_claim("user", "works_at", "Anthropic", omega=0.85, supersedes_hint="works_at")
    cands = Surfacer(store, now=NOW).unresolved_contradictions()
    assert cands == []


# --- drift_candidate ---------------------------------------------------------


def test_drift_candidate_fires_for_aged_high_omega_claim(store):
    cid = store.write_claim("user", "diet", "vegetarian", omega=1.0)
    _backdate(store, "claims", cid, "asserted_ts", days_ago=730)

    cands = Surfacer(store, now=NOW).drift_candidates(age_days=180, omega_min=0.85)

    assert len(cands) == 1
    assert cands[0].type == "drift_candidate"
    assert cid in cands[0].claim_ids
    assert "vegetarian" in cands[0].message


def test_drift_candidate_skips_recently_reasserted(store):
    """If a newer claim for the same (subject, predicate) exists, the old claim
    is not flagged — re-confirmation has happened (even if it superseded)."""
    cid_old = store.write_claim("user", "diet", "vegetarian", omega=1.0)
    _backdate(store, "claims", cid_old, "asserted_ts", days_ago=730)
    # New assertion 30 days ago (still live; no supersedes_hint so both live)
    cid_new = store.write_claim("user", "diet", "vegetarian", omega=1.0)
    _backdate(store, "claims", cid_new, "asserted_ts", days_ago=30)

    cands = Surfacer(store, now=NOW).drift_candidates(age_days=180)
    # The old one is suppressed because there's a newer assertion
    old_in_cands = any(cid_old in c.claim_ids for c in cands)
    assert not old_in_cands


def test_drift_candidate_skips_low_omega(store):
    cid = store.write_claim("user", "did", "marathon (inferred)", omega=0.30)
    _backdate(store, "claims", cid, "asserted_ts", days_ago=730)
    cands = Surfacer(store, now=NOW).drift_candidates(omega_min=0.85)
    assert cands == []


# --- emerging_entity ---------------------------------------------------------


def test_emerging_entity_fires_with_multiple_recent_mentions(store):
    alex = store.resolve_or_create_entity("Alex", entity_type="person")
    # Three claims involving Alex
    store.write_claim("user", "has_friend", "Alex", omega=0.85, object_entity="Alex")
    store.write_claim("Alex", "works_at", "Stripe", omega=0.5)
    store.write_claim("Alex", "lives_in", "SF", omega=0.5)

    cands = Surfacer(store, now=NOW).emerging_entities(min_mentions=3)
    assert len(cands) == 1
    assert cands[0].entity_id == alex.id
    assert "Alex" in cands[0].message


def test_emerging_entity_silent_for_old_or_underused(store):
    # Old entity, even with many claims
    old = store.resolve_or_create_entity("Bob", entity_type="person")
    _backdate(store, "entities", old.id, "created_ts", days_ago=60)
    store.write_claim("Bob", "lives_in", "NYC", omega=0.5)
    store.write_claim("Bob", "works_at", "X", omega=0.5)
    store.write_claim("Bob", "likes", "jazz", omega=0.5)

    # New entity, only one claim
    store.resolve_or_create_entity("Charlie", entity_type="person")
    store.write_claim("Charlie", "lives_in", "LA", omega=0.5)

    cands = Surfacer(store, now=NOW).emerging_entities(recency_days=14, min_mentions=3)
    assert cands == []


# --- co_occurrence -----------------------------------------------------------


def test_co_occurrence_fires_when_pair_mentioned_together(store):
    """Two entities that show up in the same turn 3+ times should surface."""
    store.resolve_or_create_entity("Jamie", entity_type="person")
    store.resolve_or_create_entity("Yosemite", entity_type="place")

    # 3 turns mentioning both
    for content in [
        "I went climbing with Jamie at Yosemite last weekend.",
        "Jamie wants to go back to Yosemite in spring.",
        "Showed Jamie my Yosemite photos again.",
    ]:
        store.append_turn("s1", "user", content)

    cands = Surfacer(store, now=NOW).co_occurrences(min_count=3)
    assert len(cands) == 1
    assert "Jamie" in cands[0].message and "Yosemite" in cands[0].message
    assert cands[0].evidence["count"] == 3


def test_co_occurrence_silent_below_threshold(store):
    store.resolve_or_create_entity("Jamie", entity_type="person")
    store.resolve_or_create_entity("Yosemite", entity_type="place")
    store.append_turn("s1", "user", "Jamie and I went to Yosemite once.")  # 1 mention

    cands = Surfacer(store, now=NOW).co_occurrences(min_count=3)
    assert cands == []


def test_co_occurrence_word_boundary(store):
    """'jamie' must not match 'jamies' or 'jamie's'."""
    store.resolve_or_create_entity("Jam", entity_type="thing")
    store.resolve_or_create_entity("Yo", entity_type="thing")
    # The substring 'jam' appears in 'jamie' but should NOT count
    for _ in range(5):
        store.append_turn("s1", "user", "Jamie went to Yosemite.")

    cands = Surfacer(store, now=NOW).co_occurrences(min_count=3)
    # 'Jam' and 'Yo' should NOT co-occur since they only match as substrings
    assert cands == []


# --- cooldown ----------------------------------------------------------------


def test_cooldown_suppresses_repeat(store):
    jamie = store.resolve_or_create_entity("Jamie", entity_type="person")
    _backdate(store, "entities", jamie.id, "last_seen_ts", days_ago=90)

    surfacer = Surfacer(store, now=NOW)
    first = surfacer.scan()
    assert len(first) >= 1

    surfacer.mark_shown(first[0])

    second = surfacer.scan(cooldown_days=30)
    # Same item should be filtered out
    assert not any(c.target_key == first[0].target_key and c.type == first[0].type for c in second)


def test_cooldown_expires_after_window(store):
    jamie = store.resolve_or_create_entity("Jamie", entity_type="person")
    _backdate(store, "entities", jamie.id, "last_seen_ts", days_ago=90)

    surfacer = Surfacer(store, now=NOW)
    first = surfacer.scan()
    surfacer.mark_shown(first[0])

    # Backdate the surfaced record itself to 40 days ago
    sid = store.conn.execute("SELECT id FROM surfaced ORDER BY id DESC LIMIT 1").fetchone()[0]
    _backdate(store, "surfaced", sid, "surfaced_at", days_ago=40)

    # With 30-day cooldown, it should re-appear
    second = surfacer.scan(cooldown_days=30)
    assert any(c.target_key == first[0].target_key for c in second)


def test_priority_orders_results(store):
    # Make an unresolved contradiction (priority 1.0) AND a stale relationship (priority < 0.7)
    jamie = store.resolve_or_create_entity("Jamie", entity_type="person")
    _backdate(store, "entities", jamie.id, "last_seen_ts", days_ago=90)

    e = store.resolve_or_create_entity("user")
    now_iso = NOW.isoformat(timespec="seconds")
    for obj in ["Google", "Anthropic"]:
        store.conn.execute(
            """
            INSERT INTO claims (subject_id, predicate, object_text, asserted_ts,
                                valid_from, omega, omega_origin)
            VALUES (?, 'works_at', ?, ?, ?, 0.85, '[]')
            """,
            (e.id, obj, now_iso, now_iso),
        )
    store.conn.commit()

    candidates = Surfacer(store, now=NOW).scan()
    assert candidates[0].type == "unresolved_contradiction"
