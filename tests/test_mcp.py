"""Tests for the MCP server tools.

We don't spin up the MCP transport — we call the tool functions directly,
because the @mcp.tool decorator preserves callable semantics and the
interesting logic is in the function bodies, not the protocol.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from keelson.mcp_server import (
    _set_store_for_testing,
    claim_history,
    forget_entity,
    ingest_turn,
    list_live_claims,
    recall,
    surface_scan,
    write_claim,
)
from keelson.store import KeelsonStore


def _unwrap(tool):
    """FastMCP wraps tools as FunctionTool objects exposing the original
    callable on .fn. Fall back to calling tool() if .fn isn't present."""
    return getattr(tool, "fn", tool)


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmp:
        s = KeelsonStore(Path(tmp) / "test.db")
        _set_store_for_testing(s)
        yield s
        _set_store_for_testing(None)
        s.close()


def test_ingest_turn_appends(store):
    out = _unwrap(ingest_turn)(role="user", content="Hello world.")
    assert "Appended turn #1" in out
    assert store.get_turn(1).content == "Hello world."


def test_ingest_turn_rejects_bad_role(store):
    out = _unwrap(ingest_turn)(role="alien", content="...")
    assert "Error" in out
    assert "alien" in out


def test_write_claim_basic(store):
    out = _unwrap(write_claim)(
        subject="user", predicate="works_at", object="Anthropic", omega=0.85,
    )
    assert "Wrote claim #1" in out
    live = store.live_claims(subject="user", predicate="works_at")
    assert len(live) == 1
    assert live[0].object_text == "Anthropic"


def test_write_claim_supersedes(store):
    _unwrap(write_claim)(subject="user", predicate="works_at", object="Google", omega=0.85)
    out = _unwrap(write_claim)(
        subject="user", predicate="works_at", object="Anthropic",
        omega=0.95, supersedes_hint="works_at",
    )
    assert "supersedes" in out.lower()
    live = store.live_claims(subject="user", predicate="works_at")
    assert len(live) == 1
    assert live[0].object_text == "Anthropic"


def test_recall_returns_composed_response(store):
    _unwrap(write_claim)(
        subject="user", predicate="works_at", object="Anthropic", omega=0.95,
    )
    out = _unwrap(recall)(query="Where do I work?")
    assert "Anthropic" in out


def test_recall_empty(store):
    out = _unwrap(recall)(query="Where do I work?")
    assert out.lower().startswith("i don't have")


def test_claim_history_shows_supersession(store):
    _unwrap(write_claim)(subject="user", predicate="works_at", object="Google", omega=0.85)
    _unwrap(write_claim)(
        subject="user", predicate="works_at", object="Anthropic",
        omega=0.95, supersedes_hint="works_at",
    )
    out = _unwrap(claim_history)(subject="user", predicate="works_at")
    assert "Google" in out
    assert "Anthropic" in out
    assert "LIVE" in out
    assert "PAST" in out


def test_list_live_claims_filter(store):
    _unwrap(write_claim)(subject="user", predicate="loves", object="jazz", omega=0.85)
    _unwrap(write_claim)(subject="user", predicate="works_at", object="Anthropic", omega=0.85)
    out = _unwrap(list_live_claims)(predicate="loves")
    assert "jazz" in out
    assert "Anthropic" not in out


def test_forget_entity_with_cascade(store):
    _unwrap(write_claim)(
        subject="user", predicate="has_relationship", object="Jamie",
        omega=1.0, object_entity="Jamie",
    )
    out = _unwrap(forget_entity)(name="Jamie", cascade=True)
    assert "entity=1" in out
    assert "direct claims=1" in out
    live = store.live_claims(subject="user", predicate="has_relationship")
    assert live == []


def test_forget_entity_missing(store):
    out = _unwrap(forget_entity)(name="Nobody")
    assert "No entity" in out


def test_surface_scan_empty(store):
    out = _unwrap(surface_scan)()
    assert "Nothing to surface" in out


def test_surface_scan_finds_contradiction(store):
    """Force a contradiction via direct INSERT to bypass supersession."""
    e = store.resolve_or_create_entity("user")
    from keelson.store import _now
    now_iso = _now()
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

    out = _unwrap(surface_scan)()
    assert "unresolved_contradiction" in out
    assert "Google" in out and "Anthropic" in out


def test_surface_scan_mark_shown_cooldown(store):
    # Backdate an entity to trigger stale_relationship
    from datetime import datetime, timedelta, timezone
    jamie = store.resolve_or_create_entity("Jamie", entity_type="person")
    old = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat(timespec="seconds")
    store.conn.execute("UPDATE entities SET last_seen_ts = ? WHERE id = ?", (old, jamie.id))
    store.conn.commit()

    first = _unwrap(surface_scan)(mark_shown=True)
    assert "stale_relationship" in first

    second = _unwrap(surface_scan)()
    assert "Nothing to surface" in second or "stale_relationship" not in second
