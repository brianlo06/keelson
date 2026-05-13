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


def test_backup_db_creates_file(store, tmp_path, monkeypatch):
    monkeypatch.setenv("KEELSON_DB_PATH", str(tmp_path / "active.db"))
    _unwrap(write_claim)(subject="user", predicate="works_at", object="X", omega=0.85)
    from keelson.mcp_server import backup_db
    target = tmp_path / "snap.db"
    out = _unwrap(backup_db)(target_path=str(target))
    assert target.exists()
    assert str(target) in out


def test_restore_db_swaps_state(store, tmp_path, monkeypatch):
    """Backup → mutate → restore should bring back the pre-mutation state."""
    monkeypatch.setenv("KEELSON_DB_PATH", str(tmp_path / "active.db"))
    from keelson.mcp_server import backup_db, restore_db, _set_store_for_testing

    # We need the singleton path to match the env, so close the test fixture and
    # let the server open a fresh store via get_store() at the env-pointed path
    _set_store_for_testing(None)

    _unwrap(write_claim)(subject="user", predicate="works_at", object="OLD", omega=0.85)
    snap = tmp_path / "snap.db"
    _unwrap(backup_db)(target_path=str(snap))

    # Mutate the live state away from the snapshot
    _unwrap(write_claim)(subject="user", predicate="lives_in", object="NEW", omega=0.85)

    # Restore — the lives_in claim should vanish
    out = _unwrap(restore_db)(backup_path=str(snap))
    assert "Restored from" in out
    assert "OLD" in _unwrap(list_live_claims)(predicate="works_at")
    assert _unwrap(list_live_claims)(predicate="lives_in").startswith("No live claims")


def test_restore_db_missing_file(store, tmp_path):
    from keelson.mcp_server import restore_db
    out = _unwrap(restore_db)(backup_path=str(tmp_path / "ghost.db"))
    assert "does not exist" in out


def test_extract_and_store_requires_api_key(store, monkeypatch):
    """Without ANTHROPIC_API_KEY, the tool should error politely without
    attempting an API call."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from keelson.mcp_server import extract_and_store
    out = _unwrap(extract_and_store)(content="I work at Anthropic.")
    assert "ANTHROPIC_API_KEY not set" in out


def test_extract_and_store_writes_claims_via_mocked_extractor(store, monkeypatch):
    """With a mocked extractor, the tool should append a turn AND write the
    returned claims."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    from unittest.mock import MagicMock, patch

    from keelson.mcp_server import extract_and_store
    from keelson.schema import Claim, ExtractionResult

    fake_result = ExtractionResult(
        claims=[
            Claim(
                subject="user",
                predicate="works_at",
                object="Anthropic",
                valid_from="now",
                omega=0.85,
                source_phrase="I work at Anthropic",
                supersedes_hint="works_at",
            )
        ]
    )
    fake_extractor = MagicMock()
    fake_extractor.extract.return_value = fake_result

    with patch("keelson.extractor.Extractor", return_value=fake_extractor):
        out = _unwrap(extract_and_store)(content="I work at Anthropic.")

    assert "extracted 1 claim" in out
    live = store.live_claims(subject="user", predicate="works_at")
    assert len(live) == 1
    assert live[0].object_text == "Anthropic"


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
