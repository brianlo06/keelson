"""SQLite storage for Keelson — three tables: turns, entities, claims.

Design invariants:
- `turns` is append-only. The transcript is bedrock; never delete.
- `claims` are never deleted on supersession — set `valid_to` and link
  `superseded_by`. History is preserved.
- `forgotten=1` is a soft-delete tombstone. Claims still exist in the table
  (for audit) but are excluded from default queries.
- Entity resolution is by canonical_name (case-insensitive). Aliases support
  later — for v1 we just lowercase the name.

The interesting operations live here:
- write_claim: supersedes prior live claim if supersedes_hint matches
- forget_entity: cascades through subject_id, object_entity_id, AND omega_origin
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    ts           TEXT NOT NULL,
    role         TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS turns_session_ts ON turns(session_id, ts);

CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
    content,
    content='turns',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS turns_fts_insert AFTER INSERT ON turns BEGIN
    INSERT INTO turns_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TABLE IF NOT EXISTS entities (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    type            TEXT NOT NULL DEFAULT 'unknown',
    canonical_name  TEXT NOT NULL UNIQUE,
    aliases_json    TEXT NOT NULL DEFAULT '[]',
    created_ts      TEXT NOT NULL,
    last_seen_ts    TEXT NOT NULL,
    forgotten       INTEGER NOT NULL DEFAULT 0,
    forgotten_ts    TEXT
);

CREATE INDEX IF NOT EXISTS entities_canonical ON entities(canonical_name);

CREATE TABLE IF NOT EXISTS claims (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id        INTEGER NOT NULL REFERENCES entities(id),
    predicate         TEXT NOT NULL,
    object_text       TEXT NOT NULL,
    object_entity_id  INTEGER REFERENCES entities(id),
    asserted_ts       TEXT NOT NULL,
    source_turn_id    INTEGER REFERENCES turns(id),
    valid_from        TEXT NOT NULL,
    valid_to          TEXT,
    superseded_by     INTEGER REFERENCES claims(id),
    omega             REAL NOT NULL CHECK (omega >= 0.0 AND omega <= 1.0),
    omega_origin      TEXT NOT NULL DEFAULT '[]',
    forgotten         INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS claims_live ON claims(subject_id, predicate)
    WHERE valid_to IS NULL AND forgotten = 0;
CREATE INDEX IF NOT EXISTS claims_object ON claims(object_entity_id) WHERE forgotten = 0;
CREATE INDEX IF NOT EXISTS claims_asserted ON claims(asserted_ts);

CREATE TABLE IF NOT EXISTS surfaced (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    surface_type    TEXT NOT NULL,
    target_key      TEXT NOT NULL,
    surfaced_at     TEXT NOT NULL,
    user_response   TEXT
);

CREATE INDEX IF NOT EXISTS surfaced_lookup ON surfaced(surface_type, target_key, surfaced_at);
"""


@dataclass(frozen=True)
class TurnRow:
    id: int
    session_id: str
    ts: str
    role: str
    content: str


@dataclass(frozen=True)
class EntityRow:
    id: int
    type: str
    canonical_name: str
    aliases: list[str]
    created_ts: str
    last_seen_ts: str
    forgotten: bool
    forgotten_ts: str | None


@dataclass(frozen=True)
class ClaimRow:
    id: int
    subject_id: int
    predicate: str
    object_text: str
    object_entity_id: int | None
    asserted_ts: str
    source_turn_id: int | None
    valid_from: str
    valid_to: str | None
    superseded_by: int | None
    omega: float
    omega_origin: list[int]
    forgotten: bool


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(name: str) -> str:
    return name.strip().lower()


class KeelsonStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # --- turns ---------------------------------------------------------------

    def append_turn(self, session_id: str, role: str, content: str, ts: str | None = None) -> int:
        ts = ts or _now()
        with self.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO turns (session_id, ts, role, content) VALUES (?, ?, ?, ?)",
                (session_id, ts, role, content),
            )
            return cur.lastrowid

    def get_turn(self, turn_id: int) -> TurnRow | None:
        row = self.conn.execute("SELECT * FROM turns WHERE id = ?", (turn_id,)).fetchone()
        return TurnRow(**dict(row)) if row else None

    def search_turns(self, query: str, limit: int = 10) -> list[TurnRow]:
        rows = self.conn.execute(
            """
            SELECT turns.*
            FROM turns_fts
            JOIN turns ON turns.id = turns_fts.rowid
            WHERE turns_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
        return [TurnRow(**dict(r)) for r in rows]

    # --- entities ------------------------------------------------------------

    def resolve_or_create_entity(self, name: str, entity_type: str = "unknown") -> EntityRow:
        canonical = _canonical(name)
        now = _now()
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM entities WHERE canonical_name = ?", (canonical,)
            ).fetchone()
            if row is None:
                cur = conn.execute(
                    """
                    INSERT INTO entities (type, canonical_name, aliases_json, created_ts, last_seen_ts)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (entity_type, canonical, json.dumps([name]), now, now),
                )
                ent_id = cur.lastrowid
                row = conn.execute("SELECT * FROM entities WHERE id = ?", (ent_id,)).fetchone()
            else:
                conn.execute(
                    "UPDATE entities SET last_seen_ts = ? WHERE id = ?", (now, row["id"])
                )
        return _to_entity(row)

    def get_entity(self, entity_id: int) -> EntityRow | None:
        row = self.conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
        return _to_entity(row) if row else None

    def find_entity(self, name: str) -> EntityRow | None:
        """Look up an entity by name. Does not create one if missing.
        Use this when the caller's intent is to operate on an existing entity
        (e.g. forgetting it) rather than to ensure it exists.
        """
        row = self.conn.execute(
            "SELECT * FROM entities WHERE canonical_name = ?", (_canonical(name),)
        ).fetchone()
        return _to_entity(row) if row else None

    # --- claims --------------------------------------------------------------

    def write_claim(
        self,
        subject: str,
        predicate: str,
        object: str,
        omega: float,
        source_turn_id: int | None = None,
        supersedes_hint: str | None = None,
        omega_origin: list[int] | None = None,
        object_entity: str | None = None,
        valid_from: str | None = None,
    ) -> int:
        """Write a claim. If supersedes_hint matches a live claim for the same
        subject, mark that claim superseded (set valid_to + superseded_by).
        Returns the new claim id.
        """
        subject_ent = self.resolve_or_create_entity(subject)
        object_ent_id: int | None = None
        if object_entity:
            object_ent_id = self.resolve_or_create_entity(object_entity).id

        now = _now()
        with self.transaction() as conn:
            # Find ALL live claims to supersede. There's normally at most one,
            # but if accumulated state has multiple, we supersede them all so
            # the live set converges to a single row.
            superseded_ids: list[int] = []
            if supersedes_hint:
                rows = conn.execute(
                    """
                    SELECT id FROM claims
                    WHERE subject_id = ? AND predicate = ?
                      AND valid_to IS NULL AND forgotten = 0
                    """,
                    (subject_ent.id, supersedes_hint),
                ).fetchall()
                superseded_ids = [r["id"] for r in rows]

            cur = conn.execute(
                """
                INSERT INTO claims (
                    subject_id, predicate, object_text, object_entity_id,
                    asserted_ts, source_turn_id, valid_from, omega, omega_origin
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    subject_ent.id,
                    predicate,
                    object,
                    object_ent_id,
                    now,
                    source_turn_id,
                    valid_from or now,
                    float(omega),
                    json.dumps(omega_origin or []),
                ),
            )
            new_id = cur.lastrowid

            for sid in superseded_ids:
                conn.execute(
                    "UPDATE claims SET valid_to = ?, superseded_by = ? WHERE id = ?",
                    (now, new_id, sid),
                )
            return new_id

    def get_claim(self, claim_id: int) -> ClaimRow | None:
        row = self.conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
        return _to_claim(row) if row else None

    def live_claims(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        omega_min: float = 0.0,
    ) -> list[ClaimRow]:
        """Live = valid_to IS NULL AND not forgotten."""
        sql = "SELECT c.* FROM claims c WHERE c.valid_to IS NULL AND c.forgotten = 0 AND c.omega >= ?"
        params: list = [omega_min]
        if subject:
            ent = self.conn.execute(
                "SELECT id FROM entities WHERE canonical_name = ?", (_canonical(subject),)
            ).fetchone()
            if not ent:
                return []
            sql += " AND c.subject_id = ?"
            params.append(ent["id"])
        if predicate:
            sql += " AND c.predicate = ?"
            params.append(predicate)
        sql += " ORDER BY c.asserted_ts DESC"
        rows = self.conn.execute(sql, params).fetchall()
        return [_to_claim(r) for r in rows]

    def claim_history(self, subject: str, predicate: str) -> list[ClaimRow]:
        """All claims (live + superseded, not forgotten) for a (subject, predicate),
        ordered chronologically."""
        ent = self.conn.execute(
            "SELECT id FROM entities WHERE canonical_name = ?", (_canonical(subject),)
        ).fetchone()
        if not ent:
            return []
        rows = self.conn.execute(
            """
            SELECT * FROM claims
            WHERE subject_id = ? AND predicate = ? AND forgotten = 0
            ORDER BY valid_from ASC
            """,
            (ent["id"], predicate),
        ).fetchall()
        return [_to_claim(r) for r in rows]

    # --- forgetting ----------------------------------------------------------

    def forget_entity(self, entity_id: int, cascade: bool = True) -> dict[str, int]:
        """Mark entity as forgotten. With cascade, also forget:
          - Claims where the entity is subject or object
          - Claims whose omega_origin includes this entity_id
            (i.e. the ω was acquired *through* this entity — without this
            entity, we shouldn't claim to know the fact)

        Returns counts: {'entity': 1, 'direct': N, 'omega_cascade': M}.
        """
        now = _now()
        counts = {"entity": 0, "direct": 0, "omega_cascade": 0}
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE entities SET forgotten = 1, forgotten_ts = ? WHERE id = ? AND forgotten = 0",
                (now, entity_id),
            )
            counts["entity"] = cur.rowcount

            if not cascade:
                return counts

            cur = conn.execute(
                """
                UPDATE claims SET forgotten = 1
                WHERE forgotten = 0
                  AND (subject_id = ? OR object_entity_id = ?)
                """,
                (entity_id, entity_id),
            )
            counts["direct"] = cur.rowcount

            cur = conn.execute(
                """
                UPDATE claims SET forgotten = 1
                WHERE forgotten = 0
                  AND EXISTS (
                      SELECT 1 FROM json_each(claims.omega_origin)
                      WHERE json_each.value = ?
                  )
                """,
                (entity_id,),
            )
            counts["omega_cascade"] = cur.rowcount

        return counts


# --- row converters ----------------------------------------------------------


def _to_entity(row: sqlite3.Row) -> EntityRow:
    return EntityRow(
        id=row["id"],
        type=row["type"],
        canonical_name=row["canonical_name"],
        aliases=json.loads(row["aliases_json"]),
        created_ts=row["created_ts"],
        last_seen_ts=row["last_seen_ts"],
        forgotten=bool(row["forgotten"]),
        forgotten_ts=row["forgotten_ts"],
    )


def _to_claim(row: sqlite3.Row) -> ClaimRow:
    return ClaimRow(
        id=row["id"],
        subject_id=row["subject_id"],
        predicate=row["predicate"],
        object_text=row["object_text"],
        object_entity_id=row["object_entity_id"],
        asserted_ts=row["asserted_ts"],
        source_turn_id=row["source_turn_id"],
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        superseded_by=row["superseded_by"],
        omega=row["omega"],
        omega_origin=json.loads(row["omega_origin"]),
        forgotten=bool(row["forgotten"]),
    )
