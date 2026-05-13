"""Proactive surfacing — find things worth telling the user, unprompted.

The Surfacer scans the store looking for state worth mentioning even when
the user hasn't asked. Five detectors:

  - stale_relationship: a person we used to talk about hasn't appeared in a while
  - anniversary:        a turn from N whole years ago today
  - unresolved_contradiction: multiple live claims for the same (subject, predicate)
  - drift_candidate:    a settled claim asserted long ago without re-confirmation
  - emerging_entity:    a newly-introduced entity with repeated mentions

Cooldowns prevent re-surfacing the same item. The user can mark items as
acknowledged or dismissed via mark_shown().

This is offline — pure SQL + heuristics. An LLM-backed Surfacer would compose
the messages more naturally, but the detection logic stays.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Literal

from keelson.store import ClaimRow, KeelsonStore, _now, _to_claim

SurfaceType = Literal[
    "stale_relationship",
    "anniversary",
    "unresolved_contradiction",
    "drift_candidate",
    "emerging_entity",
]


@dataclass(frozen=True)
class Surfaceable:
    """A candidate proactive surface — what the system might say, plus the
    evidence that triggered it.
    """
    type: SurfaceType
    target_key: str
    priority: float
    message: str
    entity_id: int | None = None
    claim_ids: tuple[int, ...] = ()
    turn_ids: tuple[int, ...] = ()
    evidence: dict = field(default_factory=dict)


def _parse_iso(ts: str) -> datetime:
    """Parse stored ISO timestamps. Treats naive timestamps as UTC."""
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class Surfacer:
    def __init__(self, store: KeelsonStore, now: datetime | None = None):
        self.store = store
        self.now = now or datetime.now(timezone.utc)

    # --- public API ----------------------------------------------------------

    def scan(self, cooldown_days: int = 30, top_k: int | None = None) -> list[Surfaceable]:
        """Run all detectors, apply cooldown, return candidates sorted by priority.

        Items already shown within `cooldown_days` are filtered out.
        """
        candidates: list[Surfaceable] = []
        candidates.extend(self.stale_relationships())
        candidates.extend(self.anniversaries())
        candidates.extend(self.unresolved_contradictions())
        candidates.extend(self.drift_candidates())
        candidates.extend(self.emerging_entities())

        candidates = self._apply_cooldown(candidates, cooldown_days)
        candidates.sort(key=lambda c: c.priority, reverse=True)
        if top_k is not None:
            candidates = candidates[:top_k]
        return candidates

    def mark_shown(self, candidate: Surfaceable, response: str | None = None) -> None:
        """Record that we surfaced this item. response can be 'acknowledged',
        'dismissed', or None (just shown).
        """
        with self.store.transaction() as conn:
            conn.execute(
                """
                INSERT INTO surfaced (surface_type, target_key, surfaced_at, user_response)
                VALUES (?, ?, ?, ?)
                """,
                (candidate.type, candidate.target_key, _now(), response),
            )

    # --- detectors -----------------------------------------------------------

    def stale_relationships(self, threshold_days: int = 60) -> list[Surfaceable]:
        """People (or related-person entities like user.mom) we used to talk about
        but haven't touched in `threshold_days`."""
        cutoff = (self.now - timedelta(days=threshold_days)).isoformat(timespec="seconds")
        rows = self.store.conn.execute(
            """
            SELECT * FROM entities
            WHERE forgotten = 0
              AND last_seen_ts < ?
              AND (type = 'person' OR canonical_name LIKE 'user.%')
              AND canonical_name != 'user'
            """,
            (cutoff,),
        ).fetchall()

        out: list[Surfaceable] = []
        for row in rows:
            days = (self.now - _parse_iso(row["last_seen_ts"])).days
            name = row["canonical_name"]
            display = name.split(".", 1)[1] if name.startswith("user.") else name
            priority = min(0.7, 0.3 + days / 365.0)  # caps at ~0.7
            out.append(Surfaceable(
                type="stale_relationship",
                target_key=f"entity/{row['id']}",
                priority=priority,
                message=f"Haven't heard about {display} in {days} days — everything ok?",
                entity_id=row["id"],
                evidence={"days_since_last_seen": days, "name": name},
            ))
        return out

    def anniversaries(self, max_years_back: int = 10) -> list[Surfaceable]:
        """Turns from exactly N whole years ago today."""
        out: list[Surfaceable] = []
        today = self.now.date()
        for years_back in range(1, max_years_back + 1):
            target = self._date_n_years_back(today, years_back)
            start = datetime(target.year, target.month, target.day, tzinfo=timezone.utc)
            end = start + timedelta(days=1)
            rows = self.store.conn.execute(
                """
                SELECT id, content, ts FROM turns
                WHERE ts >= ? AND ts < ? AND role = 'user'
                ORDER BY ts ASC
                """,
                (start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")),
            ).fetchall()
            for row in rows:
                year_word = "year" if years_back == 1 else "years"
                snippet = row["content"][:120]
                out.append(Surfaceable(
                    type="anniversary",
                    target_key=f"turn/{row['id']}/{years_back}y",
                    priority=0.3 + 0.05 * min(years_back, 5),
                    message=f"{years_back} {year_word} ago today you said: \"{snippet}\"",
                    turn_ids=(row["id"],),
                    evidence={"years_back": years_back, "anniversary_date": str(target)},
                ))
        return out

    def unresolved_contradictions(self) -> list[Surfaceable]:
        """Multiple live claims for the same (subject_id, predicate). With proper
        supersession this shouldn't happen, but accumulation bugs make it real."""
        rows = self.store.conn.execute(
            """
            SELECT subject_id, predicate, COUNT(*) AS n,
                   GROUP_CONCAT(id) AS claim_ids,
                   GROUP_CONCAT(object_text, ' || ') AS objects
            FROM claims
            WHERE valid_to IS NULL AND forgotten = 0
            GROUP BY subject_id, predicate
            HAVING COUNT(*) > 1
            """
        ).fetchall()

        out: list[Surfaceable] = []
        for row in rows:
            ent = self.store.get_entity(row["subject_id"])
            subject = ent.canonical_name if ent else f"entity#{row['subject_id']}"
            ids = tuple(int(x) for x in row["claim_ids"].split(","))
            out.append(Surfaceable(
                type="unresolved_contradiction",
                target_key=f"{subject}/{row['predicate']}",
                priority=1.0,
                message=(
                    f"I'm holding {row['n']} versions of {subject}'s {row['predicate']}: "
                    f"{row['objects']}. Which is current?"
                ),
                claim_ids=ids,
                entity_id=row["subject_id"],
                evidence={"count": row["n"], "objects": row["objects"]},
            ))
        return out

    def drift_candidates(self, age_days: int = 180, omega_min: float = 0.85) -> list[Surfaceable]:
        """A high-ω claim asserted long ago that has never been re-asserted.

        "You used to say you're vegetarian (~2 years ago). Still?"
        """
        cutoff = (self.now - timedelta(days=age_days)).isoformat(timespec="seconds")
        rows = self.store.conn.execute(
            """
            SELECT * FROM claims
            WHERE valid_to IS NULL
              AND forgotten = 0
              AND omega >= ?
              AND asserted_ts < ?
            ORDER BY asserted_ts ASC
            """,
            (omega_min, cutoff),
        ).fetchall()

        out: list[Surfaceable] = []
        for row in rows:
            claim = _to_claim(row)
            # Skip if there's a newer assertion for the same (subject, predicate)
            newer = self.store.conn.execute(
                """
                SELECT 1 FROM claims
                WHERE subject_id = ? AND predicate = ?
                  AND asserted_ts > ? AND forgotten = 0
                LIMIT 1
                """,
                (claim.subject_id, claim.predicate, claim.asserted_ts),
            ).fetchone()
            if newer:
                continue

            days = (self.now - _parse_iso(claim.asserted_ts)).days
            ent = self.store.get_entity(claim.subject_id)
            subject = ent.canonical_name if ent else "you"
            display_subject = "you" if subject == "user" else subject
            verb = claim.predicate.replace("_", " ")
            out.append(Surfaceable(
                type="drift_candidate",
                target_key=f"claim/{claim.id}",
                priority=0.6,
                message=(
                    f"You said {display_subject} {verb} {claim.object_text} "
                    f"about {days} days ago. Still true?"
                ),
                claim_ids=(claim.id,),
                entity_id=claim.subject_id,
                evidence={"age_days": days, "omega": claim.omega},
            ))
        return out

    def emerging_entities(self, recency_days: int = 14, min_mentions: int = 3) -> list[Surfaceable]:
        """Entities created recently with multiple claims attached — likely
        something the user wants to declare explicitly."""
        cutoff = (self.now - timedelta(days=recency_days)).isoformat(timespec="seconds")
        rows = self.store.conn.execute(
            """
            SELECT e.id, e.canonical_name, e.type, e.created_ts,
                   COUNT(c.id) AS mention_count
            FROM entities e
            LEFT JOIN claims c
              ON c.forgotten = 0
              AND (c.subject_id = e.id OR c.object_entity_id = e.id)
            WHERE e.forgotten = 0
              AND e.created_ts >= ?
              AND e.canonical_name != 'user'
              AND e.canonical_name NOT LIKE 'user.%'
            GROUP BY e.id
            HAVING mention_count >= ?
            """,
            (cutoff, min_mentions),
        ).fetchall()

        out: list[Surfaceable] = []
        for row in rows:
            display = row["canonical_name"].title()
            out.append(Surfaceable(
                type="emerging_entity",
                target_key=f"entity/{row['id']}",
                priority=0.7,
                message=(
                    f"You've mentioned {display} {row['mention_count']} times "
                    f"recently. Should I track who/what they are?"
                ),
                entity_id=row["id"],
                evidence={"mention_count": row["mention_count"], "name": row["canonical_name"]},
            ))
        return out

    # --- internals -----------------------------------------------------------

    def _apply_cooldown(self, candidates: list[Surfaceable], days: int) -> list[Surfaceable]:
        if not candidates:
            return []
        cutoff = (self.now - timedelta(days=days)).isoformat(timespec="seconds")
        rows = self.store.conn.execute(
            f"""
            SELECT surface_type, target_key, MAX(surfaced_at) AS last_shown,
                   GROUP_CONCAT(COALESCE(user_response, ''), '|') AS responses
            FROM surfaced
            WHERE surfaced_at >= ?
            GROUP BY surface_type, target_key
            """,
            (cutoff,),
        ).fetchall()
        suppressed: set[tuple[str, str]] = set()
        for r in rows:
            responses = (r["responses"] or "").split("|")
            # If the user explicitly dismissed, suppress for a long time
            # (we already filtered by cutoff so if dismissed appears here it's recent)
            if "dismissed" in responses:
                suppressed.add((r["surface_type"], r["target_key"]))
                continue
            # Otherwise just suppress until cooldown elapses
            suppressed.add((r["surface_type"], r["target_key"]))
        return [c for c in candidates if (c.type, c.target_key) not in suppressed]

    @staticmethod
    def _date_n_years_back(today: date, years: int) -> date:
        try:
            return today.replace(year=today.year - years)
        except ValueError:
            # Feb 29 on a non-leap year — fall back to Feb 28
            return today.replace(year=today.year - years, day=28)
