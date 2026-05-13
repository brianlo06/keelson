"""MCP server exposing Keelson's memory primitives as tools.

Any MCP-speaking client (Claude Code, Cursor, ChatGPT desktop, etc.) can
connect to this server and read/write personal memory through the same
SQLite store that the offline demo uses.

Persistence:
  - Database lives at $KEELSON_DB_PATH, or ~/.keelson/memory.db by default
  - Stays alive across server restarts; data accumulates over time

Run via:
  uv run keelson-mcp                       # stdio transport (default)

To register with Claude Code:
  See README.md → "Install as an MCP server"
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from keelson.compose import rule_based_compose
from keelson.retrieval import Retriever
from keelson.store import KeelsonStore
from keelson.surface import Surfacer

DEFAULT_DB_PATH = Path.home() / ".keelson" / "memory.db"

mcp = FastMCP("keelson")

_store: KeelsonStore | None = None


def get_db_path() -> Path:
    env = os.environ.get("KEELSON_DB_PATH")
    if env:
        return Path(env).expanduser()
    return DEFAULT_DB_PATH


def get_store() -> KeelsonStore:
    """Lazy singleton — opens the DB on first use and reuses it."""
    global _store
    if _store is None:
        path = get_db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        _store = KeelsonStore(path)
    return _store


def _set_store_for_testing(store: KeelsonStore | None) -> None:
    """Test hook — inject a store and bypass the default path."""
    global _store
    _store = store


# --- tools -------------------------------------------------------------------


@mcp.tool()
def ingest_turn(role: str, content: str, session_id: str = "default") -> str:
    """Append a turn to the immutable transcript.

    role must be 'user', 'assistant', or 'system'. The transcript is append-only;
    nothing in this function ever modifies an existing turn.

    Args:
        role: who said it (user / assistant / system)
        content: verbatim text of the turn
        session_id: optional grouping key. Defaults to "default".

    Returns:
        Confirmation with the new turn id.
    """
    if role not in ("user", "assistant", "system"):
        return f"Error: role must be 'user', 'assistant', or 'system'; got {role!r}"
    tid = get_store().append_turn(session_id, role, content)
    return f"Appended turn #{tid} ({role}, session={session_id!r}, {len(content)} chars)"


@mcp.tool()
def write_claim(
    subject: str,
    predicate: str,
    object: str,
    omega: float = 1.0,
    supersedes_hint: str | None = None,
    object_entity: str | None = None,
    omega_origin: list[int] | None = None,
) -> str:
    """Record a structured fact about an entity.

    The user is referred to as 'user'. Related people use 'user.<role>'
    (e.g. 'user.mom', 'user.partner'). Other people use proper names.

    omega is the provenance scalar [0..1]:
      1.00  user lived it ("I went to Tokyo")
      0.85  stated identity ("I'm a data scientist")
      0.50  reported about someone else ("My sister got married")
      0.25  inferred from context (low confidence)
      0.10  speculation / hypothetical

    supersedes_hint: if this claim updates a prior live claim for the same
      (subject, predicate), pass the predicate name here. The old claim will
      be marked superseded (valid_to set + linked via superseded_by). The
      history is preserved.

    object_entity: if the object is a person/place/thing that should be linked
      as an entity (so cascade forgetting works), pass the name here.

    omega_origin: list of entity IDs the omega "flowed through" — claims
      acquired via a specific entity. If that entity is later forgotten, this
      claim's omega cascades and the claim is scrubbed. Use [] for independent.
    """
    new_id = get_store().write_claim(
        subject=subject,
        predicate=predicate,
        object=object,
        omega=omega,
        supersedes_hint=supersedes_hint,
        object_entity=object_entity,
        omega_origin=omega_origin,
    )
    note = f" (supersedes prior live {supersedes_hint!r})" if supersedes_hint else ""
    return f"Wrote claim #{new_id}{note}"


@mcp.tool()
def recall(query: str) -> str:
    """Answer a natural-language question by looking up structured claims and
    falling back to transcript search if needed. omega-aware speech:
      omega>=0.85 direct, omega>=0.40 hedged, omega<0.40 flagged as inference.

    Examples:
      "Where do I work?"
      "Tell me about my mom"
      "What did I tell you about that climbing trip?"
    """
    store = get_store()
    result = Retriever(store).query(query)
    response = rule_based_compose(result, store=store)
    if not response.text:
        return "I don't have anything on that."
    return response.text


@mcp.tool()
def claim_history(subject: str, predicate: str) -> str:
    """Show the full history of a (subject, predicate), including superseded values.

    Use this for questions like 'where did I used to work?' or 'what have I
    believed about X over time?'
    """
    history = get_store().claim_history(subject, predicate)
    if not history:
        return f"No history for {subject}.{predicate}"
    lines = [f"History for {subject}.{predicate}:"]
    for c in history:
        status = "LIVE" if c.valid_to is None else "PAST"
        valid_to = c.valid_to or "—"
        lines.append(
            f"  [{status}] {c.object_text!r}  "
            f"from {c.valid_from[:10]} to {valid_to[:10] if valid_to != '—' else '—'}  "
            f"omega={c.omega:.2f}"
        )
    return "\n".join(lines)


@mcp.tool()
def list_live_claims(
    subject: str | None = None,
    predicate: str | None = None,
    omega_min: float = 0.0,
) -> str:
    """Inspect currently-live claims (not superseded, not forgotten).

    Filter by subject and/or predicate. omega_min filters out low-confidence
    claims. With no args, lists all live claims above omega_min.
    """
    store = get_store()
    claims = store.live_claims(subject=subject, predicate=predicate, omega_min=omega_min)
    if not claims:
        filt = f"{subject or '*'}.{predicate or '*'}"
        return f"No live claims for {filt}."
    lines = []
    for c in claims:
        ent = store.get_entity(c.subject_id)
        s_name = ent.canonical_name if ent else f"entity#{c.subject_id}"
        lines.append(f"{s_name} {c.predicate}: {c.object_text}  (omega={c.omega:.2f})")
    return "\n".join(lines)


@mcp.tool()
def forget_entity(name: str, cascade: bool = True) -> str:
    """Forget an entity. With cascade=True (default), also scrubs:
      - claims where the entity is subject or object
      - claims whose omega flowed through this entity (omega_origin contains it)

    The transcript stays intact — only structured claims are scrubbed. This is
    by design: forgetting is a structural operation on what we've extracted, not
    a censorship of what was said.
    """
    store = get_store()
    ent = store.find_entity(name)
    if ent is None:
        return f"No entity named {name!r} found."
    counts = store.forget_entity(ent.id, cascade=cascade)
    return (
        f"Forgot {name!r}: entity={counts['entity']}, "
        f"direct claims={counts['direct']}, "
        f"omega-cascade claims={counts['omega_cascade']}"
    )


@mcp.tool()
def surface_scan(top_k: int = 5, mark_shown: bool = True) -> str:
    """Find things worth telling the user proactively. Detectors:
      - stale_relationship       (person not mentioned in a while)
      - anniversary              (turn from N years ago today)
      - unresolved_contradiction (multiple live claims for one predicate)
      - drift_candidate          (old high-omega claim never re-confirmed)
      - emerging_entity          (new entity with multiple recent mentions)

    Cooldowns prevent re-surfacing the same item within 30 days.
    Set mark_shown=False to peek without starting cooldowns.
    """
    surfacer = Surfacer(get_store())
    candidates = surfacer.scan(top_k=top_k)
    if not candidates:
        return "Nothing to surface right now."

    lines = []
    for c in candidates:
        lines.append(f"[{c.priority:.2f}] ({c.type})\n  {c.message}")
        if mark_shown:
            surfacer.mark_shown(c)
    return "\n\n".join(lines)


# --- entry point -------------------------------------------------------------


def main() -> None:
    """stdio transport — what Claude Code spawns when configured."""
    mcp.run()


if __name__ == "__main__":
    main()
