"""keelson-chat — an interactive REPL backed by the same SQLite store as the MCP server.

Each user input is dispatched by simple heuristics:
  - Ends in '?'  → query (recall → compose, no storage)
  - Starts with ':' → command (see :help)
  - Otherwise → statement (ingest + extract if API key is set)

Persistence: ~/.keelson/memory.db, override with KEELSON_DB_PATH.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from keelson.compose import llm_compose
from keelson.retrieval import Retriever
from keelson.store import KeelsonStore
from keelson.surface import Surfacer

SESSION = "chat"
HELP = """\
Commands:
  :surface              Run the proactive scanner; prints top candidates
  :history SUBJ PRED    Show full history for a (subject, predicate)
  :live [SUBJ]          List live claims (optionally for one subject)
  :forget NAME          Forget an entity with full cascade
  :help                 Show this help
  :quit                 Exit

Otherwise:
  • Lines ending in '?' are treated as questions (recall, do not store)
  • All other lines are stored as user turns (and extracted if API key set)
"""


def _resolve_db_path() -> Path:
    env = os.environ.get("KEELSON_DB_PATH")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".keelson" / "memory.db"


def _make_extractor():
    """Lazy import of Extractor — only when API key is set, so the CLI can run
    completely offline."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    from keelson.extractor import Extractor
    return Extractor()


def _handle_command(line: str, store: KeelsonStore) -> bool:
    """Run a colon-prefixed command. Returns True to continue, False to exit."""
    parts = line.split()
    cmd = parts[0]

    if cmd == ":quit" or cmd == ":exit":
        return False

    if cmd == ":help":
        print(HELP)
        return True

    if cmd == ":surface":
        surfacer = Surfacer(store)
        cands = surfacer.scan(top_k=5)
        if not cands:
            print("  (nothing to surface)")
            return True
        for c in cands:
            print(f"  [{c.priority:.2f}] ({c.type})\n    {c.message}")
            surfacer.mark_shown(c)
        return True

    if cmd == ":live":
        subject = parts[1] if len(parts) > 1 else None
        claims = store.live_claims(subject=subject)
        if not claims:
            print(f"  (no live claims{' for ' + subject if subject else ''})")
            return True
        for c in claims:
            ent = store.get_entity(c.subject_id)
            s = ent.canonical_name if ent else f"entity#{c.subject_id}"
            print(f"  {s} {c.predicate}: {c.object_text}  (ω={c.omega:.2f})")
        return True

    if cmd == ":history":
        if len(parts) < 3:
            print("  Usage: :history SUBJECT PREDICATE")
            return True
        _, subj, pred = parts[0], parts[1], parts[2]
        history = store.claim_history(subj, pred)
        if not history:
            print(f"  (no history for {subj}.{pred})")
            return True
        for c in history:
            status = "LIVE" if c.valid_to is None else "PAST"
            print(f"  [{status}] {c.object_text}  ({c.valid_from[:10]} → "
                  f"{c.valid_to[:10] if c.valid_to else '—'})")
        return True

    if cmd == ":forget":
        if len(parts) < 2:
            print("  Usage: :forget ENTITY_NAME")
            return True
        name = " ".join(parts[1:])
        ent = store.find_entity(name)
        if ent is None:
            print(f"  (no entity named {name!r})")
            return True
        counts = store.forget_entity(ent.id, cascade=True)
        print(f"  Forgot {name!r}: entity={counts['entity']}, "
              f"direct={counts['direct']}, cascade={counts['omega_cascade']}")
        return True

    print(f"  Unknown command: {cmd!r}. Type :help for the list.")
    return True


def _handle_question(line: str, store: KeelsonStore) -> None:
    result = Retriever(store).query(line)
    response = llm_compose(result, store=store)
    print(f"keelson> {response.text}")


def _handle_statement(line: str, store: KeelsonStore, extractor) -> None:
    tid = store.append_turn(SESSION, "user", line)
    if extractor is None:
        print(f"  [+ turn #{tid}; no extractor — set ANTHROPIC_API_KEY to extract claims]")
        return
    try:
        result = extractor.extract(line)
    except Exception as e:
        print(f"  [+ turn #{tid}; extractor errored: {type(e).__name__}: {e}]")
        return
    if result.extraction_failed:
        print(f"  [+ turn #{tid}; extraction failed: {result.failure_reason}]")
        return
    n = 0
    for claim in result.claims:
        store.write_claim(
            subject=claim.subject,
            predicate=claim.predicate,
            object=claim.object,
            omega=claim.omega,
            source_turn_id=tid,
            supersedes_hint=claim.supersedes_hint,
        )
        n += 1
    print(f"  [+ turn #{tid}, {n} claim{'s' if n != 1 else ''}]")


def main() -> int:
    db_path = _resolve_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = KeelsonStore(db_path)
    extractor = _make_extractor()

    api_on = extractor is not None
    print(f"Keelson chat — db={db_path}, api={'on' if api_on else 'off'}")
    print("Type :help for commands, :quit to exit.\n")

    try:
        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not line:
                continue

            if line.startswith(":"):
                if not _handle_command(line, store):
                    break
                continue

            if line.endswith("?"):
                _handle_question(line, store)
            else:
                _handle_statement(line, store, extractor)
    finally:
        store.close()

    print("Bye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
