"""Walk through Keelson's 5 hard scenarios using the SQLite store directly.

No API needed. Prints state before/after each scenario.

Run with: uv run python demo.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from datetime import datetime, timedelta, timezone

from keelson.compose import rule_based_compose
from keelson.retrieval import Retriever
from keelson.store import KeelsonStore
from keelson.surface import Surfacer


def hr(title: str = "") -> None:
    print(f"\n{'=' * 78}")
    if title:
        print(f"  {title}")
        print("=" * 78)


def show_live(store: KeelsonStore, subject: str = "user") -> None:
    claims = store.live_claims(subject=subject)
    if not claims:
        print(f"  (no live claims for {subject})")
        return
    for c in claims:
        marker = f"ω={c.omega:.2f}"
        print(f"  {subject} {c.predicate:20s} {c.object_text:35s} {marker}")


def show_history(store: KeelsonStore, subject: str, predicate: str) -> None:
    history = store.claim_history(subject, predicate)
    if not history:
        print(f"  (no history for {subject}.{predicate})")
        return
    for c in history:
        status = "LIVE  " if c.valid_to is None else f"PAST  "
        valid_to = c.valid_to or "—"
        print(f"  [{status}] {c.object_text:30s} valid_from={c.valid_from} valid_to={valid_to}")


def ask(store: KeelsonStore, query: str) -> None:
    """Run the full read pipeline: parse → retrieve → compose. Print result."""
    r = Retriever(store)
    result = r.query(query)
    response = rule_based_compose(result, store=store)
    print(f"\n  Q: {query}")
    print(f"  A: {response.text}")
    print(f"     [intent={result.spec.intent}, confidence={response.confidence}, "
          f"claims={len(response.cited_claims)}, turns={len(response.cited_turns)}]")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "demo.db"
        store = KeelsonStore(db_path)

        hr("SCENARIO 1: Direct claim → live retrieval")
        print("\nUser says: 'I work at Anthropic as a researcher.'")
        t1 = store.append_turn("s1", "user", "I work at Anthropic as a researcher.")
        store.write_claim("user", "works_at", "Anthropic", omega=0.85, source_turn_id=t1,
                          supersedes_hint="works_at")
        store.write_claim("user", "role", "researcher", omega=0.85, source_turn_id=t1,
                          supersedes_hint="role")
        print("\nLive claims:")
        show_live(store)

        hr("SCENARIO 2: Contradiction handling — Google → Anthropic")
        print("\nReset: User says 'I work at Google' on day 1, then 'I work at Anthropic' on day 2.")
        store2 = KeelsonStore(Path(tmp) / "demo2.db")
        t_a = store2.append_turn("s1", "user", "I work at Google.")
        store2.write_claim("user", "works_at", "Google", omega=0.85, source_turn_id=t_a,
                           supersedes_hint="works_at")
        t_b = store2.append_turn("s1", "user", "Quick update — I just signed with Anthropic.")
        store2.write_claim("user", "works_at", "Anthropic", omega=0.85, source_turn_id=t_b,
                           supersedes_hint="works_at")

        print("\nLive query — 'Where do I work?':")
        show_live(store2)
        print("\nHistory — 'Where did I work?':")
        show_history(store2, "user", "works_at")
        store2.close()

        hr("SCENARIO 3: Long-range fuzzy recall — climbing trip")
        print("\nUser mentioned a climbing trip in passing 2 years ago.")
        store3 = KeelsonStore(Path(tmp) / "demo3.db")
        store3.append_turn("s_old", "user", "Last weekend climbing in Yosemite was incredible.")
        # ... and lots of unrelated turns
        for content in [
            "I should learn more Spanish.",
            "Watched a great movie tonight.",
            "Anthropic's new model is wild.",
            "Made pasta from scratch finally.",
        ]:
            store3.append_turn("s_old", "user", content)

        # 2 years later:
        print("\nFTS5 search for 'climbing':")
        for hit in store3.search_turns("climbing"):
            print(f"  turn#{hit.id}: {hit.content!r}")
        store3.close()

        hr("SCENARIO 4: Forget Jamie cascade — preserves Italian-food independence")
        print("\nUser had a relationship with Jamie. Italian food came up via a date.")
        print("User also independently loves Italian food (mentioned solo many times).")
        store4 = KeelsonStore(Path(tmp) / "demo4.db")

        jamie = store4.resolve_or_create_entity("Jamie", entity_type="person")
        store4.write_claim("user", "has_relationship", "Jamie",
                           omega=1.0, object_entity="Jamie",
                           supersedes_hint="has_relationship")
        store4.write_claim("user", "ate_at", "Tony's Italian (with Jamie)",
                           omega=0.85, omega_origin=[jamie.id])
        # Independently-confirmed Italian preference (omega_origin empty)
        store4.write_claim("user", "loves", "Italian food",
                           omega=1.0, omega_origin=[])

        print("\nBefore forgetting Jamie:")
        show_live(store4)

        print("\n--- User: 'forget everything about Jamie' ---")
        counts = store4.forget_entity(jamie.id, cascade=True)
        print(f"  forgotten: entity={counts['entity']}, "
              f"direct claims={counts['direct']}, "
              f"omega-cascade claims={counts['omega_cascade']}")

        print("\nAfter forgetting Jamie:")
        show_live(store4)
        print("\n  ✓ user.loves='Italian food' survives (omega_origin was empty — independent of Jamie)")
        print("  ✓ user.ate_at='Tony's Italian (with Jamie)' is gone (omega flowed from Jamie)")
        store4.close()

        hr("SCENARIO 5: Scale — 1000 turns, fast recall")
        print("\nWriting 1000 turns + 100 claims, then querying...")
        store5 = KeelsonStore(Path(tmp) / "demo5.db")
        import time

        t0 = time.perf_counter()
        for i in range(1000):
            store5.append_turn("s_big", "user", f"Turn #{i}: lorem ipsum dolor sit amet, climbing trip detail.")
        for i in range(100):
            store5.write_claim("user", f"random_pred_{i % 10}", f"value_{i}", omega=0.5)
        t1_ = time.perf_counter()
        print(f"  wrote 1000 turns + 100 claims in {(t1_ - t0)*1000:.0f}ms")

        t0 = time.perf_counter()
        hits = store5.search_turns("climbing", limit=20)
        t1_ = time.perf_counter()
        print(f"  FTS5 search for 'climbing' across 1000 turns: {len(hits)} hits in {(t1_ - t0)*1000:.1f}ms")

        t0 = time.perf_counter()
        live = store5.live_claims(subject="user")
        t1_ = time.perf_counter()
        print(f"  live_claims for 'user': {len(live)} returned in {(t1_ - t0)*1000:.1f}ms")
        store5.close()

        hr("READ PATH: natural-language queries on a live memory store")
        print("\nBuilding a richer store with mixed-ω claims...")
        store6 = KeelsonStore(Path(tmp) / "demo6.db")
        # High ω: lived experience
        store6.write_claim("user", "works_at", "Anthropic", omega=0.95, supersedes_hint="works_at")
        store6.write_claim("user", "lives_in", "Brooklyn", omega=0.95, supersedes_hint="lives_in")
        store6.write_claim("user", "age", "34", omega=0.95, supersedes_hint="age")
        store6.write_claim("user", "loves", "jazz", omega=0.85)
        store6.write_claim("user", "loves", "Italian food", omega=0.90)
        store6.write_claim("user", "loves", "climbing", omega=0.85)
        # Medium ω: user-mentioned but not core identity
        store6.write_claim("user", "considers", "applying to grad school", omega=0.55)
        # Low ω: inferred from context
        store6.write_claim("user", "did", "marathon (inferred from leg pain mention)", omega=0.30)
        # Past claim (superseded by the live Anthropic row written above)
        # Add a related person:
        mom = store6.resolve_or_create_entity("user.mom", entity_type="person")
        store6.write_claim("user.mom", "did", "retired from teaching", omega=0.50)
        store6.write_claim("user.mom", "lives_in", "Vermont", omega=0.50)
        # Old turn for FTS fallback:
        store6.append_turn("s1", "user", "I climbed in Yosemite last September with a friend.")

        print("\n--- High-confidence direct lookups ---")
        ask(store6, "Where do I work?")
        ask(store6, "What is my age?")

        print("\n--- List queries (multiple values) ---")
        ask(store6, "What do I love?")

        print("\n--- Hedged speech for medium ω ---")
        ask(store6, "What do I consider?")

        print("\n--- Inference disclaimer for low ω ---")
        # Override omega_floor for this query so the inferred marathon claim is visible:
        r_low = Retriever(store6, omega_floor=0.0)
        result_low = r_low.query("What did I do?")
        response_low = rule_based_compose(result_low, store=store6)
        print(f"\n  Q: What did I do?")
        print(f"  A: {response_low.text}")
        print(f"     [confidence={response_low.confidence}]")

        print("\n--- History query (Google → Anthropic) ---")
        ask(store6, "Where did I used to work?")

        print("\n--- Relational entity (user.mom) ---")
        ask(store6, "Tell me about my mom")
        ask(store6, "What did you last tell me about my mom?")

        print("\n--- Fuzzy fallback to transcript ---")
        ask(store6, "Remember that climbing trip?")

        print("\n--- Empty result (nothing on file) ---")
        ask(store6, "What is my favorite color?")

        store6.close()

        hr("SPEAKS UP: proactive surfacing scanner")
        print("\nBuilding a store with stale relationships, a contradiction, and an anniversary...")
        store7 = KeelsonStore(Path(tmp) / "demo7.db")
        NOW = datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc)

        # Stale: an old friend we haven't heard about
        jamie = store7.resolve_or_create_entity("Jamie", entity_type="person")
        store7.conn.execute(
            "UPDATE entities SET last_seen_ts = ? WHERE id = ?",
            ((NOW - timedelta(days=120)).isoformat(timespec="seconds"), jamie.id),
        )
        store7.conn.commit()

        # Unresolved contradiction: two live works_at claims (simulated bug state)
        user_ent = store7.resolve_or_create_entity("user")
        for obj in ["Google", "Anthropic"]:
            store7.conn.execute(
                """
                INSERT INTO claims (subject_id, predicate, object_text, asserted_ts,
                                    valid_from, omega, omega_origin)
                VALUES (?, 'works_at', ?, ?, ?, 0.85, '[]')
                """,
                (user_ent.id, obj, NOW.isoformat(timespec="seconds"),
                 NOW.isoformat(timespec="seconds")),
            )
        store7.conn.commit()

        # Anniversary: turn from exactly 1 year ago
        one_year_ago = (NOW - timedelta(days=365)).isoformat(timespec="seconds")
        store7.append_turn("s_old", "user", "Got engaged today! Best day of my life.", ts=one_year_ago)

        # Drift candidate: an old high-ω claim never re-confirmed
        cid = store7.write_claim("user", "diet", "vegetarian", omega=1.0)
        store7.conn.execute(
            "UPDATE claims SET asserted_ts = ? WHERE id = ?",
            ((NOW - timedelta(days=730)).isoformat(timespec="seconds"), cid),
        )
        store7.conn.commit()

        # Emerging entity: 3 recent claims about someone new
        store7.resolve_or_create_entity("Alex", entity_type="person")
        store7.write_claim("user", "has_friend", "Alex", omega=0.85, object_entity="Alex")
        store7.write_claim("Alex", "works_at", "Stripe", omega=0.5)
        store7.write_claim("Alex", "lives_in", "SF", omega=0.5)

        surfacer = Surfacer(store7, now=NOW)
        candidates = surfacer.scan()

        print(f"\nScanner found {len(candidates)} thing(s) worth surfacing:\n")
        for c in candidates:
            print(f"  [{c.priority:.2f}] ({c.type})")
            print(f"         → {c.message}")
            print()

        print("--- After surfacing, mark them shown to start cooldown ---")
        for c in candidates:
            surfacer.mark_shown(c)
        repeated = surfacer.scan(cooldown_days=30)
        print(f"  Second scan (with 30-day cooldown): {len(repeated)} candidate(s)")

        store7.close()

        hr("Done.")
        print("\nKeelson's three primitives, demonstrated offline:")
        print("  1. Immutable turn log → FTS5 (porter-stemmed) makes long-range fuzzy recall fast.")
        print("  2. claims + supersession → contradictions never lose history.")
        print("  3. ω + omega_origin → selective forgetting is cascade-aware.")
        print("\nRead path:")
        print("  • Rule-based parser turns NL into structured QuerySpec (intent + subject + predicate)")
        print("  • Retriever queries the store; falls back to FTS when claims thin")
        print("  • Composer hedges by ω: ≥0.85 direct, ≥0.40 'you mentioned', <0.40 'I inferred'")
        print("  • Parser and composer are pluggable — drop in LLM-backed versions later.")
        print("\nSpeaks-up path (proactive surfacing):")
        print("  • 5 detectors run pure-SQL over the store; no API call needed")
        print("  • stale_relationship · anniversary · unresolved_contradiction ·")
        print("    drift_candidate · emerging_entity")
        print("  • Cooldowns + dismissal tracking via `surfaced` table prevent repeat nags")
        print()
        store.close()


if __name__ == "__main__":
    main()
