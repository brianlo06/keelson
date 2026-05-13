"""CLI for running the read-path eval.

Each example sets up a fresh in-memory store via SetupActions, then runs
Retriever + Composer on the natural-language query and scores the response.

Uses llm_compose when ANTHROPIC_API_KEY is set, else rule_based_compose.

    uv run python -m eval.run_read              # all read categories
    uv run python -m eval.run_read --category direct_lookup
    uv run python -m eval.run_read --verbose    # print response text
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from keelson.compose import llm_compose, rule_based_compose
from keelson.retrieval import Retriever
from keelson.store import KeelsonStore
from eval.read_score import ReadScore, aggregate_read, score_read
from eval.read_types import ReadExample, SetupAction

DATA_DIR = Path(__file__).parent / "data_read"


def load_examples(category: str | None = None) -> list[ReadExample]:
    files = sorted(DATA_DIR.glob("*.jsonl"))
    if category:
        files = [f for f in files if f.stem == category]
        if not files:
            available = sorted(f.stem for f in DATA_DIR.glob("*.jsonl"))
            sys.exit(f"category {category!r} not found. available: {', '.join(available)}")

    examples = []
    for f in files:
        for line_no, line in enumerate(f.read_text().splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                examples.append(ReadExample.from_json(json.loads(line)))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"warning: skipping {f.name}:{line_no}: {e}", file=sys.stderr)
    return examples


def apply_setup(store: KeelsonStore, actions: tuple[SetupAction, ...]) -> None:
    for a in actions:
        if a.type == "claim":
            store.write_claim(
                subject=a.subject,
                predicate=a.predicate,
                object=a.object,
                omega=a.omega,
                supersedes_hint=a.supersedes_hint,
                object_entity=a.object_entity,
            )
        elif a.type == "turn":
            store.append_turn("eval", a.role or "user", a.content or "")
        else:
            raise ValueError(f"unknown setup action type: {a.type!r}")


def print_score(score: ReadScore, example: ReadExample, verbose: bool) -> None:
    if score.forbidden_hit:
        pass_fail = "FAIL"
    elif score.composite >= 0.8:
        pass_fail = "PASS"
    else:
        pass_fail = "WARN"
    print(f"  [{pass_fail}] {score.example_id:32s} composite={score.composite:.2f}  "
          f"mention={score.mention_score:.2f}  conf={'Y' if score.confidence_match else 'N'}  "
          f"cite={'Y' if score.citation_ok else 'N'}")
    if verbose:
        print(f"       query: {example.query!r}")
        print(f"       response: {score.response_text[:200]!r}")
        for failure in score.failures:
            print(f"       fail: {failure}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Keelson read-path eval suite")
    parser.add_argument("--category", help="Run only one category")
    parser.add_argument("--limit", type=int, help="Run at most N examples per category")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-example detail")
    args = parser.parse_args()

    load_dotenv()
    composer = llm_compose if anthropic.Anthropic().api_key else rule_based_compose

    examples = load_examples(args.category)
    if not examples:
        sys.exit("no read-eval examples found in eval/data_read/")

    if args.limit:
        by_cat: dict[str, list[ReadExample]] = defaultdict(list)
        for ex in examples:
            by_cat[ex.category].append(ex)
        examples = [ex for cat in by_cat.values() for ex in cat[: args.limit]]

    print(f"running {len(examples)} read example(s)...\n")

    scores_by_cat: dict[str, list[ReadScore]] = defaultdict(list)
    current_cat = None

    with tempfile.TemporaryDirectory() as tmp:
        for i, ex in enumerate(examples):
            if ex.category != current_cat:
                current_cat = ex.category
                print(f"\n=== {ex.category} ===")

            # Fresh store per example — isolation matters since setups vary
            db_path = Path(tmp) / f"{ex.id}.db"
            store = KeelsonStore(db_path)
            try:
                apply_setup(store, ex.setup)
                result = Retriever(store).query(ex.query)
                response = composer(result, store=store)
                score = score_read(ex, response)
                scores_by_cat[ex.category].append(score)
                print_score(score, ex, args.verbose)
            finally:
                store.close()

    # Summary
    print("\n\n=== SUMMARY (read-path) ===")
    print(f"{'category':<30s} {'n':>4s} {'composite':>10s} {'mention':>8s} {'conf':>6s} {'cite':>6s}")
    all_scores: list[ReadScore] = []
    for cat, scores in sorted(scores_by_cat.items()):
        agg = aggregate_read(scores)
        print(f"{cat:<30s} {int(agg['n']):>4d} {agg['composite']:>10.2f} "
              f"{agg['mention']:>8.2f} {agg['confidence']:>6.2f} {agg['citation']:>6.2f}")
        all_scores.extend(scores)

    total = aggregate_read(all_scores)
    print("-" * 80)
    print(f"{'TOTAL':<30s} {int(total['n']):>4d} {total['composite']:>10.2f} "
          f"{total['mention']:>8.2f} {total['confidence']:>6.2f} {total['citation']:>6.2f}")
    if total["forbidden_hits"]:
        print(f"\n*** {int(total['forbidden_hits'])} example(s) hit forbidden mentions ***")

    return 0 if total["composite"] >= 0.7 else 1


if __name__ == "__main__":
    sys.exit(main())
