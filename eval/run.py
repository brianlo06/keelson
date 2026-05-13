"""CLI for running the eval suite.

Usage:
    uv run python -m eval.run                       # run all categories
    uv run python -m eval.run --category direct_claim
    uv run python -m eval.run --verbose             # print per-example detail
    uv run python -m eval.run --limit 5             # run first 5 examples per file
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from keelson.extractor import Extractor
from eval.holistic import HolisticJudge, HolisticVerdict
from eval.judge import Judge
from eval.score import ExampleScore, aggregate, score_example
from eval.types import EvalExample

DATA_DIR = Path(__file__).parent / "data"


def load_examples(category: str | None = None) -> list[EvalExample]:
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
                examples.append(EvalExample.from_json(json.loads(line)))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"warning: skipping {f.name}:{line_no}: {e}", file=sys.stderr)
    return examples


def print_score(score: ExampleScore, example: EvalExample, verbose: bool) -> None:
    pass_fail = "PASS" if score.composite >= 0.8 and not score.forbidden_hit else "FAIL"
    print(f"  [{pass_fail}] {score.example_id:32s} composite={score.composite:.2f}  "
          f"P={score.precision:.2f} R={score.recall:.2f} ω={score.omega_calibration:.2f}")
    if verbose:
        print(f"       user: {example.user_message!r}")
        if score.failures:
            for failure in score.failures:
                print(f"       fail: {failure}")
        for m in score.matches:
            if m.extracted is None:
                print(f"       miss: expected {m.expected.subject} {m.expected.predicate} {m.expected.object_pattern!r}")
            elif m.verdict and m.verdict.overall < 1.0:
                print(f"       partial: matched {m.expected.predicate} -> {m.extracted.predicate} ({m.verdict.overall:.2f})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Keelson extraction eval suite")
    parser.add_argument("--category", help="Run only one category (e.g. direct_claim)")
    parser.add_argument("--limit", type=int, help="Run at most N examples per category")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-example detail")
    parser.add_argument(
        "--holistic",
        action="store_true",
        help="Also run the holistic LLM-as-judge evaluator. Adds ~1 Claude call per example.",
    )
    args = parser.parse_args()

    load_dotenv()
    client = anthropic.Anthropic()
    extractor = Extractor(client=client)
    judge = Judge(client=client)
    holistic = HolisticJudge(client=client) if args.holistic else None

    examples = load_examples(args.category)
    if not examples:
        sys.exit("no examples found in eval/data/")

    if args.limit:
        by_cat: dict[str, list[EvalExample]] = defaultdict(list)
        for ex in examples:
            by_cat[ex.category].append(ex)
        examples = [ex for cat in by_cat.values() for ex in cat[: args.limit]]

    print(f"running {len(examples)} example(s)...\n")

    scores_by_cat: dict[str, list[ExampleScore]] = defaultdict(list)
    holistic_by_cat: dict[str, list[HolisticVerdict]] = defaultdict(list)
    current_cat = None

    for ex in examples:
        if ex.category != current_cat:
            current_cat = ex.category
            print(f"\n=== {ex.category} ===")

        result = extractor.extract(ex.user_message)
        if result.extraction_failed:
            print(f"  [FAIL] {ex.id:32s} extraction failed: {result.failure_reason}")
            scores_by_cat[ex.category].append(ExampleScore(
                example_id=ex.id, category=ex.category,
                recall=0.0, precision=0.0, omega_calibration=0.0,
                predicate_correctness=0.0, composite=0.0, forbidden_hit=False,
                failures=[f"extraction failed: {result.failure_reason}"],
            ))
            continue

        score = score_example(ex, result.claims, judge)
        scores_by_cat[ex.category].append(score)
        print_score(score, ex, args.verbose)

        if holistic is not None:
            verdict = holistic.judge(ex, result.claims)
            holistic_by_cat[ex.category].append(verdict)
            if args.verbose:
                print(f"       holistic: cov={verdict.coverage:.2f} prec={verdict.precision:.2f} "
                      f"ω={verdict.omega:.2f} use={verdict.usefulness:.2f}  "
                      f"({verdict.rationale[:80]})")

    # Summary
    print("\n\n=== SUMMARY ===")
    print(f"{'category':<30s} {'n':>4s} {'composite':>10s} {'precision':>10s} {'recall':>8s} {'ω-cal':>8s}")
    all_scores: list[ExampleScore] = []
    for cat, scores in sorted(scores_by_cat.items()):
        agg = aggregate(scores)
        print(f"{cat:<30s} {int(agg['n']):>4d} {agg['composite']:>10.2f} "
              f"{agg['precision']:>10.2f} {agg['recall']:>8.2f} {agg['omega_calibration']:>8.2f}")
        all_scores.extend(scores)

    total = aggregate(all_scores)
    print("-" * 80)
    print(f"{'TOTAL':<30s} {int(total['n']):>4d} {total['composite']:>10.2f} "
          f"{total['precision']:>10.2f} {total['recall']:>8.2f} {total['omega_calibration']:>8.2f}")
    if total["forbidden_hits"]:
        print(f"\n*** {int(total['forbidden_hits'])} example(s) emitted FORBIDDEN claims (precision floor hit) ***")

    if holistic is not None:
        print(f"\n=== HOLISTIC (LLM-as-judge) ===")
        print(f"{'category':<30s} {'n':>4s} {'overall':>8s} {'cov':>6s} {'prec':>6s} "
              f"{'omega':>6s} {'use':>6s}")
        all_holistic: list[HolisticVerdict] = []
        for cat in sorted(holistic_by_cat.keys()):
            vs = holistic_by_cat[cat]
            n = len(vs)
            if not n:
                continue
            mean_overall = sum(v.overall for v in vs) / n
            mean_cov = sum(v.coverage for v in vs) / n
            mean_prec = sum(v.precision for v in vs) / n
            mean_omega = sum(v.omega for v in vs) / n
            mean_use = sum(v.usefulness for v in vs) / n
            print(f"{cat:<30s} {n:>4d} {mean_overall:>8.2f} {mean_cov:>6.2f} {mean_prec:>6.2f} "
                  f"{mean_omega:>6.2f} {mean_use:>6.2f}")
            all_holistic.extend(vs)
        if all_holistic:
            n = len(all_holistic)
            mean_overall = sum(v.overall for v in all_holistic) / n
            mean_cov = sum(v.coverage for v in all_holistic) / n
            mean_prec = sum(v.precision for v in all_holistic) / n
            mean_omega = sum(v.omega for v in all_holistic) / n
            mean_use = sum(v.usefulness for v in all_holistic) / n
            print("-" * 80)
            print(f"{'TOTAL':<30s} {n:>4d} {mean_overall:>8.2f} {mean_cov:>6.2f} {mean_prec:>6.2f} "
                  f"{mean_omega:>6.2f} {mean_use:>6.2f}")

    return 0 if total["composite"] >= 0.7 else 1


if __name__ == "__main__":
    sys.exit(main())
