# Keelson

[![Tests](https://github.com/brianlo06/keelson/actions/workflows/test.yml/badge.svg)](https://github.com/brianlo06/keelson/actions/workflows/test.yml)

Personal LLM memory architecture — three primitives, no decoration.

1. **Immutable JSONL transcript** — append-only, never edited
2. **`claims` table with supersession** — `valid_from`/`valid_to`, never delete
3. **One provenance scalar `ω ∈ [0,1]`** — modulates decay, retrieval, speech style

## Quick start

```bash
cp .env.example .env
# add your ANTHROPIC_API_KEY

uv sync
uv run python -m eval.run                  # run full eval suite
uv run python -m eval.run --category direct_claim   # one category
uv run python -m eval.run --verbose        # print per-example results
```

## Layout

```
src/keelson/
  extractor.py        # the claim extraction call
  schema.py           # claim dataclasses
  prompts/
    extract.md        # the extraction prompt (versioned)

eval/
  data/               # JSONL examples, one file per category
  run.py              # CLI entry point
  score.py            # scoring algorithm
  judge.py            # LLM-as-judge for semantic similarity
  types.py            # eval data types

tests/                # unit tests for scoring
```

## Iterating on the extraction prompt

1. Edit `src/keelson/prompts/extract.md`
2. Run `uv run python -m eval.run --verbose`
3. Find failures, hand-label corrections into `eval/data/regression.jsonl`
4. Repeat

The `regression.jsonl` category grows as you discover real bugs. Never delete entries from it.
