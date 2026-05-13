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

## Install as an MCP server

Keelson ships an MCP server so any tool that speaks Model Context Protocol —
Claude Code, Cursor, ChatGPT desktop, etc. — can read/write memory through it.

### Quick start (Claude Code)

```bash
claude mcp add keelson uv --directory /Users/brianlo/keelson run keelson-mcp
```

Or edit your MCP config (`~/.config/claude/mcp.json` or equivalent) directly:

```json
{
  "mcpServers": {
    "keelson": {
      "command": "uv",
      "args": ["--directory", "/Users/brianlo/keelson", "run", "keelson-mcp"]
    }
  }
}
```

### What it exposes

Seven tools, all backed by the same SQLite store at `~/.keelson/memory.db`
(override with `KEELSON_DB_PATH`):

| Tool | Purpose |
|---|---|
| `ingest_turn(role, content)` | Append to the immutable transcript |
| `write_claim(subject, predicate, object, omega, ...)` | Write a structured fact, with supersession |
| `recall(query)` | Natural-language Q → ω-hedged answer |
| `claim_history(subject, predicate)` | Live + superseded values |
| `list_live_claims(subject?, predicate?)` | Inspect current state |
| `forget_entity(name, cascade=True)` | Scrub with ω-cascade |
| `surface_scan(top_k=5)` | Proactive: contradictions, anniversaries, drift, stale, emerging |

### Run standalone

```bash
uv run keelson-mcp                    # stdio transport
KEELSON_DB_PATH=./test.db uv run keelson-mcp   # custom db
```
