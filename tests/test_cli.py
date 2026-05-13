"""Smoke tests for the keelson-chat REPL.

We feed inputs via monkeypatched builtins.input and verify the right side
effects on the store. The CLI's pieces are well-tested elsewhere; these
guard against import errors and the dispatch logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from keelson.store import KeelsonStore


def _run_cli_with_inputs(monkeypatch, tmp_path, inputs: list[str]) -> KeelsonStore:
    """Run cli.main() with the given input sequence, then return a fresh store
    handle to the same DB so the caller can assert on it."""
    db = tmp_path / "chat.db"
    monkeypatch.setenv("KEELSON_DB_PATH", str(db))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    it = iter(inputs + [":quit"])
    monkeypatch.setattr("builtins.input", lambda *_a, **_kw: next(it))

    from keelson.cli import main
    rc = main()
    assert rc == 0

    return KeelsonStore(db)


def test_cli_ingests_statement(monkeypatch, tmp_path):
    store = _run_cli_with_inputs(monkeypatch, tmp_path, ["I work at Anthropic"])
    assert store.get_turn(1) is not None
    assert "Anthropic" in store.get_turn(1).content
    store.close()


def test_cli_question_does_not_get_stored(monkeypatch, tmp_path):
    """Questions are reads, not writes — no turn should be appended."""
    store = _run_cli_with_inputs(monkeypatch, tmp_path, ["Where do I work?"])
    assert store.get_turn(1) is None
    store.close()


def test_cli_help_command(monkeypatch, tmp_path, capsys):
    store = _run_cli_with_inputs(monkeypatch, tmp_path, [":help"])
    out = capsys.readouterr().out
    assert ":surface" in out
    assert ":quit" in out
    store.close()


def test_cli_live_command_empty(monkeypatch, tmp_path, capsys):
    store = _run_cli_with_inputs(monkeypatch, tmp_path, [":live"])
    out = capsys.readouterr().out
    assert "no live claims" in out
    store.close()


def test_cli_unknown_command(monkeypatch, tmp_path, capsys):
    store = _run_cli_with_inputs(monkeypatch, tmp_path, [":xyzzy"])
    out = capsys.readouterr().out
    assert "Unknown command" in out
    store.close()


def test_cli_imports():
    """Module imports cleanly — no top-level errors."""
    from keelson import cli
    assert callable(cli.main)
