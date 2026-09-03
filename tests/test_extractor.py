"""Offline tests for the extractor's prompt plumbing and prompt content.

No API calls. The prompt-content tests are tripwires for guidance that a
real run proved load-bearing (kelpie FINDINGS.md, 2026-05-25): if a future
prompt edit drops it, these fail before the eval suite has to catch the
regression live.
"""

from __future__ import annotations

import pytest

from keelson.extractor import _PROMPT_TEMPLATE, _split_prompt, _strip_code_fences


# --- plumbing ---------------------------------------------------------------

def test_split_prompt_yields_system_and_user_parts():
    system, user = _split_prompt(_PROMPT_TEMPLATE)
    assert "{user_message}" in user
    assert "{user_message}" not in system
    assert len(system) > 1000  # the examples live in the cacheable prefix


def test_split_prompt_missing_separator_raises():
    with pytest.raises(ValueError):
        _split_prompt("no separator here {user_message}")


def test_strip_code_fences():
    assert _strip_code_fences('```json\n{"claims": []}\n```') == '{"claims": []}'
    assert _strip_code_fences('```\n{"claims": []}\n```') == '{"claims": []}'
    assert _strip_code_fences('{"claims": []}') == '{"claims": []}'


# --- prompt tripwires (regressions from kelpie's first real run) ------------

def test_prompt_keeps_other_subject_locative_example():
    """Regression-006: 'my sister is visiting from Berlin' must never yield
    `user lives_in Berlin`. The disambiguating example must stay in the prompt."""
    system, _ = _split_prompt(_PROMPT_TEMPLATE)
    assert "visiting from Berlin" in system
    assert "attribute facts to the right subject" in system.lower()


def test_prompt_keeps_hedged_self_disclosure_guidance():
    """Regression-007: 'I think I might be burning out' must be extracted at
    low omega, not dropped as a hedge."""
    system, _ = _split_prompt(_PROMPT_TEMPLATE)
    assert "Hedged self-disclosure" in system
    assert "burning out" in system
    assert "`experiencing`" in system or "experiencing" in system
