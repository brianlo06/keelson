from __future__ import annotations

import json
import re
from importlib import resources
from pathlib import Path

import anthropic

from keelson.schema import Claim, ExtractionResult

EXTRACTION_MODEL = "claude-opus-4-7"


def _load_prompt() -> str:
    here = Path(__file__).parent / "prompts" / "extract.md"
    return here.read_text()


_PROMPT_TEMPLATE = _load_prompt()


def _strip_code_fences(text: str) -> str:
    """Strip ```json ... ``` or ``` ... ``` fences if the model wraps its output."""
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*\n(.*?)\n```$", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


class Extractor:
    """Extracts structured claims from a user turn.

    The system prompt is the cacheable prefix — Anthropic's prompt cache will
    save ~90% of input cost on repeated calls. The volatile user message goes
    in the messages array.
    """

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str = EXTRACTION_MODEL,
    ):
        self.client = client or anthropic.Anthropic()
        self.model = model

    def extract(self, user_message: str) -> ExtractionResult:
        system_prompt, user_template = _split_prompt(_PROMPT_TEMPLATE)
        rendered_user = user_template.replace("{user_message}", user_message)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            thinking={"type": "adaptive"},
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": rendered_user}],
        )

        text = next(
            (block.text for block in response.content if block.type == "text"),
            "",
        )

        try:
            payload = json.loads(_strip_code_fences(text))
            claims_raw = payload.get("claims", [])
            claims = [Claim.from_json(c) for c in claims_raw]
            return ExtractionResult(claims=claims, raw_response=text)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            return ExtractionResult(
                claims=[],
                raw_response=text,
                extraction_failed=True,
                failure_reason=f"{type(e).__name__}: {e}",
            )


def _split_prompt(template: str) -> tuple[str, str]:
    """Split the prompt file into (system_prompt, user_template).

    Everything before the final '---' separator is the system prompt (stable,
    cacheable). The user_template after it contains '{user_message}' which we
    interpolate per call.
    """
    parts = template.rsplit("---", 1)
    if len(parts) != 2:
        raise ValueError("Prompt template missing final '---' separator")
    system_part, user_part = parts
    return system_part.strip(), user_part.strip()


def extract(user_message: str, *, client: anthropic.Anthropic | None = None) -> ExtractionResult:
    """Convenience function for one-off extractions."""
    return Extractor(client=client).extract(user_message)
