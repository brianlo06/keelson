"""Holistic LLM-as-judge evaluator for claim extractions.

The structured judge in eval/judge.py compares extracted claims against
expected claims one-by-one. This module does something different: it asks a
fresh Claude to look at the user's message + the model's extracted claims as
a whole and rate them on four dimensions, 1-10.

Why both? The structured judge catches schema-level errors (wrong predicate,
out-of-range omega, missing required claim). The holistic judge catches what
the structured one can't see — the extraction has the right shape but feels
wrong somewhere, or it nailed something subtle that wasn't in the expected
list, or the predicate names are technically right but unidiomatic.

Costs: one extra Claude call per eval example (~$0.002 with prompt caching).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anthropic

if TYPE_CHECKING:
    from keelson.schema import Claim
    from eval.types import EvalExample

HOLISTIC_MODEL = "claude-opus-4-7"


@dataclass(frozen=True)
class HolisticVerdict:
    coverage: float          # 0..1 (mapped from 1-10)
    precision: float
    omega: float
    usefulness: float
    overall: float
    rationale: str


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*\n(.*?)\n```$", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


_SYSTEM = """You are a holistic evaluator of claim-extraction quality for a
personal memory system. You receive:
  - A user's message (one turn of conversation)
  - The expected claims (what a careful extractor should produce)
  - The actual extracted claims

Score on four dimensions, each 1-10:

  coverage   — did the extractor catch the important facts the user stated?
               (10 = nothing important missed; 5 = some misses; 1 = nothing useful)
  precision  — did the extractor avoid fabrication?
               (10 = nothing made up; 5 = some bonus claims with weak basis; 1 = many fabricated)
  omega      — are ω values calibrated to provenance? Lived > stated identity >
               reported about other > inferred > speculation. (10 = all in range)
  usefulness — would these claims be useful for future recall in a personal
               memory system? Are predicates idiomatic? Object descriptions clear?
               (10 = a model recalling later would have everything it needs)

Hard rules:
  - For pure chit-chat with no factual content, the only good extraction is `[]`.
    Score `[]` as 10/10/10/10 in that case.
  - Extra claims beyond expected aren't always fabrications — bonus claims that
    capture real information from the user's message earn full precision. Only
    invented details lower the score.
  - omega should match what the user is telling you: 1.0 for lived experience,
    0.85 for stated identity, 0.5 for reported-about-other, 0.25 for inference.

Return ONLY this JSON (no preamble, no code fences):
{
  "coverage": <int 1-10>,
  "precision": <int 1-10>,
  "omega": <int 1-10>,
  "usefulness": <int 1-10>,
  "rationale": "<one sentence>"
}
"""


_USER_TEMPLATE = """USER MESSAGE:
{user_message}

EXPECTED CLAIMS:
{expected}

EXTRACTED CLAIMS:
{extracted}

Score."""


def _format_claim_list(claims) -> str:
    if not claims:
        return "  (none)"
    lines = []
    for c in claims:
        if hasattr(c, "object_pattern"):
            # ExpectedClaim
            omega = f"ω∈[{c.omega_range[0]:.2f}, {c.omega_range[1]:.2f}]"
            line = f"  - {c.subject} {c.predicate} {c.object_pattern!r}  ({omega}"
            if not c.required:
                line += ", optional"
            line += ")"
        else:
            # Claim
            line = f"  - {c.subject} {c.predicate} {c.object!r}  (ω={c.omega:.2f})"
        lines.append(line)
    return "\n".join(lines)


class HolisticJudge:
    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str = HOLISTIC_MODEL,
    ):
        self.client = client or anthropic.Anthropic()
        self.model = model

    def judge(
        self,
        example: "EvalExample",
        extracted: list["Claim"],
    ) -> HolisticVerdict:
        user_msg = _USER_TEMPLATE.format(
            user_message=example.user_message,
            expected=_format_claim_list(example.expected_claims),
            extracted=_format_claim_list(extracted),
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_msg}],
        )

        text = next(
            (block.text for block in response.content if block.type == "text"),
            "",
        )

        try:
            data = json.loads(_strip_code_fences(text))
            coverage = float(data.get("coverage", 0)) / 10
            precision = float(data.get("precision", 0)) / 10
            omega = float(data.get("omega", 0)) / 10
            usefulness = float(data.get("usefulness", 0)) / 10
            overall = (coverage + precision + omega + usefulness) / 4
            return HolisticVerdict(
                coverage=coverage,
                precision=precision,
                omega=omega,
                usefulness=usefulness,
                overall=overall,
                rationale=str(data.get("rationale", "")),
            )
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            return HolisticVerdict(
                coverage=0.0,
                precision=0.0,
                omega=0.0,
                usefulness=0.0,
                overall=0.0,
                rationale=f"holistic judge returned unparseable response: {text[:200]}",
            )
