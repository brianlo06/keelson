"""LLM-as-judge for semantic similarity between extracted and expected claims.

This module exists because string-match scoring would conflate `works_at` with
`employed_by` (real synonyms) AND also conflate `works_at Google` with
`worked_at Google` (different tenses, totally different meaning). Embedding
similarity has the same problem in the other direction.

The judge takes a (expected, extracted) pair and returns a 0..1 match score.
Cheap calls — uses Haiku-tier reasoning.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anthropic

if TYPE_CHECKING:
    from keelson.schema import Claim
    from eval.types import ExpectedClaim

JUDGE_MODEL = "claude-opus-4-7"


@dataclass(frozen=True)
class JudgeVerdict:
    overall: float
    subject_match: float
    predicate_match: float
    object_match: float
    reason: str


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*\n(.*?)\n```$", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


_JUDGE_SYSTEM = """You compare two claims about a user and judge whether they refer to the same fact.

You will receive:
- EXPECTED: a claim with subject, predicate, object pattern (semantic intent)
- EXTRACTED: a claim with subject, predicate, object (actual output)

You judge whether EXTRACTED captures the same factual content as EXPECTED.

Return ONLY a JSON object:
{
  "subject_match": <0..1>,
  "predicate_match": <0..1>,
  "object_match": <0..1>,
  "overall": <0..1>,
  "reason": "<one sentence>"
}

Scoring:
- Exact match → 1.0
- Synonyms / paraphrases of same meaning → 0.9
- Same fact, different framing → 0.7
- Related but not the same fact → 0.3
- Different fact → 0.0

IMPORTANT: temporal mismatch is a hard fail. `works_at Google` and `worked_at_previously Google` are DIFFERENT facts (one is current, one is past). predicate_match should be ≤0.2 if tense / temporal aspect differs.

Subject conventions: "user" and "the user" are the same. "user.mom" and "user's mother" are the same. Proper nouns are case-insensitive."""


_JUDGE_USER_TEMPLATE = """EXPECTED:
  subject: {expected_subject}
  predicate: {expected_predicate}
  object: {expected_object}

EXTRACTED:
  subject: {extracted_subject}
  predicate: {extracted_predicate}
  object: {extracted_object}

Judge."""


class Judge:
    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str = JUDGE_MODEL,
    ):
        self.client = client or anthropic.Anthropic()
        self.model = model

    def compare(self, expected: ExpectedClaim, extracted: Claim) -> JudgeVerdict:
        user_message = _JUDGE_USER_TEMPLATE.format(
            expected_subject=expected.subject,
            expected_predicate=expected.predicate,
            expected_object=expected.object_pattern,
            extracted_subject=extracted.subject,
            extracted_predicate=extracted.predicate,
            extracted_object=extracted.object,
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=[
                {
                    "type": "text",
                    "text": _JUDGE_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_message}],
        )

        text = next(
            (block.text for block in response.content if block.type == "text"),
            "",
        )

        try:
            data = json.loads(_strip_code_fences(text))
            return JudgeVerdict(
                overall=float(data.get("overall", 0.0)),
                subject_match=float(data.get("subject_match", 0.0)),
                predicate_match=float(data.get("predicate_match", 0.0)),
                object_match=float(data.get("object_match", 0.0)),
                reason=data.get("reason", ""),
            )
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            return JudgeVerdict(
                overall=0.0,
                subject_match=0.0,
                predicate_match=0.0,
                object_match=0.0,
                reason=f"judge returned unparseable response: {text[:200]}",
            )
