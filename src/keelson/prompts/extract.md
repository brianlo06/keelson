# Claim Extraction

You will receive ONE turn of conversation between a user and their AI assistant. Extract structured claims about the user (and people/things they mention). Be CONSERVATIVE — better to miss a claim than fabricate one.

## What is a "claim"?

A claim is a discrete fact that could be looked up later.

- "I work at Anthropic" → claim
- "My dog Penny is a corgi" → claim
- "I'm exhausted today" → ambiguous (transient state; see rules below)
- "Nice weather we're having" → NOT a claim
- "I might move to Berlin" → hedged claim (`considering: moving to Berlin`, LOW ω — see "Hedged self-disclosure")

## Output schema

Return ONLY valid JSON, no preamble or commentary:

```json
{
  "claims": [
    {
      "subject": "user" | "user.<role>" | "<entity-name>",
      "predicate": "<snake_case_relation>",
      "object": "<string or entity name>",
      "valid_from": "<ISO date | 'now' | 'unknown'>",
      "omega": <float 0..1>,
      "supersedes_hint": "<predicate to look up | null>",
      "source_phrase": "<exact substring from the turn>"
    }
  ]
}
```

If no claims, return `{"claims": []}`. NEVER invent claims to fill the array.

## Conservative extraction (about decomposition, not omission)

When a single discrete fact has multiple details, emit ONE claim that captures
them — not multiple atomic claims. The `object` field can be a phrase that
holds the full detail ("retired from teaching after 30 years" is ONE claim
about ONE event, not a `did` claim plus a `career_duration` claim).

This rule is about how to DECOMPOSE a fact, not about WHETHER to extract.
You should still:
- Extract inferred claims (with low ω like 0.25) when context implies a fact
- Extract claims about third parties (ω capped at 0.50) when the user describes them
- Extract negated claims (`is_not`, `avoids`) when the user denies something

The rule simply says: when you've decided to extract a fact, prefer one
comprehensive claim over many fragmentary ones.

## Negation

A negative statement about the user is itself a claim — with the right
predicate. Don't strip the negation off and emit the underlying positive.

- "I'm not religious" → `is_not: religious` (NOT `identifies_as: religious`)
- "I don't drink coffee" → `avoids: coffee` (NOT `drinks: coffee` or `likes: coffee`)
- "I haven't talked to my dad in years" → `estranged_from: dad`
- "I never went to college" → `did_not: attend college`

## Life events that END a state

Events that end a state are CLAIMS even when wrapped in emotional commentary.
Don't dismiss them as transient feelings.

- "I broke up with Jamie last week" → `had_relationship: Jamie` (supersedes `has_relationship`)
- "I quit my job at Google" → `worked_at_previously: Google` (supersedes `works_at`)
- "Our dog Penny passed away" → `had_pet: Penny` (supersedes `has_pet`)
- "I sold the corgi back to the breeder" → `had_pet: corgi` (supersedes `has_pet`)

The trailing emotional commentary ("it's been rough", "miss her", "vet bills
were too much") is NOT a separate claim — extract only the event itself.

## omega (ω) assignment

| Range | When |
|---|---|
| **1.00** | User directly states something they personally lived ("I went to Tokyo last June", "I'm vegan") |
| **0.85** | User states current identity / role / belief / preference ("I'm a data scientist", "I love jazz") |
| **0.70** | User states a current intention or plan ("I start a new job Monday") |
| **0.50** | User reports a fact about someone else ("My sister got married") |
| **0.25** | You infer something not directly stated |
| **0.10–0.15** | Hedged self-disclosure ("I think I might be burning out"); speculation, dream content, hypothetical |

When uncertain, default to the LOWER value.

## Hedged self-disclosure — extract at LOW ω, don't drop

When the user hedges a statement about their OWN ongoing state or a live
possibility ("I think…", "I might…", "maybe I'm…"), the hedge lowers ω —
it does not erase the claim. Someone saying "I think I might be burning
out" is disclosing something load-bearing; store it, marked uncertain.

- "I think I might be burning out" → `experiencing: possible burnout` (ω 0.15)
- "I might apply to grad school" → `considering: applying to grad school` (ω 0.25)
- "Maybe I'm just not a city person" → `considering: not being a city person` (ω 0.15)

Distinguish this from momentary states: "I'm tired right now" is transient
(do not extract); "I think I might be burning out" describes an ongoing
condition the user is tentatively naming (extract, low ω). Pure
counterfactuals ("what if I'd taken that job") remain non-claims.

## Predicates

Prefer canonical snake_case forms when they fit; invent new ones if needed:

`works_at`, `worked_at_previously`, `role_at`, `lives_in`, `lived_in_previously`, `born_in`, `has_relationship`, `has_sibling`, `has_pet`, `has_child`, `likes`, `dislikes`, `loves`, `hates`, `is`, `was`, `identifies_as`, `did`, `attended`, `visited`, `believes`, `prefers`, `avoids`, `age`, `said`, `considering`, `experiencing`.

## Time

- "now" → `"now"`
- "last June" → `"2025-06"` (use current year as anchor unless specified)
- "when I was a kid" → `"unknown"` (note in `source_phrase`)
- If unsure → `"unknown"` (do not guess)

## Supersession

If the new claim plausibly replaces an existing one, set `supersedes_hint` to the predicate to look up. The application handles the actual update.

- "I just started at Anthropic" → `supersedes_hint: "works_at"`
- "I broke up with Jamie" → `supersedes_hint: "has_relationship"`
- "I moved to Brooklyn" → `supersedes_hint: "lives_in"`

If the user is adding (not replacing), leave it `null`.

## Other-subject claims

When the claim is about a person other than the user, use `user.<role>` for related people (`user.mom`, `user.sister`, `user.coworker`) or the proper noun for non-relations. omega is at most 0.50 for these.

CRITICAL — attribute facts to the right subject. When a sentence mentions
both the user and someone else, decide WHOSE fact each detail is before
choosing `subject`. "My sister is visiting from Berlin" contains a locative
about the SISTER (she lives in Berlin), not the user — emitting
`subject: user, predicate: lives_in, object: Berlin` would be a
fabrication. Prepositional phrases attached to the other person
("visiting from X", "calling from X", "who works at Y") belong to that
person's subject, at ω ≤ 0.50.

## DO NOT extract

- Chit-chat, greetings, weather talk
- Hypotheticals, "what if" scenarios
- Jokes (unless tagged as serious)
- Transient emotional states ("I'm tired right now") — UNLESS the user signals they want it remembered ("write that down", "I want to track this")
- The AI's own statements about the user
- Questions or queries (these are reads, not writes)

## Examples

### Direct identity statement

USER: "I'm 34, work at Anthropic as a researcher, and I have a corgi named Penny."

```json
{
  "claims": [
    {"subject": "user", "predicate": "age", "object": "34", "valid_from": "now", "omega": 1.00, "supersedes_hint": "age", "source_phrase": "I'm 34"},
    {"subject": "user", "predicate": "works_at", "object": "Anthropic", "valid_from": "now", "omega": 0.85, "supersedes_hint": "works_at", "source_phrase": "work at Anthropic"},
    {"subject": "user", "predicate": "role_at", "object": "researcher at Anthropic", "valid_from": "now", "omega": 0.85, "supersedes_hint": "role_at", "source_phrase": "as a researcher"},
    {"subject": "user", "predicate": "has_pet", "object": "Penny (corgi)", "valid_from": "now", "omega": 1.00, "supersedes_hint": null, "source_phrase": "I have a corgi named Penny"}
  ]
}
```

### Contradiction / update

USER: "Just got the offer — I'm leaving Google for Anthropic next month."

```json
{
  "claims": [
    {"subject": "user", "predicate": "worked_at_previously", "object": "Google", "valid_from": "now", "omega": 1.00, "supersedes_hint": "works_at", "source_phrase": "leaving Google"},
    {"subject": "user", "predicate": "works_at", "object": "Anthropic", "valid_from": "2026-06", "omega": 0.70, "supersedes_hint": "works_at", "source_phrase": "for Anthropic next month"}
  ]
}
```

### About someone else (be conservative — one claim, ω≤0.50)

USER: "My mom just retired from teaching after 30 years."

```json
{
  "claims": [
    {"subject": "user.mom", "predicate": "did", "object": "retired from teaching after 30 years", "valid_from": "now", "omega": 0.50, "supersedes_hint": "current_role", "source_phrase": "My mom just retired from teaching"}
  ]
}
```

Note: a single discrete event is ONE claim. The duration ("after 30 years") is
part of the event description, not a separate `career_duration` claim. For
third parties, ω is capped at 0.50 — you know what the user told you, not
whether it's accurate.

### Someone else's fact stays theirs

USER: "My sister Mara is visiting from Berlin this week."

```json
{
  "claims": [
    {"subject": "user", "predicate": "has_sibling", "object": "Mara (sister)", "valid_from": "now", "omega": 1.00, "supersedes_hint": null, "source_phrase": "My sister Mara"},
    {"subject": "user.sister", "predicate": "lives_in", "object": "Berlin", "valid_from": "now", "omega": 0.50, "supersedes_hint": null, "source_phrase": "visiting from Berlin"}
  ]
}
```

CRITICAL: "visiting from Berlin" tells you where MARA lives. Do NOT emit
`subject: user, predicate: lives_in, object: Berlin` — the user's location
is unstated here. The sibling relationship is the user's own fact (ω 1.00);
where the sibling lives is a reported third-party fact (ω 0.50).

### Hedged self-disclosure (extract, low ω)

USER: "I think I might be burning out."

```json
{
  "claims": [
    {"subject": "user", "predicate": "experiencing", "object": "possible burnout", "valid_from": "now", "omega": 0.15, "supersedes_hint": null, "source_phrase": "I think I might be burning out"}
  ]
}
```

Note: the hedge ("think", "might") lowers ω to 0.10–0.15; it does not make
this a non-claim. For a journaling user, tentatively-named ongoing states
are exactly what should be remembered.

### Negation: "I'm not X"

USER: "I'm not religious, but I do meditate daily."

```json
{
  "claims": [
    {"subject": "user", "predicate": "is_not", "object": "religious", "valid_from": "now", "omega": 0.85, "supersedes_hint": null, "source_phrase": "I'm not religious"},
    {"subject": "user", "predicate": "practices", "object": "meditation (daily)", "valid_from": "now", "omega": 0.85, "supersedes_hint": null, "source_phrase": "I do meditate daily"}
  ]
}
```

CRITICAL: do NOT emit a positive claim (`identifies_as: religious`) when the
user is negating that very thing. The negation IS the claim.

### Ending a relationship

USER: "I broke up with Jamie last week. It's been rough."

```json
{
  "claims": [
    {"subject": "user", "predicate": "had_relationship", "object": "Jamie", "valid_from": "now", "omega": 1.00, "supersedes_hint": "has_relationship", "source_phrase": "I broke up with Jamie"}
  ]
}
```

Note: "It's been rough" is transient emotional commentary — NOT a claim.
The breakup itself IS a claim and supersedes any prior `has_relationship`.

### Hedge / non-claim

USER: "I might apply to grad school next year, not sure yet."

```json
{
  "claims": [
    {"subject": "user", "predicate": "considering", "object": "applying to grad school", "valid_from": "now", "omega": 0.25, "supersedes_hint": null, "source_phrase": "I might apply to grad school"}
  ]
}
```

### Chit-chat — extract nothing

USER: "Hey! How are you today?"

```json
{"claims": []}
```

### Query, not a claim — extract nothing

USER: "What do you remember about my mom?"

```json
{"claims": []}
```

### Inferred claim (low ω)

USER: "Finally getting feeling back in my legs after yesterday."

```json
{
  "claims": [
    {"subject": "user", "predicate": "did", "object": "intense physical activity (inferred)", "valid_from": "yesterday", "omega": 0.25, "supersedes_hint": null, "source_phrase": "feeling back in my legs after yesterday"}
  ]
}
```

### Negation supersedes

USER: "Actually I'm not vegetarian anymore — eating meat again."

```json
{
  "claims": [
    {"subject": "user", "predicate": "was", "object": "vegetarian", "valid_from": "now", "omega": 1.00, "supersedes_hint": "diet", "source_phrase": "I'm not vegetarian anymore"},
    {"subject": "user", "predicate": "diet", "object": "eats meat", "valid_from": "now", "omega": 1.00, "supersedes_hint": "diet", "source_phrase": "eating meat again"}
  ]
}
```

---

Now extract claims from this turn:

USER: {user_message}
