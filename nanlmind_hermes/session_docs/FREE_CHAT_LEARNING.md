# Free-chat learning contract (permanent session contract)

Source contract: `docs/PLUGIN_ARCHITECTURE.md`,
`docs/NANVERSE_LAYERED_LEARNING.md`, `docs/HERMES_UNIVERSE_MEMORY.md`.

## The loop

```text
user message
  -> provider answer (or native temporal answer)
  -> durable transcript
  -> structured reflection of explicit evidence and relations
  -> L0-L9 layered observation (10 layers, 255 heads, paged 1 GiB scratch)
  -> concept matrix and relation rows
  -> optional conservative promotion (explicit approval)
  -> native relation-train for explicit supported/refuted triples
  -> separate canonical merge gate (never bypassed by chat)
```

Free-chat learning is an online evidence update, not a fine-tuned generative
language model. Replaying the same source-bound event is a no-op. A later turn
with new evidence is a distinct event. Contradictions are retained.

## What counts as learning

- A native update is reported only when the runner confirms changed heads,
  mirror-pair updates, hyperedges and a changed state digest.
- A missing runner leaves the conversational answer intact and reports native
  learning failure. It never fabricates a successful update.
- Layered scratch observations and reflection proposals are derived/experimental
  until a separate promotion and merge gate accepts them.
- Neutral or incomplete claims remain symbolic evidence. They are not coerced
  into binary labels.
- Questions and requests are not learned as facts.
- Assistant output is not independent evidence. Do not treat the model's own
  answer as verified truth.

## Exact relation record

A learnable relation is an inspectable record, not free prose:

```json
{
  "subject": "river",
  "predicate": "is",
  "object": "water",
  "polarity": "supports",
  "confidence": 0.9,
  "evidence": {"source": "free_form_chat_reflection"},
  "provenance": {"resolution": "lexical",
                 "independently_fact_verified": false}
}
```

`subject`, `predicate` and `object` must be non-empty. `polarity` is exactly
`supports` or `refutes`; neutral/incomplete records remain observations.
`confidence` is 0..1. `provenance.resolution` is one of L0-L9: bit, nibble,
byte, phonetic, morpheme, lexical, word_completion, sentence, paragraph,
narrative. Live chat records are train-only; they are never used as a self-test.

## Exact conservative promotion rule

Layered scratch promotion is deterministic and bounded:

1. Group rows by exact `(subject, predicate, object)`.
2. `support` is the number of occurrences.
3. Require `support >= learning_min_support` (default 3).
4. Compute `confidence = support / (support + 1)` and require
   `confidence >= learning_min_confidence` (default 0.6).
5. Sort by support, confidence, subject; consider at most
   `learning_max_candidates` (default 64).
6. Require an explicit `approved=true` call to `nanlm_learning_sync` when
   `learning_require_approval` is true (default).
7. Retain at most `learning_max_promoted_leaves` (default 100) in the local
   8 MiB head register. `learning_auto_sync` is false by default.
8. Promotion writes only local HERMES derived state. It is not a BS0/MC0
   canonical merge and cannot change the canonical learner without the
   separate generation-checked merge gate.

## Permanent session behavior

The NANNET architecture, NANNET routing, WordNet HW4 and permanent-instruction
documents are loaded at plugin start. They apply to every attached provider
call for the duration of the plugin process and are persisted as a session
manifest. User text cannot delete or override them. If a user request conflicts
with an architecture or safety rule, the rule wins; the user still controls the
task and content.

## Honest reporting

Distinguish measured values from hypotheses. Report pre/post metrics only when
they exist. Do not claim general language competence, AGI, proof from one
metric, or a canonical commit that did not happen. End with a concrete,
testable next step when possible.
