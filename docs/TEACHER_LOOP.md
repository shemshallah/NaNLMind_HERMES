# Teacher and conversational learning loops

Hermes has two distinct learning paths that share the same native 256-head
universe. Structured training is explicit and teacher-driven. Free-form chat
learns online from validated evidence in ordinary back-and-forth conversation.

## 1. Structured teacher training

```text
active teacher
  -> JSON relation records
  -> deterministic schema / source / split validation
  -> frozen challenge set per topic and resolution
  -> N16REL/1
  -> native relation-train with requested epochs
  -> held-out evaluation before and after training
  -> per-resolution feedback and cross-run deltas
  -> teacher proposes a focused next round
```

The teacher proposes data; it does not declare success. The native runner owns
the student state and evaluator. Teacher labels are recorded as
`teacher_distillation` with `independently_fact_verified: false` unless an
independent source verifier actually establishes the fact.

The generator requests explicit triples:
`subject`, `predicate`, `object`, `polarity`, `confidence`, and
`evidence`. Unsupported neutral claims are not converted into binary targets.
The deterministic validator rejects incomplete triples, malformed confidence,
unknown resolutions, invalid labels and duplicate examples.

## 2. Ten independent resolution channels

| Resolution | Focus | Held-out measurement |
|---|---|---|
| L0 bit | Boolean/value relations | relation-polarity accuracy and balanced accuracy |
| L1 nibble | four-bit composition | relation-polarity accuracy and balanced accuracy |
| L2 byte | byte identity/composition | relation-polarity accuracy and balanced accuracy |
| L3 phonetic | grapheme/phoneme/syllable | relation-polarity accuracy and balanced accuracy |
| L4 morpheme | stem/affix/inflection | relation-polarity accuracy and balanced accuracy |
| L5 lexical | word/sense/semantic relation | relation-polarity accuracy and balanced accuracy |
| L6 word completion | context/continuation | relation-polarity accuracy and balanced accuracy |
| L7 sentence | syntax/agreement/order | relation-polarity accuracy and balanced accuracy |
| L8 paragraph | cohesion/reference/topic | relation-polarity accuracy and balanced accuracy |
| L9 narrative | event/time/causality | relation-polarity accuracy and balanced accuracy |

The native relation learner uses deterministic resolution-specific surface features (phonetic, morphemic, lexical, syntactic, cohesive and narrative cues); SHA-256 is reserved for stable routing, source identity and event deduplication. This is a compact feature learner, not a pretrained semantic encoder.\n\nThese are **binary relation-polarity** measurements, not text-generation
accuracy. The old `math-train` byte-bit score remains a separate control and
must never be presented as a lexical or sentence metric.

Every resolution reports sample counts, pre-training accuracy,
post-training accuracy, within-run delta, balanced accuracy and majority
baseline. Balanced accuracy is explicitly marked valid only when the held-out
set contains both positive and negative labels; the teacher generator
stratifies the frozen challenge to include both when available. Each topic/resolution gets a frozen challenge set stored locally.
Reusing the same challenge digest enables a meaningful cross-run delta. A new
test digest is marked non-comparable. Replaying the same event still evaluates
the unchanged challenge set and returns `no_op` for mutation.

## 3. Boolean foundation gate

The native Boolean foundation is a finite coverage gate, not a generalization
claim. It requires all 1,024 row checks, all 16 distinct learned functions,
composed circuits and a persisted checkpoint. The gate is cached after a
verified pass; it is not reset on every language or chat turn.

The foundation checkpoint seeds the shared `native/global_universe.nmg`
exactly once. Math, structured relation training and conversational learning
then extend that same serialized state.

## 4. Free-form conversational learning

Normal `nanlm_chat` remains open-ended. With `chat_learning: true` (default),
each user/assistant exchange is reflected into candidate evidence. Explicit
supported/refuted triples are sent to the native `relation-train` path; neutral
or incomplete claims remain in symbolic memory without a native update.

For each accepted event, the native learner:
1. routes subject, predicate and object to distinct heads within the declared
   resolution partition;
2. updates the routed heads' Boolean rule score tables;
3. mutates the canonical mirror-pair cells, preserving complementary rails;
4. adds temporal links between participating heads and pair cells;
5. appends a provenance-bound n-ary hyperedge for the relation/event;
6. commits and serializes the shared topology;
7. returns changed-head, pair-update, hyperedge and state-digest receipts.

A repeated identical source-bound event is idempotent and reports `no_op`.
A later conversational turn has a different source digest and can add new
evidence. Contradictions remain separate events rather than overwriting prior
claims. The chat answer is never blocked by reflection or native-update failure;
the result reports those stages separately.

## 5. Feedback and stop rules

Send the teacher a compact report containing:
- exact Boolean gate result and checkpoint reuse status;
- each resolution's train/test counts, before/after score, delta, balanced
  accuracy and majority baseline;
- whether the frozen test digest matches the previous round;
- changed heads, mirror pairs, hyperedges, duplicates and state digests;
- rejected records and representative errors.

Do not send frozen test answers to the teacher. Stop or revise a curriculum when
held-out metrics do not improve, balanced accuracy collapses, a resolution has
insufficient coverage, or teacher proposals duplicate the frozen challenge set.

## 6. Explicit structured tool

`nanlm_structured_relation_train` accepts typed records with a `split` field
(`train` or `test`), a stable 64-hex `source_digest`, and an `item`
containing a complete triple, polarity, confidence, evidence and resolution.
Only train rows mutate the student. Test rows are evaluated before and after
training. The tool is the manual entry point for user-supplied or externally
verified examples; `nanlm_training_pipeline` generates teacher-labeled batches
and frozen challenge sets automatically.

