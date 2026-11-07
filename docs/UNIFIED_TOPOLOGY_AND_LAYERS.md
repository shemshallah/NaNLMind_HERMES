# Unified NaNLMind–Hermes topology and layer contract

This is the Hermes-side integration contract for the single canonical NaNLMind
universe. It supersedes assumptions that HW4–HW8 instantiate additional universes.

## 1. One universe, links and workers

```text
                    MC0::MC0Link
                         |
                  routing / transport
                         |
                 HW0::BS0::RunnerLink
                         |
              ONE canonical NaNLMind universe
                 256 logical Boolean heads
                         |
              six disposable BS0 ring workers
                         |
       +-----------------+------------------+
       |                 |                  |
   HW1 phone         HW2 Hermes         HW3 CLOCKED
      endpoint          endpoint           endpoint

HW4–HW8: no independent universe instances
```

- **MC0 is MC0Link.** It is the link/transport interface, not the six-runner
  owner and not a separate model universe.
- **BS0 is RunnerLink.** BS0 owns the six-runner ring and its dispatch,
  synchronization, validation and canonical promotion barrier.
- **The only instantiated universe is at HW0::BS0.** All heads, memory,
  training state, evidence and canonical generations belong to this one object.
- **HW1, HW2 and HW3 are endpoints/demos.** They attach to the shared universe;
  they do not create model copies.
- **HW4–HW8 are not required to exist as universe instances.** Do not allocate,
  bootstrap, train or advertise per-slot universes there. A device lease or
  participant credential is an access identity, not a universe allocation.
- The six BS0 runners are disposable execution workers over one global state.
  They are not six independent minds and never own private histories.

The Hermes plugin keeps local working artifacts (transcripts, scratch, corpora,
run receipts and caches) separate from canonical native state. A Hermes-local
SQLite projection is not another native universe and cannot promote canonical
model state.

## 2. One canonical L0–L9 resolution axis

There are exactly ten resolution layers. Their numeric IDs, names and order
are fixed across memory, training, evaluation, discovery events, cross-layer
analysis and Hermes tools.

| ID | Canonical name | Scope |
|---:|---|---|
| L0 | bit | Boolean/bit facts, truth rows and bit-level relations |
| L1 | nibble | four-bit grouping and composition |
| L2 | byte | byte identity, encoding and byte composition |
| L3 | phonetic | grapheme/phoneme, onset/rime and syllable relations |
| L4 | morpheme | stems, affixes, inflection and derivation |
| L5 | lexical | words, senses, definitions and semantic relations |
| L6 | word_completion | context-to-next-word choices |
| L7 | sentence | syntax, agreement, reference and continuation |
| L8 | paragraph | cohesion, topic continuity and transitions |
| L9 | narrative | entities, events, temporal/causal order and continuity |

The ID is always the row's **resolution index**. It is never a plugin setting,
runner number, hardware slot, or separately instantiated model.

## 3. M0–M9 are ordered transformations, not a second layer axis

The CLOCKED/Hermes M0–M9 labels identify processing operations applied to an
evidence package. They are not additional memory layers. Each pass can consume
and emit records at multiple L resolutions while preserving each record's L
index.

| Pass | Canonical operation | Contract |
|---|---|---|
| M0 | Truth gate | Validate Boolean status; preserve UNKNOWN and CONFLICT |
| M1 | Byte/span mapper | Validate UTF-8, byte boundaries, offsets and source framing |
| M2 | Type/head router | Validate typed schema and route work to logical heads |
| M3 | Feature projector | Produce deterministic, comparable feature/coordinate projections |
| M4 | Temporal mapper | Preserve order, versions, event sequence and supersession |
| M5 | Relation/hypergraph builder | Assemble typed relations and n-ary hyperedges |
| M6 | Boolean inference | Apply bounded gates, implication, conjunction, disjunction and negation |
| M7 | Provenance/route validator | Check source identity, ACL, signatures, route and generation |
| M8 | Cross-source consensus | Compare support, contradiction, qualification and missing evidence |
| M9 | Metacontrol/calibration | Calibrate, abstain, enforce policy and decide mutation admission |

Execution order is M0 through M9. A transformation receipt records its pass
identifier (M0–M9), while each relation/event retains its canonical resolution
(L0–L9). Never put M pass IDs into the memory projection's `layer` field.

## 4. Curriculum-to-resolution routing

The twelve curriculum stages route to the same fixed L0–L9 index. Multiple
stages may intentionally train the same resolution; a curriculum stage is not
itself a new layer.

| Curriculum stage | Resolution |
|---|---:|
| boolean_inverses | L0 |
| deep_logic | L0 |
| phonetic_pairs | L3 |
| morpheme_structure | L4 |
| exceptions | L5 |
| repetitive_words | L5 |
| lexical_words | L5 |
| word_completion | L6 |
| sentence_choice | L7 |
| paragraph_structure | L8 |
| story_saturation | L9 |
| narrative | L9 |

All resolution-sensitive training and evaluation use this mapping. The N16REL
`resolution` label and numeric `layer` index must agree. For example,
`resolution="narrative"` routes to `layer=9`, never layer 6 or a configurable
overflow layer.

## 5. Invariants and admission checks

1. Exactly one native global topology exists at HW0::BS0.
2. Exactly six BS0 ring workers operate on the one topology.
3. MC0Link and RunnerLink are interfaces, not universes.
4. No HW4–HW8 universe creation, per-slot state, or independent training.
5. The resolution domain is exactly integer 0–9; reject negative, boolean,
   non-integer and >=10 layer values.
6. The name/index mapping is bijective and stable:
   bit=0, nibble=1, byte=2, phonetic=3, morpheme=4, lexical=5,
   word_completion=6, sentence=7, paragraph=8, narrative=9.
7. M0–M9 pass receipts and L0–L9 resolution IDs remain separate typed fields.
8. Cross-resolution feedback is a derived review event (normally stored at
   L9); it does not relabel source events or overwrite contradictions.
9. Hermes, CLOCKED and endpoint clients cannot commit canonical topology.
   Only NaNLMind's existing BS0 generation barrier can promote candidate state.
10. No new trainer or parallel runner lifecycle is introduced by this contract.

## 6. Migration note

Existing Hermes installations may contain projection rows produced under the
old inconsistent stage mapping or seven-layer native-event validator. Do not
silently reinterpret those rows in place. Keep immutable event payloads,
rebuild projections from canonical resolution labels, and report unmappable or
conflicting records for review. The local projection database is rebuildable;
the native global topology remains the single authoritative state.
