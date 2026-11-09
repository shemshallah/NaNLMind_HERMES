# Unified NaNLMind–Hermes topology and layer contract

This is the Hermes-side integration contract for the single canonical NaNLMind
universe. It supersedes assumptions that HW4–HW8 instantiate additional universes.

## 0. Canonical build authority

The native build and topology contract lives in NaNLMind:
`spec/unified-build.v1.json`, exposed by `build/nanlm-universal unification` and
`make unified-check`. NaNROUTER is the transport, point-0 ingress, routing and
control plane; it does not instantiate or fork the native universe. HERMES is a
read-only adapter and candidate producer.

HERMES reads the contract through `nanlmind_hermes.unification` at
`NANLMIND_ROOT`, a sibling `../NaNLMind` checkout, or the configured
`nanlmind_root`; it never substitutes hardcoded constants as authority. When
the canonical binary is available, HERMES also verifies its `unification` line.
See [UNIFIED_BUILD_ADAPTER.md](UNIFIED_BUILD_ADAPTER.md).

## 1. One universe, links and workers

```text
                    MC0::MC0Link
                         |
                  routing / transport
                         |
                 HW0::RunnerLink → BS0::RunnerRing
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

HW4+: allocatable participant addresses attached to HW0::BS0
```

- **MC0 is MC0Link.** It is the link/transport interface, not the six-runner
  owner and not a separate model universe.
- **HW0 is RunnerLink.** HW0 is the hardware-level runner interface.
- **BS0 is the RunnerRing.** BS0 owns the six-runner ring and its dispatch,
  synchronization, validation and canonical promotion barrier.
- **The only instantiated universe is at HW0::BS0.** All heads, memory,
  training state, evidence and canonical generations belong to this one object.
- **HW1, HW2 and HW3 are endpoints/demos.** They attach to the shared universe;
  they do not create model copies.
- **HW4+ is the allocatable participant space.** HW4–HW8 have no special
  reservation; the BS0 allocator may assign any free address from HW4 onward.
  They are not runner identities, model slots or universe instances.
- **Every HW4+ lease is a scoped participant address only.** A lease identifies
  an external participant attached to the shared HW0::BS0 universe; it does not
  allocate model state, a private head topology or a second universe.
- The six BS0 runners are disposable execution workers over one global state.
  They are not six independent minds and never own private histories.

The Hermes plugin keeps working artifacts (transcripts, scratch,
corpora, run receipts and caches) under its configured state directory. Its
`universe/` subdirectory is the native state/checkpoint path for the one
HW0::BS0 universe, not another universe. All BS0 workers and attached Hermes
clients must use the same canonical state path; do not configure per-runner or
per-endpoint native state directories. Hermes-local SQLite projections remain
derived caches and cannot promote canonical model state.

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

### Runtime integration status

The Hermes head-process path now executes and records all ten M0–M9 passes:
M0–M5 deterministic preflight, M6 delegated to NaNLMind's read-only native
`relation-eval`, and M7–M9 response/provenance/consensus/metacontrol validation.
The append-only `transform_trace` scratch lane records digests and bounded
summaries, not raw evidence. M5 relation/hyperedge projections are candidate
diagnostics; the native service remains authoritative and re-derives inference
atoms from the immutable evidence package. Tests exercise the complete chain
with a controlled head-cortex response.

This is local integration validation, not a claim of a live external CLOCKED
gateway plus production native-binary run. Candidate learning remains isolated
behind a separate `relation-train` candidate endpoint.

There is a naming collision to resolve before wiring the chain: the core
router's `runner/nannet_router/macro.py` calls its ten address/bin mapping
labels `MAPPING_LAYERS` (bit-to-head through meta-policy). Those are routing-map
stages, not this evidence-processing chain. Do not reinterpret their numeric
IDs as these M0–M9 operations or as L0–L9 memory resolutions. The existing core
`runner/training_pipeline.py` and `runner/hermes_pipeline.py` remain
authoritative; this integration must call them rather than replace their
training lifecycle.

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
3. MC0Link and HW0 RunnerLink are interfaces, not universes.
4. No per-participant universe creation, private state, or independent training;
   HW4+ addresses are routing identities for the one HW0::BS0 universe.
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
11. NaNLMind owns the canonical build/topology contract and native binary;
    NaNROUTER is transport/control-plane only; HERMES is a read-only adapter
    and candidate producer.
12. The router role overlay (250 curriculum heads, root/control reservations)
    and the native triad field (255 members + field head 255) are views of the
    same 256 physical heads. They do not create extra heads or universes.

## 6. Migration note

Existing Hermes installations may contain projection rows produced under the
old inconsistent stage mapping or seven-layer native-event validator. Do not
silently reinterpret those rows in place. Keep immutable event payloads,
rebuild projections from canonical resolution labels, and report unmappable or
conflicting records for review. The local projection database is rebuildable;
the native global topology remains the single authoritative state.
