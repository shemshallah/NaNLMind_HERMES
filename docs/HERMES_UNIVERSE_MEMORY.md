# Hermes universe memory and trainer architecture

## The object being trained

There is one native `GlobalTopology`, not 256 independent models. Its 256 palindromic `HeadView` instances address head records, cells and scratchpads in the same shared root `Universe`. The head index is a routing/learning address, not a separate neural network.

The native `math-train` command remains a bounded byte-bit transition task. Its held-out score is a byte control, not lexical or sentence accuracy.

The native `relation-train` command is a separate typed relation-polarity objective. It consumes source-bound `N16REL/1` records with train/test splits and a resolution tag L0-L9. It routes each relation through three distinct heads, updates their Boolean score tables, mutates three canonical mirror-pair cells, adds temporal links, and appends a six-member n-ary hyperedge containing the subject/predicate/object heads and their pair cells. It reports pre-training and post-training held-out relation-polarity accuracy, balanced accuracy, majority baseline and delta independently for every populated resolution. These scores are not text-generation accuracy and teacher labels are not independently verified facts.

The native topology serializer persists the global chart and its Universe contents, including the per-head scratchpads where Boolean scores are stored, the shared hypergraph, and the append-only ledger. Each trained head adds a learned hyperedge connecting its fixed-point head cell to its adjacent pair-cell address; edge weight is the held-out accuracy in Q16.16. A final ledger event commits the before/after hypergraph digests. A subsequent invocation restores the serialized topology before applying new updates. Hermes reports this as a `serialized_global_topology` checkpoint; it must not be confused with a conventional neural-weight file or with the legacy Context2 bootstrap, which remains in-memory.

## What is stored

The durable substrate has two complementary stores:

1. **Native global state** — the serialized NaNLMind topology, including head/cell state and scratchpad scores. This is the live computational object that the native trainer resumes.
2. **Additive discovery ledger** — immutable, content-addressed mathematical events, source digests, provenance, confidence, evidence, typed graph edges and n-ary hyperedges. This preserves discoveries and contradictions across runs and peer merges.

The ledger is not a neural-weight checkpoint. Equations and operator expressions are inert structured data; they are never executed when ingested.

## What a head contributes

Each head contributes a candidate rule and evidence for a defined byte-bit task. The head index and palindromic address remain attached to the discovery. All 256 heads are required for a native math-training trace; incomplete or duplicate head traces and discoveries without valid native hyperedge IDs are rejected before merge. The native topology's head count is fixed at 256; Hermes does not permit a reduced head count for native discoveries. Reorganization preserves each discovery's native head attribution rather than replacing it with a subject hash.

## Merge protocol

A merge is set union over immutable event IDs, followed by deterministic projection rebuild:

- validate schema, source digest, payload bounds, head/layer range, and provenance;
- accept only signed/authorized deltas at the NANNET boundary;
- deduplicate by event ID;
- retain conflicting claims and their evidence;
- rebuild projections from the complete event set;
- evaluate before promoting any derived conclusion.

This is an associative, commutative, idempotent merge at the event-set layer. It is not a last-writer-wins register and not an averaging of fictitious weights. Remote nodes exchange compact deltas (new events plus base/root digest), never corpora or native process memory by default. Hermes `nanlm_memory_merge` performs the local idempotent event-set union; authenticated transport and global generation/base-root reconciliation remain the NANNET merge boundary's responsibility.

## Continuous reorganization

Reorganization does not mutate discoveries. The linguistic substrate has ten independent resolution coordinates: L0 bit, L1 nibble, L2 byte, L3 phonetic, L4 morpheme, L5 lexical, L6 word completion, L7 sentence, L8 paragraph and L9 narrative. These are not aliases for seven semantic buckets. Native head provenance is retained when claims are rerouted. The evidence assessor aggregates support/refutation weights by subject; contradictory subjects are promoted to L9 and receive a versioned assessment record with contributing event IDs. Assessment records are rebuildable projections, not new claims. This is currently deterministic evidence review, not yet a learned top-down evaluator that changes lower-layer training. Future head/layer scores may influence routing only when the scoring algorithm and version are recorded.

A contradiction is a first-class relation. The evaluator can compare conditions, derive counterexamples, lower confidence, or mark a claim superseded in a separate assessment event. It cannot erase the earlier claim.

## Current training loops

Structured teacher path:

`teacher -> typed L0-L9 records -> schema/evidence/split validation -> frozen topic/resolution challenge -> N16REL/1 -> native relation-train -> heads + mirror pairs + temporal links + hyperedges -> pre/post metrics -> append-only memory merge`

Free-form chat path:

`user/assistant exchange -> structured reflection -> supported/refuted explicit triples -> N16REL/1 online batch -> same native relation-train -> shared topology commit -> mutation receipt`

Both paths use the same `native/global_universe.nmg`. The verified Boolean foundation checkpoint seeds that shared state exactly once. The foundation gate is cached after a verified pass so it cannot reset the learner on every chat turn or training round.

The legacy byte-bit trainer remains available as a separate control. It is no longer used as a proxy for lexical, word-completion, sentence, paragraph or narrative learning.

## Persistence and authority

- Native serialized topology: resumable computational state.
- SQLite WAL event ledger: durable additive discoveries and evidence.
- Rebuildable projections: disposable routing views, never the source of truth.
- Native chart integration: each head's learned relation is now inserted into the native hypergraph and committed through the global ledger; Hermes separately preserves the structured evidence event and native hyperedge reference in its additive event ledger.
- NANNET: authorization and delta coordination only; Koyeb is not the trainer or corpus store.

## Required gates

1. Deterministic event IDs and duplicate replay.
2. Contradiction retention.
3. Merge order/permutation equivalence.
4. Reorganization idempotence.
5. Crash/reopen durability.
6. Held-out evaluation proves no train/test leakage.
7. Native trainer emits exactly 256 unique, source-bound head discoveries.
8. A second native run restores and extends the same serialized global topology.
9. Every future native hypergraph mutation is represented in the global ledger.

Do not describe the current Boolean transition task as general symbolic mathematics or claim that seven-layer feedback training is complete.

## Resolution-aware routing (shared by chat and runners)

Every accepted discovery may carry `provenance.resolution`. Reorganization maps
that resolution into the same seven-layer projection for every producer:

| Layer | Resolution(s) |
|---|---|
| 0 | bit, nibble, byte |
| 1 | phonetic |
| 2 | morpheme |
| 3 | lexical (roots, prefixes, suffixes, word identity) |
| 4 | word completion |
| 5 | sentence |
| 6 | paragraph, narrative/story |

The resolution tag is routing metadata, not a claim that the native Boolean
trainer has learned every linguistic scale. The append-only event and its source
provenance remain unchanged; only the deterministic active projection is rebuilt.
Unknown or absent resolution tags retain the semantic-kind default. Contradictions
still move to the highest review layer while retaining their origin head.

## Android / Termux dependency profile

The base package intentionally has no third-party runtime dependencies. Chat,
N16TXT processing, SQLite mathematical memory, corpus generation and the local
runner adapter use the Python standard library. Install `cryptography` only on a
platform where its wheel/build is supported, using `pip install -e ".[auth]"`.
The NANNET identity, signature, token and encrypted-payload commands require that
optional extra; they fail as unavailable functionality rather than preventing the
Hermes chat/training plugin from importing. Do not replace cryptographic
operations with homemade or unauthenticated substitutes on Android.

## Shared routing for chat, extraction and native runners

All producers append events to the same immutable memory ledger. Conversational
reflection and corpus extraction may attach a linguistic `provenance.resolution`
tag, which selects the corresponding language layer. Native 256-head runner
discoveries retain their explicit native layer and head during reorganization;
they are not silently collapsed to the semantic-kind default. When evidence
for one subject conflicts, the review projection still promotes both sides to
the highest review layer while preserving original head/layer provenance.

The active Hermes model can now generate an twelve-stage curriculum corpus through
`nanlm_generate_curriculum`: Boolean inverses, deep logic, exceptions,
repetitive/common words, consonant/vowel phonetic pairings, morpheme structure,
individual lexical words with roots and affixes, word-completion alternatives,
sentence choice, paragraph structure and short narratives. Short stories also
appear immediately after deep logic to saturate the language field early. It writes canonical N16TXT/1 plus a content-addressed manifest.
This creates training material; it does not by itself prove that the native
256-head trainer has learned language. The native trainer remains responsible
for its own measured held-out results and committed hyperedges.

## Deterministic surface measurements from conversation

Each successful chat turn appends exact, source-digest-bound measurements for
UTF-8 bit counts, nibble frequencies, byte frequencies, orthographic token
frequencies, sentence segmentation and paragraph segmentation. These are
observations of the actual user/assistant exchange, not inferred semantics.
They route into the shared bit/nibble/byte, lexical, sentence and paragraph
layers. Phonetic, morphemic, word-completion and narrative interpretation stays
in the separate structured language-model reflection pass and is explicitly
treated as a hypothesis when it is not directly observable.

## Read/write loop: runners and chat experience the same universe

Both curriculum generation and conversational reflection receive a compact,
read-only snapshot of active claims and evidence assessments from the shared
SQLite universe before proposing new material. Curriculum prompts select
context from the layer relevant to the current stage; chat reflection receives
a bounded cross-layer context. Their outputs are then appended to the same
event ledger and deterministically reorganized. Native 256-head runner
discoveries also preserve their native head/layer routes in that projection.
This is a shared read/append/reorganize interface, not a claim that the native
C++ learner directly executes SQL or has absorbed every symbolic memory item
into its internal parameters.


## Cross-resolution mathematical feedback

Every discovery may carry a resolution label and, when appropriate, a stable `alignment_id` or `concept_id` in provenance. The same concept can therefore be represented at several granularities without pretending that its bit-level, phonetic, lexical and narrative descriptions are identical data.

The cross-resolution analyzer constructs a symmetric 10×10 co-occurrence matrix over the ten resolution coordinates. For each aligned concept it compares cross-level pairs, records polarity agreement and support/refutation contradictions, computes polarity entropy, and calculates the descriptive consistency score `(aligned - contradicted) / (aligned + contradicted)` (zero when no comparable pair exists). Neutral pairs remain visible but do not count as support/refutation agreement.

Each multi-resolution concept produces a deterministic equation record with an n-ary hyperedge whose participants are the original immutable event IDs. The record stores pair evidence, resolution indices, coverage and score; it is routed to the tenth review channel. It never replaces, averages away or deletes the original claims. Explicit alignment IDs can join layer-specific subjects; without one, equal subjects are compared. The operation is invoked after Hermes chat reflection and native math training, and is callable by runners through `nanlm_memory_cross_resolution`.

This creates several useful mathematical feedback paths: coarse-to-fine constraint checking, fine-to-coarse evidence aggregation, cross-level contradiction detection, resolution coverage measurement, and uncertainty/entropy tracking. It is a deterministic structural analysis layer, not a learned differentiable gradient or proof of semantic equivalence. The next experiment should test whether feedback scores predict held-out word, sentence and narrative improvements.

## Native relation record protocol

`N16REL/1` is a UTF-8 file with tab-separated fields:

`split, resolution, polarity, confidence_q16, subject_hex, predicate_hex, object_hex, source_sha256`.

Only `train` rows mutate state. `test` rows are measured before and after
training. The source digest is part of the event identity: exact replay is
idempotent, while a later conversational turn that supplies the same relation
from a different source remains a separate evidence event. Contradictions are
retained as separate hyperedges.

The ten resolutions remain distinct in the native report:
bit, nibble, byte, phonetic, morpheme, lexical, word completion, sentence,
paragraph and narrative. Missing test coverage is not reported as zero or
copied from another resolution.

