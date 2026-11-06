# Hermes universe memory and trainer architecture

## The object being trained

There is one native `GlobalTopology`, not 256 independent models. Its 256 palindromic `HeadView` instances address head records, cells and scratchpads in the same shared root `Universe`. The head index is a routing/learning address, not a separate neural network.

The native `math-train` command currently trains a bounded, explicit task: each head learns a Boolean truth-table rule for a byte-bit transition, using a deterministic 80/20 sequential train/held-out split. It emits one JSONL discovery per head with the source SHA-256, head address, selected Boolean rule, all 16 operator scores, sample counts and held-out accuracy. This is a first working native learning path, not yet a general mathematical reasoner or a full seven-layer learning loop.

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

Reorganization does not mutate discoveries. It creates a deterministic active projection: observations L0, entities L1, relations L2, operators L3, equations L4, hypotheses and constraints L5, and counterexamples L6. Native head provenance is retained when claims are rerouted. The evidence assessor aggregates support/refutation weights by subject; contradictory subjects are promoted to L6 and receive a versioned assessment record with contributing event IDs. Assessment records are rebuildable projections, not new claims. This is currently deterministic evidence review, not yet a learned top-down evaluator that changes lower-layer training. Future head/layer scores may influence routing only when the scoring algorithm and version are recorded.

A contradiction is a first-class relation. The evaluator can compare conditions, derive counterexamples, lower confidence, or mark a claim superseded in a separate assessment event. It cannot erase the earlier claim.

## Current training loop

`corpus -> native 256-head trainer -> per-head JSONL discoveries -> schema/source/head validation -> append-only event merge -> deterministic layer reorganization -> held-out metrics`

The native trace is now a stable structured-discovery stream for the Boolean transition task. The next research step is to extend the native trainer to emit richer typed operators, equations, hyperedges and explicit evaluation/counterexample events, then add cross-layer evaluation and bounded feedback.

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

The active Hermes model can now generate a eleven-stage curriculum corpus through
`nanlm_generate_curriculum`: Boolean inverses, deep logic, exceptions,
repetitive/common words, phonetic/morpheme patterns, roots and affixes,
word-completion alternatives, sentence choice, paragraph structure and short
narratives. It writes canonical N16TXT/1 plus a content-addressed manifest.
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
