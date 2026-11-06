# Hermes universe memory and trainer architecture

## The object being trained

There is one native `GlobalTopology`, not 256 independent models. Its 256 palindromic `HeadView` instances address head records, cells and scratchpads in the same shared root `Universe`. The head index is a routing/learning address, not a separate neural network.

The native `math-train` command currently trains a bounded, explicit task: each head learns a Boolean truth-table rule for a byte-bit transition, using a deterministic 80/20 sequential train/held-out split. It emits one JSONL discovery per head with the source SHA-256, head address, selected Boolean rule, all 16 operator scores, sample counts and held-out accuracy. This is a first working native learning path, not yet a general mathematical reasoner or a full seven-layer learning loop.

The native topology serializer persists the global chart and its Universe contents, including the per-head scratchpads where Boolean scores are stored. A subsequent invocation restores the serialized topology before applying new updates. Hermes reports this as a `serialized_global_topology` checkpoint; it must not be confused with a conventional neural-weight file or with the legacy Context2 bootstrap, which remains in-memory.

## What is stored

The durable substrate has two complementary stores:

1. **Native global state** — the serialized NaNLMind topology, including head/cell state and scratchpad scores. This is the live computational object that the native trainer resumes.
2. **Additive discovery ledger** — immutable, content-addressed mathematical events, source digests, provenance, confidence, evidence, typed graph edges and n-ary hyperedges. This preserves discoveries and contradictions across runs and peer merges.

The ledger is not a neural-weight checkpoint. Equations and operator expressions are inert structured data; they are never executed when ingested.

## What a head contributes

Each head contributes a candidate rule and evidence for a defined byte-bit task. The head index and palindromic address remain attached to the discovery. All 256 heads are required for a native math-training trace; incomplete or duplicate head traces are rejected before merge. The native topology's head count is fixed at 256; Hermes does not permit a reduced head count for native discoveries.

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

Reorganization does not mutate discoveries. It creates a deterministic active projection: observations L0, entities L1, relations L2, operators L3, equations L4, hypotheses and constraints L5, and counterexamples L6. Routing is recomputed from the complete immutable event set. This is currently a deterministic structural reorganization, not yet a learned top-down evaluator that changes lower-layer training. Future head/layer scores may influence routing only when the scoring algorithm and version are recorded.

A contradiction is a first-class relation. The evaluator can compare conditions, derive counterexamples, lower confidence, or mark a claim superseded in a separate assessment event. It cannot erase the earlier claim.

## Current training loop

`corpus -> native 256-head trainer -> per-head JSONL discoveries -> schema/source/head validation -> append-only event merge -> deterministic layer reorganization -> held-out metrics`

The native trace is now a stable structured-discovery stream for the Boolean transition task. The next research step is to extend the native trainer to emit richer typed operators, equations, hyperedges and explicit evaluation/counterexample events, then add cross-layer evaluation and bounded feedback.

## Persistence and authority

- Native serialized topology: resumable computational state.
- SQLite WAL event ledger: durable additive discoveries and evidence.
- Rebuildable projections: disposable routing views, never the source of truth.
- Native chart integration: discoveries are currently persisted in the native state and separately in Hermes' event ledger; direct native hypergraph insertion of each event remains a subsequent integration step.
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
