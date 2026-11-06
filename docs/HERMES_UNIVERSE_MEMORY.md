# Hermes universe memory and trainer architecture

## Why this is not a weight file

The current native `bootstrap` path trains an in-memory Context2Model and exits without serializing it. A log, corpus digest, or run manifest is not a checkpoint. Hermes therefore must not label these artifacts as learned persistent state.

The durable learning substrate is an additive mathematical memory:

1. **Discovery event** — a typed claim, equation, operator, constraint, observation, hypothesis, or counterexample.
2. **Provenance** — source corpus digest, run ID, head, layer, derivation, and evidence.
3. **Typed graph** — entities and relations represented as edges/hyperedges, with equations retained as inert structured expression trees.
4. **Projection** — deterministic routing of event IDs into head/layer views. Projections are disposable and rebuildable.
5. **Evaluation** — held-out tests add evidence events; they do not overwrite old claims.

## What a head contributes

A head contributes candidate discoveries and routing/attention metadata, not a private model. The current core repository defines 256 palindromic heads, not 255. Hermes must read the active head count from the native universe when that API is available; until then it supports a configurable count (default 256) and must never silently drop a head.

Each discovery is attributed to a head and layer. The stable event ID is SHA-256 over canonical payload plus source digest. Repeated delivery is idempotent. Two contradictory claims remain separate events with opposite polarity or distinct conditions; confidence is evidence metadata, not an overwrite priority.

## Merge protocol

A merge is set union over immutable event IDs, followed by deterministic projection rebuild:

- validate schema, source digest, payload bounds, head/layer range, and provenance;
- accept only signed/authorized deltas at the NANNET boundary;
- deduplicate by event ID;
- retain conflicting claims and their evidence;
- rebuild projections from the complete event set;
- evaluate before promoting any derived conclusion.

This is an associative, commutative, idempotent merge at the event-set layer. It is not a last-writer-wins register and not an averaging of fictitious weights. Remote nodes exchange compact deltas (new events plus base/root digest), never corpora or native process memory by default.

## Continuous reorganization

Reorganization does not mutate discoveries. It creates a new active projection: observations near L0, entities L1, relations L2, operators L3, equations L4, hypotheses/constraints L5, and counterexample/critical review at the highest layer. The projection function is deterministic and versioned. Future native head/layer scores may influence routing only if their scoring algorithm and version are recorded; provenance remains immutable.

A contradiction is a first-class relation. The evaluator can compare conditions, derive counterexamples, lower confidence, or mark a claim superseded in a separate assessment event. It cannot erase the earlier claim.

## Training loop

`corpus -> native runner -> candidate extraction -> event validation -> append -> merge -> reorganize -> held-out evaluation -> evidence events`

The extractor is a separate adapter because the current native CLI does not yet expose a stable structured-discovery stream. Until that interface exists, Hermes can ingest explicit JSON discovery records, but must not pretend that arbitrary stdout text has been converted into mathematics.

## Persistence layers

- SQLite WAL event ledger: durable local additive memory and idempotent ingest.
- Rebuildable projections: head/layer routing, never the source of truth.
- Native chart/NaN16 integration: future adapter to commit validated event IDs and graph edges into the single global topology.
- NANNET: authorization and delta coordination only; Koyeb is not the trainer or corpus store.

## Required gates

1. Deterministic event IDs and duplicate replay test.
2. Contradiction retention test.
3. Merge order/permutation equivalence test.
4. Reorganization idempotence test.
5. Crash/reopen durability test.
6. Held-out evaluation proves no train/test leakage.
7. Native adapter proves every committed graph mutation is represented in the global ledger.

Until native serialization and the adapter are complete, report `memory_persistent=true` only for the event ledger and `native_checkpoint_present=false` for Context2Model.
