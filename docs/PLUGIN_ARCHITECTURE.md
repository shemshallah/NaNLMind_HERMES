# Plugin architecture

## Two interaction modes

Hermes exposes two distinct ways to learn. Both update the same native
`GlobalTopology`; neither creates a parallel semantic learner.

### 1. Structured teacher training

```text
/nanlm logic|words|sentences|pipeline
  -> mandatory exact Boolean foundation gate
  -> active teacher proposes typed relation records
  -> schema / evidence / polarity / resolution validation
  -> frozen topic-specific held-out set per selected L0-L9 resolution
  -> N16REL/1 train + test records
  -> native relation-train [epochs]
  -> 256 routed heads + canonical mirror-pair cells
  -> temporal links + provenance-bound n-ary hyperedges
  -> before/after held-out metrics per resolution
  -> cross-run comparison on the same frozen test digest
  -> append-only memory merge and experiment manifest
```

Teacher records are distillation labels, not independently verified facts.
The deterministic validator checks schema, bounds, source provenance, split
separation and duplicate structure. Boolean truth-table correctness remains
owned by the exact native foundation gate, not by teacher confidence.

### 2. Free-form conversational learning

```text
user <-> assistant back-and-forth
  -> durable transcript
  -> structured reflection of explicit evidence and relations
  -> keep neutral / incomplete claims as symbolic evidence only
  -> supported/refuted subject-predicate-object claims
  -> N16REL/1 online train events
  -> same persistent native GlobalTopology
  -> routed heads + canonical mirror-pair mutations
  -> temporal links + typed hyperedges with source digest
  -> native mutation receipt returned with the chat answer
```

This is enabled by `chat_learning: true` (default). It is an online evidence
update, not an automatic teacher curriculum or a claim that a full generative
language model has been fine-tuned. Replaying the same source-bound event is an
explicit no-op; a new turn has a new source digest and remains a distinct event.
Contradictory claims are retained as separate hyperedges.

## Native task and measurement boundaries

The legacy `math-train` task remains byte-bit transition prediction. Its
held-out score must never be relabeled as word, sentence or narrative accuracy.

The new `relation-train` objective predicts relation polarity from the routed
predicate head. It reports independent L0-L9 counts, pre-training accuracy,
post-training accuracy, within-run delta, balanced accuracy and majority
baseline. These are relation-polarity metrics for teacher/chat labels, not text
generation metrics or independent factual verification.

Structured training uses a frozen challenge file per topic and resolution.
When the same challenge digest is reused, Hermes reports cross-run deltas.
If the test digest changes, cross-run accuracy is marked non-comparable rather
than presented as a trend. A no-op replay still evaluates the frozen test set
against the unchanged state.

## Shared native state

The verified Boolean foundation checkpoint seeds
`native/global_universe.nmg` exactly once. Math training, structured relation
training and free-form chat learning all resume that same serialized topology.
The immutable foundation receipt is reused rather than rerunning a reset gate
on every chat turn.

Native relation events mutate:
- routed Boolean heads and their scratch scores;
- canonical mirror-pair cells, including complementary temporal rails;
- temporal links between participating heads and pair cells;
- n-ary hyperedges containing subject, predicate, object and pair references;
- the global append-only ledger and serialized topology.

The SQLite evidence ledger remains an auditable projection/event store; it is
not substituted for native head or mirror-pair state.

## Stores

| store | owner | contents |
|---|---|---|
| conversation transcripts | plugin state | JSONL user/assistant rows |
| mathematical memory | SQLite WAL | immutable discoveries, edges, hyperedges, assessments |
| frozen challenges | plugin state | per-topic/per-resolution held-out relation records |
| native topology | native runner | serialized head/cell/scratchpad state, mirror pairs, hypergraph, ledger |
| experiment manifests | plugin state | corpus, command, metrics, state digests, paths |
| NANNET identity | plugin state | encrypted Ed25519 private key and tokens |

## Trust and failure boundaries

- Provider output is untrusted. Structured teacher records carry
  `independently_fact_verified: false` unless a real verifier supplies proof.
- Free-form chat only sends explicit supported/refuted triples to native
  learning. Neutral or incomplete records are not coerced into binary labels.
- Native learning is reported only after the runner confirms changed heads,
  mirror-pair updates, hyperedges and a changed state digest.
- Exact replay returns `no_op`; it is not reported as new learning.
- A missing runner leaves the chat answer intact but reports native learning
  failure; it never fabricates a successful update.
- NANNET authorizes and coordinates; it is not the local trainer or corpus store.

