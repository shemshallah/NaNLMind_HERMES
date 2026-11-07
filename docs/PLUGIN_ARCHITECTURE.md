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
predicate head. The native relation learner uses deterministic resolution-specific surface features (phonetic, morphemic, lexical, syntactic, cohesive and narrative cues); SHA-256 is reserved for stable routing, source identity and event deduplication. This is a compact feature learner, not a pretrained semantic encoder. It reports independent L0-L9 counts, pre-training accuracy,
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


## 3. Autonomous CLOCKED mode and NaN head cortex

CLOCKED by Rezolve.ai is an external structured-reasoning service. Its upstream
SDK, API key, gateway, endpoint orchestration, idempotency and service tests
belong in the dedicated
[NaNLMind_CLOCKED repository](https://github.com/shemshallah/NaNLMind_CLOCKED).
NaNLMind retains only its receiving-language schemas, NANNET evidence consumer,
native Boolean heads, M0-M9 transforms, relation learner and canonical
six-runner training/commit path.

Do not confuse the Rezolve.ai service with NaNLMind's existing
`ClockedAdapter` in `runner/hermes_pipeline.py`, which binds a training
session to the NaNLMind logical epoch/clock. They are separate systems and
must have distinct adapter names and credentials.

HERMES remains the user-facing coordinator. It autonomously proposes a
schema-bound mode/task plan, then a deterministic policy checks allowed
endpoints, source ACLs, budgets, idempotency, credentials and approval. The
default is approval; automatic CLOCKED calls require an explicit
budgeted-auto policy. Ordinary local chat remains local.

For multi-source tasks, HERMES uses separate typed CLOCKED calls, not a
fictional parallel CLOCKED Pipeline. Each result keeps its endpoint-native
type, source identity, digest, cost and provenance. A revisioned scratch
stores writer-separated lanes for user input, HERMES plans, CLOCKED results,
normalized evidence, conflict maps, NaN16 packages, head responses, synthesis
drafts, answer validation and final output.

The recommended answer cycle has two NaNLMind passes:

1. Freeze the multi-source evidence scratch, encode it as a typed NaN16
   package, and submit it through the existing NANNET/HW3:0 route. NaNLMind
   processes it through M0-M9 and the existing Boolean-derived head/relation
   learner.
2. HERMES synthesizes a natural-language draft from the evidence and relation
   graph, then submits that draft with its claim-to-source map for a second
   head-validation pass before delivery.

The HERMES head cortex is a lightweight scheduler/client over the existing
256 NaNLMind logical head IDs. It reuses HeadView/HeadRunner, HeadGate,
LogicSessionTrainer and N16REL relation learning. It does **not** clone the
six-runner ring, leader election, leases, generation barriers, Koyeb volume
administration or canonical checkpoints. HERMES's existing `runner.py`
remains a narrow local native-process wrapper.

Self-modification is candidate-only: heads may propose bounded Boolean-table,
gate, relation-feature or routing-bias mutations in an isolated sandbox.
Truth-table regressions, deep-logic circuits, held-out tasks, counterexamples,
shuffled-label controls, no-CLOCKED baselines and replay checks must pass.
Only the NaNLMind evaluator and six-runner generation barrier can promote a
canonical mutation. CLOCKED agreement is not ground truth.

The detailed HERMES implementation design is in
[docs/CLOCKED_AUTONOMOUS_HEAD_CORTEX.md](CLOCKED_AUTONOMOUS_HEAD_CORTEX.md).
The complete cross-repository specification, M0-M9 profile, scratch protocol,
multi-source graph and service boundary are in
[NaNLMind_CLOCKED/docs/HERMES_AUTONOMOUS_CLOCKED_HEAD_CORTEX.md](https://github.com/shemshallah/NaNLMind_CLOCKED/blob/main/docs/HERMES_AUTONOMOUS_CLOCKED_HEAD_CORTEX.md).

The first implementation slice is committed: HERMES now has a hash-chained
scratch journal, deterministic mode/task planner, scoped CLOCKED gateway
client, multi-source executor, evidence codec, head-cortex client and
two-pass synthesis tool. The CLOCKED service exposes the bounded `/v1/tasks`
endpoint. NaNLMind provides the read-only native `relation-eval` command and
a NANNET-authorized head-cortex HTTP adapter.

The services are not yet deployed or connected through the live Router/Koyeb
route. Live upstream SDK calls, production scratch transport and candidate
self-modification/promotion remain outstanding. The native relation-eval path
does not mutate head state; candidate mutation remains sandbox-only future
work.
