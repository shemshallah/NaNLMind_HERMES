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


## 3. CLOCKED structured-observation integration (design)

CLOCKED by Rezolve.ai is an external structured-reasoning API, not another NaNLMind model or a second semantic topology. Note that NaNLMind already has a class named `ClockedAdapter` in its staged training corpus pipeline; that existing adapter binds a session to the NaNLMind logical epoch/clock and is not the Rezolve.ai HTTP API client. Keep the new service adapter distinctly named (for example `ClockedApiAdapter`) and reuse the existing pipeline/conversation trainer where compatible. Its Decide, Extract, Fill, PII/Protect, Injection/Guard, Script/Inspect, Grounding/Verify, Compare, Rerank and Pipeline results should enter the same native `GlobalTopology` as typed, provenance-bound observations.

The canonical cross-repository design is maintained in
[`NaNLMind/docs/CLOCKED_HERMES_BRAINSTORM.md](https://github.com/shemshallah/NaNLMind_CLOCKED/blob/main/docs/CLOCKED_HERMES_BRAINSTORM.md).
That document defines the proposed NaNLMind_CLOCKED project, HW3/HW3:0 addressing,
the M0–M9 CLOCKED profile, exact byte/span and mirror-pair mapping, scratch
protocol, logic-first training, baselines/controls, CLI, authentication and
Router/Runner Ring/Koyeb boundaries.

### Integration invariants

- HERMES chat remains the default; CLOCKED mode is a separate interface over
  the same plugin adapter and scratch protocol.
- HW3 is the CLOCKED service identity; HW3:0 is its NANNET interface. Router
  node point 3, BSR wire IDs and internal head IDs remain distinct namespaces.
- The CLOCKED adapter uses the official `clocked-sdk` contract behind one
  narrow service boundary. NANNET credentials and CLOCKED API keys are
  independent and must never be substituted for one another.
- CLOCKED results are typed teacher observations. They are not automatically
  independently verified facts and must not be flattened into generic text.
- Preserve exact source bytes, byte offsets, field schemas, probability
  distributions, evidence spans, response digests, SDK/API version and cost
  metadata.
- Use the existing 256-head topology, 8 MiB register chart, canonical mirror
  pairs, temporal rails and M0–M9 mapping layers. Do not create a parallel
  CLOCKED brain or a competing clock.
- Logic foundation and frozen controls precede CLOCKED-specific training.
  Report teacher imitation, independent correctness and calibration separately.
- A shared scratch uses immutable revisions and writer-separated lanes. HERMES
  authors prompts; CLOCKED appends results; NaNLMind appends its own response;
  HERMES synthesizes and continues.
- Router owns HW3 route/ACL generations and Koyeb volume bindings. Koyeb is
  gateway/control/storage infrastructure, not the semantic learner.
- The six-runner ring owns distributed head execution and commits one validated
  global handoff. No partial runner state is promoted.
- Sponsored demo access requires an explicitly authorized server-side CLOCKED
  proxy and quotas. Otherwise users supply their own CLOCKED API key. NANNET
  pairing alone does not grant CLOCKED credits.

This is a design-only integration point. It does not claim that HW3:0, the
CLOCKED proxy, CLOCKED mode, the scratch bridge or CLOCKED-specific training
has been implemented or deployed.
