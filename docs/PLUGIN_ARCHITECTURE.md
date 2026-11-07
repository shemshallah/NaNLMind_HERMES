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
SDK, API key, gateway, endpoint orchestration, idempotency, encrypted result
cache and service tests belong in the dedicated
[NaNLMind_CLOCKED repository](https://github.com/shemshallah/NaNLMind_CLOCKED).
NaNLMind retains only its receiving-language schemas, scoped observation
publisher, native Boolean heads, relation learner and canonical learning
infrastructure. The NaNLMind-side files under \`integrations/clocked/\` do not
import or call the upstream CLOCKED SDK.

Do not confuse the Rezolve.ai service with NaNLMind's existing
\`ClockedAdapter\` in \`runner/hermes_pipeline.py\`, which binds a training
session to the NaNLMind logical epoch/clock. They are separate systems and
must have distinct adapter names and credentials.

HERMES is the user-facing coordinator. It asks its configured LLM to choose
among fixed CLOCKED modes using the user request and source descriptors, not
raw source bodies. HERMES maps the selected mode to a fixed endpoint allowlist;
invalid or unavailable proposals fall back to deterministic policy. Source
transmission remains approval-gated. Ordinary local chat remains local.

For multi-source tasks, HERMES uses separate typed CLOCKED calls, not a
fictional parallel CLOCKED Pipeline. Each result retains endpoint type, source
identity and provenance. A revisioned, hash-chained scratch stores user input,
plans, sanitized CLOCKED results, extracted evidence, explicit cross-source
conflicts, NaN16 packages, head responses, synthesis drafts, validation and
final output. PII, injection and script responses are allowlist-sanitized
before scratch persistence; raw sensitive values and script/prompt bodies are
not copied into those lanes.

The implemented answer cycle is:

1. HERMES proposes a mode and bounded task plan. After approval, the CLOCKED
   gateway executes the typed endpoint calls. Results and conflict signals are
   appended to shared scratch.
2. HERMES synthesizes a draft from the source evidence and endpoint results.
   It submits the evidence package and draft to the authenticated NaNLMind
   head-cortex receiving interface. NaNLMind maps CLOCKED endpoint observations,
   conflict-map entries and answer claims into typed relations and evaluates
   them with its native Boolean heads. HERMES composes the final response from
   the original evidence plus the head trace, preserving UNKNOWN and CONFLICT.
3. With separate explicit approval, HERMES can convert CLOCKED observations
   already in scratch into weak-supervision N16REL records. NaNLMind clones the
   base topology and runs native \`relation-train\` only against that candidate.
   CLOCKED outputs are marked as teacher observations, never ground truth.
   Candidate evaluation uses a caller-declared test split, verifies its digest
   and topology digests, and rejects overlap with the candidate's training
   fingerprints.

The HERMES head cortex is a lightweight client/orchestrator over the existing
256 NaNLMind logical head IDs. It does not copy the native runner implementation
or create a second six-runner ring. Native Boolean-head state, relation
learning, mirror-pair updates, temporal links and hyperedges remain owned by
NaNLMind. Candidate training writes only an isolated candidate topology; it
never changes canonical state.

Candidate evaluation reports accuracy, coverage, balanced accuracy, confusion
counts and per-resolution metrics. A positive result is only a review signal.
Automated mutation search, broader negative controls, candidate review UI and
six-runner generation-barrier promotion remain outstanding. No candidate is
canonical until that separate promotion gate exists and passes.

The detailed HERMES implementation design is in
[docs/CLOCKED_AUTONOMOUS_HEAD_CORTEX.md](CLOCKED_AUTONOMOUS_HEAD_CORTEX.md).
The cross-repository architecture, M0-M9 target profile, scratch protocol,
multi-source graph and service boundary are in
[NaNLMind_CLOCKED/docs/HERMES_AUTONOMOUS_CLOCKED_HEAD_CORTEX.md](https://github.com/shemshallah/NaNLMind_CLOCKED/blob/main/docs/HERMES_AUTONOMOUS_CLOCKED_HEAD_CORTEX.md).

Production CLOCKED credentials, live upstream end-to-end verification and
deployment of the CLOCKED gateway and NaNLMind head-cortex receiver through the
live Router/Koyeb HW3:0 route remain deployment work. The current native
head-cortex relation path is implemented and tested, but the full M0-M9
transformation sequence is not yet demonstrated end-to-end.
