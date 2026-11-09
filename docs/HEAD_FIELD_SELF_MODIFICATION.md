# 256-head field and scratch-first self-modification

**Status:** implementation specification. This document is normative for new
work on the local 256-head field. It does not itself change runtime behavior.

**Scope:** the HERMES-side, derived 256-head field, its per-head L0-L9 scratch
state, and the only permitted path from scratch to the local main register.

**Out of scope and untouched:** the native NaNLMind topology, the six-runner
canonical promotion barrier, canonical BS0/MC0 state, and any canonical merge.
The local main register described here is a derived, unmerged store.

Normative terms `MUST`, `MUST NOT`, `SHOULD` and `MAY` are used in the usual
sense. A requirement is not satisfied by documentation, a boolean flag, or a
caller's assertion; it must be enforced by code and covered by a test.

---

## 0. One-page model

```text
per-head L0-L9 scratch (immutable records, bounded rings)
        |
        |  stage candidate: one head only, base generation pinned
        v
candidate in scratch  ---> Boolean evaluation (16 masks, exact 64/64)
        |                         |
        | fail                    | pass
        v                         v
     DENIED                 explicit approval record
                                  |
                                  | approval digest + base generation
                                  v
                         generation compare-and-swap
                                  |
                         stale ---+--- current
                           |              |
                           v              v
                        DENIED      main-register commit
                                          |
                                          v
                                  one new generation
```

The field loop is separate from the commit loop:

```text
per-head local rule/score/activation
        -> bounded classical field term
        -> deterministic bucket router
        -> many heads active per context (never one global head)
        -> per-head scratch updates
```

Self-modification never bypasses the order:

```text
scratch-stage
  -> Boolean evaluation
  -> explicit approval
  -> generation CAS
  -> main-register commit
```

No stage may skip ahead. No field score, corpus run, chat turn, metric, or
caller convenience may reverse this order.

---

## 1. Current implementation baseline and gaps

The repository already contains parts of the target surface. The specification
distinguishes what exists from what must be added.

| Surface | Current file | Current status | Required by this spec |
|---|---|---|---|
| 256 head records and 32,640 pair cells | `nanlmind_hermes/resident/cells.py` | implemented: permanent `head_id`/`identity`, Q16.16 score/state, per-head generation, pair rails | remain the hot projection; never the self-modification authority |
| 256-head activation field | `nanlmind_hermes/resident/head_field.py` | partial: additive per-head score, normalized activation vector, entropy/active-head metrics, pair coupling, scratch proposals with `approved=True` | add per-head rule/score/activation learner state and a bounded coupled field term; route commits through the new pipeline |
| L0-L9 per-head scratch | `nanlmind_hermes/resident/scratch_layers.py` | partial: 256 x 10 independent bounded rings, canonical JSON values, staged tokens, generation CAS, in-memory main backend | add semantic layer names, total budget enforcement, per-head and field snapshots, digest-bound approval, rollback and crash recovery |
| Boolean base | `nanlmind_hermes/boolean_wave.py` | implemented: 16 multilinear functions, exact on `{0,1}`, `verify_exact()` 64/64 | freeze; use as the only Boolean evaluator in candidate admission |
| 255-head layered ensemble | `nanlmind_hermes/layered_learning.py` | implemented: L0-L9 projections, 255 content heads, scratch register, relation promotion | add 256-slot field routing and per-head scratch hooks; do not treat its ensemble as a second main register |
| Sparse 256-head next-letter baseline | `nanlmind_hermes/head_population.py`, `docs/HEAD_POPULATION.md` | implemented; documents rank-1 collapse when all 255 heads train on every context | use sparse, bucketed multi-head routing; keep head 0 as a bounded prior |
| Activation benchmark | `scripts/head_field_benchmark.py` | implemented: synthetic CPython benchmark over a 2 MiB resident-cell buffer, four activation modes, entropy/collision metrics, explicit limitations | use as a diagnostic, not a training or native-performance proof |
| Existing tests | `tests/test_resident_head_field.py`, `tests/test_resident_scratch_layers.py`, `tests/test_boolean_wave.py`, `tests/test_layered_learning.py`, `tests/test_head_population.py` | passing prototypes | extend with the acceptance matrix in section 9 |

### 1.1 Gaps that prevent claiming the target contract

1. `ResidentHeadField.activate` writes the hot head score without a Boolean
   evaluation or a generation-fenced main commit. It is a runtime accumulator,
   not the learner-state write path.
2. `ResidentHeadField.propose` / `apply_proposal` use `approved=True` as the
   approval test. That is a caller-controlled boolean, not an approval record
   bound to a candidate digest, an evaluation digest and a base generation.
3. `ScratchLayers.commit_staged` has a correct generation CAS pattern, but its
   approval test is also a bare boolean and it does not run Boolean evaluation.
4. There is no per-head learned rule, calibrated score policy, or activation
   policy. `cells.py` stores a score and a state word, not a rule.
5. There is no coupled field term. `field_activation()` normalizes existing
   positive scores; it does not compute a mean-field coupling.
6. `field_snapshot()` is a bounded view of scratch values. There is no
   immutable per-head snapshot with a digest chain, no field snapshot bound to
   a head root, and no rollback target.
7. The total memory budget is not enforced across geometry and payload size.
   A `max_value_bytes` of 64 KiB would exceed 2 GiB at full ring occupancy.
8. There is no corpus/free-chat integration that demonstrates many heads are
   active rather than one global head.
9. There is no secret/path/raw-pointer validator on candidate, approval or
   snapshot payloads.
10. There is no main-register crash-recovery contract for this path.

These gaps are implementation work, not reasons to weaken the contract.

---

## 2. Non-negotiable Boolean logic base

### 2.1 The 16 functions

The canonical evaluator is `nanlmind_hermes/boolean_wave.py`. The truth table
is indexed by `k = 2a + b`:

```python
def table(mask):
    return tuple((mask >> k) & 1 for k in range(4))

def boolean(mask, a, b):
    return (mask >> ((a << 1) | b)) & 1

def multilinear(mask, a, b):
    f00, f01, f10, f11 = table(mask)
    return (f00 * (1 - a) * (1 - b)
            + f01 * (1 - a) * b
            + f10 * a * (1 - b)
            + f11 * a * b)
```

The 16 masks are complete and mandatory. The truth tuples below are in
`(f(0,0), f(0,1), f(1,0), f(1,1))` order.

| mask | truth tuple | name | exact extension |
|---:|---|---|---|
| 0 | 0,0,0,0 | ZERO | `0` |
| 1 | 1,0,0,0 | NOR | `(1-a)(1-b)` |
| 2 | 0,1,0,0 | NOT A AND B | `(1-a)b` |
| 3 | 1,1,0,0 | NOT A | `1-a` |
| 4 | 0,0,1,0 | A AND NOT B | `a(1-b)` |
| 5 | 1,0,1,0 | NOT B | `1-b` |
| 6 | 0,1,1,0 | XOR | `a + b - 2ab` |
| 7 | 1,1,1,0 | NAND | `1-ab` |
| 8 | 0,0,0,1 | AND | `ab` |
| 9 | 1,0,0,1 | XNOR | `1-a-b+2ab` |
| 10 | 0,1,0,1 | B | `b` |
| 11 | 1,1,0,1 | A implies B | `1-a+ab` |
| 12 | 0,0,1,1 | A | `a` |
| 13 | 1,0,1,1 | B implies A | `1-b+ab` |
| 14 | 0,1,1,1 | OR | `a + b - ab` |
| 15 | 1,1,1,1 | ONE | `1` |

`NOT` is the unary complement `1-a`; masks 3 and 5 are the two-input
projections that realize it on one argument.

### 2.2 Exactness requirement

For every mask in `0..15` and every `(a, b)` in `{0,1} x {0,1}`:

```text
multilinear(mask, a, b) == boolean(mask, a, b)
```

That is 16 x 4 = 64 mandatory cases. The implementation MUST call
`verify_exact()` and MUST require `correct == 64`, `total == 64`, `exact is
True` before any candidate can be approved.

The fuzzy extension on `[0,1]` is a scoring device, not a new Boolean algebra:

- `OR(0.5, 0.5) = 0.75` and `AND(0.5, 0.5) = 0.25`; neither is idempotent at
  non-vertices.
- `XOR(a, a) = 2a - 2a^2`, which is zero only at `a = 0` or `a = 1`.
- Magnitude carries logic; phase is a separate temporal channel. There is no
  phase-preserving Boolean algebra claim.

### 2.3 XOR/AND/OR base

The mandatory anchor masks are:

```text
XOR = 6
AND = 8
OR  = 14
```

They are the reference base because the project depends on their exact
truth tables and their multilinear extensions. They are not, by themselves,
functionally complete without a constant or a complement; this document does
not claim otherwise. Completeness comes from retaining all 16 explicit masks,
not from a learned approximation of them.

Every composite head rule MUST reduce to these exact masks at its leaves. A
candidate may add, remove or reorder nodes inside its allowed rule graph, but
it MUST NOT change:

- the mask numbering;
- the meaning of mask 6, 8 or 14;
- the 64-case exactness obligation;
- the global Boolean evaluation order.

### 2.4 Boolean admission rules

The following are hard gates. They are checked before approval is even
possible.

| ID | Requirement |
|---|---|
| B-1 | `verify_exact()` must return 64/64 with `exact=True`. |
| B-2 | XOR(6), AND(8) and OR(14) truth tables and extensions must match the table above. |
| B-3 | A candidate MUST NOT change the global Boolean base digest. |
| B-4 | For `score`, `activation` and `rollback` scopes, the candidate's hard `TRUE`/`FALSE` output must equal the base output on the mandatory suite. For `rule` scope, the candidate's declared truth table must be exactly reproducible from its canonical mask DAG; a rule candidate may change a head's Boolean decision only through that evaluated table, never through the field term. |
| B-5 | The field term may not resolve `UNKNOWN` into `TRUE`/`FALSE` or collapse `CONFLICT` into a single value. A rule candidate must preserve and report `UNKNOWN`/`CONFLICT` cases explicitly. |
| B-6 | The field term MUST NOT be an input to the hard Boolean result. It is applied only after the result is computed. |
| B-7 | The candidate must be deterministic: same input, same rule state, same field-off result. Time, random values, process identity, locale and filesystem state must not affect the Boolean result. |
| B-8 | The candidate must not skip, short-circuit, or downgrade any of the 16 masks, including masks it does not use in its own output rule. |

The base digest is computed once from the immutable table and stored in the
main-register header:

```text
BOOLEAN_BASE_DIGEST =
  sha256(canonical_json([(mask, table(mask)) for mask in range(16)]))
```

A commit whose stored base digest differs from the current evaluator's digest
MUST be denied with `boolean_base_changed`. Migrating the base is a separate
explicit operation, never a candidate mutation.

### 2.5 How new field logic cannot replace or weaken the base

A head rule is a bounded DAG:

```text
HeadRule := {
  version: integer,
  nodes: [RuleNode],          # 1..16 nodes
  output: node_id,
  table_digest: sha256
}

RuleNode := {
  id: integer,
  mask: 0..15,
  left: FeatureRef | node_id,
  right: FeatureRef | node_id,
  thresholds: [Q16.16, Q16.16]   # only for FeatureRef inputs
}
```

This permits new compositions while keeping every leaf canonical. A
composition is not a new truth primitive; its truth table must be reproducible
from the canonical 16 masks.

The field term is not part of this DAG at all. It is an activation bias:

```text
activation_i = clip(local_activation_i + field_impulse_i, 0, 1)
```

It can change which heads participate, but it cannot:

- alter a rule node's mask;
- alter a hard Boolean output;
- turn `UNKNOWN` into `TRUE` or `CONFLICT` into `FALSE`;
- replace a mandatory mask evaluation;
- make the 64-case suite optional;
- lower the evaluation report's required coverage.

The strongest statement is structural: **there is no code path from the field
term to the Boolean result**. A field score is consulted only by the router and
the activation vector. The Boolean evaluator receives the rule DAG, the
binarized feature inputs and the exact mask table; it never receives `J`,
`lambda`, `entropy`, `resonance`, or a corpus score.

---

## 3. Per-head independent learners and the coupled field term

### 3.1 Head IDs and roles

The field has exactly 256 logical head IDs, `0..255`.

| ID range | Role | Corpus training | Identity |
|---|---|---|---|
| 0 | void/prior anchor | excluded; fixed prior rule | permanent; never renumbered or replaced |
| 1..255 | content learners | permitted through scratch candidates | permanent; one ID per learner |

The existing 255-head ensemble in `layered_learning.py` indexes the content
heads `1..255`. The field adds head 0 explicitly. Do not renumber the existing
ensemble, and do not treat the ensemble's aggregate distribution as a 256th
head.

All 256 IDs are stateful. Head 0 is a stateful prior with a fixed rule; it is
not a global distribution and it must not dominate content activation.

### 3.2 Per-head learner state

Each head owns one `HeadLearnerState` record. The record is independent: it
does not contain references to another head's mutable state.

```json
{
  "schema": "nanlm.head-learner-state/1",
  "head": 42,
  "role": "content",
  "generation": 17,
  "rule": {
    "schema": "nanlm.head-rule/1",
    "version": 3,
    "nodes": [
      {"id": 0, "mask": 8, "left": {"feature": 0, "threshold": 32768},
       "right": {"feature": 1, "threshold": 32768}}
    ],
    "output": 0,
    "table_digest": "64-hex-sha256"
  },
  "score": {
    "value_word": 7864,
    "bias_word": 0,
    "calibration": {"kind": "identity", "version": 1},
    "updated_tick": 123456
  },
  "activation": {
    "threshold_word": 26214,
    "temperature_word": 32768,
    "bucket_bias_word": 0,
    "usage": 12,
    "reward_word": 655
  },
  "layer_roots": {
    "L0": "64-hex-sha256",
    "L1": "64-hex-sha256",
    "L2": "64-hex-sha256",
    "L3": "64-hex-sha256",
    "L4": "64-hex-sha256",
    "L5": "64-hex-sha256",
    "L6": "64-hex-sha256",
    "L7": "64-hex-sha256",
    "L8": "64-hex-sha256",
    "L9": "64-hex-sha256"
  },
  "provenance": {
    "created_tick": 123000,
    "updated_tick": 123456,
    "source_ids": ["corpus:sha256:..."]
  },
  "state_digest": "64-hex-sha256"
}
```

Rules for this record:

- `head` is immutable. A candidate that changes it is denied.
- `role` is immutable after genesis.
- `rule` is data, never executable code. No Python source, bytecode, shell
  text, URL, pointer or object reference is permitted.
- `score.value_word`, thresholds, temperatures and rewards are signed or
  unsigned Q16.16 integers. Persisted numbers are integers, not floats, so a
  replay produces identical bytes.
- `layer_roots` bind the head to its ten scratch layers by digest.
- `state_digest` is the SHA-256 of the canonical JSON of all fields except
  `state_digest` itself.
- A missing or malformed field fails closed; it does not default to a global
  distribution.

### 3.3 Local rule, score and activation

Each head computes three distinct things. They must not be conflated.

**Rule.** The rule answers a Boolean or bounded truth-table question. It is
evaluated with canonical masks only. Its output is one of `TRUE`, `FALSE`,
`UNKNOWN`, `CONFLICT`, plus a rule-trace digest.

**Score.** The score is a bounded learner preference, not a probability and not
truth. A content head updates its own score only from its own selected
observations:

```text
score_word_i <- clip(
    score_word_i + eta_word * (reward_word_i - baseline_word_i),
    -32768, 32767
)
```

`reward_word_i` is derived from a declared local objective (for example, a
next-letter hit or a relation-support count). It is not a semantic truth label.
`eta_word` is bounded by the head's activation policy. Head 0 is never updated
by this rule.

**Activation.** The activation is the head's participation weight:

```text
local_i = sigmoid(
    (score_i + context_bias_i - threshold_i) / temperature_i
)
local_i = clip(local_i, 0, 1)
```

Each head owns its own `threshold_i`, `temperature_i`, `bucket_bias_i`, usage
counter and reward accumulator. There is no global threshold and no global
temperature.

**Isolation contract.** A head update may read:

- the input features for the current observation;
- its own rule, score and activation state;
- the public field vector (activation values only).

It may not read another head's rule, score, activation policy, scratch payload,
provenance or snapshot. It may not write another head's state. The field
aggregator reads only the public activation vector; it does not read private
scratch bodies.

### 3.4 Coupled field term

The field is a classical, deterministic mean-field coupling over the public
activation vector. It is not a global head, not a semantic memory and not a
quantum process.

Let `a_i` be the local activation of head `i`, `N = 256`, and `J` a sparse
symmetric coupling matrix with `J_ii = 0`.

```text
mean = (1 / N) * sum_j a_j

impulse_i = clip(
    (1 / max(1, degree_i)) * sum_{j in neighbours(i)}
        J_ij * (a_j - mean),
    -1, 1
)

field_i = damping * lambda * impulse_i
a_i' = clip(a_i + field_i, 0, 1)
```

Defaults and bounds:

| Parameter | Default | Bound |
|---|---:|---:|
| `lambda` | 0.15 | 0.0 .. 0.50 |
| `damping` | 0.5 | 0.0 .. 1.0 |
| field impulse cap | 0.25 | 0.0 .. 0.25 |
| neighbours per head | 8 | 0 .. 8 |
| total edges | 0 at genesis | 0 .. 1,024 |
| `|J_ij|` | 0 | 0.0 .. 1.0 |
| sum `|J_ij|` per head | 0 | 0.0 .. 4.0 |
| synchronous rounds | 1 | 1 .. 4 |

`head 0` participates as an anchor input, but its activation is capped at
`0.05` and its own field-induced change is capped at `0.01`. It never receives
corpus training.

The mean subtraction is deliberate: a shared common mode is the failure that
produced rank-1 collapse in `docs/HEAD_POPULATION.md`. Subtracting the mean
keeps the field sensitive to differences between heads rather than to one
global level.

The field term is **not** applied during Boolean evaluation, and no inference
call mutates `J`. Persistent coupling changes are field candidates and follow
the same stage -> evaluate -> approve -> CAS path as head candidates. A pair
rail observation may be collected as evidence, but it does not silently become
an active field coupling.

The default is one synchronous round. More rounds are an experiment, must be
bounded by the table above, and must not claim convergence. "Resonance" here
means a deterministic classical mean-field correlation, nothing more.

### 3.5 Deterministic multi-head routing

A single global winner is forbidden. Routing uses content buckets so that many
heads can participate without one head absorbing the corpus.

- Heads `1..255` are split into 15 content buckets of 17 heads each:
  bucket `b` covers `1 + 17b .. 17 + 17b`.
- Head 0 is the anchor and is always represented, but with the 0.05 cap.
- For each layer `L0..L9`, one winner per content bucket is selected.
- Default active set: head 0 plus 15 content winners = 16 heads per layer.
- Minimum: 8 heads per layer. Maximum: 64 distinct heads per turn across all
  layers.
- If the requested budget is below 16, keep head 0 and the highest-scoring
  remaining winners; rotate the skipped buckets by the context digest so no
  bucket starves. If the budget is above 16, select additional winners per
  bucket in rank order, still capped at 64 distinct heads per turn.

The routing score for head `i`, layer `l`, context `x` is:

```text
route_i = a_i
        + 0.10 * hash01("route|" + l + "|" + context_digest + "|" + i)
        + 0.05 * normalized_reward_i
        - 0.10 * normalized_usage_i
```

`hash01` is a stable cryptographic hash (for example BLAKE2b), not Python's
process-randomized `hash()`. `normalized_usage` is bounded per bucket. The
bucket structure guarantees coverage; the hash and usage terms prevent one
head from winning every context.

Every layer uses a different salt, so L0-L9 activate different populations of
heads. Across ten layers and many observations, the field should populate many
distinct head IDs. Coverage is measured, not assumed.

---

## 4. L0-L9 per-head scratch layers

### 4.1 Canonical layers

The layer names and indices are fixed by
`docs/UNIFIED_TOPOLOGY_AND_LAYERS.md`:

| ID | Name | Scope |
|---:|---|---|
| L0 | bit | bit/Boolean facts and truth rows |
| L1 | nibble | four-bit grouping and composition |
| L2 | byte | byte identity and composition |
| L3 | phonetic | onset/rime and phonetic relations |
| L4 | morpheme | stems, affixes and derivation |
| L5 | lexical | words, senses, definitions and lexical relations |
| L6 | word_completion | context-to-next-choice evidence |
| L7 | sentence | syntax, agreement and continuation |
| L8 | paragraph | cohesion, topic continuity and transitions |
| L9 | narrative | events, temporal/causal order and continuity |

A layer is a resolution axis. It is not a head, a runner, a hardware slot or a
main-register generation.

### 4.2 Per-head logical partition

Every head owns ten logical layer partitions. The implementation may share
physical pages, but the API treats `(head, layer)` as an isolated partition:

```text
scratch key = (head_id, layer_id, revision, generation)
```

The default `ScratchLayers` geometry is:

| Item | Default |
|---|---:|
| heads | 256 |
| layers per head | 10 |
| pages per layer | 2 |
| records per page | 8 |
| slots per layer | 16 |
| payload limit per record | 1 KiB |
| max possible record payload | 40 MiB |
| per-head slots | 160 |
| per-head default payload ceiling | 160 KiB |

`maximum_value_bytes` may be raised only if the constructor proves:

```text
HEAD_COUNT * LAYER_COUNT * slots_per_layer * max_value_bytes
    <= SCRATCH_BUDGET_BYTES
```

`SCRATCH_BUDGET_BYTES` is exactly `1 GiB`. At the default 1 KiB payload this is
40 MiB. At 64 KiB payload the product is about 2.5 GiB, so the configuration
MUST be rejected before allocation. This is one of the concrete no-2-GiB gates.

Records are immutable once written. The ring may recycle an old slot when a
layer is full; the write receipt MUST report `evicted: true` and the evicted
record digest must remain in the bounded audit history. A slot reuse is an
explicit eviction, not an in-place mutation of the old record.

### 4.3 Record schema

Every scratch record is canonical JSON with these fields:

```json
{
  "schema": "nanlm.resident-scratch-layer/1",
  "head": 42,
  "layer": 5,
  "layer_name": "L5",
  "kind": "observation | candidate | snapshot | tombstone",
  "revision": 7,
  "generation": 19,
  "parent_digest": "64-hex-sha256-or-null",
  "value": {},
  "value_digest": "64-hex-sha256"
}
```

The record MUST NOT contain:

- a filesystem path, home directory, drive letter, `file://` URL or `..`
  segment;
- a token, key, password, private-key block or authorization header;
- a raw pointer, `id()` value, ctypes address, mmap offset or unpickled object;
- executable source or an importable module name.

The `head`, `layer`, numeric bounds, canonical JSON, digest and payload size are
validated before storage. A malformed record is denied with `invalid_record`;
nothing is partially written.

### 4.4 Per-head snapshot

A per-head snapshot is a compact, immutable summary of one head's learner
state and its ten layer roots. It does not copy raw scratch text.

```json
{
  "schema": "nanlm.head-snapshot/1",
  "head": 42,
  "generation": 17,
  "parent_generation": 16,
  "state_digest": "64-hex-sha256",
  "rule_digest": "64-hex-sha256",
  "score_word": 7864,
  "activation": {
    "threshold_word": 26214,
    "temperature_word": 32768,
    "usage": 12
  },
  "layer_roots": {
    "L0": "64-hex-sha256",
    "L1": "64-hex-sha256",
    "L2": "64-hex-sha256",
    "L3": "64-hex-sha256",
    "L4": "64-hex-sha256",
    "L5": "64-hex-sha256",
    "L6": "64-hex-sha256",
    "L7": "64-hex-sha256",
    "L8": "64-hex-sha256",
    "L9": "64-hex-sha256"
  },
  "created_tick": 123456,
  "snapshot_digest": "64-hex-sha256"
}
```

A head snapshot is bounded to 16 KiB. It must be reproducible byte-for-byte
from the same state and generation.

### 4.5 Field snapshot

A field snapshot binds all 256 heads to one field generation and one coupling
state.

```json
{
  "schema": "nanlm.head-field-snapshot/1",
  "generation": 17,
  "head_root": "64-hex-sha256",
  "heads": [
    {"head": 0, "state_digest": "64-hex-sha256"},
    {"head": 1, "state_digest": "64-hex-sha256"}
  ],
  "couplings": [
    {"a": 3, "b": 7, "weight_word": 1638}
  ],
  "field": {
    "lambda_word": 9830,
    "damping_word": 32768,
    "rounds": 1
  },
  "metrics": {
    "active_heads": 16,
    "entropy_ratio": 0.41,
    "field_entropy_norm": 0.24
  },
  "snapshot_digest": "64-hex-sha256"
}
```

`head_root` is the Merkle root over the 256 head state digests in ID order. A
field snapshot is valid only if:

1. all 256 IDs `0..255` are present exactly once;
2. each `state_digest` matches a retained head snapshot;
3. `head_root` recomputes exactly;
4. every coupling edge has distinct endpoints, valid bounds and no duplicates;
5. the field parameters are within the table in section 3.4.

Field snapshots are capped at 4 MiB and retained at most two deep. Head
snapshots are retained at most two deep per head. The total snapshot budget is
part of the 1 GiB scratch budget, not an additional unbounded store.

### 4.6 Snapshot integrity and rollback lineage

- Snapshots are content-addressed by `snapshot_digest`; there is no mutable
  "current snapshot" pointer.
- A snapshot may be evicted only after two newer snapshots for the same head
  or field. An evicted snapshot is unavailable for rollback and must be
  reported as `snapshot_unavailable`.
- A rollback target must be an ancestor of the current generation. A snapshot
  from a different lineage is denied.
- Snapshot reads verify the digest before returning data. A digest mismatch
  fails closed; the store does not fall back to a previous snapshot silently.
- Snapshots never contain raw text, secrets, paths or pointers; they contain
  digests and bounded numeric state.

---

## 5. Scratch-first self-modification pipeline

### 5.1 State machine

```text
IDLE
  -> STAGED          (scratch record written; main untouched)
  -> EVALUATED       (Boolean evaluation passed; report digest fixed)
  -> APPROVED        (explicit approval record bound to candidate+report)
  -> COMMITTED       (generation CAS succeeded; new main generation)
  -> ROLLED_BACK     (a later, explicit, forward commit restored a snapshot)

Any state may transition to DENIED with a reason code.
A stale base or CAS conflict transitions to STALE; the token is dead.
A crash during commit is recovered from the previous valid slot.
```

No state may be skipped. `STAGED -> APPROVED` is invalid. `EVALUATED ->
COMMITTED` is invalid. `APPROVED -> COMMITTED` is valid only after a successful
CAS.

### 5.2 Stage

`stage_candidate` writes one candidate record to scratch and returns a staged
token. It MUST NOT write the main register.

A head candidate contains:

```json
{
  "schema": "nanlm.head-candidate/1",
  "candidate_digest": "64-hex-sha256",
  "head": 42,
  "scope": "rule | score | activation | rollback_head",
  "base_generation": 17,
  "base_state_digest": "64-hex-sha256",
  "base_boolean_digest": "64-hex-sha256",
  "proposal": {},
  "proposer_id": "session:...",
  "created_tick": 123456,
  "evidence": ["sha256:..."]
}
```

A field candidate contains:

```json
{
  "schema": "nanlm.head-field-candidate/1",
  "candidate_digest": "64-hex-sha256",
  "scope": "coupling | rollback_field",
  "base_generation": 17,
  "base_head_root": "64-hex-sha256",
  "delta_edges": [
    {"a": 3, "b": 7, "weight_word": 1638}
  ],
  "proposer_id": "session:...",
  "created_tick": 123456
}
```

Stage rules:

- A head candidate targets exactly one `head` in `0..255`. Head 0 rule,
  role and identity changes are denied.
- A field candidate may not contain head rule, score or activation edits.
- A head candidate may not contain coupling edits. A field candidate may not
  contain per-head edits.
- `base_generation` and `base_state_digest` (or `base_head_root`) are pinned at
  stage time from a fresh read of the main register.
- The candidate is bounded: at most 16 rule nodes; at most 64 field-edge
  changes; at most 16 KiB of canonical JSON for a head candidate; at most
  256 KiB for a field candidate.
- The candidate is validated for schema, bounds, secret/path content,
  raw-pointer keys, executable content and unsupported keys before it is
  stored.
- Staging reads the base but never mutates main. A staging failure leaves no
  partial candidate.

### 5.3 Boolean evaluation

`evaluate_candidate` runs on an ephemeral copy of the head state with the
candidate applied. It does not write main and does not consume approval.

Mandatory evaluation report:

```json
{
  "schema": "nanlm.head-evaluation/1",
  "candidate_digest": "64-hex-sha256",
  "base_generation": 17,
  "base_state_digest": "64-hex-sha256",
  "boolean_base_digest": "64-hex-sha256",
  "exact": {"correct": 64, "total": 64, "passed": true},
  "mandatory_masks": {"6": true, "8": true, "14": true},
  "rule_table_exact": true,
  "hard_output_parity": true,
  "parity_required": true,
  "unknown_conflict_preserved": true,
  "field_not_in_boolean_path": true,
  "deterministic_replay": true,
  "regression": {
    "cases": 256,
    "delta_accuracy": 0.0,
    "delta_coverage": 0.0,
    "failures": []
  },
  "failures": [],
  "evaluation_digest": "64-hex-sha256"
}
```

Evaluation MUST include:

1. All 64 canonical mask/input cases.
2. XOR/AND/OR exact-form checks.
3. A check that the candidate's declared rule table is exactly reproducible
   from its canonical mask DAG. For `score`, `activation` and `rollback`
   scopes, also require base/candidate hard-output parity; for `rule` scope,
   parity may be false because the rule is intentionally changed, and
   `parity_required` must be false.
4. A field-off/field-on check proving the hard Boolean result is unchanged by
   the field term.
5. `UNKNOWN` and `CONFLICT` preservation.
6. A bounded held-out regression suite for the head's declared objective.
7. A deterministic replay of the full suite.
8. A schema/bounds/invariant check.
9. A check that the candidate cannot alter the base digest.

`evaluation_digest` binds the candidate digest, base generation, base state
digest, Boolean base digest, suite version and report body. If any input
changes, the report is stale and approval is invalid.

A failed evaluation sets the candidate state to `DENIED` and records the
failure. It never reaches approval.

### 5.4 Explicit approval

Approval is a structured, single-use record, not `approved=True`.

```json
{
  "schema": "nanlm.head-approval/1",
  "approval_id": "64-hex-sha256",
  "decision": "approve",
  "candidate_digest": "64-hex-sha256",
  "evaluation_digest": "64-hex-sha256",
  "head": 42,
  "scope": "rule",
  "base_generation": 17,
  "base_state_digest": "64-hex-sha256",
  "proposer_id": "session:...",
  "approver_id": "operator:...",
  "approver_kind": "user | operator",
  "approved_tick": 123500,
  "expires_tick": 127100,
  "nonce": "opaque-single-use-id"
}
```

Approval rules:

- `decision` MUST be exactly `approve`.
- `approver_id` MUST differ from `proposer_id`. A model or the proposing tool
  cannot approve its own candidate.
- The record MUST bind both the candidate digest and the evaluation digest.
- It MUST bind the exact base generation and base state digest.
- It expires after a bounded TTL (default 1 hour; maximum 24 hours).
- The nonce is single-use. A replay is denied even if all digests still match.
- A bare boolean, a generic "yes", a prior approval for a different head, or an
  approval whose scope does not cover the candidate is denied.
- A stale, failed, expired, already-used or mismatched approval is denied.
- Approval does not write main. It only authorizes one CAS attempt for one
  candidate.

Approval failure leaves the candidate staged but not committed; the caller may
stage a new candidate if the base is still current. A CAS conflict invalidates
the approval, because the approved base no longer exists.

### 5.5 Generation CAS

The main register is a per-head 256-record store with a monotonic generation.
A commit is a compare-and-swap, not a compare-and-set:

```text
commit(expected_generation, expected_state_digest, candidate, approval)
```

Under one exclusive lock:

1. Read both main slots. Choose the highest valid generation.
2. If the two valid slots have the same generation but different digests,
   fail closed with `split_brain` and freeze writes.
3. Require `current_generation == expected_generation`.
4. Require `current_state_digest == expected_state_digest`.
5. Re-open and re-verify the candidate and approval records.
6. Verify approval signature/binding, nonce, expiry and scope.
7. Apply the candidate to a copy of the affected head only.
8. Recompute the head state digest, head root and field root.
9. Write the full new generation to the alternate slot.
10. `fsync` the slot and its directory.
11. Only then publish the new generation and append the audit receipt.
12. Mark the approval nonce consumed and the candidate token committed.

If any step fails before publication, the old slot remains authoritative and
the new slot is never selected. If the process crashes mid-write, recovery
selects the highest valid generation; a torn or invalid slot is ignored. A
corrupt pair never silently falls back to genesis.

The main register MUST have:

- exactly 256 per-head records;
- one fixed generation counter;
- a maximum page size of 8 MiB;
- per-head record cap of 16 KiB;
- a field record cap of 2 MiB;
- a separate Boolean base digest;
- a separate head root;
- no filesystem path in the record or status output.

The main-register generation is namespaced. It is not
`chart_storage.py`'s snapshot generation, not `ResidentUniverse._generation`,
and not a native canonical generation. Do not compare or reuse one as another.

### 5.6 Main-register commit

The commit is one transaction for one head (or for the field record), not a
batch of heads. A head commit:

- changes exactly one head's rule, score or activation;
- preserves every other head's state digest;
- preserves head 0 unless the candidate is explicitly denied for head 0;
- updates the affected head's `layer_roots` only if the candidate cites valid
  scratch records;
- recomputes the field root because the head root changed;
- writes the hot `ResidentCells` projection only after main publication.

If the hot projection fails after main publication, the main register remains
authoritative and the projection is marked `projection_stale`. The next read
rebuilds from main or serves the main value. A projection failure must not
write a second main generation.

Pair rails are evidence accumulators, not head-identity state. A pair
observation may update the hot pair rail without a main commit, but it must not
change any head's rule, score, activation policy, identity or head generation
for learner-state purposes. If pair evidence is to influence the field term,
it must first become an approved field candidate.

### 5.7 Denial rules

Denial is fail-closed, audited and non-mutating. The main register and the hot
head projection remain unchanged.

| Code | Condition | Result |
|---|---|---|
| `approval_required` | no structured approval | deny |
| `approval_mismatch` | candidate/eval/head/scope digest mismatch | deny |
| `approval_expired` | `approved_tick > expires_tick` | deny |
| `approval_replay` | nonce already consumed | deny |
| `self_approval` | approver equals proposer or is a model | deny |
| `stale_generation` | expected generation differs from current | deny; token dead |
| `base_digest_mismatch` | expected state digest differs | deny; token dead |
| `boolean_eval_failed` | 64-case or regression failure | deny |
| `boolean_base_changed` | base digest differs from evaluator | deny; freeze |
| `cross_head_scope` | candidate touches more than one head | deny |
| `identity_rewrite` | head ID, identity or role changes | deny |
| `head_zero_immutable` | head 0 rule/role rewrite | deny |
| `field_scope_violation` | field candidate contains head edits, or vice versa | deny |
| `secret_or_path` | path, secret, token or credential detected | deny |
| `raw_pointer` | pointer/address/object-reference key detected | deny |
| `canonical_merge` | request attempts canonical merge/promotion | deny |
| `memory_budget` | scratch, main, snapshot or audit budget exceeded | deny |
| `split_brain` | equal-generation slots disagree | freeze |
| `snapshot_unavailable` | rollback target evicted or unknown | deny |
| `field_root_mismatch` | field snapshot does not match current head root | deny |
| `projection_stale` | hot projection lags main | serve main; do not re-commit |

No field score, entropy value, benchmark result, corpus size or user urgency
overrides these denials. There is no emergency bypass flag.

### 5.8 Rollback semantics

Rollback is a new forward commit, not a time-travel write and not a generation
decrement.

**Head rollback.** Stage a `rollback_head` candidate that names a retained head
snapshot. Evaluate it with the same Boolean suite, approve it, and CAS it. The
new generation has the old head state and the current states of all other
heads. The field root is recomputed. If the hybrid state fails evaluation, the
rollback is denied.

**Field rollback.** Stage a `rollback_field` candidate that names a retained
field snapshot. It is permitted only when every current head state digest
matches the snapshot's `head_root`. A field rollback cannot silently overwrite
newer head state; if the heads have moved, the snapshot is stale and denied.

**Crash rollback.** Recovery reads both slots, chooses the highest valid
generation, and never selects a partial slot. If both slots are corrupt, the
register freezes read-only and reports `main_register_frozen`; it does not
recreate a genesis state.

**Approval rollback.** Consumed approval nonces stay consumed. Rollback does
not revive an old approval. The rollback needs its own candidate, evaluation,
approval and CAS.

**Audit rollback.** Every denial, stage, evaluation, approval, commit and
rollback appends an audit record. The audit store is bounded (default 64 MiB)
and must never contain secrets or paths.

---

## 6. Invariants and enforcement

These invariants are normative. A change that violates one is rejected even if
it improves a benchmark.

| ID | Invariant | Enforcement |
|---|---|---|
| I-1 | No cross-head identity rewrite. One candidate touches one head; head IDs, roles and identities are immutable; a head cannot read or write another head's private state. | candidate schema, per-head CAS, `identity_address()` checks in `cells.py`, `tests/test_resident_head_field.py` |
| I-2 | No unapproved main write. Only the structured approval + CAS path may publish a main generation. | `resident/head_mutation.py`, `resident/head_main.py`, tests |
| I-3 | No canonical merge. This path may not call canonical promotion, `apply_candidate` for canonical state, the six-runner barrier, or any merge endpoint. | `canonical_merge_allowed=False`, code review and tests |
| I-4 | No 2 GiB overbootstrap. The naive `256 x 8 MiB = 2 GiB` allocation is forbidden. Scratch is a shared bounded ring or 1 GiB maximum, main is 8 MiB total, snapshots are bounded and the audit is bounded. | constructor budget check, `tests/test_resident_head_field_invariants.py` |
| I-5 | No raw pointers. Persisted candidates, approvals, snapshots and audit records contain IDs, enums, Q16.16 integers and SHA-256 digests only. No process address, `id()`, ctypes pointer, mmap offset or unpickled object. Existing `leaf_chain`/`relation_chain` are store-internal handles and are not accepted from candidate payloads. | schema validator, tests |
| I-6 | No secret/path leakage. No filesystem path, home directory, token, key, password or authorization header enters scratch, main, snapshots, status or audit. | validator + redaction tests |
| I-7 | Bounded memory. Scratch <= 1 GiB; main <= 8 MiB; per-head main <= 16 KiB; snapshots <= 16 MiB; audit <= 64 MiB; one record <= 64 KiB; one field edge set <= 1,024 edges; one turn <= 64 distinct heads. | budget checks and tests |
| I-8 | Immutable Boolean base. All 16 masks, exact 64/64, XOR/AND/OR truth tables and the base digest are immutable. | `boolean_wave.py`, evaluation report, tests |
| I-9 | Field logic cannot gate Boolean logic. The field term is not an input to the hard Boolean result. | code path and `field_not_in_boolean_path` test |
| I-10 | Unknown and conflict survive. `UNKNOWN` never becomes `FALSE`; `CONFLICT` is never averaged away. | evaluation report, relation tests |
| I-11 | Deterministic replay. Same base, candidate, approval and field settings produce identical bytes and the same result; no `hash()`, clock, randomness or locale dependence in routing or state. | stable hash, integer state, replay tests |
| I-12 | Monotonic generation. A successful commit increments generation by exactly one; stale, duplicate or rollback attempts never decrement; one approval authorizes one successful CAS. | main CAS, token/approval ledger, tests |
| I-13 | Snapshot lineage. Every rollback target is retained, digest-valid and an ancestor; field rollback requires the exact head root. | snapshot store, tests |
| I-14 | No silent split brain. Two valid slots with the same generation but different digests freeze writes. | recovery test |

### 6.1 Authority model (important)

There is one authority for learned head state: the **main register**. The
`ResidentCells` head record is a **hot projection** used for runtime activation
and identity checks. The scratch layers are **staging**, not authority.

```text
scratch layers  --stage-->  candidate
                               |
                               v
                         Boolean evaluation
                               |
                               v
                         approval record
                               |
                               v
                         main register (authority)
                               |
                               v
                         ResidentCells (projection)
```

Do not introduce a second authoritative generation. If `ScratchLayers`
`MemoryMainBackend` is used, it is a test/development backend for the same
main-register protocol, not an additional production authority. Production
must use one main-register implementation and one generation counter.

---

## 7. Corpus learning and free chat: many heads, not one global head

### 7.1 The failure this must avoid

`docs/HEAD_POPULATION.md` records the baseline failure: with `active=255`, every
head trains on every context, all heads converge to the global distribution,
and the head Gram rank is 1. A larger head count is not diversity. The field
must use sparse, bucketed routing and measure specialization.

### 7.2 Corpus learning

```text
corpus observation
  -> L0..L9 feature projections
  -> bucket router selects 16 heads per layer (head 0 + 15 content winners)
  -> selected heads update their own scratch state
  -> per-layer scratch record (no main write)
  -> batch support accumulation
  -> stage up to 64 head candidates
  -> Boolean evaluation per candidate
  -> explicit approval per candidate
  -> generation CAS -> main commit
```

Corpus rules:

- Head 0 is never trained on corpus content.
- A head is updated only when it wins its bucket for that layer and context.
- No global average distribution is written into every head.
- Scratch writes are bounded by the ring and the 1 GiB budget.
- A batch may stage candidates only for heads that have enough support,
  confidence and novelty. Defaults follow `LearningPolicy`: support >= 3,
  confidence >= 0.6, at most 64 candidates.
- A corpus run does not commit main automatically. It may stop at `STAGED` or
  `EVALUATED`; `approved=true` is not a corpus default.
- Field evidence is staged as a separate field candidate. Pair observations do
  not silently mutate active coupling.

### 7.3 Free chat

- Each chat turn performs one bounded observation when chat learning is
  enabled.
- The turn routes 16 heads per layer across L0-L9, capped at 64 distinct heads
  per turn.
- Head 0 appears as the prior with at most 5% of activation mass.
- Free chat writes scratch and returns digest/trace receipts; it does not
  commit main and does not create an approval automatically.
- A user may explicitly stage a candidate from the turn, but the ordinary chat
  path must not do so.
- The reply may read the field status as routing context; it must not present
  activation, resonance or entropy as a truth claim.

### 7.4 Activation guarantees and measurements

For each probe context, define:

```text
A(x) = selected head IDs
p_i  = a_i / sum_j a_j        (a_i = 0 when not active)
H_act(x) = -sum_i p_i ln p_i / ln(256)
```

Measurement protocol:

- Use a held-out probe set of 1,024-4,096 unique contexts, not training text.
- Run with the main register frozen and no approval.
- Report per-layer and per-turn populations, not only a global average.
- Compare against controls: no-field, single-head, uniform-field and
  all-heads-active. The all-heads-active control is expected to collapse.

Required diagnostics:

| Metric | Definition | Default target |
|---|---|---|
| `active_heads_per_layer` | winners in one layer | 8 .. 16 |
| `active_heads_per_turn` | distinct heads across L0-L9 | 8 .. 64 |
| `distinct_heads_window` | distinct content heads over a 4,096-observation window | >= 128 |
| `head0_share` | share of activation mass on head 0 | <= 0.05 |
| `max_head_share` | largest single-head share in a window | <= 0.10 |
| `activation_entropy_norm` | mean `H_act(x) / ln(256)` | >= 0.35 |
| `field_entropy_norm` | mean entropy of `p_i^field = |field_i| / sum|field|`, divided by `ln(256)`; 0 when field is inactive | >= 0.20 when active |
| `effective_rank` | `exp(H(eigenvalues of head Gram))`, normalized | > 1.5; target >= 2.0 |
| `mean_cosine` | mean pairwise cosine of head activation profiles | <= 0.50 |
| `state_digest_collisions` | exact duplicate head state digests | 0 |
| `rule_fingerprint_collisions` | duplicate rule-table digests among populated heads | <= 5% |
| `route_set_collision_rate` | fraction of unique contexts whose top-K sets have Jaccard >= 0.9 | <= 0.25 |
| `canonical_aliasing_rate` | pair-address aliasing in the field | 0 |
| `rail_key_collision_rate` | pair rail-key collisions | 0 |

`rank` and `effective_rank` are diagnostic, not semantic capacity. Numeric
eigenvalue rank can be inflated by tiny numerical eigenvalues; always report
the tolerance and the effective rank. `mean_cosine` alone can look low while
effective rank is 1, so report both.

### 7.5 Collision classes

Distinguish three different collisions:

1. **Identity/state collision.** Two distinct heads have the same canonical
   state digest or the same permanent identity. This is always a failure.
2. **Rule collision.** Two heads use the same rule-table digest. Some sharing
   is allowed, but widespread duplication means the learners are not
   independent. Report the count and the populated-head denominator.
3. **Route collision.** Different contexts select the same top-K set. Some
   repeated contexts are expected; measure only unique, deduplicated contexts.
   A high route-collision rate means the router has collapsed to one global
   population.

Address aliasing in the pair field (`canonical_aliasing_rate`,
`rail_key_collision_rate`) is a separate structural check and must remain zero
for the pair layout.

---

## 8. Point-by-point integration checklist

Each item names the implementation file(s), the acceptance evidence, and the
required order. Items 1-6 must be completed before items 7-10 can be enabled in
chat.

| # | Work | Files | Done when |
|---:|---|---|---|
| 1 | Freeze the Boolean base and record its digest. | `nanlmind_hermes/boolean_wave.py`, `tests/test_boolean_wave.py` | 64/64 exactness; masks 6/8/14 checked; base digest stored in evaluation and main header; no semantic edits to the 16 masks. |
| 2 | Define bounded per-head learner state. | `nanlmind_hermes/resident/head_learner.py` (new), `nanlmind_hermes/resident/head_field.py` | 256 independent records; each has own rule/score/activation; head 0 role fixed; state digest and schema tests. |
| 3 | Add the classical coupled field term and bucket router. | `nanlmind_hermes/resident/head_field.py` | sparse `J`, mean-field impulse, bounded defaults, 15 content buckets, 8-64 active heads; field-off control; no field input to Boolean result. |
| 4 | Add per-head L0-L9 scratch semantics and budgets. | `nanlmind_hermes/resident/scratch_layers.py` | 256 x 10 layers; canonical L0-L9 names; 1 GiB total cap; default 40 MiB; 64 KiB record cap; eviction receipts; secret/path/pointer validation. |
| 5 | Add immutable per-head and field snapshots. | `nanlmind_hermes/resident/scratch_layers.py`, `nanlmind_hermes/resident/head_field.py` | head snapshot <= 16 KiB; field snapshot <= 4 MiB; head root; two-deep retention; digest verify; snapshot tests. |
| 6 | Add the per-head main register and generation CAS. | `nanlmind_hermes/resident/head_main.py` (new), `nanlmind_hermes/chart_storage.py` (pattern reference only) | 8 MiB total; 256 records; per-head 16 KiB; dual-slot recovery; exact generation/digest CAS; split-brain freeze. |
| 7 | Implement stage -> Boolean evaluation -> approval -> CAS -> commit. | `nanlmind_hermes/resident/head_mutation.py` (new) | state machine; evaluation report; structured approval; one successful CAS per approval; bare `approved=True` denied; denial table covered. |
| 8 | Implement rollback and crash recovery. | `nanlmind_hermes/resident/head_main.py`, `nanlmind_hermes/resident/head_mutation.py` | head rollback is forward commit; field rollback requires exact head root; torn write recovery; no genesis fallback; approval nonces stay consumed. |
| 9 | Wire corpus learning to the field router. | `nanlmind_hermes/layered_learning.py`, `scripts/corpus_training.py`, `scripts/head_field_benchmark.py` | corpus activates many heads; scratch writes only; up to 64 candidates staged; no automatic main write; metrics recorded. |
| 10 | Wire free chat to the field router. | `nanlmind_hermes/chat.py`, `nanlmind_hermes/tools.py`, `nanlmind_hermes/__init__.py` | one bounded observation per turn; 16 heads per layer; no automatic approval/commit; chat still works when field is disabled. |
| 11 | Expose the local field tools and manifest entries. | `nanlmind_hermes/__init__.py`, `plugin.yaml`, `docs/TOOLS_REFERENCE.md` | status/stage/evaluate/approve/commit/rollback tools are scoped and approval-gated; manifest test passes. |
| 12 | Add independence, entropy and collision diagnostics. | `nanlmind_hermes/resident/head_field.py`, `scripts/head_field_benchmark.py`, `scripts/head_field_diagnostics.py` (new) | reports active heads, entropy, effective rank, mean cosine, state/rule/route collisions; held-out probe; no path in output. |
| 13 | Enforce memory budgets. | `nanlmind_hermes/resident/scratch_layers.py`, `nanlmind_hermes/resident/head_main.py` | constructor rejects any config above 1 GiB scratch, 8 MiB main, 16 MiB snapshots, 64 MiB audit; 2 GiB request denied before allocation. |
| 14 | Enforce the no-canonical-merge guard. | `nanlmind_hermes/resident/head_mutation.py`, `nanlmind_hermes/resident/universe.py` (reference) | field commit marks `canonical_commit: false`; no merge/promotion call; tests assert canonical merge disabled. |
| 15 | Add secret/path/pointer validation. | `nanlmind_hermes/resident/scratch_layers.py`, `nanlmind_hermes/resident/head_mutation.py` | candidates with keys such as `path`, `file`, `pointer`, `address`, `api_key`, `token`, `BEGIN PRIVATE KEY` are denied; status/audit contains no paths. |
| 16 | Label the Python mirror honestly. | `scripts/head_field_benchmark.py`, `README.md` (future edit), this document | reports `implementation: "python-mirror"` and `native_performance_claim: false`; no throughput comparison to native. |
| 17 | Update the public status contract. | `nanlmind_hermes/resident/head_field.py`, `nanlmind_hermes/resident/head_main.py`, `docs/` | status includes generation, active heads, entropy, collisions, budgets, `canonical_merge_allowed=false`; no secret/path. |
| 18 | Run the full acceptance matrix and record results. | `tests/test_resident_head_field.py`, `tests/test_resident_scratch_layers.py`, new test files in section 9 | all required tests pass; diagnostics JSON is reproducible; no test writes canonical state. |

Implementation order is not optional: items 1-6 establish the immutable base,
the per-head state, the scratch and the CAS; item 7 is the only path that may
publish; items 8-10 add recovery and activation; items 11-18 expose and audit.

---

## 9. Acceptance tests

Test names are normative for the new files; existing tests may be extended but
must not be weakened.

### 9.1 Boolean base

File: `tests/test_boolean_wave.py` (existing, extend)

- `test_all_sixteen_functions_are_exact_on_booleans`
- `test_xor_and_or_reference_truth_tables`
- `test_boolean_base_digest_is_stable`
- `test_field_term_is_not_in_boolean_path`
- `test_candidate_cannot_change_base_digest`

Acceptance: 64/64 exact; masks 6/8/14 match; a candidate that edits a mask or
skips a case is denied before approval; field-on and field-off hard outputs are
identical.

### 9.2 Independent head state and field term

File: `tests/test_resident_head_field.py` (existing, extend)

- `test_256_heads_are_independent_stateful_learners`
- `test_head_zero_is_bounded_prior_and_not_corpus_trained`
- `test_activation_is_head_local_and_additive`
- `test_coupled_field_term_is_bounded_and_uses_public_activations`
- `test_field_term_cannot_change_rule_or_score`
- `test_head_identity_is_never_rewritten`
- `test_bucket_router_activates_many_heads`

Acceptance: mutating one head leaves every other state digest unchanged; head 0
shares at most 5%; field impulse respects caps; 15 content buckets plus head 0
produce 8-64 active heads; no global head wins all contexts.

### 9.3 Scratch layers and snapshots

File: `tests/test_resident_scratch_layers.py` (existing, extend)

- `test_all_256_heads_have_ten_layers`
- `test_layers_are_independent_per_head`
- `test_scratch_geometry_and_payload_budget`
- `test_scratch_never_writes_main`
- `test_per_head_snapshot_digest_and_bounds`
- `test_field_snapshot_head_root`
- `test_snapshot_retention_and_eviction`
- `test_scratch_rejects_secret_path_and_pointer`

Acceptance: 256 x 10 partitions; default geometry <= 40 MiB; 64 KiB payload
configuration rejected before allocation; snapshot sizes within 16 KiB/4 MiB;
digest mismatch fails closed; secret/path/pointer test data is denied.

### 9.4 Main register and CAS

File: `tests/test_resident_head_main.py` (new)

- `test_main_register_is_eight_mib_and_256_records`
- `test_commit_requires_exact_generation_and_digest`
- `test_stale_generation_denies_without_write`
- `test_duplicate_commit_denies`
- `test_split_brain_freezes_writes`
- `test_torn_write_recovers_previous_slot`
- `test_rollback_is_forward_generation`
- `test_hot_projection_never_becomes_second_authority`

Acceptance: only one slot is published; stale/duplicate writes leave main
unchanged; torn new slot is ignored; corrupt pair freezes; rollback increments
generation; hot projection failure does not create a second commit.

### 9.5 Approval and denial

File: `tests/test_resident_head_mutation.py` (new)

- `test_stage_does_not_touch_main`
- `test_boolean_evaluation_precedes_approval`
- `test_bare_approved_true_is_denied`
- `test_approval_binds_candidate_and_evaluation_digest`
- `test_approval_replay_is_denied`
- `test_expired_approval_is_denied`
- `test_self_approval_is_denied`
- `test_cross_head_candidate_is_denied`
- `test_field_scope_and_head_scope_cannot_mix`
- `test_cas_conflict_invalidates_approval`
- `test_denial_leaves_state_unchanged`

Acceptance: no main write before a valid structured approval; evaluation
failure cannot be approved; every denial returns a reason code and leaves main
and hot cells unchanged.

### 9.6 Invariants and security

File: `tests/test_resident_head_field_invariants.py` (new)

- `test_no_2_gib_overbootstrap`
- `test_no_secret_or_path_in_status_or_audit`
- `test_no_raw_pointer_in_candidate_or_snapshot`
- `test_no_canonical_merge_call`
- `test_no_unapproved_main_write`
- `test_memory_budgets_are_enforced`
- `test_unknown_and_conflict_are_preserved`
- `test_head_identity_collisions_are_zero`

Acceptance: a 2 GiB geometry or payload request is rejected before allocation;
status/audit do not contain paths or secrets; pointer-shaped fields are denied;
canonical merge remains disabled; memory caps hold under full ring occupancy.

### 9.7 Corpus and free-chat activation

File: `tests/test_resident_head_field_activation.py` (new), extend
`tests/test_layered_learning.py` and `tests/test_chat.py`

- `test_corpus_activates_many_heads_not_one_global_head`
- `test_free_chat_activates_many_heads_per_turn`
- `test_head_zero_share_is_bounded`
- `test_activation_entropy_and_effective_rank_are_reported`
- `test_route_state_and_rule_collisions_are_measured`
- `test_no_field_and_single_head_controls`
- `test_chat_does_not_auto_commit`

Acceptance: corpus window covers >= 128 distinct content heads; free chat
activates 8-64 per turn; head 0 <= 5%; max head share <= 10%; entropy >= 0.35;
effective rank > 1.5; state collisions zero; no automatic main write.

### 9.8 Integration and label tests

Files: `tests/test_plugin_manifest.py`, `tests/test_packaging.py`,
`scripts/head_field_benchmark.py`

- `test_field_tools_are_manifested_and_approval_gated`
- `test_python_mirror_label_is_not_native_performance`
- `test_benchmark_report_has_no_canonical_merge`
- `test_diagnostics_report_schema_and_no_paths`

Acceptance: manifest lists the field tools; benchmark output labels
`python-mirror`; no native speed claim; no canonical merge; diagnostics JSON is
stable and path-free.

### 9.9 Definition of done

- [ ] All existing tests still pass.
- [ ] 64/64 Boolean exactness with masks 6/8/14.
- [ ] 256 head records, each with independent rule/score/activation state.
- [ ] 256 x 10 L0-L9 scratch partitions.
- [ ] Per-head and field snapshots with digest verification.
- [ ] Stage -> Boolean evaluation -> structured approval -> generation CAS ->
      main commit enforced in code.
- [ ] Denial table covered.
- [ ] Head and field rollback are forward commits; crash recovery tested.
- [ ] No cross-head identity rewrite.
- [ ] No unapproved main write.
- [ ] No canonical merge.
- [ ] No 2 GiB overbootstrap; budgets tested.
- [ ] No raw pointers; no secret/path leakage.
- [ ] Corpus and free chat activate many heads, measured and reported.
- [ ] Python-mirror/native-performance, activation/truth and classical-resonance
      boundaries are stated in user-facing output.

---

## 10. Honest boundaries

### 10.1 The Python mirror is not native performance

The 256-head field described here is a Python coordinator over bounded records.
It is not the native NaNLMind kernel, not a native vectorized head engine, and
not a phone/handset performance claim. `scripts/head_field_benchmark.py`
already states that its measurements are host CPython over a 2 MiB bytearray,
not native code, not an mmap file and not model training or inference. Keep
that label. Do not compare its update rate with native runner throughput, and
do not use it to claim a latency or battery result.

Python adds object overhead, GIL scheduling and `fsync` cost. The logical
budgets in this document are not an RSS guarantee. Measure physical memory,
file bytes and latency separately on each target.

### 10.2 Activation is not semantic truth

`a_i`, `score_i`, `field_i`, entropy and resonance are routing and ranking
signals. They are not truth values, entailment, source authority, confidence or
calibration. A strongly active head is not a correct head. A high-entropy field
is not a better answer. A low-entropy field is not a proof of focus.

The Boolean layer proves only that a rule follows from its premises under the
declared gates. It does not prove the premises are true. `UNKNOWN` remains
unknown; `CONFLICT` remains conflict. Never average them into a generic
confidence score. The field term cannot resolve either state.

### 10.3 Resonance is classical

The field term is a deterministic mean-field correlation over public activation
values. Pair rails use additive classical phase/amplitude/weight accumulators.
There is no quantum process, no quantum tunneling, no nonlocal effect, no
consciousness claim and no noosphere. "Resonance" is a metaphor for a classical
coupling pattern. A field entropy number is a distribution statistic, not a
physical entropy and not evidence of semantics.

### 10.4 Head count is not independence

256 logical heads are not 256 independent models, 256 native threads or 256
guaranteed experts. `docs/HEAD_POPULATION.md` shows that training all heads on
every context collapsed the head Gram rank to 1. Sparse bucket routing and the
metrics in section 7 reduce that failure mode; they do not prove semantic
specialization. Independence must be measured on held-out probes and compared
with controls. State collisions, rule collisions and route collisions can all
hide a single global behavior.

### 10.5 Memory and persistence

The scratch ring is bounded and sparse by design. Its default is small; a
larger paged backend may use the existing 1 GiB logical space. The hard cap is
1 GiB for all scratch and snapshot material, 8 MiB for main, and 64 MiB for
audit. These are logical/allocated limits, not a promise that Python RSS stays
under the same number. A 2 GiB overbootstrap copy or a `256 x 8 MiB` mirror is
forbidden. The phone profile must work with no 2 GiB file present.

### 10.6 Approval and security

Approval is a policy gate, not a proof of human intent and not a secure
enclave. A host-level attacker with write access can bypass local files. The
schema validation for secrets and paths is best-effort data hygiene, not a
complete DLP system. Do not put credentials in scratch, prompts, snapshots,
logs or audit records. File permissions and OS isolation remain part of the
deployment boundary.

### 10.7 Scope of the main register

The main register is a local derived store. A successful commit is not a
canonical NaNLMind commit, not a six-runner promotion and not a merge. The
canonical merge gate remains closed. Rollback is a new forward generation; it
does not rewind history, revive approvals or restore a canonical state.

### 10.8 No raw pointers, but Python objects exist

Persisted records contain no pointers. In memory, Python necessarily uses
object references. Those references must not be serialized, accepted from a
candidate, written to a snapshot, or treated as stable identity. Identity is
the integer head ID and its permanent address in `cells.py`, not `id(obj)`.

### 10.9 Boolean completeness wording

XOR, AND and OR are mandatory reference functions, not a claim that those three
operators alone form a complete basis. The implementation retains all 16
explicit truth tables and the unary complement. Any wording that says "three
gates generate everything" without constants or complement is inaccurate and
must not be used.

### 10.10 What this document does not claim

It does not claim that the current prototype already satisfies the contract,
that the field improves answer quality, that the Python path matches native
speed, that activation is meaning, or that a local commit is canonical. It
defines the implementation target and the evidence required before those
claims could be made.

---

## Appendix A. Default constants

| Constant | Value | Meaning |
|---|---:|---|
| `HEAD_COUNT` | 256 | logical head IDs 0..255 |
| `CONTENT_HEADS` | 255 | IDs 1..255 |
| `LAYER_COUNT` | 10 | L0..L9 |
| `BUCKETS` | 15 | content buckets of 17 heads |
| `ACTIVE_PER_LAYER` | 16 | head 0 + 15 content winners |
| `ACTIVE_PER_TURN_MAX` | 64 | distinct heads across L0..L9 |
| `FIELD_LAMBDA` | 0.15 | mean-field strength |
| `FIELD_DAMPING` | 0.5 | field damping |
| `FIELD_CAP` | 0.25 | activation impulse cap |
| `FIELD_MAX_EDGES` | 1,024 | sparse coupling limit |
| `FIELD_MAX_DEGREE` | 8 | neighbours per head |
| `HEAD0_ACTIVATION_CAP` | 0.05 | prior share limit |
| `SCRATCH_BUDGET_BYTES` | 1 GiB | hard scratch + snapshot cap |
| `SCRATCH_DEFAULT_BYTES` | 40 MiB | default ring payload |
| `MAIN_REGISTER_BYTES` | 8 MiB | total main register |
| `MAIN_PER_HEAD_BYTES` | 16 KiB | per-head record cap |
| `HEAD_SNAPSHOT_BYTES` | 16 KiB | per-head snapshot cap |
| `FIELD_SNAPSHOT_BYTES` | 4 MiB | field snapshot cap |
| `SNAPSHOT_RETAIN` | 2 | per head and field |
| `AUDIT_BUDGET_BYTES` | 64 MiB | bounded audit |
| `MAX_RECORD_BYTES` | 64 KiB | one scratch record |
| `MAX_TEXT_BYTES` | 1 MiB | one observation |
| `MAX_RELATIONS` | 512 | relations per observation |
| `MAX_STAGED_CANDIDATES` | 64 | per pass |
| `APPROVAL_TTL` | 1 hour | default; 24 hours maximum |

## Appendix B. Approval/receipt example

```json
{
  "schema": "nanlm.head-commit-receipt/1",
  "status": "committed",
  "head": 42,
  "candidate_digest": "64-hex-sha256",
  "evaluation_digest": "64-hex-sha256",
  "approval_id": "64-hex-sha256",
  "base_generation": 17,
  "new_generation": 18,
  "previous_state_digest": "64-hex-sha256",
  "new_state_digest": "64-hex-sha256",
  "boolean_base_digest": "64-hex-sha256",
  "canonical_commit": false,
  "main_updated": true,
  "projection": "current | stale",
  "committed_tick": 123456
}
```

A denial returns `committed: false`, `main_updated: false`, the same base
generation and a reason code. It never returns a path or a secret.

## Appendix C. Status example

```json
{
  "schema": "nanlm.head-field-status/1",
  "generation": 18,
  "heads": 256,
  "layers": 10,
  "active_heads": 16,
  "active_head_count": 16,
  "head0_share": 0.02,
  "max_head_share": 0.07,
  "activation_entropy_norm": 0.41,
  "field_entropy_norm": 0.23,
  "effective_rank": 12.4,
  "mean_cosine": 0.11,
  "state_digest_collisions": 0,
  "rule_fingerprint_collisions": 0,
  "route_set_collision_rate": 0.08,
  "scratch_bytes": 1234567,
  "scratch_budget_bytes": 1073741824,
  "main_generation": 18,
  "main_register_bytes": 8388608,
  "canonical_merge_allowed": false,
  "main_register_frozen": false,
  "field": {
    "edges": 12,
    "lambda": 0.15,
    "rounds": 1
  }
}
```

The status contract contains no filesystem path, token, key or raw pointer.

## Appendix D. Reason-code set

```text
approval_required
approval_mismatch
approval_expired
approval_replay
self_approval
stale_generation
base_digest_mismatch
boolean_eval_failed
boolean_base_changed
cross_head_scope
identity_rewrite
head_zero_immutable
field_scope_violation
secret_or_path
raw_pointer
canonical_merge
memory_budget
split_brain
snapshot_unavailable
field_root_mismatch
projection_stale
invalid_record
unknown_head
unknown_layer
unknown_token
token_consumed
```

A denial may include a human-readable message, but the reason code is the
stable machine contract and must not leak content.
