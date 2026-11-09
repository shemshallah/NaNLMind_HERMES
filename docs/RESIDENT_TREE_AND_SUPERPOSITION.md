# Resident tree, mirror links and resonance queries

Status: **design/proposed contract**. The tree, record linking, resonance query
and superposition collapse described here are not implemented. The existing
math modules named below are real, but they are not wired into a resident
universe. This document does not report a benchmark, accuracy or test result.

“Resonance” here means deterministic multi-channel scoring of a typed
candidate subgraph. “Superposition” means a linear combination of candidate
amplitudes followed by a deterministic collapse. Neither term implies physical
quantum behavior.

## 1. Tree layout

The resident tree is an immutable, content-addressed graph rooted at one
per-universe hub. It reuses the existing `leaf:<sha256>` and `hub:<name>`
semantics from `nanlmind_hermes/mth_leaves.py`; it does not invent a second
leaf format.

```text
hub:resident.root
├── hub:resident.lexical
│   ├── leaf:<sha>  "copper"          (token/lexical leaf)
│   └── leaf:<sha>  "element"
├── hub:resident.relation
│   └── rel:<sha>                     (typed subject-predicate-object)
├── hub:resident.pair
│   ├── pair:<rank>                   (unordered distinct heads)
│   │     s+ / s-, mirror_rank, mirror_addr
│   └── pair:<rank>/mirror            (linked mirror record)
├── hub:resident.head
│   ├── head:17
│   └── head:136                      (reverse8(17))
├── hub:resident.evidence
│   └── leaf:<sha>                    (source-bound evidence)
├── hub:resident.corpus
│   └── leaf:<sha>                    (optional training chunk)
└── hub:resident.function
    └── hub:mth.* / kernel:mth.*      (existing MTH tree; referenced)
```

Namespaces:

| name | meaning | mutable? |
|---|---|---|
| `leaf:<sha256>` | immutable content-addressed leaf | no |
| `hub:<name>` | immutable typed namespace node | no |
| `rel:<sha256>` | immutable relation record | no |
| `he:<sha256>` | immutable hyperedge record | no |
| `pair:<rank>` | pair cell for `0 <= rank < 32640`, distinct heads | append-only |
| `pair:<rank>/mirror` | rank-mirror cell | append-only |
| `head:<0..255>` | head record | append-only |
| `head:<h>/mirror` | `reverse8(h)` link | derived |

Local `u32` handles are an internal index detail. Portable APIs and bundles use
the string/digest names above. A handle from one file is never valid in
another file or bundle.

The hub parent chain must be acyclic and must terminate at
`hub:resident.root`. Every leaf has a parent; every relation, pair and hyperedge
has a hub. Deleting or rewriting a hub is forbidden; a new hub requires an
explicit versioned create operation.

## 2. Leaf record

A leaf remains immutable and content-addressed. The resident record adds index
fields without changing the existing MTH leaf identity:

```text
leaf:<sha256>
  kind:       token | lexical | evidence | corpus | hub | relation-view
  parent:     hub:<name> or leaf:<sha256>
  head:       u16 0..255
  mirror:     reverse8(head)
  resolution: L0..L9
  feature:    256 float32 (optional, but exactly 256 if present)
  token:      token digest (optional)
  payload:    arena offset/length/digest (optional)
  source:     SHA-256 of the source that produced the leaf
  links:      relation/hyperedge/pair entries are index postings, not leaf
              bytes; the leaf itself is never mutated
```

Rules:

- A leaf's content hash covers its canonical semantic payload. Two leaves with
  the same digest must have identical bytes; a collision with different bytes
  is `leaf_digest_conflict`.
- Updating a word, definition or evidence row creates a new leaf and a new
  relation. It never edits the old leaf.
- The leaf's relation/hyperedge counts are derived from the index, not trusted
  from a mutable counter.
- A relation may reference a `hub:mth.*` leaf as an operator/function leaf, but
  execution still belongs to `mth_leaves.py`/N16I; the resident tree does not
  execute native code.

## 3. Relation link closure

Every relation record is an immutable typed edge with a **mandatory closure**.
For a relation `r`:

```text
r.subject   -> leaf:S
r.predicate -> leaf:P
r.object    -> leaf:O

heads:
  hs = S.head, hp = P.head, ho = O.head
  ms = reverse8(hs), mp = reverse8(hp), mo = reverse8(ho)

pair cells (distinct heads):
  pair_sp = rank_of_pair(min(hs,hp), max(hs,hp))
  pair_so = rank_of_pair(min(hs,ho), max(hs,ho))
  pair_po = rank_of_pair(min(hp,ho), max(hp,ho))

mirrors:
  mirror_sp = 32639 - pair_sp
  mirror_so = 32639 - pair_so
  mirror_po = 32639 - pair_po
  mirror_addr(pair) = reverse16(canonical_addr(pair))

hyperedge:
  he = <content-addressed hyperedge>
  primary members = [head:hs, head:hp, head:ho,
                     pair:pair_sp, pair:pair_so, pair:pair_po]
  secondary mirror members = [pair:mirror_sp, pair:mirror_so, pair:mirror_po]
```

`pair_so` is the **primary** pair cell for the relation; `pair_sp` and
`pair_po` are mandatory as well. A relation with only one or two pair links is
rejected with `incomplete_pair_links`. The hyperedge's primary arity is six for
an `N16REL/1` relation: three head leaves plus three pair cells. The three
mirror cells are secondary links, not extra primary arity.

```text
                 leaf:S ───── pair:SP ───── leaf:P
                   \             |             /
                    \        pair:SO          /
                     \           |           /
                      \      pair:PO        /
                       \         |         /
                        [ hyperedge HE ]
                  members: head:S, head:P, head:O,
                           pair:SP, pair:SO, pair:PO
                  mirror members: pair:SP', pair:SO', pair:PO'
                  rails on every pair: s+ (forward), s- (reverse)
```

The resident relation contract requires three **distinct** routed heads,
matching native `N16REL/1` training. If a source observation routes two or more
roles to the same head, it is stored as an observation leaf and cannot be
inserted as a relation until rerouted; the insertion is rejected with
`duplicate_relation_head`. The 256 fixed-point cells `(a,a)` remain chart/head
state, not relation pair vertices. This keeps the pair-rank domain exact and
avoids a sentinel that other code could misread as a valid rank.

### 3.1 Mandatory fields

| field | required | validation |
|---|---|---|
| `relation_id` | yes | content-addressed; duplicate ID must match bytes |
| `subject_leaf`, `predicate_leaf`, `object_leaf` | yes | all resolve; no dangling ref |
| `subject_head`, `predicate_head`, `object_head` | yes | 0..255; route matches leaves |
| `subject_mirror_head`, `predicate_mirror_head`, `object_mirror_head` | yes | `reverse8` of the head |
| `pair_sp`, `pair_so`, `pair_po` | yes | valid rank; `pair_at` matches heads |
| `mirror_sp`, `mirror_so`, `mirror_po` | yes | `32639 - pair_rank`; mirror record exists |
| `pair_addr_sp/so/po` | yes | `canonical_addr`/`reverse16` match |
| `hyperedge_id` | yes | exists; six primary members match |
| `resolution` | yes | integer 0..9, not bool |
| `polarity` | yes | supports/refutes/neutral |
| `confidence` | yes | finite, 0..1 |
| `source_digest` | yes | 64 hex; binds the evidence source |
| `generation` | yes | producer generation; not trusted as canonical |
| `record_digest` | yes | verifies canonical bytes |

The relation record is 512 bytes in the resident file; the portable bundle uses
the same semantic fields in canonical JSONL with content digests instead of
local handles.

### 3.2 Link insertion is append-only

Inserting a relation appends:

1. the relation record;
2. its hyperedge record;
3. new pair records or pair deltas;
4. index postings from each leaf/head/pair/hyperedge to the relation.

It does not mutate the leaf records or rewrite an existing relation. A
correction is a new relation with a new source digest and, if appropriate, a
`supersedes` evidence link. Contradictory relations remain visible.

## 4. Mirror and rails

The three mirrors are distinct:

| mirror | definition | domain |
|---|---|---|
| address mirror | `reverse16(canonical_addr)` | 16-bit chart address |
| rank mirror | `32639 - rank` | 0..32639 pair rank |
| head mirror | `reverse8(head)` | 0..255 head |

`reverse8` has 16 palindromic fixed heads (`0x00`, `0x11`, …, `0xFF`), so those
head mirrors are self-links. The rank mirror has no fixed point because 32,639
is odd. Do not use `255 - i` complement pairing as a rank mirror; it is a
different involution and fails as a general temporal mirror.

Each pair cell carries the two temporal rails:

```text
s+ : forward rail, past -> future
s- : reverse rail, future -> past
```

A commit advances the frame and copies the committed `s+` word into `s-`.
For complex pair couplings, `A_mirror = conj(A)`. For integer rails, the
mirror cell carries the corresponding copied word. `mirror_residual` is the
maximum violation; it must be zero for exact integer rails and within the
declared float tolerance for complex couplings (the design default is `1e-6`,
but no measurement is claimed here).

The mirror link is not a substitute for a relation. A relation must still cite
the primary pair, the mirror pair, the hyperedge and the leaves. Mirroring is a
structural consistency check; it does not prove the relation is true.

## 5. Pair graph and hyperedges

The 32,640 distinct-head pair cells are the vertices of the Johnson graph
`J(256,2)`:

- vertex count: `C(256,2) = 32,640`;
- degree: `2*(256-2) = 508`;
- two vertices are adjacent when their pairs share a head;
- the cycle-space dimension is `32,640 - 255 = 32,385`;
- the 256 fixed-point cells `(a,a)` are chart/head state, not vertices of this
  pair graph and not relation pair cells;
- no `32,640 x 32,640` dense matrix is materialized.

This is the same graph used by `pair_quantum.py` and `cycle_space.py`. Pair
adjacency is a candidate-generation path, not a claim that every adjacent pair
is semantically related.

A hyperedge is an n-ary membership record:

```text
he:<sha256>
  arity: 2..8 primary members
  members[8]:        leaf/head/pair/hyperedge refs
  member_kinds[8]:   typed
  mirror_members[8]: secondary mirror refs (not counted in arity)
  relation_id:       the relation that created it (if any)
  evidence_ref:      source/evidence handle
  source_digest:     required
  record_digest:     required
```

An N16REL relation hyperedge has exactly the six primary members shown in §3.
Every member must resolve. A hyperedge with a dangling member, duplicate
primary member, arity below 2 or above 8, or an unresolved mirror is rejected.

### 5.1 Query working-set budget

These are design targets, not measurements:

| item | default | hard |
|---|---:|---:|
| candidate cap `k` | 256 | 4,096 |
| query feature vector | 1 KiB (256 f32) | 1 KiB |
| query phase vector | 1 KiB (64 complex f32 pairs) | 1 KiB |
| observed pair edges | 4,096 | 32,640 |
| pair adjacency expansion per seed | 508 | 508 |
| per-query score table | 1 MiB | 16 MiB |
| per-query working set | 8 MiB | 32 MiB |
| dense pair matrix | 0 B | 0 B |

The pair graph is always sparse. A query must return `candidate_limit` rather
than allocate an unbounded result set, and it must never materialize a
`32,640 x 32,640` matrix. Page-cache usage follows
[UNIFIED_RESIDENT_UNIVERSE.md](UNIFIED_RESIDENT_UNIVERSE.md).

## 6. Query pipeline: exact plus resonance

The resident query is not an exact-key lookup with a decorative score. Exact
lookup is one candidate source and one scoring channel; the default resonant
mode also uses features, phase, cycle and gauge channels.

```text
query (text / leaf refs / feature / phase / observed pairs)
  |
  +--> exact triple index (S, P, O)
  +--> token/leaf postings
  +--> feature bucket index (bounded)
  +--> pair adjacency J(256,2)
  +--> hyperedge membership
  |
  v
candidate union, dedupe, cap K (default 256, max 4096)
  |
  v
per-channel scores: exact, dot, cosine, phase, cycle, gauge, rail
  |
  v
normalize + weighted fusion
  |
  +--> ranked candidates (deterministic)
  +--> superposition amplitudes / collapse (optional)
  |
  v
response: candidates, scores, evidence, generation, abstain reason
```

A query object:

```text
Query {
  text?: string,
  leaf_refs?: [leaf-id],
  token_ids?: [token-id],
  feature?: float32[256],
  phase?: complex[64] or real[64][2],
  observed_pairs?: [(rank, complex)],
  mode: "exact" | "resonant" | "auto",
  k: int <= 4096,
  generation: pinned u64,
  weights?: {exact, dot, cosine, phase, cycle, gauge, rail}
}
```

Candidate generation is bounded. `mode="exact"` returns only exact index
matches and is the compatibility path for deterministic tests.
`mode="resonant"` returns exact candidates plus resonant candidates and scores
them. `mode="auto"` uses exact candidates first and re-ranks them resonantly,
but may add candidates only if the exact set is below `k` and the query is not
marked exact-only. A response must say which mode was used.

## 7. Scoring channels

All scores are normalized to `[0,1]` before fusion. A non-finite score, a
dimension mismatch or a zero-norm vector is a clean failure/abstention, never
NaN propagation.

### 7.1 Exact

```text
exact = 1.0   if S, P and O all match
exact = 0.5   if S and P match, O is unconstrained or wildcard
exact = 0.25  if only S matches
exact = 0.0   otherwise
```

The exact channel is a feature, not a veto. A relation with a better exact
match can still lose to a strongly supported resonant candidate; the response
must expose both channel scores.

### 7.2 Dot and cosine

For 256-dimensional feature vectors `x`, `y`:

```text
dot(x,y)    = Σ_i x_i y_i
cos(x,y)    = dot(x,y) / (||x||_2 ||y||_2)
```

If either norm is zero, `cos = 0` and the candidate is marked `degenerate`.
Dot is normalized for fusion by the maximum absolute dot among the candidate
set, or by `||x|| ||y||` when a stable bound is required; the chosen
normalization is recorded. Cosine has a natural `[0,1]` mapping after the
negative range is handled explicitly: `cos01 = (cos + 1)/2`.

Dot/cosine measure geometric similarity in the chosen feature projection. They
do not establish semantic equivalence, truth or entailment.

### 7.3 Phase

For unit phasors `q_j`, `p_j`, the gauge-invariant score is

```text
phase_gauge = |Σ_j conj(q_j) p_j|^2 / N^2
```

with `N` channels. The real cosine/sine equivalent is

```text
C = Σ_j cos(q_j - p_j)
S = Σ_j sin(q_j - p_j)
phase_gauge = (C^2 + S^2) / N^2
```

The naive phase-sensitive control is

```text
phase_naive = Re Σ_j conj(q_j) p_j / N
```

A global phase rotation of every query channel leaves `phase_gauge` unchanged
and changes `phase_naive`. Independent per-channel noise degrades both; the
gauge form removes only the global nuisance phase. This is classical signal
processing, matching `phase_memory.py`.

### 7.4 Cycle

For an observed edge field `e` on the pair graph, split it into a vertex
gradient and a divergence-free cycle:

```text
divergence(e)[i] = Σ_j e_ij
project_cycle(e) = e - gradient(potential)
```

The cycle score between query cycle `q_c` and prototype cycle `p_c` is the
normalized real inner product:

```text
cycle_raw = Re <q_c, p_c> / (||q_c|| ||p_c||)
cycle01   = (cycle_raw + 1) / 2
```

`project_cycle` must produce near-zero divergence (the existing
`cycle_space.py` tests use a numerical tolerance; no new measurement is
claimed). Cycle consistency is a structural consistency check, not factual
truth.

### 7.5 Gauge

Given query and prototype edge fields and an observed edge mask, fit a nuisance
vertex potential `u` minimizing

```text
Σ_observed (d_e - (u_j - u_i))^2 + ridge * ||u||^2
```

where `d_e = query_e - prototype_e`. The residual after the fit is
`gauge_residual`; the naive residual is `Σ_observed d_e^2`. The gauge score is
the monotone map

```text
gauge01 = 1 / (1 + gauge_residual)
```

The gauge-aware residual removes an arbitrary vertex gauge; it does not create
information. It is the strong baseline from `gauge_memory.py`, not a claim that
the prototype is true.

### 7.6 Rail

For a candidate relation whose pair cells carry `s+`/`s-`, rail compatibility is

```text
rail01 = 1 / (1 + |s+ - s-| / (1 + |s+| + |s-|))
```

or a declared equivalent. The rail score is only available when the pair
records are present and mirror-consistent; otherwise the channel is absent and
the fusion weights are renormalized. `s+` and `s-` are temporal, not mirror
evidence.

## 8. Fusion, abstention and superposition

### 8.1 Fusion

Default weights (design defaults, not tuned claims):

```text
exact 0.35, cosine 0.20, dot 0.10, phase 0.15,
cycle 0.10, gauge 0.10, rail 0.00 (optional)
```

For present channels `c` with weights `w_c`:

```text
score = Σ_c w_c * score_c / Σ_c w_c
```

The response records the weights, the per-channel scores and any missing
channels. Exact mode uses only `exact`; resonant mode uses all available
channels. `rail` is opt-in because it depends on temporal state.

Abstain when:

- there are no candidates;
- every channel is missing or below `min_score` (design default `0.25`);
- the top-two fused scores differ by less than `margin` (design default
  `0.05`) and the query requests a decisive answer;
- a required dimension/norm check fails;
- the pinned generation no longer has a valid index root.

Abstention is a valid result. It is not a zero score disguised as an answer.

### 8.2 Superposition

For a candidate set with relation amplitudes `a_i` (supports positive,
refutes negative, neutral zero), linear accumulation is

```text
A(c) = Σ_{i: candidate c} a_i
E(c) = |A(c)|^2
p(c) = E(c) / Σ E
```

Opposite signs cancel; duplicate event IDs are deduplicated before summing, so
replay does not amplify a claim. The merge is commutative and associative at
the event-set layer. If `Σ E = 0` (all amplitudes cancel or are zero), the
result is `abstain`, not a division by zero. This is classical signed weighted
voting.

Collapse:

- **MAP:** `argmax_c |A(c)|`, deterministic.
- **Born-style sample:** draw from `p(c)` with a PRNG seeded by
  `SHA256(state_digest || generation || query_id)`. The same state, generation
  and query produce the same collapse. This is a deterministic classical
  sample, not a quantum measurement.
- **Abstain:** if the top-two energy margin is below the configured threshold,
  or if the top candidate is tied, return `abstain` rather than an arbitrary
  winner.

The response must say `collapse: "map" | "seeded_sample" | "abstain"` and
include the seed commitment, not a claim that a physical superposition was
measured.

## 9. What this differs from a key-value store

| dimension | key-value store | resident tree |
|---|---|---|
| key | byte string | typed leaf/relation/pair/head ref |
| value | opaque blob | immutable leaf + relation + hyperedge subgraph |
| write | overwrite/delete | append immutable record; new generation |
| query | exact key lookup | exact + token/feature/pair/hyperedge candidate union |
| ranking | none or caller-supplied | deterministic multi-channel scores |
| contradictions | last writer wins | retained as separate relations and evidence |
| result | one value or miss | ranked candidates, channel scores, provenance, abstain |
| consistency | key-level | generation-bound snapshot and CAS |
| missing data | miss | `abstain` with a reason, never a fabricated match |
| semantics | none | typed links; still not truth |

A key-value store can be an index inside the resident universe (exact lookup
is one channel), but the resident universe is not reducible to “key → blob”.
Conversely, the extra structure does not make it a model, a proof system or a
database with arbitrary transactions.

## 10. What it does not prove

- **Not truth.** A relation can be false, contradictory or teacher-generated
  noise. Confidence is evidence metadata, not a calibrated probability.
- **Not semantic understanding.** Dot/cosine similarity is geometric, not
  entailment. Phase and gauge scores are signal-processing invariants, not
  meaning.
- **Not new information.** Gauge fitting removes a nuisance potential; cycle
  projection removes a gradient; neither creates information.
- **Not quantum.** Phase, superposition and collapse are classical arithmetic
  and deterministic sampling. No quantum hardware, speedup, entanglement or
  consciousness is claimed.
- **Not a proof system.** Structural consistency and mirror invariants do not
  prove a mathematical theorem or a factual claim.
- **Not learning success.** A ranked retrieval or a collapsed answer is not
  evidence that a model learned, generalized or improved on held-out data.
- **Not canonical.** The tree is local-derived. It cannot promote BS0 state.
- **Not a security boundary by itself.** Hashing and digests detect integrity
  changes; they do not authenticate a source unless a signature/receipt is
  verified. Access control is a separate capability check.
- **Not an exact-recall guarantee.** Candidate generation is bounded; a
  resonant query can miss a valid relation that is not in the candidate set.
  Exact mode is the compatibility path.
- **Not an accuracy improvement.** No matched experiment shows that resonance
  beats exact lookup or any baseline. Such a claim requires a pre-registered
  held-out comparison with abstention and false-positive accounting.
- **Not a performance claim.** No latency, throughput, memory or accuracy
  number is asserted here.

## 11. Failure modes

| failure | detection | response |
|---|---|---|
| dangling subject/predicate/object leaf | ref resolution | `dangling_reference`; reject relation |
| incomplete pair links | missing `pair_sp/so/po` | `incomplete_pair_links` |
| invalid pair rank | `pair_at` range/order | `pair_rank_invalid` |
| pair/mirror mismatch | rank/address/mirror check | `pair_mirror_mismatch` |
| duplicate head in N16REL | head equality | `duplicate_relation_head` |
| hyperedge member mismatch | arity/member set | `hyperedge_mismatch` |
| hub cycle | parent walk | `hub_cycle` |
| leaf content collision | digest bytes | `leaf_digest_conflict` |
| feature dimension/non-finite | validation | `feature_invalid` |
| phase dimension mismatch | validation | `phase_dimension_mismatch` |
| zero-norm vector | norm check | score 0, `degenerate` |
| NaN/Inf score | finite check | `score_nonfinite`; abstain |
| candidate overflow | `> k` | `candidate_limit`; bounded truncation with reason |
| stale index | generation/root | `index_stale`; rebuild |
| generation mismatch | pinned snapshot | `generation_mismatch` |
| cross-universe ref | namespace/owner check | `cross_universe_reference` |
| superposition tie | margin check | `abstain`, no arbitrary winner |
| mirror residual too high | threshold | `mirror_violation`; candidate excluded |
| unsupported mode | enum validation | `query_mode_invalid` |

## 12. Invariants

1. Leaves, relations and hyperedges are immutable; updates append, never
   overwrite.
2. Every relation resolves subject/predicate/object leaves, three distinct-head
   pair cells, three mirrors and one hyperedge.
3. Every pair rank is in `0..32639`; `pair_at` matches the cited heads; the rank
   mirror is `32639 - rank`; the address mirror is `reverse16`.
4. Every hyperedge member resolves; primary arity is 2..8; an N16REL hyperedge
   has exactly six primary members.
5. Hub parents form a DAG terminating at `hub:resident.root`.
6. No leaf/relation/hyperedge is visible across a generation boundary unless
   pinned by that generation.
7. Scores are finite. Zero norms, dimension mismatches and missing channels are
   explicit, not NaN or silent zeros.
8. Exact candidates and resonant candidates are distinguishable in the
   response; resonance never fabricates an exact match.
9. `phase_gauge` is invariant under a global phase; `gauge_residual` removes a
   vertex potential; cycle projection removes the gradient. These are controls,
   not proof of meaning.
10. Superposition addition is commutative and associative; replay of the same
    event IDs does not double-count; collapse with the same
    `(state_digest, generation, query_id)` is reproducible.
11. Every response carries `proof:false`, `truth_claim:false`,
    `quantum_claim:false` and `canonical_claim:false`.
12. No dense `32,640 x 32,640` matrix is allocated; pair adjacency is sparse.
13. The resident tree never executes a relation payload as native code, shell
    command or arbitrary pointer.

## 13. Acceptance tests (proposed, not executed)

These tests do not exist yet. They are the required gates for a resident tree
and query implementation. No pass/fail result is asserted.

| ID | test | assertion |
|---|---|---|
| T-01 | `test_resident_tree.py::test_relation_closure` | six-member hyperedge; three pair/mirror links; all refs resolve |
| T-02 | `test_resident_tree.py::test_dangling_leaf` | missing subject/predicate/object -> `dangling_reference` |
| T-03 | `test_resident_tree.py::test_pair_mirror` | wrong rank/address/mirror -> `pair_mirror_mismatch` |
| T-04 | `test_resident_tree.py::test_duplicate_heads` | equal routed heads are rejected as a relation, not given a fabricated pair rank |
| T-05 | `test_resident_tree.py::test_hyperedge_arity` | 6 members for N16REL; duplicate/missing member rejected |
| T-06 | `test_resident_tree.py::test_hub_dag_and_immutability` | hub cycle and leaf rewrite rejected |
| T-07 | `test_resident_tree.py::test_leaf_digest_collision` | same digest/different bytes -> `leaf_digest_conflict` |
| T-08 | `test_resident_query.py::test_exact_parity` | exact mode matches the legacy ledger query for the same rows |
| T-09 | `test_resident_query.py::test_resonant_includes_exact` | resonant mode exposes exact plus resonant channel scores |
| T-10 | `test_resident_query.py::test_dot_cosine_edges` | zero norm -> 0/degenerate; cosine range correct |
| T-11 | `test_resident_query.py::test_phase_gauge` | global phase leaves `phase_gauge` unchanged; naive control changes |
| T-12 | `test_resident_query.py::test_gauge_residual` | synthetic prototype + vertex potential: gauge residual < naive residual |
| T-13 | `test_resident_query.py::test_cycle_projection` | cycle projection divergence within tolerance |
| T-14 | `test_resident_query.py::test_superposition_order_and_tie` | A→B = B→A; exact tie -> `abstain` |
| T-15 | `test_resident_query.py::test_seeded_collapse` | same digest/generation/query -> same collapse; changed seed -> allowed change |
| T-16 | `test_resident_query.py::test_generation_visibility` | old snapshot cannot see new relation/pair/hyperedge |
| T-17 | `test_resident_query.py::test_cross_universe_ref` | foreign ref rejected |
| T-18 | `test_resident_query.py::test_non_claims` | response has `proof:false`, `truth_claim:false`, `quantum_claim:false`, `canonical_claim:false` |
| T-19 | `test_resident_query.py::test_bounded_candidates` | candidate count capped; no dense pair matrix allocation |
| T-20 | `test_resident_query.py::test_pair_graph_degree` | J(256,2) degree 508 and cycle dimension 32,385 |

T-01, T-03, T-04, T-05, T-11, T-12, T-14, T-15 and T-18 are mandatory before
any query API is exposed.

## 14. Exact file/module ownership

| path | status | responsibility |
|---|---|---|
| `nanlmind_hermes/resident/tree.py` | proposed | hub/leaf/relation/pair/mirror/hyperedge closure and insertion |
| `nanlmind_hermes/resident/records.py` | proposed | relation/hyperedge/pair/leaf record codecs |
| `nanlmind_hermes/resident/index.py` | proposed | exact triples, postings, pair adjacency, hyperedge membership, feature buckets |
| `nanlmind_hermes/resident/query.py` | proposed | candidate union, fusion, abstention, generation pin |
| `nanlmind_hermes/resident/resonance.py` | proposed | dot/cosine/phase/cycle/gauge/rail scoring wrappers |
| `nanlmind_hermes/resident/superposition.py` | proposed | signed amplitude accumulation and deterministic collapse |
| `nanlmind_hermes/mth_leaves.py` | **exists** | immutable `leaf:`/`hub:` semantics and N16I execution receipts |
| `nanlmind_hermes/gauge_memory.py` | **exists** | gauge residual/retrieval math |
| `nanlmind_hermes/phase_memory.py` | **exists** | phase/gauge scoring |
| `nanlmind_hermes/cycle_space.py` | **exists** | gradient/cycle decomposition and cycle inner product |
| `nanlmind_hermes/temporal_spine.py` | **exists** | rank mirror `32639-r` and coupling update |
| `nanlmind_hermes/boolean_wave.py` | **exists** | Boolean amplitude functions and polarity caution |
| `nanlmind_hermes/word_writer.py` | **exists** | pair rank/payload packing and word-leaf update |
| `nanlmind_hermes/concept_matrix.py` | **exists** | pair-cell concept accumulation |
| `tests/test_resident_tree.py` | proposed | T-01..T-07 |
| `tests/test_resident_query.py` | proposed | T-08..T-20 |

The resident modules must be wrappers/adapters over these existing math modules,
not reimplementations. In particular, do not duplicate `mth_leaves.py` leaf
format, `temporal_spine.py` rank mirror, or the N16I execution path.

## 15. Integration status and non-claims

- **Implemented today:** the existing leaf/hub, gauge, phase, cycle, rank-mirror
  and pair-cell modules named in §14, plus the resident constants. None is wired
  into a resident tree or query.
- **Proposed:** all `resident/tree.py`, `records.py`, `index.py`, `query.py`,
  `resonance.py` and `superposition.py` behavior and all tests above.
- **Not claimed:** a trained semantic model, a proof system, a quantum system,
  a canonical merge, a performance result, an accuracy result, or a working
  end-to-end intake.
- **Correct description:** a bounded, immutable typed graph with deterministic
  exact and resonance retrieval, explicit provenance, generation visibility and
  honest abstention.
