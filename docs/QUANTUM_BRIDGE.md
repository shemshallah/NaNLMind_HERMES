# Quantum-memory bridge

`nanlmind_hermes/quantum_bridge.py` is a read-only composition layer over the
existing classical gauge, cycle, phase, pair-space, spectral and robust-recovery
modules. It does **not** create a new model, a new quantum system, or live
measurements.

Every value it emits is:

- classified as `local/derived`;
- bound to a deterministic `input_sha256`;
- produced without writing canonical state;
- produced without calling NANNET, the router, or CLOCKED;
- accompanied by an `available`/`reason` pair when the input does not support
  that section.

It makes no quantum-speedup or quantum-proof claim. The word "quantum" in the
surrounding module names refers to mathematical analogies already documented by
those modules; the bridge does not upgrade them into physical claims.

## Files

- `nanlmind_hermes/quantum_bridge.py` — the bridge implementation.
- `tests/test_quantum_bridge.py` — deterministic, mutation, fail-closed and
  bounded-output tests.
- `docs/QUANTUM_BRIDGE.md` — this contract.

Existing modules and `nanlmind_hermes/__init__.py` are unchanged.

## Input contract

```python
from nanlmind_hermes.quantum_bridge import QuantumMemoryBridge

bridge = QuantumMemoryBridge(
    memory_rows=rows,              # optional bounded iterable of dicts
    concept_graph=graph,           # optional ConceptMatrix or edge graph
    semantic_nodes=nodes,          # optional WordNet-shaped mapping
    semantic_start=0,              # required with semantic_nodes
)
summary = bridge.summary()         # small deterministic digest
full = bridge.to_dict()            # full deterministic summary
```

### `memory_rows`

Each item is either:

1. a flattened plugin-memory payload with `subject`, `predicate`, `object`,
   and optional `kind`, `confidence`, `provenance`, `expression`; or
2. a record of the shape `{"item": {...payload...}, "source_digest": ...}`.

A usable row must have non-empty subject, predicate and object after stable
case-folding. Labels are bounded to 96 characters; longer labels keep a
16-hex-digit digest suffix so distinct long labels do not collapse silently.
The bridge copies/reads the row; it never edits it. `expression` is not
evaluated, emitted, or interpreted. Rows are skipped, not guessed, when:

- the item is not a mapping;
- subject, predicate or object is missing;
- subject and object are the same (a self-loop cannot be a pair edge);
- confidence/weight is not positive.

The default confidence is `1.0` when the field is absent. Weights are clamped
to `[0, 1000]`. Unordered sets are rejected rather than silently ordered by
hash iteration, preserving deterministic output.

### `concept_graph`

Accepted forms:

- a `ConceptMatrix` instance — its `pairs`, `concepts`, `statements` and
  `saturation()` output are read, never mutated;
- a mapping with `nodes` and an `edges` / `relations` / `pairs` iterable;
- an iterable of edge mappings or `(subject, object)` /
  `(subject, predicate, object)` sequences.

Edge mappings accept the primary keys `subject`, `predicate`, `object`, plus
the aliases `s`, `p`, `o`, `source`, `target`, `from`, `to`, `relation`,
`role`, `weight` and `confidence`.

`ConceptMatrix` cells without typed roles become the predicate `related_to`.
Cells with roles emit one relation per role, sharing the cell weight. This is a
read-only projection; the matrix itself is never observed or modified.

### `semantic_nodes` / `semantic_start`

`semantic_state` is optional and is never inferred from flat memory rows. To
enable it, pass a mapping of WordNet-shaped records (`hypernyms` and/or
`pointers`) and a start offset present in that mapping. The bridge calls the
existing `semantic_state` functions and emits only norms and hashes, not the
full 256-wide vectors.

### Bounds

| argument | default | hard range |
|---|---:|---:|
| `max_rows` | 64 | 1..256 |
| `max_graph_edges` | 64 | 1..256 |
| `max_vertices` | 12 | 2..16 |
| `max_triangles` | 8 | 1..32 |
| `max_channels` | 8 | 1..16 |
| `max_patterns` | 8 | 2..16 |
| `semantic_dimensions` | 32 | 8..256 |
| `semantic_depth` | 3 | 1..8 |

Only the first `max_rows` input entries are inspected; truncation is reported
as `rows_truncated` and `rows_received`. Concept-graph edges are bounded by
`max_graph_edges`. Output lists (vertices, triangles, channels, provenance
rows, top concepts/pairs, patterns) are bounded by these arguments.

## Output contract

`to_dict()` returns a fresh deep copy. `summary()` returns a smaller digest of
the same deterministic values. Both are safe to JSON-encode.

Top-level fields:

- `schema` — `nanlmind-hermes.quantum-memory-bridge/1`;
- `locality` — always `local/derived`;
- `canonical_state_modified` — always `false`;
- `claims` — `quantum_speedup`, `quantum_proof`, `physical_quantum_system`
  always `false`;
- `provenance` — `input_sha256`, bounds, counts, used row bindings, skipped-row
  reasons, graph/semantic digests, derivation note;
- `integration` — modules reused, external calls (`[]`), and the unwired list;
- `sections` — the six section objects below.

Every section and every marked sub-block also contains:

- `locality: "local/derived"`;
- `derived: true`;
- `input_sha256` matching the top-level digest;
- `available` and `reason` (reason is empty only when available).

### `sections.gauge_cycle`

- `graph`: derived vertex count/list, observed edge count, complete-graph edge
  count, `cycle_rank` (`cycle_space.cycle_dimension`),
  `fundamental_cycle_count` (`(n-1)(n-2)/2` for the complete subgraph),
  triangle count, and observed fraction. `cycle_rank`,
  `fundamental_cycle_count` and `triangle_count` are explicitly scoped to the
  complete graph over the selected vertices; the computed invariant for the
  actual complete field is in `invariants`.
- `invariants` (available only for a complete observed pair field):
  `cycle_rank`, computed fundamental cycle count, fundamental-flux norm,
  gradient/cycle norms, max cycle divergence, gradient-cycle inner product,
  reconstruction error. Requires the complete pair field because the Hodge
  split fills missing edges with zero otherwise.
- `gauge_fixed_retrieval` (available with at least two provenance groups and at
  least three shared observed pairs): per-group cyclic scalar prototypes,
  aggregate query, gauge-aware best group and residual, naive best group and
  residual, gauge margin, and per-group residuals.
- `codec` (available only for a complete pair field): bounded random cycle
  basis, decoded coefficients and reconstruction error from
  `cycle_codec.decode_cycle` / `estimate_gauge` / `encode`.
- `cycle_state`: local ephemeral `CycleState.observe`, energy, divergence, and
  a fixed local vertex-gauge perturbation residual. Nothing is persisted.

### `sections.phase_coherence`

- `triangles_used` and bounded triple-coherence statistics from
  `cycle_space.triple_coherence` on the three encoded edges of each complete
  triangle;
- mean absolute triangle flux from `cycle_space.triangle_flux`;
- `phase_locking_order` computed from unit triangle fluxes using
  `phase_memory.complex_gauge_score`;
- `phase_memory` (available with at least two provenance groups and two common
  nonzero triangle-flux channels): `PhaseMemory` gauge/naive scores, best
  group, global-phase invariance delta, and real-channel agreement;
- `typed_loops`: bounded loop-trace diagnostic from `TypedCycleOperators` and
  `relation_operators.matmul`, using a representative predicate per edge.

The phase angle is a deterministic SHA-256 label encoding of the predicate
string, not a measured physical phase. This is stated in the output itself.

### `sections.pair_spectral`

- `pair_space` (available at three or more vertices): exact Johnson graph
  `J(n,2)` spectrum, degree, pair count, revival period, return probability at
  the revival period, and the ideal Grover query probability for the bounded
  derived subgraph. This is exact arithmetic for the subgraph, not a claim
  about the canonical K256 graph and not a speedup.
- `spectral` (available with at least two observed pair weights): bounded
  symmetric adjacency matrix, `WaveMatrix.status`, Fiedler values/gap and
  partition counts from `spectral.fiedler_partition`. Zero entries mean no
  observed relation, not a fabricated coupling.

### `sections.robust_recovery`

Available only for a complete observed pair field. It computes:

- `two_hop_median` and `two_hop_mean` over the input field;
- raw MAEs to the observed field;
- MAEs to the projected gradient (the Hodge gradient reference from the same
  input);
- median/mean disagreement;
- cycle and gradient norms, `flatness_ratio`, and a `gauge_flat` flag.

The projected gradient is a reference, not ground truth. No recovery accuracy
is claimed.

### `sections.concept_graph`

Reports the concept-graph source, node/edge counts, truncation, statements,
bounded top concepts and top pairs, role counts, and a graph SHA-256. It is
available when a readable concept graph was supplied; otherwise it fails closed
with a reason.

### `sections.semantic_state`

Reports whether an explicit WordNet-shaped mapping was supplied, ancestor-edge
count, ordinary/cycle feature norms and SHA-256 digests, dimensions, depth, and
input node count. The full vectors are deliberately not emitted.

## Exact APIs reused

| Module | APIs called by the bridge |
|---|---|
| `gauge_memory` | `project_cycle`, `gauge_residual`, `naive_residual`, `retrieve`, `divergence` |
| `cycle_space` | `edge_count`, `cycle_dimension`, `triangle_count`, `decompose`, `reconstruct`, `fundamental_fluxes`, `triangle_flux`, `triple_coherence`, `divergence`, `cycle_inner_product` |
| `cycle_codec` | `pairs`, `random_cycle_basis`, `decode_cycle`, `estimate_gauge`, `gauge_values`, `encode` |
| `cycle_state` | `CycleState.observe`, `snapshot`, `with_gauge`, `energy` |
| `phase_memory` | `PhaseMemory.store`, `scores`, `retrieve`, `rotate`, `complex_gauge_score`, `real_gauge_score` |
| `pair_quantum` | `pair_count`, `johnson_spectrum`, `johnson_degree`, `revival_period`, `return_probability`, `best_grover_queries` |
| `robust_recovery` | `two_hop_median`, `two_hop_mean`, `mean_absolute_error` |
| `spectral` | `WaveMatrix.status`, `fiedler_partition` (which internally uses `graph_laplacian` and `smallest_eigen`) |
| `semantic_state` | `ancestor_edges`, `ordinary_state`, `cycle_state` |
| `typed_cycle` | `TypedCycleOperators.path`, `reverse_path` |
| `relation_operators` | `matmul` |
| `concept_matrix` | `ConceptMatrix.pairs`, `concepts`, `statements`, `saturation` |

### Not imported

`joint_semantic_decoder` does not exist as `nanlmind_hermes/joint_semantic_decoder.py`.
The repository contains `scripts/joint_semantic_decoder.py`, a training entry
point that imports NumPy and reads a WordNet archive. It has no importable
decoder API or bundled weights, so the bridge does not claim to run it. This is
an honest integration gap, not a hidden feature.

## Fail-closed rules

| Condition | Result |
|---|---|
| no usable relations / fewer than three vertices | gauge, phase, pair-space and robust sections `available=false` |
| fewer than three concept vertices | pair-space and robust sections `available=false` |
| incomplete observed pair field | Hodge invariants, cycle codec and robust recovery `available=false`; graph metadata may still be available |
| fewer than two provenance groups | gauge-fixed retrieval `available=false` |
| fewer than three shared observed pairs | gauge-fixed retrieval `available=false` |
| no complete triangle of complex edges | phase section `available=false` |
| fewer than two common triangle-flux channels | phase-memory readout `available=false` |
| no ConceptMatrix/graph | concept-graph section `available=false` |
| no explicit semantic nodes/start | semantic-state section `available=false` |
| malformed concept graph | graph section `available=false` with a reason; memory rows are still processed |

Unsupported inputs therefore return flags and reasons, not invented numbers.

## Determinism and non-mutation

- Relations are sorted by a stable key before aggregation.
- Vertices are selected by `(count descending, label ascending)` and indexed by
  sorted label.
- Groups, triangles, channels, top concepts and top pairs are sorted.
- `random_cycle_basis` is seeded with the configured integer `seed`.
- Raw-row and relation digests are order-independent for the selected set.
- `to_dict()` returns a deep copy; callers cannot mutate the bridge's stored
  summary through it.
- No input row, graph, `ConceptMatrix`, or `semantic_nodes` mapping is written.

## What remains unwired

These are concrete integration points outside this bridge:

1. **Chat** — `tools.chat` and the plugin's `handle_chat` do not call
   `QuantumMemoryBridge`.
2. **Learning status** — `nanlm_learning_status` does not emit these sections.
3. **Tool exposure** — `plugin.yaml` and the tool schema in
   `nanlmind_hermes/__init__.py` have no `nanlm_quantum_bridge` tool.
4. **Memory feed** — `MemoryStore.query()` rows are not automatically passed to
   the bridge; a caller must select a bounded window.
5. **Persistence** — the bridge returns `input_sha256` but does not write a
   durable provenance record.
6. **Joint decoder** — no importable library path or checked-in weights exist;
   see the note above.
7. **NANNET/router/CLOCKED** — intentionally not called; any future exposure
   must go through the existing authorization gates rather than this module.

## Honest limits

- The derived graph is small and bounded (at most 16 vertices by default
  configuration). It is not the live 256-head universe.
- Scalar edge weights are confidence-derived; they are not learned semantic
  couplings.
- Phase angles are label encodings, not physical phases. Coherence and
  phase-locking values therefore describe the encoding, not a quantum state.
- The typed loop trace uses the default, hand-specified SU(2) relation
  operators. It is a diagnostic, not learned loop dynamics.
- The cycle codec uses a seeded random cycle basis, not a semantic basis.
  Reconstruction error checks the code path, not meaning.
- Robust recovery compares against a projected gradient from the same input,
  not held-out ground truth.
- Spectral values come from bounded power iteration on a small weighted graph;
  they are approximate and depend on the derived graph.
- The Johnson/Grover values are exact mathematical controls for the bounded
  subgraph. They are not evidence of a physical system or algorithmic
  advantage.
- No output is a live measurement, a canonical state, or a proof.

## Verification

Run:

```bash
python3 -m unittest tests.test_quantum_bridge -v
```

The tests cover:

- a complete synthetic four-vertex row set that exercises gauge/cycle,
  phase/coherence, pair/spectral and robust-recovery sections;
- empty and unsupported inputs returning `available=false` with reasons;
- deterministic output across repeated calls and reversed row order;
- deep-copy output and no mutation of rows, graphs, or `ConceptMatrix` input;
- bounded output under small `max_rows`, `max_vertices`, `max_triangles`,
  `max_channels`, `max_patterns` and `max_graph_edges` values;
- concept-graph-only and optional semantic-state paths.
