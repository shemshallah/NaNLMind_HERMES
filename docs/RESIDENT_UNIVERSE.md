# Resident universe: in-memory state and primitive interfaces

Status: partly implemented. The local-derived one-file store is now
`nanlmind_hermes/resident/universe.py`; see
[RESIDENT_IMPLEMENTATION.md](RESIDENT_IMPLEMENTATION.md) for the actual map,
APIs and tests. The wider proposal below still describes future integration of
the live chart, SQLite history and MTH indexes.

## What is resident today

| State | Where it lives now | Resident? |
|---|---|---|
| 8 MiB chart / address table / pair table | `Chart8MiB` in `LocalRunner`, cached per state dir | yes, per process |
| Temporal register frame/rails | `TemporalRegister`, chart extension + JSON sidecar | yes, per process |
| Head scratch register | `HeadScratchRegister` + chart extension + `language/mth_heads.json` | yes, per process |
| Word n-gram/temporal model | `TemporalMirrorLM` Python dicts loaded from `language/temporal_mirror.json` | yes, per process |
| 255-head ensemble, concept matrix, local head register | `LayeredLearningCoordinator` cached per state dir | yes, per process |
| Relation/hyperedge evidence | SQLite `mathematical_memory.sqlite3`, queried with a bounded limit | **no**, rebuilt/queried per call |
| L0-L9 scratch leaves | sparse `scratch_register/pages/*.n16` files | **no**, paged on demand |
| MTH stack / edge index | rebuilt in `_native_synthesize` per turn from memory rows | **no**, transient |
| 2 GiB overbootstrap map | Bootstrap runner only | **never** on Android/peer |

So the plugin has a resident 8 MiB chart and cached model/ensemble objects, but
not one resident typed universe containing every relation. This is the gap.

## Proposed `ResidentUniverse`

One process-wide object per `state_dir`, opened at plugin boot and shared by
chat, layered learning, MTH leaves and tools.

```text
ResidentUniverse
  chart            8 MiB Chart8MiB (cells, address/pair tables, registers)
  heads            typed head records / Boolean score tables
  pairs            C(256,2) pair cells, two rails, integer/NaN payloads
  relations        compact relation index (subject,predicate,object,polarity)
  hyperedges       n-ary membership index
  temporal         vocab/uni/bi/tri/mirror transition tables
  concepts         32,640-cell concept matrix
  scratch_index    page/record index, payload paged out
  mth_stack        resident L0-L9 edge/vocab index
  generation       monotonic state generation + digest
```

### Memory policy

- Chart: fixed 8 MiB.
- Pair cells: 32,640 × 2 rails × compact integers (target under 1 MiB).
- Relations: cap resident records (default 65,536); cold records remain in
  SQLite and are loaded on demand. Never load the 2 GiB overbootstrap map.
- Scratch leaves: keep only an index in RAM; payload pages stay paged/sparse.
- All mutations append and bump generation; readers use generation-bound views.
- Checkpoint after bounded intervals, not every word; crash recovery reopens the
  last generation and rejects torn snapshots.

### API shape

```text
ResidentUniverse.open(ctx)
  .lookup(subject, predicate=None, object=None) -> typed rows
  .append_relation(record, source_digest) -> receipt + generation
  .pair(left, right, operator) -> new immutable leaf/hash
  .route(address) -> resolved typed handle
  .snapshot() / .checkpoint() / .close()
```

All writes go through validation, provenance, generation and capability checks.
No tool receives a raw Python object or arbitrary memory pointer.

## Primitive alignment

- Relations and cells use typed integer/NaN16 descriptors, not untyped dicts
  that later get reinterpreted.
- Native kernels execute through the MTH/N16I path; the resident layer never
  substitutes Python arithmetic for a native result.
- Data leaves/hubs define the type and constraints; resident indexes only make
  lookup fast.
- Every operation returns a receipt: generation, source digest, execution mode,
  and whether anything was persisted.

## External-system interfaces

Two deliberately separate adapters:

1. **NaN/N16 interface** — signed frames, typed leaf descriptors, N16D/N16I
   dispatch, capability-scoped routes, immutable result leaves. No arbitrary
   native pointers or shell.
2. **Normal-code interface** — bounded JSON/HTTPS/stdio adapters with explicit
   schemas, allowlists, auth scopes and audit records. External systems are
   capabilities, not implicit memory access.

Both adapters read/write the same resident universe through the API above. A
model can propose an adapter call, but an authorization policy executes it and
returns a receipt. This is how the system can eventually learn to interface with
other systems without letting self-modification escape the sandbox.

## Integration phases

1. Add `ResidentUniverse` over the existing chart + SQLite + layered state.
2. Route `_memory_store`, `_memory_context`, `structured_candidates`, MTH stack
   and `mth_leaves` through the resident indexes.
3. Add generation-bound snapshots and invalidation tests.
4. Add compact relation/pair encodings and memory caps.
5. Add NaN/N16 adapter and normal-code adapter with capability checks.
6. Only then consider promotion/merge gates; canonical merge stays off.

## Honest boundary

A resident universe improves latency and primitive coherence. It does not create
a second model, a second memory owner, quantum behavior, a proof system or
canonical state. The 2 GiB overbootstrap map must not be copied to the edge.
