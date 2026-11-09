# MTH data leaves, hubs and constrained pairing

Status: implemented with local tests. This is a data/constraint layer, not a
canonical merge or mathematical proof system.

## Leaf shape

Each leaf is immutable and content-addressed:

```text
leaf:<32 hex>
  kind: vector | scalar | matrix | relation
  hub: hub:mth.vector | hub:mth.scalar | hub:mth.matrix | hub:mth.relation
  parent, generation, immutable=true
  payload
  function_definition
  action_relations
  identities
  constraints
  constants
  wordnet
  content_sha256
  nan_image: real NaN16 .nan text + SHA-256 + word count
```

The `.nan` image is executable on the Python NaNVM0 boundary: it prints `L`
then halts, and its data region contains the LEAF magic, content hash, metadata
JSON and payload. This makes every leaf inspectable/executable without the
native C compiler; native leaf images can also be run by the NaN0S core.

## Hub tree

```text
hub:mth.root
  ├── hub:mth.vector      (256 float32, WordNet "vector" link)
  ├── hub:mth.scalar
  ├── hub:mth.matrix
  ├── hub:mth.relation
  ├── hub:mth.operator    (admitted operator table)
  ├── hub:mth.wordnet     (canonical HW4:0 route)
  ├── hub:mth.constraints (deny-by-default invariants)
  └── hub:mth.function    (all 91 MTH kernels)
        ├── hub:mth.function.graph   (20 kernels)
        ├── hub:mth.function.vector  (13 kernels)
        ├── hub:mth.function.matrix  (3 kernels)
        ├── ... 44 family hubs total
        └── kernel:mth.*             (one immutable leaf each)
```

Vector leaf payloads accept either exactly 256 finite `values` or a finite
`fill` value repeated across all 256 elements (for example
`{"fill": 1.0}`). The fill is explicit; there is no silent short-array
broadcast.

`nanlm_mth_tree action=map` creates the function tree. Each kernel leaf records
the kernel id, family, precision, shape, workspace, source/fixture digests,
marker, status/CI claim, execution mode, and links to its family hub,
`hub:mth.function`, `hub:mth.operator`, the appropriate vector/matrix/scalar
hub, `hub:mth.wordnet` and `hub:mth.constraints`. `action=audit` verifies:

- expected 91 vs linked 91, with no missing/extra/duplicate aliases;
- source and fixture digests match the bundled manifest;
- every kernel leaf has its family hub -> function root -> root ancestry;
- every kernel leaf's real NaN16 image executes and prints its `L` marker.

The current local audit: **91/91 linked, 44 family hubs, 0 missing, 0 orphans,
0 digest mismatches, 91/91 images verified, complete=true**. The full per-kernel
map is in [MTH_FUNCTION_TREE.md](MTH_FUNCTION_TREE.md).

The vector hub carries:

- dimension 256, float32, sparse/no dense-per-word matrix;
- identities: add commutativity/associativity/identity, sub(x,x)=0,
  dot symmetry/non-negativity, cosine symmetry/range;
- a WordNet HW4 link to `nan://GM0:SC0:MC0:BS0:0:HW4:0` and the `vector` term.

WordNet definitions are lexical evidence, not execution authority. If the
archive/word is absent, the link is declared but unresolved; the leaf remains
honest about that.

## Pairing

`left::right` is resolved through the hub rules:

| operator | input kinds | output | execution |
|---|---|---|---|
| `connect` | vector::vector | relation | none; creates a relation leaf |
| `dot` | vector::vector | scalar | unified N16I `mth.vector.dot256.f32` |
| `cosine` | vector::vector | scalar | unified N16I `mth.vector.cosine256.f32` |
| `add` | vector::vector | vector | unified N16I `mth.vector.add256.f32` |
| `sub` | vector::vector | vector | unified N16I `mth.vector.sub256.f32` |

The four math operators use N16I as the unified general-operand path on the
host-built/bundled portable NaN0S core. Direct native execution
(`execute_vector_kernel_native`) is retained only as an explicit
diagnostic/parity helper; it is not the default pairing path.

A successful math pairing creates a **new** hash leaf with the result,
kernel/program/operator ids, inputs, hub links, identities and execution
receipt. The receipt records the N16I path. Input leaves remain byte-identical.

A denied pairing returns `status: denied` with a stable code and does not
create a leaf or execute the kernel:

- `unknown_leaf`
- `hub_pairing_denied`
- `illegal_pairing` (e.g. `vector::scalar`)
- `unknown_operator` (e.g. `vector44::illegalpairing`)
- `operator_not_admitted`
- `native_execution_failed` (stable code for failed kernel execution)
- `invariant_violation`
- `identity_violation`

Example:

```json
{"left":"vector1","right":"vector2","operator":"add"}
=> {"status":"ok","hashleaf":"leaf:...","leaf":{"kind":"vector", ...}}
```

```json
{"left":"vector1","right":"vector44","operator":"illegalpairing"}
=> {"status":"denied","error":{"code":"unknown_operator"}}
```

## Self-modification guard

- No API mutates a leaf or hub in place.
- Every accepted operation creates a new content-addressed leaf.
- Hubs and their constraints cannot be rewritten or deleted.
- Pairing is deny-by-default by kind/operator table.
- Dimension, finiteness, cosine range, dot non-negativity and
  symmetry/antisymmetry identities are checked before a result leaf exists.
- Audit records are append-only; denied attempts are logged without payloads.
- Execution receipts (N16I or native diagnostic) are local/derived and do not
  authorize NANNET routes, leases, merges or canonical head/commit updates.

## Evidence

`tests/test_mth_leaves.py` covers hub seeding, NaN-executable leaf images,
idempotent leaf creation, connect/math pairing, input immutability, illegal and
unknown pairing denial, audit records, and the tool handlers.
