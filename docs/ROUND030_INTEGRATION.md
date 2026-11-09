# Round 030 Integration: native mirror root, raw NaN lanes, resident cycles

Status: integration note for `experiments/round030` and the resident Python
stack.  Round 030 is a **classical 16-category WordNet synset classifier**, not
a quantum experiment, a generative language model, or evidence that IEEE NaN
encodings accelerate learning.  This note maps the published result onto the
repository without upgrading its claims.

## 1. Artifact map

| Path | Role |
| --- | --- |
| `experiments/round030/README.md` | protocol, results, limitations |
| `experiments/round030/results.json` | three seeds, before/after gauge readouts |
| `experiments/round030/tuned_results.json` | validation-selected schedules, 101 shifts |
| `experiments/round030/native_audit.json` | root size, gradient/gauge/snapshot audit |
| `experiments/round030/native_cycle.cpp` | 8 MiB native root and C ABI |
| `experiments/round030/root_{17,29,41}.bin` | `NANCYC30` marker plus exact root |
| `experiments/round030/run.py`, `tune.py`, `native_audit.py` | experiment drivers |
| `experiments/round030/SHA256SUMS` | artifact integrity |
| `nanlmind_hermes/nan_geometry.py` | binary64 NaN payload lane geometry |
| `nanlmind_hermes/nan16de.py` | 64-bit NaN16 double-endian envelope |
| `nanlmind_hermes/resident/cells.py` | 256 heads, 32,640 pair cells, two rails |
| `nanlmind_hermes/resident/cycle_observable.py` | raw-u32 cycle view over rails |
| `nanlmind_hermes/resident/cycle_training.py` | fixed-point cycle trainer |
| `nanlmind_hermes/resident/cycle_hypergraph.py` | cycles as resident hyperedges |
| `nanlmind_hermes/resident/head_field.py` | independent 256-head activation field |
| `nanlmind_hermes/resident/cycle_tools.py`, `field_tools.py` | bounded tool handlers |
| `scripts/nan_cycle_benchmark.py` | integer cycle/raw/control benchmark |
| `scripts/resident_benchmark.py`, `head_field_benchmark.py` | store and head-field harnesses |
| `tests/test_round030_integration.py` | repository-facing contract tests |

`SHA256SUMS` covers the README, JSON, source, logs and three root snapshots.

## 2. Native C++ root versus the raw NaN lane layer

These are **two different storage layers** and must not be conflated.

**Native C++ root.**  `native_cycle.cpp` defines 256 head records of 128 bytes
(`32,768` bytes), 32,640 unordered pair records of 256 bytes, two 128-byte
directional lanes per pair, and a contiguous root of `8,388,608` bytes,
enforced by `static_assert(sizeof(Memory)==ROOT);`.  The pair index is
`i*(511-i)/2+j-i-1` for `i < j`.  Each lane is `float f[32]` or `float r[32]`;
24 of those words per class lane are used.  Training, forward reads and gauge
shifts all use the host floating-point unit.  The C++ root does not store or
interpret raw NaN payload bits.

**Raw NaN lane layer.**  `nan_geometry.py` treats one IEEE-754 binary64 word as
an integer bit pattern: sign, 11 exponent bits and 52 payload bits, with a
4-bit tag and 48-bit signed fixed-point value in the payload.  No identity,
gauge, cycle or lane-codec path converts to Python `float`.  A 128-byte lane is
a 16-byte header plus 14 little-endian 64-bit words, using the same 256-head /
32,640-pair root arithmetic.  `nan16de.py` separately defines a 64-bit NaN16
double-endian envelope with an exact 32-bit payload.

**Integration rule:** treat the C++ root as the trained float32 prototype and
the raw NaN payload modules as a bit-preserving storage/geometry layer.  Do not
describe one as the implementation of the other, and do not infer NaN-payload
speed or semantic superiority from Round 030.

## 3. Where the weights live

The effective semantic weights are triangle circulations.  For class `c` and
dimension `d`, with `a = 1 + 2*d` and `b = a + 1`:

```
w[c,d] = x(0,a,c) + x(a,b,c) - x(0,b,c)
```

`nan_get` returns these 16 x 24 = 384 effective coefficients.  Each effective
coefficient is physically distributed over three directed edges inside the
8 MiB root.  The three-edge representation has gauge redundancy; it is not
additional independent capacity.  The trained parameters live in the pair-lane
floats, not in a separate NumPy tensor.  `run.py` reconstructs logits as
`X @ W.T` and checks the result against the native readout.

The `root_*.bin` snapshots are not pickled tensors.  They are an 8-byte
`NANCYC30` marker followed by the exact 8 MiB root.  The effective classifier
is recoverable from the snapshot without a second learned-parameter file.

## 4. Gauge shift

The gauge transform is additive at vertices:

```
x(i,j) -> x(i,j) + g[i] - g[j]
```

For the anchored triangle, the three gauge terms telescope:

```
x(0,a)+g[0]-g[a] + x(a,b)+g[a]-g[b] - x(0,b)-g[0]+g[b]
= x(0,a) + x(a,b) - x(0,b)
```

`nan_gauge` applies this to every native pair lane.  The resident
`CycleField.gauge_shift` applies the same integer relation modulo `2**32`;
`GaugeStore` applies it over arbitrary Python integers.

Acceptance has two levels.  **Exact:** integer `GaugeStore` and resident
`CycleField` readouts are bit-exact under any 256-word gauge; tests compare all
32,385 fundamental cycles.  **Float32 native:** the C++ root cancels the same
terms in float32, so residuals are small but not zero.  Published audit error is
`4.1653402149677277e-07`; the tuned 101-shift logit change stays below `3e-4`.
These are float32 tolerances, not exact integer identities.

## 5. Snapshot and restore

The native format is minimal: 8 bytes `NANCYC30` plus the exact `8,388,608`
byte `Memory` layout, for `8,388,616` bytes total.  `nan_snapshot` writes the
marker and complete root; `nan_restore` checks the marker and reads it back.
The audit reports bitwise restore, and seed runs report
`restore_max_logit_error = 0.0`.

Limits stay explicit: same-architecture and little-endian only; no concurrency
control, transaction, CRC or crash recovery; optional `mlock` is not a
CPU-register-residency guarantee; snapshots are experimental state, not durable
user data.  The resident Python `CycleField.snapshot` is a different shape: an
immutable mapping of 32,640 pair records and 32,385 cycle samples, not a native
memory image.

## 6. Training and gradients

Native `nan_epoch` performs one shuffled softmax cross-entropy pass.  Forward
logits come from the three-edge circulations; `p = softmax(logits)`; the
gradient is `p[c] - 1[c == target]`; `add_gradient` distributes
`-learning_rate * gradient * x[d]` over the three edges with signed skew
writes, preserving mirror antisymmetry.  The native audit compares one
analytical update with readback and reports `1.862645149230957e-09` maximum
absolute error.  The tuned result matches a capacity-matched no-intercept
linear classifier within `1.14e-05` to `5.72e-06` maximum logit difference in
the seed runs.

The resident Python trainer is separate and integer-only.  `IntegerLaneField`
is the deterministic reference; `CellsLaneField` writes signed Q16.16 words
directly into resident rails; `CycleObservableLaneField` adapts an unsigned
raw-u32 `CycleField`.  `CycleTrainer` validates a whole batch, writes both
mirror lanes, advances generation once, and can reject a step that increases
loss.  `gradient()` returns `predict - target` exactly, and
`finite_difference_check()` is exact integer arithmetic.  With
`normalize_gradient=False` each of three edges moves by the native-like full
update; with `normalize_gradient=True` the update is distributed over the three
edges for exact reconstruction experiments.

## 7. Resident cells, cycles, hyperedges and field heads

**Cells.**  `ResidentCells` owns 256 head records and 32,640 pair cells, each
with forward and reverse 32-byte rail records.  The raw cycle layer reads and
writes the 32-bit `weight` word without float conversion and leaves head
identity, phase, amplitude and relation metadata untouched.

**Cycle field.**  `CycleField` is a view over resident rails:
`get_oriented`/`set_oriented` preserve exact words; `cycle(i,j,root)` is
`x(root,i)+x(i,j)-x(root,j) mod 2**32`; `triangle(a,b,c)` is the oriented
three-head circulation; `gauge_shift` updates both directions of every pair;
`batch_cycles()` is bounded by 32,385 fundamental cycles; `snapshot()` is
immutable and detached.  This is the bridge between the native triangle readout
and resident cells: exact for integers, approximate in native float32.

**Hyperedges.**  `cycle_hypergraph.CycleHypergraph` attaches a cycle observable
to existing stores: one typed hyperedge per cycle, references to the three
existing pair cells, a deterministic cycle leaf and payload digest, mirror-safe
cycle identity, idempotent re-add, conflict rejection on changed
value/payload/source, and bounded snapshot, audit and neighbor traversal.  Pair
cells are referenced, never duplicated.  `canonical_merge` remains false;
membership is bookkeeping, not a truth claim.

**Head field and field learning.**  `ResidentHeadField` gives each of 256 heads
an independent Q16.16 activation score and pair couplings.  Proposals are
scratch records; an explicit approval commits an accepted proposal.
`field_learning.FieldLearner` adds a bounded per-sentence activation/coupling
channel beside the temporal model.  Both declare head activation as evidence,
not semantic truth, and neither enables canonical merge.

**Tools.**  `cycle_tools.py` exposes bounded `status`, `snapshot`, `cycle`,
`gauge`, `set`, `train` and `query` handlers; `field_tools.py` exposes the
corresponding head-field handlers.  They validate properties, require literal
`approved=true` for writes, reject pointer/offset leakage, cap batch and output
sizes, and return `canonical_merge.enabled = false` and
`canonical_promotion = false` on every response.  They are dependency-injected
handlers, not a new authority layer.

## 8. Benchmarks and measurement boundaries

`scripts/nan_cycle_benchmark.py` compares synthetic integer lanes: correct
cycle readout, raw single-edge readout, wrong-triangle control, conventional
integer edge-table cycle, uniform-gauge control and per-lane-noise control.
Cycle and conventional integer readouts tie exactly; raw and wrong-triangle
readouts fail under nonuniform gauges; noise defeats both.  This is a
geometry/control harness, not a model-quality or native-speed claim.

`scripts/resident_benchmark.py` measures a synthetic resident-store memory and
latency harness (budget, mapping, ingest, query, checkpoint, reopen).  It does
not train the Round 030 model or establish phone performance.
`scripts/head_field_benchmark.py` provides deterministic head-field output and
an explicit `canonical_merge_performed = false` scope.

The Round 030 ctypes number (`7.74 ms` for 1,000 Python-crossed predictions) is
an audit convenience, not a batched native benchmark; it is hardware- and
container-dependent.

## 9. Honest limitations

- **Conventional classifier is strong.**  No-intercept logistic regression on
  the same clean 24D features reaches `42.3214%` mean test accuracy versus
  `42.0833%` for the tuned native cycle readout.  The native geometry does not
  beat a correct conventional linear model.
- **External text encoder.**  TF-IDF vocabulary, n-grams and 24D TruncatedSVD
  are fitted outside the C++ root.  The native root learns only the final
  16-class linear map.
- **Synthetic gauge.**  The shifts are intentionally generated to satisfy
  additive vertex structure.  They are not natural corpus perturbations and do
  not show robustness to semantic noise.
- **Not a generative language model.**  Round 030 is a 16-category WordNet
  synset classifier; it does not generate text or establish LM quality.
- **No NaN-arithmetic acceleration claim.**  The C++ bridge uses float32 lanes; the binary64 NaN payload modules are a distinct bit-preserving layer.
- **No memory-efficiency or quantum claim.**  The 8 MiB root is fixed; triangle cancellation is classical linear algebra.
- **Snapshot scope.**  Same-architecture, non-transactional, non-concurrent,
  with no endian portability guarantee.
- **Corpus scope.**  Results depend on the external Round 025 WordNet JSONL and
  the published category/seed split; the corpus is not embedded here.
- **Validation, not test, tuning.**  Hyperparameters used validation cross-entropy, but this remains a small three-seed study.

## 10. Acceptance gates and commands

An integration is acceptable only when all of these hold:

1. `SHA256SUMS` verifies for every listed artifact.
2. Raw root is exactly `8,388,608` bytes; snapshots are `8,388,616` bytes and
   start with `NANCYC30`; pair count is `32,640` with `65,280` directed
   mappings.
3. Integer gauge tests compare all `32,385` fundamental cycles exactly.
4. Native float32 gauge drift is below published audit bounds; raw-edge readout
   changes materially while cycle class predictions stay unchanged.
5. Native snapshot/restore is bitwise and restores identical logits.
6. Gradient check error is below `1e-6`; the published audit value is
   `1.862645149230957e-09`.
7. `CANONICAL_MERGE` and semantic-truth flags remain false in cycle training,
   cycle tools, field learning and field tools.
8. Raw NaN payload lane tests remain bit-preserving and separate from the C++
   float32 bridge.
9. The suite reports pass/skip/error precisely; only genuinely absent optional
   modules or compilers may skip.
10. No generative-LM, quantum, phone-performance, memory-efficiency or
    NaN-acceleration claim is inferred from the classifier result.

Run the integration suite:

```bash
python3 -m unittest tests.test_round030_integration -v
```

Expected in this container: all 14 tests pass.  Without a C++20 compiler the
`NativeSharedLibraryTests` class skips; absent optional Python modules also
skip rather than fail.  A present artifact or module that violates an
invariant fails.

Reproduce the published experiment (requires C++20, NumPy, scikit-learn and the
Round 025 WordNet corpus JSONL):

```bash
cd experiments/round030
c++ -std=c++20 -O3 -shared -fPIC native_cycle.cpp -o libnative_cycle.so
OPENBLAS_NUM_THREADS=2 python3 run.py --seeds 17 29 41
OPENBLAS_NUM_THREADS=2 python3 tune.py
OPENBLAS_NUM_THREADS=2 python3 native_audit.py
```

Run resident geometry and control benchmarks:

```bash
python3 scripts/nan_cycle_benchmark.py \
  --cycles 48,256,32385 \
  --json /data/data/com.termux/files/usr/tmp/nan-cycle.json \
  --markdown /data/data/com.termux/files/usr/tmp/nan-cycle.md

python3 scripts/resident_benchmark.py --profile low --quick \
  --json /data/data/com.termux/files/usr/tmp/resident-low.json \
  --markdown /data/data/com.termux/files/usr/tmp/resident-low.md

python3 scripts/head_field_benchmark.py
```

These outputs are local diagnostics, not a substitute for the Round 030 corpus
experiment or a held-out language-model evaluation.

## 11. Review checklist

- Native root and raw NaN lane layer are described as separate mechanisms.
- Weights are described as distributed pair-lane circulations, not a detached
  tensor.
- Gauge language distinguishes exact integers from float32 tolerance.
- Snapshot limitations are stated next to the claimed restore result.
- Cells, cycle fields, hyperedges and head fields are linked by file name and
  responsibility, not merged into one authority.
- Tools and benchmarks retain approval, bound and no-claim flags.
- Limitations name the strong conventional control, external encoder,
  synthetic gauge and non-generative scope.
- No result is presented as a language-model, quantum, phone or NaN-arithmetic
  advantage.
