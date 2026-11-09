# ROUND 029 integration: raw NaN geometry in the resident stack

Status: **integration contract and audit**, 2026-11-08, repository
`/data/data/com.termux/files/home/NaNLMind_HERMES`.

Source artifacts: `experiments/round029/nan_round029/README.md`,
`summary.json`, `results.json`, `native_results.json`, `experiment.py` and
`native_mirror.cpp`. The integration tests are
`tests/test_round029_total_integration.py`.

This document separates three things that must not be conflated:

* **measured here** — a value produced by the test command in this checkout;
* **artifact** — a value recorded by the Round 029 experiment or native proof;
* **target** — desired production wiring that this test-only change does not
  implement.

Round 029 is a controlled WordNet gloss-classification experiment with a
deliberately distribution-shifted vertex gauge. It is not a general language
model result, not a quantum result, and not evidence that IEEE NaN arithmetic
is a computational resource. The resident stack stores semantic edge fields as
Q16.16 integers on pair rails; the repository's raw NaN16 payload codec is a
separate transport representation.

## 1. Point-by-point integration map

| Round 029 element | Resident representation | Tested contract | Status |
|---|---|---|---|
| Raw NaN payload codec | `nan16de.NanWord`; `nanvm.make_word` / `word_payload` | all 16 truth tables, byte round-trip, exact 32-bit payload, far/near heads | implemented, tested |
| 256 heads | `resident/cells.py`: 256 x 32-byte head records | permanent `head_id` / `identity` are not rewritten by execution | implemented, tested |
| 32,640 pairs | 32,640 x 64-byte pair cells, two 32-byte rails | every unordered pair has one dense index and one `reverse16` mirror | implemented, tested |
| Rank mirror | `temporal_spine.rank_of_pair`, `pair_at`, `mirror_rank` | full 32,640 rank bijection; `mirror_rank(r)=32639-r` | implemented, tested |
| Address mirror | `cells.address_of`, `canonical_address`, `reverse16` | `canonical=min(addr,reverse16(addr))`; mirror is `reverse16` | implemented, tested |
| Oriented edge field | pair rails `s+` / `s-`, Q16.16 weight, amplitude, phase | skew writes `(a,b,+x)` / `(b,a,-x)` produce exact antisymmetry | implemented, tested |
| Fundamental triangle flux | `anti(0,a)+anti(a,b)-anti(0,b)` from live rails | 48 triangles cancel two independent random vertex gauges | implemented, tested |
| Cycle vs raw readout | `cycle=oa+ab-ob`; `raw=ab` | 64 x 48 synthetic vectors: cycle error < 1e-12, raw error > 1 | implemented, tested |
| Exact gauge proof | `cycle_space.triangle_flux` | 12 random gauges, 48 triangles, flux equals `z_k` within 1e-12 | implemented, tested |
| Leaves | `resident/leaves.py`, content-addressed 112-byte records | field snapshots and payloads already persist through leaves | implemented; cycle-through-leaf is target |
| Relations / hyperedges | `resident/relations.py`; default six-member hyperedge | relation binds subject/predicate/object, pair, mirror and relation members | implemented, tested |
| Head field | `resident/head_field.py`; `field_learning.py` | additive, independent heads; scratch proposal before commit | implemented, tested |
| Temporal LM | `temporal_lm.py`; interpolated 1/2/3-gram counts | temporal channel learns the same sentence independently | implemented, tested |
| Cycle learning bridge | test-local bridge over temporal + field + `CycleState` | one observation updates temporal vocab, head field and cycle state | **contract only**, not production |
| Persistence | `resident/universe.py`; pair rails in the mmap | cycle rails and relation hyperedge survive close/reopen | implemented, tested |
| Tools | `resident/field_tools.py` | literal approval, bounds, read-only queries, generation CAS | implemented, tested |
| Benchmark | `scripts/head_field_benchmark.py` | fixed-seed deterministic fingerprint; timing excluded | implemented, tested |
| Canonical merge | all resident/field receipts and benchmark scope | every canonical/promotion flag is false | implemented, tested |

Important layout distinction:

* Python resident cells use **2,097,152 bytes**: 8,192 head bytes + 2,088,960
  pair bytes.
* The Round 029 C++ proof uses **8,388,608 bytes**: 256 x 128-byte heads +
  32,640 x 256-byte pairs.
* They share the same pair mapping and triangle mathematics, not the same bytes,
  cache behaviour or performance. The native proof is not the Python layout.

## 2. Exact formulas

### 2.1 Pair address and two separate mirrors

For `0 <= a,b < 256`, `a != b`:

```text
reverse8(x)  = bit-reverse of the low 8 bits of x
reverse16(x) = bit-reverse of all 16 bits of x
addr(a,b)    = (a << 8) | reverse8(b)
canonical(a,b) = min(addr(a,b), reverse16(addr(a,b)))
mirror_address = reverse16(canonical)
pair_index(a,b) = dense index of canonical(a,b) in [0, 32640)
```

The address mirror is not the rank mirror:

```text
rank(i,j)       = i*(511-i)/2 + (j-i-1),   0 <= i < j <= 255
pair_at(rank)   = (i,j)
mirror_rank(r)  = 32639 - r
mirror_pair(r)  = pair_at(32639 - r)
```

`canonical/mirror_address` and `rank/mirror_rank` are both involutions, but they
are different maps. The tests assert all 32,640 pair addresses have one unique
canonical cell and that both involutions are complete.

### 2.2 Rail field and triangle flux

A real edge value `x` is stored as a signed Q16.16 word:

```text
word = round(x * 65536), saturating signed 32-bit
x    = word / 65536
```

For a skew observation, store `+x` on `(a,b)` and `-x` on `(b,a)`. Define:

```text
w(a,b)    = signed weight read in the a -> b direction
anti(a,b) = 0.5 * (w(a,b) - w(b,a))
F(a,b)    = anti(0,a) + anti(a,b) - anti(0,b)
```

`F(a,b)` is the fundamental triangle flux relative to head 0. It is a derived
observable from live rails, not a new fact and not a quantum observable.

### 2.3 Round 029 edge construction and exact cancellation

For semantic coordinate `z_k`, use the disjoint triangle
`(0, a, b)` with `a=1+2k`, `b=a+1`:

```text
x_0a = g_0 - g_a
x_ab = z_k + g_a - g_b
x_0b = g_0 - g_b

F_k = x_0a + x_ab - x_0b = z_k
```

The zero-sum gauge coefficients cancel exactly. Equivalently, with oriented
incidence `B`, observed edge field `x = z + B^T g`, and any cycle vector `c`
satisfying `Bc = 0`:

```text
c^T x = c^T z + (Bc)^T g = c^T z
```

For the complete graph on 256 heads:

```text
edges              = C(256,2) = 32,640
gradient dimension = 255
cycle dimension    = 32,640 - 255 = 32,385
```

Round 029 trains on a 48-dimensional cycle subspace of that 32,385-dimensional
space. The full cycle space is not trained by the experiment.

### 2.4 Round 029 readouts

Given `oa`, `ab`, `ob` rows:

```text
cycle                = oa + ab - ob
raw                  = ab
wrong_triangle       = oa + ab - roll(ob, 1)
correct_conventional = oa + ab - ob       # mathematically identical to cycle
```

A conventional graph implementation of the same invariant is expected to tie
the NaN implementation exactly. That tie is a result of the experiment, not a
defect of this integration.

### 2.5 Resident field and scratch-first proposal

The independent head field stores an additive Q16.16 score per head. Field
activation clamps negative scores to zero and normalizes:

```text
p_h = max(0, score_h) / sum_j max(0, score_j)
entropy = -sum_h p_h log(p_h)
```

`FieldLearner` uses token hashing (`fnv_head(token) % 256`), an 8-bit mirror
head, bounded active heads/couplings/writes, and a two-phase write:

1. `propose(...)` stages only scratch state;
2. invariant checks verify shape, finiteness, non-negativity, bounds, budgets
   and that the main field is unchanged;
3. only an accepted proposal calls `apply_proposal` and per-head `activate`.

A head activation is exposure evidence, not semantic truth. `CANONICAL_MERGE`
and `HEAD_ACTIVATIONS_ARE_SEMANTIC_TRUTH` are both `False`.

### 2.6 Relations and hyperedges

The default resident relation hyperedge contains six member kinds:

```text
leaf (subject), leaf (predicate), leaf (object),
pair (dense pair index), mirror (reverse16 address), relation (content id)
```

The pair member is the relation's canonical cell index; the mirror member is
the 16-bit reverse address. The test asserts that both members are present and
match the relation record after append.

## 3. Measured versus target

### 3.1 Round 029 artifact (mean of seeds 17, 29, 41; 1,152 held-out synsets)

| Readout | Accuracy | Cross-entropy (nats) |
|---|---:|---:|
| True cycle triangle, distribution-shift gauge | 52.691% | 1.63859 |
| Raw directed pair, distribution-shift gauge | 10.966% | 11.67543 |
| Wrong triangle indexing, distribution-shift gauge | 10.706% | 11.62996 |
| Conventional correct triangle observable | 52.691% | 1.63859 |
| Raw pair, zero gauge train and test | 52.691% | 1.63859 |
| Learned all-three-edge control, gauge-augmented | 26.823% | 2.65367 |
| Correct triangle, gauge-augmented | 52.691% | 1.63859 |

The cycle, conventional-correct and zero-gauge raw conditions tie when they
carry the same invariant and the same clean signal. The learned all-edge
control does not catch up under this training budget.

### 3.2 Recorded native artifact (`summary.json`)

| Quantity | Value |
|---|---:|
| Root bytes | 8,388,608 |
| Unique pair records | 32,640 |
| Max float32 triangle gauge error | 1.86265e-06 |
| Max reverse antisymmetry error | 0 |
| Finite-difference gradient error | 6.01865e-05 |
| Loss before local step | 0.0658141 |
| Loss after local step | 0.0322489 |
| Triangle reads | 500,000 |
| Read time in one Linux run | 1.40741 ms |
| Linux `mlock` status | 0 |
| Native gate | passed |

A local rerun in this checkout also compiled and passed the native structural
gate. Its timing and exact noise values differ run to run; they are not asserted.
The read time is host-specific and is not an Android, phone or portable
performance claim.

### 3.3 Measured integration gates in this checkout

| Gate | Measured |
|---|---|
| Full test command | 16 tests, 16 pass, 0 skip, 0 error, 19.713 s |
| Pair mapping | 32,640 unique indices, 32,640 canonical addresses, 32,640 ranks |
| Rank mapping | complete `pair_at(rank_of_pair)=identity`; mirror involution complete |
| Pure cycle readout | max cycle error 3.7748e-15 over 64 x 48 samples; raw mean error 3.4028 |
| Resident rails, first gauge | max flux error 1.6377e-05 over 48 triangles |
| Resident rails, second gauge | max flux error 2.8999e-05 over 48 triangles |
| Resident reverse antisymmetry | exactly 0 on stored skew edges |
| Cycle persistence | flux and both rail weights identical after reopen; generation/digest unchanged |
| Hyperedge membership | 6 members: 3 leaves, pair, mirror, relation; pair/mirror ids match |
| Scratch-first field | accepted proposal writes 23 head updates + 23 couplings in the test sentence |
| Tools | missing approval denied; stale generation denied; valid commit accepted; queries read-only |
| Benchmark | same fixed-seed fingerprint; different seed changes fingerprint |
| Canonical merge | false in field, persistence, tools and universe receipts |

Targets that are not met by this change:

| Target | Why it is still open |
|---|---|
| Production cycle bridge inside `FieldLearner` / `TemporalMirrorLM` | the test bridge is local; `TemporalMirrorLM._prob` does not consume cycle flux |
| Cycle state in the journaled commit chain | direct pair-rail writes are made durable by `flush`; they are not individually journal commits |
| Full 32,385-dimensional cycle operator training | Round 029 and these tests use a 48-dimensional subspace |
| Natural, non-synthetic gauge-shift task | Round 029 measures a specified generated gauge distribution |
| Android/handset measurement | all Python timings here are host CPython/Termux; native timing is a Linux C++ run |
| Repair of the `ResidentUniverse.cycle_flux()` convenience wrapper | in this checkout it references `address_of` without importing it and raises `NameError`; tests use the equivalent live-rail `pair_state` math |

The last row is a known code gap, not a passing claim. It is recorded so a green
test run is not mistaken for a complete production wrapper.

## 4. Test coverage and lazy skips

Command:

```sh
cd /data/data/com.termux/files/home/NaNLMind_HERMES
python3 -m unittest tests.test_round029_total_integration -v
```

Covered classes:

| Class | Contract |
|---|---|
| `RawNanPayloadCodecTests` | 16 truth tables, payload round-trip, NaN16DE byte codec, NaNVM payload/head agreement, packed rank/amplitude/phase state |
| `MirrorPairMappingTests` | all 32,640 canonical cells, dense indices, rank bijection, address-vs-rank mirror separation |
| `GaugeGeometryTests` | exact triangle cancellation; cycle vs raw under sigma=3 gauge; wrong indexing differs |
| `ResidentCycleFieldTests` | 48 triangles on real Q16.16 rails; two gauges; exact antisymmetry; permanent identities |
| `CycleHyperedgeMembershipTests` | pair and mirror members bound into the relation hyperedge |
| `CyclePersistenceTests` | cycle rails, relation hyperedge, generation and digest survive universe reopen |
| `CycleLearningBridgeTests` | temporal vocab + independent head field + `CycleState` update together; no merge |
| `FieldToolApprovalBoundsTests` | approval required, head/delta/token bounds, stale CAS, read-only query, no mutation on denial |
| `NativeCompareTests` | helper skips without a summary; mismatch is a failure; compile/run only when a compiler exists |
| `BenchmarkDeterminismTests` | fixed-seed deterministic fingerprint and no native/phone/merge claim |
| `CanonicalMergeFalseTests` | explicit false authority metadata |

Skip policy:

* a genuinely absent Python module is skipped by `load_module`;
* native comparison is skipped when `summary.json`, `native_mirror.cpp` or a
  C++20 compiler is absent;
* native compilation, execution or JSON mismatch is a **failure**, never a
  silent skip;
* the benchmark test skips only when the resident cells module cannot provide
  the 2 MiB region.

Result of the command on this host: **16 passed, 0 skipped, 0 errors**. The
native test ran because `g++` was available.

## 5. Acceptance gates

### 5.1 Integration tests

```sh
cd /data/data/com.termux/files/home/NaNLMind_HERMES
python3 -m unittest tests.test_round029_total_integration -v
```

Acceptance: exit status 0; every non-skipped test passes; skip reasons are
limited to absent optional modules/toolchain; no errors. A native structure or
numeric mismatch must fail.

### 5.2 Artifact integrity

```sh
cd experiments/round029/nan_round029
sha256sum -c SHA256SUMS
```

Acceptance: all eight listed artifacts match their recorded hashes.

### 5.3 Native proof (optional toolchain)

```sh
cd experiments/round029/nan_round029
g++ -std=c++20 -O3 -Wall -Wextra native_mirror.cpp -o native_mirror
./native_mirror
```

Acceptance: JSON has `root_bytes=8388608`, `mirror_pairs=32640`,
`max_gauge_error <= 3e-6`, `max_reverse_error=0`,
`finite_difference_error < 2e-3`, `loss_after_step < loss_before_step`,
`reads=500000`, `passed=true`. Timing and `mlock_status` are reported but not
used as acceptance because they depend on host load and privileges.

### 5.4 Semantic experiment (optional full dependencies)

```sh
cd experiments/round029/nan_round029
python3 experiment.py --seeds 17 29 41
python3 strong_control.py
```

These require the extracted WordNet 3.1 corpus, Python 3, NumPy, PyTorch and
scikit-learn, and a path edit for `SRC`. They are not run by the integration
test command. Acceptance is agreement with `summary.json` within the recorded
seed/sample definitions, not a new performance claim.

## 6. Honest non-claims

1. **The conventional graph ties.** A conventional implementation of the same
   correct triangle observable reaches 52.691% and CE 1.63859, exactly matching
   the cycle readout. Round 029 demonstrates an invariance useful under a
   specified gauge shift, not intrinsic NaN superiority.
2. **No language-model win.** Raw pairs fail under the shifted gauge, but a
   correct conventional invariant ties. The learned all-edge control reaches
   26.823%, not 52.691%. These are 16-way WordNet gloss-classification results,
   not general prose comprehension, perplexity or generation quality.
3. **No quantum claim.** Cycle projection, gauge cancellation and triangle flux
   are classical linear algebra. Phase/amplitude accumulators are data
   structures, not physical quantum states or speedups.
4. **C++ timing is not Android timing.** The 1.40741 ms read result is one Linux
   container run of 500,000 reads over an 8 MiB C++ root. It is not a Termux,
   Android, phone or production throughput measurement.
5. **The native root is not resident in CPU registers.** It is an 8 MiB heap
   allocation. `mlock` depends on host privileges and memory limits; success in
   one run does not guarantee residency or portability.
6. **Native layout differs from Python layout.** The C++ proof uses 128-byte
   heads and 256-byte pairs; `resident/cells.py` uses 32-byte heads and 64-byte
   cells. Same mapping/math, different storage and performance.
7. **Raw NaN payload codec is not semantic storage.** `NanWord`/NaNVM encode a
   64-bit word with far/near NaN heads and a 32-bit payload. Resident semantic
   edges are Q16.16 rail words. Passing both codec tests does not mean the pair
   rails are IEEE-NaN arithmetic.
8. **The gauge shift is synthetic and specified.** Train/validation use sigma
   0.02; test uses unseen vertex gauge sigma 3.0. This measures robustness to a
   chosen distribution shift, not naturally occurring dictionary noise.
9. **The cycle bridge is a contract, not the production learner.** The test
   bridge uses production `TemporalMirrorLM`, `FieldLearner` and `CycleState`,
   but production `_prob` still ignores cycle flux. End-to-end cycle-conditioned
   next-token scoring remains a target.
10. **Persistence is not full journaling.** `observe_pair` writes rails directly;
    `flush` makes them durable, but the cell API does not append one journal
    record per rail update. A crash between pair write and a later commit is not
    covered by a per-pair transaction.
11. **No canonical merge or semantic truth.** All resident/field receipts and
    tools report canonical merge/promotion false. Head activation is exposure
    evidence; relation extraction is not independently fact-verified.
12. **Benchmarks are synthetic and deterministic.** The head-field benchmark
    uses a fixed synthetic activation workload on a 2 MiB bytearray. Its
    fingerprint determinism is not model quality or native speed.
13. **No full-cycle-space training.** The trained/verified subspace is 48
    dimensions; the complete graph has 32,385 cycle dimensions.

## 7. Reproduction summary

| Item | Path |
|---|---|
| Round 029 source and readings | `experiments/round029/nan_round029/` |
| Integration tests | `tests/test_round029_total_integration.py` |
| Resident cells / pairs | `nanlmind_hermes/resident/cells.py` |
| Leaves | `nanlmind_hermes/resident/leaves.py` |
| Relations / hyperedges | `nanlmind_hermes/resident/relations.py` |
| Head field | `nanlmind_hermes/resident/head_field.py` |
| Field learning | `nanlmind_hermes/resident/field_learning.py` |
| Field persistence | `nanlmind_hermes/resident/field_persistence.py` |
| Field tools | `nanlmind_hermes/resident/field_tools.py` |
| Universe | `nanlmind_hermes/resident/universe.py` |
| Temporal LM | `nanlmind_hermes/temporal_lm.py` |
| Cycle codec / space / state | `nanlmind_hermes/cycle_codec.py`, `cycle_space.py`, `cycle_state.py` |
| Gauge memory | `nanlmind_hermes/gauge_memory.py` |
| Rank mirror | `nanlmind_hermes/temporal_spine.py` |
| Raw NaN codec | `nanlmind_hermes/nan16de.py`, `nanlmind_hermes/nanvm.py` |
| Benchmark | `scripts/head_field_benchmark.py` |

The integration is accepted only when the test command exits 0, the artifact
hashes match, the native structural gate passes when a compiler is present, and
the false canonical-authority flags remain false. Anything else is a target or
an open defect, not a Round 029 success.
