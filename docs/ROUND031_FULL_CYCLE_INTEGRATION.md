# ROUND 031 full-cycle integration: native root, external encoder and the required 32,385-dimensional extension

Status: **measured artifact plus integration contract**, 2026-11-08, repository
`/data/data/com.termux/files/home/NaNLMind_HERMES`.

Source artifacts: `experiments/round031/nan_round031/native_cycle.cpp`,
`train.py`, `audit.py`, `transfer.py`, `final_audit.py`, `results.json`,
`audit.json`, `transfer_results.json`, `README.md`, `SHA256SUMS`, snapshots and
checkpoints. The integration test is `tests/test_round031_total_integration.py`.

Three categories stay separate:

* **measured here** — produced by the test command in this checkout;
* **artifact** — recorded by the Round 031 run or its native audit;
* **target** — desired production wiring this test-only change does not build.

Round 031 is a byte-level recurrent language-model experiment. It is not a
language-model win, not a quantum result, not an Android measurement and not
evidence that IEEE NaN arithmetic accelerates computation.

## 1. What Round 031 actually measures

Two runs per seed, 100 updates each, on 1,800 WordNet noun glosses per split.
The conventional model owns a 16 x 16 `nn.Parameter`; the native model reads the
same 256 values from an 8 MiB C++ root and updates that root through
`nan_apply`. The neural encoder, GRU and decoder remain external PyTorch
parameters in both models.

Held-out WordNet bits per byte, lower is better:

| Seed | conventional 16 x 16 | native root (256 weights) |
|---:|---:|---:|
| 17 | 3.8946724630734084 | 4.007561834657775 |
| 29 | 3.7829323637814896 | 3.847606375431593 |

The conventional model wins on both seeds. Native test accuracy is 22.97% and
25.62%; conventional accuracy is 25.67% and 26.65%. Both evaluate 13,200
held-out bytes. The native model has 40,496 neural parameters versus 40,752 for
the conventional model, exactly the 256 pair weights moved into the root.

The synthetic gauge test moved the root by at most 7.988419383764267e-07
(seed 17) and 7.192138582468033e-07 (seed 29), below the 1e-6 gate, with
unchanged classes and a passing mirror audit. Disabling memory worsened test
bits per byte to 4.061406929163623 and 3.928569746730655, without consistently
lowering accuracy.

## 2. Exact Round 031 integration path

### 2.1 Native root memory

`native_cycle.cpp` fixes `N=256`, `D=16`, `K=16`, `PAIRS=32640` and
`ROOT=8388608`:

```text
Memory
  Head heads[256]       256 * 128 bytes =  32,768 bytes
  Pair pair[32,640]     32,640 * 256 bytes = 8,355,840 bytes
  total                                      8,388,608 bytes
Pair
  float f[32]           forward lane, 32 float32 words
  float r[32]           reverse lane, 32 float32 words
```

The Python `Native` facade loads `libnative_cycle.so` with `ctypes`; training
calls `nan_new(0)`, so the root is allocated and zeroed but not mlocked. The ABI
is `nan_new`, `nan_get(ptr,dst[256])`, `nan_apply(ptr,grad,lr)`, `nan_gauge`,
`nan_snapshot`, `nan_restore`, `nan_audit` and `nan_free`. `nan_get` writes
channel-major `dst[c*D+d]`; Python transposes into `W[d,c]`. A pair has two
lanes and dense index `idx(i,j)=i*(511-i)/2+j-i-1` for `i<j`.

### 2.2 External encoder and decoder

The external model is deliberately conventional:

```text
x_t              byte id in [0, 256)
h_t              GRUCell(Embedding(256, 48), 48)
key(h_t)         Linear(48, 16)
val(z)           Linear(16, 48, bias=False)
rel              tanh(val(tanh(key(h_t) @ W)))
logits           Linear(48, 256)(h_t + rel)
```

`W` is either the conventional `nn.Parameter` or the root read through
`nan_get`. The root does not encode bytes, words, positions or GRU state; the
external network does. Round 031 is therefore not a fully memory-native
language model.

### 2.3 Gradient interface

PyTorch computes ordinary cross-entropy over 256-way byte logits. For the
native model, `W` is a detached clone with `requires_grad_(True)`, so
backpropagation produces `W.grad` through `key` and `val`. The pair parameter is
excluded from AdamW. The root update is:

```text
nan_apply(ptr, grad, lr=0.003)
  for c in 0..15:
    for d in 0..15:
      amount = -lr * grad[d,c]
      add_gradient(d, c, amount)
```

For dimension `d`, `a=1+2d`, `b=a+1`, and:

```text
add_gradient(d,c,amount):
  edge(0,a,c) += amount/3
  edge(a,b,c) += amount/3
  edge(0,b,c) -= amount/3
```

The three-edge distribution is the minimum Euclidean-norm update under the
constraint that the triangle sum changes by exactly `amount`. It is not a dense
gradient step over all unused pair edges. Neural parameters use AdamW at 0.002
with gradient-norm clipping 1.0; the native root uses fixed-step SGD at 0.003.
Nominal parameter counts are matched; optimizers are not.

### 2.4 Training, evaluation and checkpoints

Training samples 8 x 48-byte windows from training glosses with wraparound and
one optimizer step per batch. Validation runs at steps 1, 25, 50, 75 and 100,
capped at 100 documents; test runs after the best validation step. Evaluation
resets recurrent state at document boundaries, caps each document at 96 bytes,
and reports `bits_per_byte = cross_entropy_sum / n_bytes / ln(2)` and
`accuracy = argmax_correct / n_bytes`. The best neural state and corresponding
root snapshot are retained. The native
run then applies `nan_gauge(seed+500, 2.0)`, re-evaluates, restores the
snapshot, evaluates with `W=0`, and saves `root_<seed>.bin`. Gauge and
zero-memory probes are diagnostics, not additional training.

## 3. Required full 32,385-dimensional extension

Round 031 exposes 16 triangles x 16 channels = 256 effective weights. The
complete 256-head graph has a much larger cycle space. The resident extension
must make that space exact and addressable without pretending the native
artifact already trains it.

### 3.1 Complete-graph cycle geometry

```text
heads                  256
undirected pairs       C(256,2) = 32,640
gradient dimension     256 - 1 = 255
cycle dimension        32,640 - 255 = 32,385
triangles              C(256,3) = 2,763,520
fundamental root cycles C(255,2) = 32,385
```

For raw oriented words `x(a,b)` and root 0:

```text
c(i,j,0) = x(0,i) + x(i,j) - x(0,j)      (mod 2**32)
0 < i < j < 256
```

A vertex gauge is `x(a,b) += g(a) - g(b)`. Substitution cancels every term:

```text
c'(i,j,0) = c(i,j,0) + (g0-gi) + (gi-gj) - (g0-gj) = c(i,j,0)
```

The cancellation is exact integer arithmetic modulo `2**32`, not a float
tolerance. The second orientation is the modular mirror
`x(b,a) = -x(a,b) mod 2**32`.

### 3.2 Raw-word storage contract

The resident `CycleField` reads and writes the 32-bit weight word of each pair
rail directly. It does not reinterpret the word as Q16.16, float32 or a NaN
payload. All 32-bit patterns are preserved, including `0x7FC00000`,
`0x7FC00001`, `0x7F800001`, `0xFFC00000`, `0x80000000` and `0xFFFFFFFF`.
Amplitude, phase, relation chain, source/target and mirror metadata are left
untouched, and writes never touch the 256 head records.

The tests write a deterministic word to every canonical pair, then verify all
32,640 pair words and all 32,385 cycle words; mirror selected NaN bit patterns
and check adjacent payloads stay distinct; and apply a large random gauge,
verifying all 32,385 cycle words are bit-identical while a raw edge changes.

### 3.3 Fixed-point full-basis training

`CycleTrainer` operates on signed lane values with an exact Q16.16 loss:
`residual r = c(i,j,0) - target`, `L = 1/2*r**2`, `gradient = r`. One test
constructs all 32,385 root-cycle observations and performs a single full-basis
step on an `IntegerLaneField` with learning rate 1/1024, gradient normalization,
batch limit 40,000 and pair limit 70,000. The step is accepted, observes 32,385
cycles, touches more than 32,000 pairs, and reduces the exact integer loss. This
is a fixed-point integration of the full basis, not a reproduction of the
float32 Round 031 run.

### 3.4 Hypergraph, leaves and incidence

For each root cycle `(0,i,j)` the cycle hypergraph derives a leaf id, a
hyperedge id and nine members: one cycle leaf, three pairs `(0,i)`, `(i,j)`,
`(0,j)`, three vertices `0`, `i`, `j`, one signed value and one digest. A
symbolic sweep maps all 32,385 cycles to unique IDs and covers all 32,640 pairs.
A real store materializes 260 hyperedges, more than the 256 native weights, and
verifies membership, snapshot count and audit. Full bulk append is not run
because `_find_cycle_view` scans existing edges; symbolic all-cycle mapping is
tested, bulk materialization remains a target.

### 3.5 Persistence, reopen and no 256-only path

The persistence test fills all 32,640 pair rails in a real 16 MiB
`ResidentUniverse`, flushes, records generation and digest, closes, reopens, and
compares a digest over all 32,640 pair records and all 32,385 cycle records.
Pair words, cycle words, generation and digest are identical, and
`canonical_merge.enabled` remains false.

The resident API must not substitute the native 16 x 16 readout for the full
basis. Tests assert `ROOT_CYCLE_COUNT = 32,385`; a 257-cycle batch contains the
256-cycle batch plus one new cycle; full snapshots contain 32,385 cycles; limits
above 32,385 are rejected; the trainer accepts 300 observations and updates more
than 256 pairs; and a real hypergraph store holds 260 cycle hyperedges.

## 4. Memory budget

| Region | Exact bytes | Notes |
|---|---:|---|
| Native heads | 32,768 | 256 x 128-byte head records |
| Native pair cells | 8,355,840 | 32,640 x 256 bytes |
| Native root | 8,388,608 | exactly 8 MiB |
| Native snapshot file | 8,388,616 | 8-byte `NANCYC30` marker plus root |
| Native pair cell / lane | 256 / 128 | two lanes of 32 float32 words |
| Active native weights | 1,024 | 256 float32 effective weights |
| Active native pair cells | 48 | 16 disjoint triangles x 3 edges |
| Resident cell region | 2,097,152 | 8,192 head bytes + 2,088,960 pair bytes |
| Resident raw pair words | 261,120 | 32,640 pairs x 2 directions x 4 bytes |
| Resident cycle values | derived | 32,385 values read, not a second array |
| Materialized cycle hyperedge | about 809 | relation 304 + hyperedge 96 + arena about 297 + leaf 112 |

The native root is 4x larger than the resident cell region and uses a different
layout; they share geometry, not bytes or timing. The full basis is a derived
view over pair rails, not a reduction of native storage or a conversion of 256
weights into 32,385 trained weights. The 260-cycle hypergraph test uses about
0.2 MiB, not the roughly 26 MiB a full materialization needs.

## 5. Connection to resident cells, leaves, hyperedges and field heads

* 256 heads and 32,640 pairs: `resident/cells.py` keeps permanent head
  identity and two 64-byte-cell rails; full mapping and mirrors are tested.
* Raw oriented weights and triangle flux: `resident/cycle_observable.py`
  preserves unsigned 32-bit words and exposes all 32,385 cycles.
* Full-basis training: `resident/cycle_training.py` accepts the 32,385-cycle
  step; `resident/cycle_hypergraph.py` maps every root cycle to a leaf and
  nine-member hyperedge.
* Persistence: `resident/universe.py` keeps pair and cycle words equal after
  close/reopen; the optional snapshot and native bridges expose themselves
  lazily.
* Field heads: the independent additive head field is not trained by Round 031
  and is not written by cycle lanes. Cycle flux is still not fed into the
  production field-head or temporal decoder.

## 6. Honest limitations

1. **Round 031 has 256 effective weights.** Only 16 triangles x 16 channels are
   exposed and updated; the other 32,129 cycle degrees and most pair lanes are unused.
2. **The conventional model wins.** Both seeds have lower test bits per byte with the ordinary 16 x 16 tensor. The root shows direct training and gauge invariance, not a benefit.
3. **The gauge test is synthetic.** `nan_gauge` adds a specified random vertex potential with sigma 2.0, not natural dictionary noise or arbitrary corruption.
4. **This is not a language win.** It is 256-way next-byte prediction on 1,800 WordNet glosses per split with 13,200 held-out bytes, not prose comprehension, natural-book perplexity or generation quality.
5. **The native root is float32, not a raw NaN layer.** It stores IEEE float values with ordinary arithmetic; the raw NaN16 codec and resident raw-word cycle field are separate storage layers.
6. **The encoder is external.** The byte GRU, key/value projections and output head remain PyTorch parameters; Round 031 is not fully memory-native.
7. **Optimizers are not matched.** Native uses fixed-step SGD 0.003; neural uses AdamW 0.002. Matched nominal counts do not imply matched optimization.
8. **The sample is not representative WordNet.** The first 1,800 noun glosses in corpus order are selected; evaluation resets recurrent state per document and truncates at 96 bytes.
9. **The Alice transfer test is tiny.** The untouched test split is 288 bytes; 30-update validation improvement is not strong generalization.
10. **The root is allocated, not locked.** `nan_new(0)` does not mlock; the 6-7 microsecond read is one Linux ctypes probe, not Android residency or portable latency.
11. **Full-basis training is a fixed-point test.** The 32,385-observation step proves the resident integral interface, not that the float32 native run trained the full cycle space.
12. **Bulk hypergraph materialization is not demonstrated.** The symbolic all-cycle map is complete; the real store test stops at 260 cycles.
13. **No canonical merge or semantic truth.** Training, hypergraph, snapshot, persistence and tool authority flags are false; cycle observations are evidence, not facts.
14. **No quantum claim.** Cycle projection, gauge cancellation and triangle sums are classical linear algebra.

## 7. Acceptance gates and commands

### 7.1 Integration test gate

```sh
cd /data/data/com.termux/files/home/NaNLMind_HERMES
python3 -m unittest tests.test_round031_total_integration -v
```

Measured result in this checkout:

```text
Ran 26 tests in 32.310s
OK
26 passed, 0 skipped, 0 errors, 0 failures
```

The ten classes cover the full mapping and words (`FullCycleBasisTests`), the
above-256 path (`No256OnlyPathTests`), NaN-bit preservation
(`RawWordPreservationTests`), all-cycle gauge invariance
(`GaugeInvarianceTests`), full-basis loss reduction (`FullBasisTrainingTests`),
reopen equality (`PersistenceReopenTests`), all-cycle incidence plus 260 real
hyperedges (`CycleHypergraphTests`), artifacts and checksums
(`Round031ArtifactTests`), optional module surfaces
(`OptionalFullCycleModulesTests`) and false authority flags
(`CanonicalMergeFalseTests`). A present module or artifact that violates the
contract fails; only a genuinely absent optional module or artifact skips. No
test writes to the repository.

### 7.2 Artifact checksum gate

```sh
cd /data/data/com.termux/files/home/NaNLMind_HERMES/experiments/round031/nan_round031
sha256sum -c SHA256SUMS
```

Measured result: all 23 entries report `OK`: `native_cycle.cpp`, `train.py`,
`audit.py`, `transfer.py`, `final_audit.py`, `README.md`, `results.json`,
`audit.json`, `transfer_results.json`, `train.log`, `transfer.log`, the four
model checkpoints, the four transfer checkpoints and the four root snapshots.
Every `root_*.bin` is 8,388,616 bytes: 8-byte `NANCYC30` marker plus the
8,388,608-byte root. The gate fails if a file is missing or a digest differs.

### 7.3 Artifact consistency gate

The artifact test asserts 1,800 glosses per split; 13,200 held-out and 7,061
validation bytes; conventional bits per byte below native on both seeds; native
gauge deviation at most 1e-6 with unchanged accuracy and audit true; native
memory-disabled bits above enabled; 10,000 reads, restore difference 0.0 and
mirror audit true; 117,570 records, 87,655 two-hop candidates, 1,500 checked,
zero direct hits; validation improvements after 30 Alice updates; and a
288-byte Alice test with finite bits per byte.

### 7.4 Native reproduction gate — optional toolchain

```sh
cd /data/data/com.termux/files/home/NaNLMind_HERMES/experiments/round031/nan_round031
c++ -std=c++20 -O3 -fPIC -shared native_cycle.cpp -o libnative_cycle.so
OPENBLAS_NUM_THREADS=1 python3 train.py --steps 100 --seeds 17 29
OPENBLAS_NUM_THREADS=1 python3 audit.py
OPENBLAS_NUM_THREADS=1 python3 transfer.py
OPENBLAS_NUM_THREADS=1 python3 final_audit.py
```

These require the original Round 025 WordNet `corpus.jsonl` and Round 022 Alice
files at their recorded paths. Acceptance is agreement with recorded definitions
and artifact structure, not a new performance claim. The integration test does
not compile or rerun the language model.

Accept Round 031 only as a direct native-root training demonstration with 256
effective weights, exact gauge invariance, snapshot/mirror integrity, and exact
full-cycle resident geometry. Do not accept it as a conventional-model win, a
full 32,385-dimensional native training result, a language generalization
result, an Android measurement, a raw-NaN claim or a canonical merge.
The integration is accepted only when the test command exits 0 with no
unexpected skips, all 23 artifact hashes match, and every canonical-merge flag
remains false. Anything else is an open target or a defect, not a Round 031
success.
