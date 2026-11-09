# Round 032 Total Integration: sparse retrieval over the 8 MiB mirror root

Status: integration note for `experiments/round029` through
`experiments/round032` and the resident Python stack. Round 032 is an exact
symbolic retrieval layer over known WordNet hypernym links, **not** a trained
retrieval-augmented generative language model. This note records what is wired
together, what each round actually measured, and which claims remain unsupported.

## 1. What the four rounds establish

| Round | Main artifact | Measured result | Correct scope |
| --- | --- | --- | --- |
| 029 | `nan_round029/` | 52.691% cycle accuracy vs 10.966% raw pair under a synthetic gauge shift; conventional triangle ties at 52.691% | Controlled gauge-invariance feature test |
| 030 | `round030/` | 42.0833% native-root classifier after gauge vs 8.2143% raw readout; conventional linear reaches 42.3214% | Native pair-lane training prototype |
| 031 | `nan_round031/` | native 4.00756 / 3.84761 bits per byte vs ordinary 3.89467 / 3.78293 on two seeds | Direct native-root language-training comparison |
| 032 | `round032/` | 50,268 typed facts, 250,000/250,000 posting and hash hits, 2,000/2,000 two-hop hits, 0 direct hits | Exact indexed retrieval benchmark |

The system is cumulative only at the level of shared geometry and storage
discipline. The rounds do **not** form a single trained end-to-end model:

* Round 029 uses batched Python tensors and a proof-of-implementation C++ root.
* Round 030 trains classifier weights directly in the native pair lanes.
* Round 031 inserts a 256-effective-weight native root into a byte-level GRU.
* Round 032 stores retrieval postings beside a materially identical 8 MiB root,
  but does not feed retrieved facts into a decoder.

## 2. Raw NaN payload geometry

`nanlmind_hermes/nan_geometry.py` treats an IEEE-754 binary64 word as an integer
bit pattern, not as a floating-point number. Sign, exponent and payload fields
are preserved; a tagged 48-bit fixed-point value can live inside the payload.
`nanlmind_hermes/nan16de.py` defines a separate 64-bit double-endian envelope
with an exact 32-bit payload and all sixteen Boolean truth tables. These layers
are bit-preserving storage and codec experiments.

The 8 MiB C++ roots in Rounds 029-032 are **different** from that raw payload
layer. They store IEEE `float` words and Q16.16 integers. No Round 032 retrieval
path performs payload-preserving NaN arithmetic. It is therefore incorrect to
describe the native float root as an implementation of the NaN payload codec,
or to infer NaN-specific speed from these benchmarks.

## 3. The full 32,385-cycle basis

For a complete graph on 256 heads, the number of unordered pairs is

```
C(256, 2) = 32,640
```

Choosing head 0 as the root removes 255 root edges from the independent edge
set. The star fundamental-cycle basis therefore has

```
32,640 - 255 = C(255, 2) = 32,385
```

independent cycles. Round 029 used only 48 triangles, and Round 031 exposed only
**256 effective triangle weights** through a 16x16 readout. The remaining
`32,385 - 256 = 32,129` cycle degrees were unused in that language-model run.

`nanlmind_hermes/resident/cycle_observable.py` addresses all 32,385 root cycles
as `x(0,i) + x(i,j) - x(0,j)` modulo `2**32`.
`nanlmind_hermes/resident/full_cycle_training.py` allocates the complete
32,385-word coefficient vector in Q16.16 units, refuses a 256-word truncated
window, and supports sparse feature maps without shrinking the basis.
`FullCycleMemory`, `FullCycleTrainer`, a root-star gauge, a full-width exact
gradient and a snapshot/restore path are present and covered by tests.

The complete basis is an architectural capacity statement, not evidence that
all cycles are usefully trained. Round 032 retrieval is symbolic and does not
train those coefficients from next-token loss.

## 4. Native 8 MiB mirror root

All native rounds use the same record arithmetic:

* 256 head records x 128 bytes = 32,768 bytes.
* 32,640 unordered pair records x 256 bytes = 8,355,840 bytes.
* Total root = 8,388,608 bytes = exactly 8 MiB.
* Each pair record contains independent forward and reverse 128-byte lanes.
* The dense pair index is `i*(511-i)/2 + j-i-1` for `i < j`.

The mirror identity is directional: writing `x(i,j)` and `x(j,i)` reaches two
lanes of one pair cell. The resident Python `ResidentCells` region uses the same
256-head / 32,640-pair counts with Q16.16 rail weights, amplitude, phase,
identity and generation metadata.

Round 032 additionally stores a growing hyperedge table outside the
immutable-location root. This is deliberate: the root is fixed-size and
address-stable, while posting lists, fact ids and adjacency grow dynamically.
The root's float lanes receive a demonstrative triangle update, but Round 032
does not train them from retrieval or language loss.

`mlock` succeeded in the reported Linux container. That is a host-specific
observation, not a guarantee for Android, and 8 MiB is not CPU-register-resident.

## 5. Cycle training and gauge invariance

The invariant is exact for integer words. Let `x(i,j)` be an oriented lane and
let a vertex gauge add `g[i] - g[j]` to every edge. Then

```
x(0,i)+x(i,j)-x(0,j)
-> x(0,i)+g[0]-g[i] + x(i,j)+g[i]-g[j] - x(0,j)-g[0]+g[j]
 = x(0,i)+x(i,j)-x(0,j)
```

All gauge terms telescope. In the float32 native root the same cancellation is
numerical rather than bit-exact. Round 029 reports maximum gauge error
`1.86e-6`; Round 030 reports effective-coefficient error near `4.17e-7` and
101-shift logit change below `3e-4`; Round 031 reports maximum effective-weight
deviation below `8e-7` with unchanged predicted classes.

The raw resident `CycleField` works modulo `2**32`, so its cancellation is
exact for every 32-bit word. `FullCycleTrainer.apply_root_star_gauge` moves pair
lanes while leaving all 32,385 cycle readouts and the trained coefficient digest
unchanged. Round 032's C++ triangle learns `0.0116667 -> 1.01167`; a random gauge
leaves the learned `1.01167` intact to float32 precision.

These gauge shifts are **synthetic and additive by construction**. They test a
specified invariance, not naturally occurring distribution shift.

## 6. Native snapshots

Round 030 and Round 031 use the `NANCYC30` snapshot format: an 8-byte marker
followed by the exact 8,388,608-byte root, for 8,388,616 bytes total. Restore
checks the marker and reads the complete root back. Round 030 reports bitwise
restore and zero maximum logit error; Round 031 reports
`snapshot_max_diff = 0.0` after gradient updates and a gauge shift.

This is experimental same-architecture state: little-endian and
compiler-layout dependent, with no CRC, transaction log, crash recovery or
concurrent-writer protection; it is neither durable user data nor a portable
model format.

Round 032 has no persistent hyperedge snapshot in its C++ archive. Persistence
is provided instead by the resident universe described below, which is a
different container and file format.

## 7. Sparse retrieval postings in Round 032

Round 032 hashes a concept id into an anchored head pair:

```
k = splitmix64(concept_id + 0x9e3779b97f4a7c15)
a = 1 + k mod 255
b = 1 + (k >> 15) mod 255
canonical = sorted(a, b)
```

The anchored pair owns an inverted posting list of typed facts. A fact records
source, destination, relation, anchor pair, strength and the next posting id.
Lookup uses the pair index to reach the head of the list, then compares explicit
source and relation fields to resolve collisions. The comparing hash map indexes
the same 50,268 facts.

Published measurements:

| Quantity | Value |
| --- | ---: |
| Stored facts | 50,268 |
| Membership entries counted | 50,268 |
| Pair-map audit | pass |
| Root bytes | 8,388,608 |
| Query repetitions | 250,000 |
| Posting hits | 250,000 |
| Hash hits | 250,000 |
| Posting time | 9.77183 ms |
| Hash time | 2.29752 ms |

The hash comparator is faster in this microbenchmark. There is **no unique
performance advantage** established for the mirror geometry. The posting lists
are headed in a separate fixed index array, not embedded directly in the root
record, and there is no concurrent writer or WAL in this round.

The archive contains 49,321 distinct `(source, relation)` keys; 937 keys have
more than one destination. For unique keys the Python integration test checks
that posting and hash lookup return the same fact. For duplicate keys it checks
that both paths return a member of the same destination set. This distinction is
important: the C++ posting walk returns the newest record while the exact hash
map retains the first insertion for a repeated key.

## 8. Two-hop traversal and long-context control

`queries.tsv` contains 2,000 chains sampled from already stored links. The C++
traversal follows source -> middle -> goal with the same relation type and
verifies the goal destination. All 2,000 queries succeed; `direct_found` is 0,
so these are genuine two-edge chains rather than repeated parent lookups. This
is symbolic graph lookup, not learned composition.

The long-context test inserts 512 synthetic distractor facts between an original
mention and a query. Results:

| Condition | Correct |
| --- | ---: |
| Short recent context only | 0 / 1,000 |
| Pair-index conceptual retrieval | 1,000 / 1,000 |
| Hash-index retrieval | 1,000 / 1,000 |

Both indexes retrieve facts outside the short window. The control shows that a
window-limited reader cannot recover facts absent from its window; it does not
test a neural decoder, attention mechanism or generative answer quality.

## 9. Typed hyperedges and retrieval persistence

`nanlmind_hermes/resident/cycle_hypergraph.py` represents a fundamental cycle as
a typed hyperedge whose members are references to three pair cells, a cycle
leaf, vertices, a value and provenance. Canonical cycle ids are deterministic,
repeated identical content is idempotent, conflicting content is rejected, and
the layer never rewrites pair cells to repair identity.

The resident universe stores relations as hyperedges with leaf, pair, mirror and
relation members. Round 032 integration tests append eight typed hypernym
relations derived from `facts.tsv`, read them by subject and predicate, close the
universe, reopen the same file, and verify unchanged generation and digest,
surviving payloads and hyperedge ids, retrieval of the expected object, three
leaf members plus pair/mirror/relation members, bounded pair and mirror ids, and
counts of 32,640 pairs and 32,385 root cycles.

This proves repository persistence of typed retrieval records. It does not make
the C++ Round 032 posting table crash-safe, concurrent or transactionally
durable.

## 10. Field and scratch-first heads

`resident/head_field.py` gives every head an independent activation score and
derives a normalized 256-entry field with entropy, active-set and peak metrics.
`resident/scratch_layers.py` gives every head ten L0-L9 scratch layers. A scratch
proposal is immutable and staged; committing to a main slot requires explicit
approval and a generation compare-and-swap.

The Round 032 integration test retrieves a fact, activates its anchored heads,
stores the anchored triangle, applies a 256-word gauge, and checks unchanged
cycle readout with zero gauge residual, 64 bit-identical sampled root cycles,
byte-identical 256 head records, and unchanged activation, identity and field
metrics.

This is a real cross-layer contract at the data-arithmetic level. It is not a
learned coupling between retrieval, head field and decoder.

## 11. Temporal language model

`nanlmind_hermes/temporal_lm.py` maintains a bounded temporal word graph with
recency weighting, beam completion and correction/statistics hooks. It can be
exercised independently of the resident cycle geometry. Round 031 separately
trains a byte-level GRU whose memory matrix is replaced by the native root.

The repository still lacks the connection implied by "retrieval-augmented
language model": Round 032 facts are not injected into the temporal LM or the
Round 031 decoder, and retrieved hyperedges do not condition next-token logits.
Any description of the current system as an end-to-end retrieval-augmented
generator would overstate the evidence.

## 12. Resident universe and tools

`resident/universe.py` provides a bounded one-file memory-mapped universe with
head, pair, leaf, relation, hyperedge, payload, journal, feature and scratch
sections. It tracks generation and digest, validates section bounds, supports
reopen, and reports `canonical_merge = {enabled: false, attempted: false}`.

Persistence shapes are distinct: `NANCYC30` is a fixed 8 MiB native float/Q16
root; the resident universe is a variable container for leaves, typed relations,
hyperedges and retrieval records; cycle persistence and snapshot modules store
deterministic resident cycle state and bounded projections, not a memory image.

`resident/cycle_tools.py` and `resident/field_tools.py` expose bounded handlers
with approval and size limits. Tool calls can observe or propose, but canonical
merge, unrestricted main-state mutation and unbounded scans are refused.
`FullCycleTrainer` receipts and hypergraph audits both carry
`canonical_merge = false`; the hypergraph's `merge`/`promote` path raises
`CycleDenied` with code `canonical_merge_disabled`.

## 13. Total integrated data flow

The current, honest flow is:

```
raw NaN payload modules --> raw bit geometry/codecs (separate from float roots)
        v
resident cells: 256 heads, 32,640 pair cells, two rails
        +--> cycle observable: 32,385 exact modulo-2**32 root cycles
        +--> head field: 256 independent activations
        +--> scratch layers: staged proposals, approved main commits
        v
full-cycle trainer: all 32,385 Q16.16 coefficients
        +--> native root: 8 MiB float32 prototype; NANCYC30 snapshot
        +--> resident universe: leaves, typed hyperedges, retrieval postings
        v
Round 032 retrieval: exact known-link postings and two-hop lookup
```

The arrows describe shared layouts and tested adapters, not one jointly
optimized model.

## 14. Acceptance gates

`tests/test_round032_total_integration.py` runs sixteen gates:

| Gate | Contract |
| --- | --- |
| SHA manifest | Seven listed files match `SHA256SUMS` |
| Geometry | 8 MiB = 256 head records + 32,640 pair records |
| Native results | Counts, times, cycle update and gauge are self-consistent |
| Inputs | 50,268 facts and 2,000 queries parse as typed triples |
| README claims | Limitations remain explicit and bounded |
| Mirror map | 32,640 unique symmetric pair indexes |
| Posting/hash equality | Unique keys agree; duplicates agree on membership |
| 250k query pattern | Both indexes find all 250,000 sampled lookups |
| Two-hop | 2,000 chains found, 0 direct hits |
| Long context | Published 1,000/1,000 retrieval and 0/1,000 short control reproduced |
| Hyperedge persistence | Eight typed relations survive universe reopen |
| Cycle/field integration | Gauge preserves cycle, memories, field and head bytes |
| Full basis | 32,385 coefficients accepted; 256-word windows refused |
| Full training | Sparse step, gauge and snapshot keep full dimension |
| Round 031 contrast | Published 256-weight, 32,129-unused window is confirmed |
| Canonical merge | False in basis, snapshot, hypergraph and trainer receipts |

Observed: `python3 -m unittest tests.test_round032_total_integration -v`
reports `Ran 16 tests in 10.756s`, `OK`. That is 16 passed, 0 skipped and 0
errors. Lazy skips exist for genuinely absent optional modules or artifacts;
none were absent in this checkout.

## 15. Reproducible commands

```bash
# Round 029
cd experiments/round029/nan_round029
python experiment.py --seeds 17 29 41
python strong_control.py
g++ -std=c++20 -O3 -Wall -Wextra native_mirror.cpp -o native_mirror && ./native_mirror

# Round 030
cd experiments/round030
c++ -std=c++20 -O3 -shared -fPIC native_cycle.cpp -o libnative_cycle.so
OPENBLAS_NUM_THREADS=2 python run.py --seeds 17 29 41
OPENBLAS_NUM_THREADS=2 python tune.py
OPENBLAS_NUM_THREADS=2 python native_audit.py

# Round 031
cd experiments/round031/nan_round031
c++ -std=c++20 -O3 -fPIC -shared native_cycle.cpp -o libnative_cycle.so
OPENBLAS_NUM_THREADS=1 python train.py --steps 100 --seeds 17 29
OPENBLAS_NUM_THREADS=1 python audit.py
OPENBLAS_NUM_THREADS=1 python transfer.py
OPENBLAS_NUM_THREADS=1 python final_audit.py

# Round 032
cd experiments/round032
python prepare.py /path/to/round025/corpus.jsonl --out data --limit 50000
c++ -O3 -std=c++20 -Wall -Wextra native_recursive.cpp -o native_recursive
./native_recursive data/facts.tsv data/queries.tsv > results.json
python long_context.py
sha256sum -c SHA256SUMS
cd ../..
python3 -m unittest tests.test_round032_total_integration -v
```

Rounds 029-031 paths refer to the prior Round 025 WordNet corpus and Round 022
Alice files. Those external inputs are intentionally not duplicated; edit the
documented path constants when reproducing elsewhere.

## 16. Honest non-claims

* **Hash is faster.** The conventional hash map takes 2.29752 ms for 250,000
  lookups versus 9.77183 ms for the posting walk. No mirror-geometry speed
  advantage is claimed.
* **No decoder integration yet.** Retrieved facts do not condition a neural
  decoder, the temporal LM or the Round 031 GRU. That remains the end-to-end
  task.
* **Conventional models win or tie.** Round 031's ordinary matrix has lower bits
  per byte on both seeds; Round 030's conventional linear classifier is slightly
  stronger; Round 029's conventional triangle ties because it computes the same
  invariant.
* **Gauge shifts are synthetic.** The invariance is real for the specified
  additive transform, but naturally gauge-shifted language data was not measured.
* **No quantum effect.** Nothing here uses quantum hardware, amplitudes or
  speedups. "Gauge" denotes an algebraic redundancy, not physics.
* **No learned retrieval generalization.** Round 032 retrieves links already in
  the fact table. Sampling queries from stored links is not unseen-word inference.
* **No canonical truth or portable speed.** A typed hyperedge records provenance
  and membership, not canonical truth; merge is disabled. Native timings are
  single-host observations, so `mlock` success and microsecond reads do not imply
  Android throughput, energy efficiency or register residence.
* **No fully trained end-to-end system.** The full 32,385-cycle trainer exists,
  but Round 032 does not train it from next-token loss and Round 031 does not
  retrieve from Round 032 postings.

## 17. Remaining integration work

The next defensible steps are: feed typed hyperedges into a decoder and evaluate
book-disjoint language modeling; train all 32,385 cycles with matched optimizers
and budgets; embed posting heads and add crash-safe snapshots; measure target
Android hardware; and report unseen-link generalization separately from stored
fact lookup.

The integration claim is therefore narrow: the repository contains consistent
mirror-pair geometry, exact integer cycle observables, a complete 32,385-cycle
training module, native-root prototypes, persistent typed hyperedges and a
reproducible exact retrieval benchmark. It does not yet contain a demonstrated
retrieval-augmented language-model advantage.
