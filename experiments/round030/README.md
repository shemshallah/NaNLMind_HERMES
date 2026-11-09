# Round 030 — Native gauge-invariant semantic learning

## What this round actually establishes

This is a **native-root-trained** semantic classifier. Unlike Round 029, the model parameters themselves occupy pair-lane floats inside one 8,388,608-byte C++ allocation. At each supervised example, forward logits read anchored triangle circulations directly from the root; cross-entropy gradients update the three involved edge values, maintaining exact antisymmetry of mirror directions. The 256-head layout, 32,640 unique pairs and dual independent 128-byte directional lanes are retained.

This is a **classical** geometry test, not a quantum experiment or a natural-language generator. It does *not* show that IEEE NaN encodings inherently accelerate learning: records hold float32 values in a C++ struct, not raw payload-preserving NaN arithmetic. Input features are learned outside the root using training-only TF-IDF/TruncatedSVD, and a conventional linear classifier has comparable or stronger accuracy.

## Source and evaluation

Source: the user-uploaded `wn3.1.dict.zip` converted to `nan_round025/corpus.jsonl` (117,570 usable WordNet records in the prior round). Per seed, 16 well-represented noun categories, 150 training, 35 validation and 35 test synsets per category. All partitions are **synset-disjoint**. TF-IDF (including vocabulary) and 24D TruncatedSVD are fitted only on training glosses. Held-out test sets contain 560 synsets per seed.

Three seeds: 17, 29, 41. Native softmax SGD hyperparameters chosen by **validation cross-entropy**, not test performance, from four training schedules. Native classifier: 16 categories × 24 learned dimensions = 384 gauge-invariant effective weights, each encoded as an oriented triangle circulation in the physical pair lanes. Its expanded 3-edge representation has gauge redundancy, which is not additional independent learnable capacity. An ordinary 16×24 linear classifier (no intercept) is the primary conventional control.

### Measurements

- Mean tuned test accuracy before and after a large synthetic gauge shift: **42.0833%**; identical class predictions (three seeds).
- Raw directed-pair readout after this gauge shift: **8.2143%** accuracy (same trained root, no retraining).
- Ordinary no-intercept logistic regression on the same clean text features: **42.3214%** (separate experiment in `results.json`).
- Native mathematical gradient check: absolute difference from analytical update **1.86e-9**.
- Invariance after one gauge shift: max effective coefficient error **4.17e-7** in the native audit.
- Invariance after 101 successive gauge shifts: max logit difference < **3e-4**, float32, and class predictions unchanged in the test.
- Mirror antisymmetry passed. Bitwise snapshot/restore passed.
- ctypes 1,000-prediction wall time in this container: **7.74 ms**, includes Python crossing overhead, not a native batched benchmark. Hardware dependent.
- Model weights remain recoverable from snapshots `root_{seed}.bin`, without separately saving a learned parameter tensor.

The gauge disturbances are **artificial and intentionally chosen** to satisfy additive vertex-gradient structure. The result demonstrates a meaningful, rigorously specified *invariance* advantage over a raw pair readout, but it does **not** prove better semantic comprehension, general language modeling, memory efficiency, or supremacy over a correct ordinary graph implementation of the identical triangle invariant.

## Reproduce

Requires C++20, NumPy, scikit-learn and the WordNet corpus JSONL from Round 025.

```bash
c++ -std=c++20 -O3 -shared -fPIC native_cycle.cpp -o libnative_cycle.so
OPENBLAS_NUM_THREADS=2 python run.py --seeds 17 29 41
OPENBLAS_NUM_THREADS=2 python tune.py
OPENBLAS_NUM_THREADS=2 python native_audit.py
```

The example code reads the corpus at `/mnt/data/nan_round025/corpus.jsonl`. Edit `DATA` in `run.py` to match its local location after extracting. Source references remain WordNet 3.1 rather than embedding the entire original dictionary into this package.

Snapshot `NANCYC30` consists of an 8-byte marker plus the exact 8 MiB raw native layout. It is intended for this experimental same-architecture prototype only; it is not concurrency-safe, endian-neutral, transactional or crash-safe. Locking is optional (`nan_new(0)` for reproducibility); the container's RAM residency is not a guarantee of CPU register residence.

## Next experiments that could actually establish a language benefit

1. Jointly train the text encoder and native pair-operator field using batched sparse backward writes (currently encoder is frozen after SVD).
2. Train an autoregressive decoder conditioning on the effective cycle readouts and *typed, retrieved* hyperedges. Use full-book-disjoint Gutenberg evaluation and compare to a model with conventional gauge-invariant features.
3. Compare graph regularization and geometry with equal FLOPs/parameters against strong GNN/recurrent/attention baselines. Include read/write bandwidth, cache misses, Android energy, and allocation costs.
4. Corpus ingestion with full provenance, stable concept IDs, dynamic graph growth and WAL/CRC snapshots; preserve opposed assertions as attributed claims.
5. Use the already-defined native NaN16 VM for future guest-side computing, distinct from this C++ host prototype.
