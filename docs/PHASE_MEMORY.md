# Phase memory: gauge invariance without quantum claims

## What Round 011 established

- 64 prototypes × 64 phase channels × 3,000 queries with a random global
  phase per query: gauge-invariant retrieval **100%**, naive phase-sensitive
  retrieval **41.37%**.
- A 128-channel real cosine/sine implementation agreed with the complex
  implementation on **all 3,000 predictions** (maximum score difference
  ~2.2e-16).
- Independent phase noise eventually destroys recall. In the 32-channel,
  1,024-pattern sweep, gauge accuracy was 100% at σ=0.25 rad, 95.8% at σ=1.0,
  and 0.7% at σ=2.0 (near chance).
- A WordNet 26-category text classifier at 68.09% accuracy produced identical
  predictions after reversible complex phase encoding. Encoding ordinary
  probabilities as phases is not independent evidence of cycle learning.
- Bell/CHSH and interferometer results were classical simulations; they do not
  demonstrate physical quantum behavior in NaN memory.

## Hermes implementation

`nanlmind_hermes/phase_memory.py` stores unit phasors and scores:

```
gauge:  |Σ_j conj(q_j) p_j|^2
naive:   Re Σ_j conj(q_j) p_j
real:   (Σ cos(q_j-p_j))^2 + (Σ sin(q_j-p_j))^2
```

The gauge score is invariant under a global phase; the real score is exactly
the same computation in cosine/sine channels. This is classical signal
processing and a useful memory mechanism, not quantum computation.

Small deterministic sweep (`scripts/phase_memory_bench.py`, 64 patterns, three
seeds, 60 queries each):

| dimensions | noise σ (rad) | gauge | naive | real agreement |
|---:|---:|---:|---:|---:|
| 16 | 0.25 | 1.000 | 0.383 | 1.000 |
| 16 | 1.00 | 0.839 | 0.289 | 1.000 |
| 16 | 2.00 | 0.033 | 0.028 | 1.000 |
| 32 | 0.25 | 1.000 | 0.417 | 1.000 |
| 32 | 1.00 | 0.994 | 0.322 | 1.000 |
| 32 | 2.00 | 0.072 | 0.056 | 1.000 |
| 64 | 0.25 | 1.000 | 0.444 | 1.000 |
| 64 | 1.00 | 1.000 | 0.383 | 1.000 |
| 64 | 2.00 | 0.122 | 0.078 | 1.000 |

These numbers match the external pattern: global phase is nuisance, independent
phase noise is the limit, and real channels are sufficient.

## Persistent gauge-preserving state

`nanlmind_hermes/cycle_state.py` keeps complex edge values and projects every
update onto the divergence-free cycle subspace. Adding an arbitrary vertex
potential and reprojecting leaves the stored cycle content unchanged. A
snapshot/restore path supports persistence.

This is the memory layer; it does not create information and does not yet learn
edge operators.

## Next

Train typed edge operators from real WordNet relations, compose them only where
the relation semantics justify composition, and compare against a matched model
without the cycle/composition objective. The decisive target is new-word
definition inference, not a mathematically elegant loop.
